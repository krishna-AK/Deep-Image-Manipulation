import math

import torch
import torch.nn.functional as F
import util

from models.networks.stylegan2_layers import StyledConv, ConvLayer, EqualLinear, ToRGB


class UpsamplingBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim,
                 blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True,
                                blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False,
                                use_noise=use_noise)

    def forward(self, x, style):
        return self.conv2(self.conv1(x, style), style)


class ResolutionPreservingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim):
        super().__init__()
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=False)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=False, bias=False)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = self.skip(x)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)


class UpsamplingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim, blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True, blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False, use_noise=use_noise)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=True, bias=True)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)


class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)


class StyleGAN2ResnetGenerator(torch.nn.Module):


    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument("--netG_scale_capacity", default=1.0, type=float)
    #     parser.add_argument(
    #         "--netG_num_base_resnet_layers",
    #         default=2, type=int,
    #         help="The number of resnet layers before the upsampling layers."
    #     )
    #     parser.add_argument("--netG_use_noise", type=util.str2bool, nargs='?', const=True, default=True)
    #     parser.add_argument("--netG_resnet_ch", type=int, default=256)
    #
    #     return parser

    def __init__(self,opt):
        super().__init__()

        self.spatial_code_ch = opt.spatial_code_ch
        self.global_code_ch = opt.global_code_ch
        self.num_classes = opt.num_classes
        self.netG_num_base_resnet_layers = opt.netG_num_base_resnet_layers
        self.netG_use_noise = opt.netG_use_noise
        self.netG_scale_capacity = opt.netG_scale_capacity
        self.netG_no_of_upsamplings = opt.netG_no_of_upsamplings

        num_upsamplings =   self.netG_no_of_upsamplings
        blur_kernel = [1, 3, 3, 1]

        self.global_code_ch = self.global_code_ch + self.num_classes

        self.add_module(
            "SpatialCodeModulation",
            GeneratorModulation(self.global_code_ch, self.spatial_code_ch))

        in_channel = self.spatial_code_ch
        for i in range(self.netG_num_base_resnet_layers):
            # gradually increase the number of channels
            out_channel = (i + 1) / self.netG_num_base_resnet_layers * self.nf(0)
            out_channel = max(self.spatial_code_ch, round(out_channel))
            layer_name = "HeadResnetBlock%d" % i
            new_layer = ResolutionPreservingResnetBlock(
                in_channel, out_channel, self.global_code_ch)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        for j in range(num_upsamplings):
            out_channel = self.nf(j + 1)
            layer_name = "UpsamplingResBlock%d" % (2 ** (4 + j))
            new_layer = UpsamplingResnetBlock(
                in_channel, out_channel, self.global_code_ch,
                blur_kernel, self.netG_use_noise)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        last_layer = ToRGB(out_channel, self.global_code_ch,
                           blur_kernel=blur_kernel)
        self.add_module("ToRGB", last_layer)

    def nf(self, num_up):
        ch = 64 * (2 ** (self.netG_no_of_upsamplings - num_up))
        ch = int(min(256, ch) * self.netG_scale_capacity)
        return ch

    def forward(self, spatial_code, global_code):
        spatial_code = util.normalize(spatial_code)
        global_code = util.normalize(global_code)

        x = self.SpatialCodeModulation(spatial_code, global_code)
        for i in range(self.netG_num_base_resnet_layers):
            resblock = getattr(self, "HeadResnetBlock%d" % i)
            x = resblock(x, global_code)

        for j in range(self.netG_no_of_upsamplings):
            key_name = 2 ** (4 + j)
            upsampling_layer = getattr(self, "UpsamplingResBlock%d" % key_name)
            x = upsampling_layer(x, global_code)
        rgb = self.ToRGB(x, global_code, None)

        return rgb