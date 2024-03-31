import torch
import torch.nn.functional as F
import torch.nn as nn
import util

from models.networks.stylegan2_layers import ResBlock, ConvLayer, EqualLinear





class StyleGAN2ResnetEncoder(torch.nn.Module):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument("--netE_scale_capacity", default=1.0, type=float)
    #     parser.add_argument("--netE_num_downsampling_sp", default=4, type=int)
    #     parser.add_argument("--netE_num_downsampling_gl", default=2, type=int)
    #     parser.add_argument("--netE_nc_steepness", default=2.0, type=float)
    #     return parser

    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1]

        self.from_rgb = nn.Sequential()
        self.from_rgb.add_module("FromRGB", ConvLayer(3, self.nc(0), 1))

        self.till_mid_point = nn.Sequential()
        for i in range(self.opt.netE_num_downsampling_sp):
            self.till_mid_point.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True)
            )

        # spatial features
        nchannels = self.nc(self.opt.netE_num_downsampling_sp)
        self.to_spatial_code = nn.Sequential()
        self.to_spatial_code.add_module("ToSpatialCode1",ConvLayer(nchannels,nchannels,
                                                                  1,activate=True,bias=True))
        self.to_spatial_code.add_module("ToSpatialCode2", ConvLayer(nchannels, self.opt.spatial_code_ch,
                                                                   1, activate=False, bias=True))

        # Global features
        self.to_global_code = nn.Sequential()
        for i in range(self.opt.netE_num_downsampling_gl):
            idx_from_beginning = self.opt.netE_num_downsampling_sp + i
            self.to_global_code.add_module(
                "ConvLayerDownBy%d" % (2 ** idx_from_beginning),
                ConvLayer(self.nc(idx_from_beginning),
                          self.nc(idx_from_beginning + 1), kernel_size=3,
                          blur_kernel=[1], downsample=True, pad=0)
            )

        nchannels = self.nc(self.opt.netE_num_downsampling_sp +
                            self.opt.netE_num_downsampling_gl)
        self.global_code_output = nn.Sequential()
        self.global_code_output.add_module("output_dense",EqualLinear(nchannels, self.opt.global_code_ch))

    def nc(self, idx):
        nc = self.opt.netE_nc_steepness ** (5 + idx)
        nc = nc * self.opt.netE_scale_capacity
        nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)


    def forward(self, x, extract_features=False):
        x = self.from_rgb(x)
        midpoint = self.till_mid_point(x)
        sp = self.to_spatial_code(midpoint)

        if extract_features:
            padded_midpoint = F.pad(midpoint, (1, 0, 1, 0), mode='reflect')
            feature = self.to_global_code[0](padded_midpoint)
            assert feature.size(2) == sp.size(2) // 2 and \
                feature.size(3) == sp.size(3) // 2
            feature = F.interpolate(
                feature, size=(7, 7), mode='bilinear', align_corners=False)

        x = self.to_global_code(midpoint)
        x = x.mean(dim=(2, 3))
        gl = self.global_code_output(x)

        sp = util.normalize(sp)
        gl = util.normalize(gl)
        if extract_features:
            return sp, gl, feature
        else:
            return sp, gl
