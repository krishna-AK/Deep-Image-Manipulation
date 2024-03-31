import torch
import torch.nn.functional as F
import torch.nn as nn

import util
from models.CapsuleEncoder import PrimaryCapsuleLayer, CapsuleLayer, ReshapeCapsuleOutput
from models.networks.stylegan2_layers import ConvLayer, ResBlock, EqualLinear


# Assuming ResBlock, ConvLayer, EqualLinear are already defined as before
# Assuming PrimaryCapsuleLayer and CapsuleLayer are defined based on previous discussions

class CapsuleBasedEncoder(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        blur_kernel = [1, 2, 1]

        # From RGB
        self.from_rgb = nn.Sequential(
            ConvLayer(3, self.nc(0), 1)
        )

        # Till Mid Point (remains unchanged)
        self.till_mid_point = nn.Sequential()
        for i in range(self.opt.netE_num_downsampling_sp):
            self.till_mid_point.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel, reflection_pad=True)
            )

        # Replace spatial code with Capsule Network and include ReshapeCapsuleOutput
        nchannels = self.nc(self.opt.netE_num_downsampling_sp)
        self.to_spatial_code = nn.Sequential(
            ConvLayer(nchannels, nchannels, 3, activate=True, bias=True, downsample=False),
            PrimaryCapsuleLayer(num_capsules=32, in_channels=nchannels, out_channels=32, kernel_size=9, stride=2),
            CapsuleLayer(num_capsules=self.opt.spatial_code_ch // 16, num_route_nodes=32 * 6 * 6, in_channels=8, out_channels=8, routing_iterations=3),
            # Here we add the ReshapeCapsuleOutput module
            # Assuming the target dimensions based on your network's requirements
            ReshapeCapsuleOutput(target_height=8, target_width=8)  # Adjust target_height and target_width as needed
        )

        # Global features part remains unchanged
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
        self.global_code_output.add_module("output_dense", EqualLinear(nchannels, self.opt.global_code_ch))


    def nc(self, idx):
        nc = self.opt.netE_nc_steepness ** (5 + idx)
        nc = nc * self.opt.netE_scale_capacity
        nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)

    def forward(self, x, extract_features=False):
        x = self.from_rgb(x)
        midpoint = self.till_mid_point(x)

        # For capsule network, ensure to reshape or process the output accordingly
        sp = self.to_spatial_code(midpoint.view(midpoint.size(0), -1, midpoint.size(-1)))

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
