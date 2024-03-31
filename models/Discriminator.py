# from models.networks import BaseNetwork
import torch.nn

from models.networks.stylegan2_layers import Discriminator as OriginalStyleGAN2Discriminator


class StyleGAN2Discriminator(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.stylegan2_D = OriginalStyleGAN2Discriminator(
            opt.crop_size,
            2.0,
            blur_kernel=[1, 3, 3, 1]
        )

    def forward(self, x):
        pred = self.stylegan2_D(x)
        return pred

    def get_features(self, x):
        return self.stylegan2_D.get_features(x)

    def get_pred_from_features(self, feat, label):
        assert label is None
        feat = feat.flatten(1)
        out = self.stylegan2_D.final_linear(feat)
        return out


