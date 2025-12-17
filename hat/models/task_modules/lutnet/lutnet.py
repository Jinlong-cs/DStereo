import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from .base_module import SepLUT
from .function import Interp1d, Quantize


def sample_pdf(lut_size, bins, cdf):
    u = torch.linspace(0, 1, lut_size).float().to(bins.device)
    u = u[1:-1]  # exclude the first 0 and the last 1

    inds = torch.searchsorted(cdf, u, right=False)

    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, lut_size - 1)

    inds_sampled = torch.stack([below, above], dim=1)
    inds_sampled = inds_sampled.view(
        -1,
    )
    cdf_g = torch.gather(cdf, 0, inds_sampled)
    cdf_g = cdf_g.view(-1, 2)
    bins_g = torch.gather(bins, 0, inds_sampled)
    bins_g = bins_g.view(-1, 2)
    denom = cdf_g[:, 1] - cdf_g[:, 0]
    denom[denom < 1e-5] = 1

    samples = bins_g[:, 0] + (u - cdf_g[:, 0]) / denom * (
        bins_g[:, 1] - bins_g[:, 0]
    )

    return samples


@OBJECT_REGISTRY.register
class LutNet(nn.Module):
    """
    A module of LutNet.

    Args:
        in_channels (int): The number of input channels.
        output_bit (int): The bit depth of the output image.
        lut_size (int): The number of vertices in the LUT.
        down_size (tuple): The size of the downsampled image.
        pregamma (float): The gamma correction factor.
        track_coarse_steps (int): The number of steps to track the coarse LUT.
        norm_symmetric (bool): Whether to normalize the output image to \
            [-1, 1].
        alpha (float): The momentum of the moving average.
    """

    def __init__(
        self,
        in_channels,
        output_bit=16,
        lut_size=64,
        down_size=None,
        pregamma=None,
        track_coarse_steps=None,
        norm_symmetric=False,
        alpha=0.9,
    ) -> None:
        super(LutNet, self).__init__()
        self.in_channels = in_channels
        self.output_bit = output_bit
        self.lut_size = lut_size
        self.down_size = down_size
        self.pregamma = pregamma
        self.track_coarse_steps = track_coarse_steps
        self.norm_symmetric = norm_symmetric
        self.alpha = alpha

        self.lut = SepLUT(
            in_channels,
            n_vertices=lut_size - 1,  # exclude the first 0
            n_base_feats=8,
            fine_sample=True,
        )

        coarse_y = self._lut_init(lut_size - 1)
        coarse_y = torch.tensor(coarse_y).float()
        coarse_x = torch.linspace(0, 1, lut_size).float()
        fine_y = self._lut_init(lut_size - 1)
        fine_y = torch.tensor(fine_y).float()
        fine_x = torch.linspace(0, 1, lut_size).float()

        self.register_buffer("coarse_x", coarse_x)
        self.register_buffer("coarse_y", coarse_y)
        self.register_buffer("fine_x", fine_x)
        self.register_buffer("fine_y", fine_y)

        self.register_buffer("track_coarse_step", torch.tensor(0).int())

        self.interp = Interp1d.apply
        self.quantize = Quantize.apply

    def _lut_init(self, lut_size):
        x = np.linspace(0, 2 ** 20 - 1, lut_size)
        y = 9.05 * np.power(10, np.log10(x * 50 + 1) / 2)
        y = y / (2 ** 16 - 1)
        return y

    def forward(self, img):
        one = torch.ones(1).to(img.device)
        zero = torch.zeros(1).to(img.device)

        if self.pregamma is not None:
            img = torch.pow(img, self.pregamma)

        if not self.training:
            fine_y = torch.cat([zero, self.fine_y], dim=0)
            fine_img = self.interp(img, self.fine_x, fine_y)
            if self.norm_symmetric:
                fine_img = (fine_img - 0.5) * 2

            return fine_img

        if self.down_size is not None:
            x = F.interpolate(
                img, size=self.down_size, mode="bilinear", align_corners=False
            )
        else:
            x = img

        lut_coarse, lut_fine = self.lut(x)

        if (
            self.track_coarse_steps is None
            or self.track_coarse_step < self.track_coarse_steps
        ):
            lut_coarse = torch.cumsum(lut_coarse, dim=1) / torch.sum(
                lut_coarse, dim=1, keepdim=True
            )
            lut_coarse_center = lut_coarse.mean(dim=0)
            lut_coarse_center = (
                self.coarse_y * (1 - self.alpha)
                + lut_coarse_center * self.alpha
            )
            lut_coarse_center_ = torch.cat([zero, lut_coarse_center], dim=0)

            if self.track_coarse_steps is not None:
                self.track_coarse_step = self.track_coarse_step + 1
            self.coarse_y = lut_coarse_center.detach().clone()

            fine_x = sample_pdf(
                self.lut_size, self.coarse_x, lut_coarse_center_.detach()
            )
            fine_x = torch.cat([zero, fine_x, one], dim=0)
            self.fine_x = fine_x.detach().clone()
        else:
            lut_coarse_center = self.coarse_y
            lut_coarse_center_ = torch.cat([zero, lut_coarse_center], dim=0)
            fine_x = self.fine_x

        lut_fine = torch.cumsum(lut_fine, dim=1) / torch.sum(
            lut_fine, dim=1, keepdim=True
        )
        lut_fine_center = lut_fine.mean(dim=0)
        lut_fine_center = (
            self.fine_y * (1 - self.alpha) + lut_fine_center * self.alpha
        )
        self.fine_y = lut_fine_center.detach().clone()

        lut_fine_center_ = torch.cat([zero, lut_fine_center], dim=0)

        if (
            self.track_coarse_steps is None
            or self.track_coarse_step < self.track_coarse_steps
        ):
            fuse_x = torch.zeros(2 * self.lut_size - 2).to(img.device)
            fuse_x[0 : self.lut_size] = self.coarse_x
            fuse_x[self.lut_size :] = fine_x[1:-1]
            fuse_y = torch.zeros(2 * self.lut_size - 2).to(img.device)
            fuse_y[0 : self.lut_size] = lut_coarse_center_
            fuse_y[self.lut_size :] = lut_fine_center_[1:-1]

            sort_fuse_x, sort_index = torch.sort(fuse_x)
            sort_fuse_y = fuse_y[sort_index]

            img = self.interp(img, sort_fuse_x, sort_fuse_y)
            img = self.quantize(img, self.output_bit)
        else:
            img = self.interp(img, fine_x, lut_fine_center_)
            img = self.quantize(img, self.output_bit)

        if self.norm_symmetric:
            img = (img - 0.5) * 2

        return img

    def set_qconfig(self):
        self.qconfig = None
