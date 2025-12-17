import torch.nn as nn


class BasicBlock(nn.Module):
    """
    A basic block for SepLUT.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The kernel size of the convolutional layer.
        stride (int): The stride of the convolutional layer.
        norm (bool): Whether to use the instance normalization.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, norm=False
    ) -> None:
        super(BasicBlock, self).__init__()
        list = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        if norm:
            list.append(nn.InstanceNorm2d(out_channels, affine=True))

        self.model = nn.Sequential(*list)

    def forward(self, x):
        return self.model(x)


class SepLUT(nn.Module):
    """
    A modified implementation of "SepLUT: Separable Image-adaptive \
    Lookup Tables for Real-time Image Enhancement".

    Args:
        in_channels (int): The number of input channels.
        n_vertices (int): The number of vertices in the LUT.
        n_base_feats (int): The channel number of base features.
        fine_sample (bool): Whether to sample the fine LUT.
        coarse_vertices (int): The number of vertices in the coarse LUT.
        fine_vertices (int): The number of vertices in the fine LUT.
    """

    def __init__(
        self,
        in_channels,
        n_vertices,
        n_base_feats=8,
        fine_sample=False,
        coarse_vertices=None,
        fine_vertices=None,
    ) -> None:
        super(SepLUT, self).__init__()
        self.n_vertices = n_vertices
        self.n_base_feats = n_base_feats
        self.fine_sample = fine_sample

        backbone = [
            BasicBlock(
                in_channels, n_base_feats, kernel_size=3, stride=2, norm=True
            )
        ]
        n_feats = n_base_feats
        for _ in range(3):
            backbone.append(
                BasicBlock(
                    n_feats, n_feats * 2, kernel_size=3, stride=2, norm=True
                )
            )
            n_feats *= 2
        backbone.append(BasicBlock(n_feats, n_feats, kernel_size=3, stride=2))
        backbone.append(nn.Dropout(0.5))
        backbone.append(nn.AdaptiveAvgPool2d(2))
        self.backbone = nn.Sequential(*backbone)

        out_channels = n_feats * 2 * 2

        if self.fine_sample:
            if coarse_vertices is None:
                self.lut_coarse = nn.Sequential(
                    nn.Linear(out_channels, n_vertices), nn.Sigmoid()
                )
                self.lut_fine = nn.Sequential(
                    nn.Linear(out_channels, n_vertices), nn.Sigmoid()
                )
            else:
                assert fine_vertices is not None, "fine_vertices should be set"
                self.lut_coarse = nn.Sequential(
                    nn.Linear(out_channels, coarse_vertices), nn.Sigmoid()
                )
                self.lut_fine = nn.Sequential(
                    nn.Linear(out_channels, fine_vertices), nn.Sigmoid()
                )
        else:
            self.lut1d_generator = nn.Sequential(
                nn.Linear(out_channels, n_vertices), nn.Sigmoid()
            )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        if self.fine_sample:
            lut_coarse = self.lut_coarse(x)
            lut_fine = self.lut_fine(x)
            return lut_coarse, lut_fine
        else:
            lut1d = self.lut1d_generator(x)
            return lut1d
