# Copyright (c) Horizon Robotics, All rights reserved.

import torch
from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d


class CTC(torch.nn.Module):
    """CTC module."""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
        ctc_loss: nn.Module = None,
    ):
        """Construct CTC module.

        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = ConvModule2d(
            eprojs, odim, kernel_size=1, norm_layer=None, act_layer=None
        )
        self.dropout = nn.Dropout(p=dropout_rate)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = (
            ctc_loss
            if ctc_loss is not None
            else torch.nn.CTCLoss(reduction=reduction_type, zero_infinity=True)
        )

    def forward(
        self,
        hs_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        # hs_pad = hs_pad.squeeze(2).permute(0, 2, 1)
        ys_hat = self.ctc_lo(self.dropout(hs_pad))
        return ys_hat

    def cal_ctc_loss(
        self,
        ys_hat,
        hlens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_lens: torch.Tensor,
    ):
        # ys_hat: (B, L, D) -> (L, B, D)
        # B C 1 T -> B C T -> B T C
        ys_hat = ys_hat.squeeze(2).permute(0, 2, 1)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """Argmax of frame activations.

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def fuse_model(self):
        self.ctc_lo.fuse_model()
