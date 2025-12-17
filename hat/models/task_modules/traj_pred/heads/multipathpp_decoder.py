# Copyright (c) Horizon Robotics. All rights reserved.
#
# For further details about model implementation, inputs/outputs, etc.,
# please see this developer guide:
# https://horizonrobotics.feishu.cn/docs/doccnBr8XXeUQG3MeNHAr0tcZxb
from typing import Dict, List

import torch
import torch.nn as nn


def bivariate_gaussian_activation(inputs: torch.Tensor) -> torch.Tensor:
    """Output parameters of bivariate Gaussian distribution."""
    mu_x = inputs[..., 0:1]
    mu_y = inputs[..., 1:2]
    sig_x = inputs[..., 2:3]
    sig_y = inputs[..., 3:4]
    rho = inputs[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


class MTPPlusDecoder(nn.Module):
    def __init__(
        self,
        num_modes: int,
        op_len: int,
        use_variance: bool,
        encoding_size: int,
        hidden_size: int,
    ):
        """Prediction decoder for MTP.

        Args:
            num_modes: number of output trajectory modes K.
            op_len: prediction horizon.
            hidden_size: hidden layer size.
            encoding_size: size of incoming context encoding.
            use_variance: whether to output variance params along with
                mean predicted locations.
        """

        super().__init__()

        self.num_modes = num_modes
        self.op_len = op_len
        self.use_variance = use_variance
        self.op_dim = 5 if self.use_variance else 2

        self.hidden = nn.Linear(encoding_size, hidden_size)
        self.traj_op = nn.Linear(hidden_size, op_len * self.op_dim)
        self.prob_op = nn.Linear(hidden_size, 1)

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, agg_encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for MTP.

        Args:
            agg_encoding: [N, num_modes, encodign_size] sized aggregated
                context encoding.

        Returns:
            predictions: dictionary with the following keys:
                - 'traj': K predicted trajectories.
                - 'probs': K corresponding probabilities.
        """
        h = self.leaky_relu(self.hidden(agg_encoding))
        batch_size = h.shape[0]
        traj = self.traj_op(h)
        probs = self.log_softmax(self.prob_op(h))
        traj = traj.reshape(
            batch_size, self.num_modes, self.op_len, self.op_dim
        )
        probs = probs.squeeze(dim=-1)

        traj = (
            bivariate_gaussian_activation(traj) if self.use_variance else traj
        )

        predictions = {"traj": traj, "probs": probs}

        return predictions


class MultiHeadMTPPlusDecoder(nn.Module):
    def __init__(
        self,
        num_header: int,
        num_modes: int,
        op_len: int,
        use_variance: bool,
        encoding_size: int,
        hidden_size: int,
    ):
        """Initialize the module.

        Args:
            num_header: number of heads to use.
            num_modes: number of output trajectories for each prediction.
            op_len: prediction horizon as number of steps.
            use_variance: whether to output variance params along with
                mean predicted locations.
            encoding_size: size of incoming context encoding.
            hidden_size: hidden layer size.
        """
        super().__init__()

        self.num_header = num_header
        self.mtpplus_headers = nn.ModuleList()

        for _ in range(self.num_header):
            self.mtpplus_headers.append(
                MTPPlusDecoder(
                    num_modes,
                    op_len,
                    use_variance,
                    encoding_size,
                    hidden_size,
                )
            )

    def forward(self, agg_encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward model.

        Args:
            agg_encodings: a list of tensors, one for each head. Each size is
                [N, num_modes, encoding_size].
        """
        return [
            decoder(agg_encoding)
            for decoder, agg_encoding in zip(
                self.mtpplus_headers, agg_encodings
            )
        ]
