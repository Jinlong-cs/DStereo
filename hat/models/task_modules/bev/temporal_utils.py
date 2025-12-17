# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Sequence, Union

import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from torch.quantization import QuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = [
    "ANCTemporalSimpleFusion",
    "ANCSpatialGRU",
]


class ANCSpatialGRUCell(nn.Module):
    """SpatialGRUCell.

    A GRU cell that takes an input tensor [BxCxHxW]
    and an optional previous state and passes a
    convolution gated recurrent unit over the data.

    Args:
        input_dim: Number of channels of input tensor.
        hidden_dim: Number of channels of hidden state.
        kernel_size: Size of the convolution kernel.
        h_w: Size of the image.
        bias: Whether or not to add the bias.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Sequence],
        h_w: Sequence,
        bias: bool = True,
    ):
        super(ANCSpatialGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2

        self.conv_update = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            bias=bias,
            padding=padding,
        )
        self.conv_reset = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            bias=True,
            padding=padding,
        )

        self.conv_state_tilde = ConvModule2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=padding,
            bias=bias,
            norm_layer=nn.BatchNorm2d(hidden_dim),
            act_layer=nn.ReLU(inplace=True),
        )
        self.cat_x_state = hnn.quantized.FloatFunctional()
        self.state_tilde_cat = hnn.quantized.FloatFunctional()

        self.update_gate_sig = nn.Sigmoid()
        self.reset_gate_sig = nn.Sigmoid()

        self.one = nn.Parameter(
            torch.Tensor([1.0]).float(), requires_grad=False
        )
        self.one_quant = QuantStub()

        self.reset_sub = hnn.quantized.FloatFunctional()
        self.state_tilde_mul = hnn.quantized.FloatFunctional()

        self.out_sub = hnn.quantized.FloatFunctional()
        self.out_mul1 = hnn.quantized.FloatFunctional()
        self.out_mul2 = hnn.quantized.FloatFunctional()
        self.out_add = hnn.quantized.FloatFunctional()

        self.init_rnn_state = nn.Parameter(
            torch.zeros(1, self.hidden_dim, h_w[0], h_w[1]),
            requires_grad=False,
        )
        self.rnn_quant = QuantStub()

    def init_hidden(self, batch_size):
        init_rnn_state = (
            self.init_rnn_state.repeat(batch_size, 1, 1, 1)
            if batch_size > 1
            else self.init_rnn_state
        )
        init_rnn_state = self.rnn_quant(init_rnn_state)
        return init_rnn_state

    def forward(self, x, state):
        # Compute gates
        x_and_state = self.cat_x_state.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = self.update_gate_sig(update_gate)
        reset_gate = self.reset_gate_sig(reset_gate)

        # Compute proposal state, activation is defined in norm_act_config
        # (can be tanh, ReLU etc)
        one = self.one_quant(self.one)
        state_tilde = self.conv_state_tilde(
            self.state_tilde_cat.cat(
                [
                    x,
                    self.state_tilde_mul.mul(
                        self.reset_sub.sub(one, reset_gate), state
                    ),
                ],
                dim=1,
            )
        )

        output = self.out_add.add(
            self.out_mul1.mul(self.out_sub.sub(one, update_gate), state),
            self.out_mul2.mul(update_gate, state_tilde),
        )

        return output

    def fuse_model(self):
        self.conv_state_tilde.fuse_model()


@OBJECT_REGISTRY.register
class ANCTemporalSimpleFusion(nn.Module):
    """Simple temporal fusion module which contains 'cat' and 'add'.

    Args:
        in_channels: Number of channels of input tensor.
        num_frames: Number of frames to fused.
        fusion_method: fusion method, only support 'cat' and 'add'.
        use_decay: Whether to apply the strategy of historical frame decay.
        decay_rate: historical frame decay's rate.
            The process is fusion(current_frame,historacal_frame)*decay + current_frame*(1-decy).  # noqa
    """

    def __init__(
        self,
        in_channels: int,
        num_frames: int,
        fusion_method: str = "cat",
        use_decay: bool = False,
        decay_rate: float = 0.2,
    ):
        super(ANCTemporalSimpleFusion, self).__init__()
        assert fusion_method in ["add", "cat"]
        self.num_frames = num_frames
        self.fusion_method = fusion_method
        projection_inchannels = in_channels
        if self.fusion_method == "cat":
            self.cat = nn.quantized.FloatFunctional()
            projection_inchannels = in_channels * num_frames
        elif self.fusion_method == "add":
            self.adds = nn.ModuleList()
            for _i in range(num_frames - 1):
                self.adds.append(nn.quantized.FloatFunctional())
        self.projection = ConvModule2d(
            projection_inchannels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.use_decay = use_decay
        self.decay_rate = decay_rate
        if self.use_decay:
            self.adds_decay = nn.quantized.FloatFunctional()
            self.mul_one_sub_decay = nn.quantized.FloatFunctional()
            self.mul_decay_rate = nn.quantized.FloatFunctional()
            self.one_sub_decay = 1.0 - self.decay_rate

    @staticmethod
    def parse_pre_feats(pre_feats):
        """Parse previous features during inference."""
        return pre_feats[0]

    def forward_once(self, cur_frame: torch.Tensor, fused_feat: list = None):
        """Forward function for recurrent mode.

        Args:
            cur_frame: the feature of the current frame with
                shape of (B, C, H, W).
            fused_feat: the history fused feature with
                shape of (B, C, H, W).
        """
        if fused_feat is None:
            return cur_frame, cur_frame

        fused_feat = _as_list(fused_feat)
        return self.forward([cur_frame] + fused_feat)

    def forward(self, frames: list):
        """Forward function.

        Args:
            frames: the origin features of frames.
        """
        assert (
            len(frames) == self.num_frames
        ), "The number of frames is not matched."
        if self.fusion_method == "add":
            fusion_data = frames[0]
            for i, pre_frame in enumerate(frames[1:]):
                fusion_data = self.adds[i].add(fusion_data, pre_frame)
        elif self.fusion_method == "cat":
            fusion_data = self.cat.cat(frames, dim=1)
        else:
            raise NotImplementedError()

        fusion_data = self.projection(fusion_data)

        # apply the strategy of historical frame decay.
        if self.use_decay:
            decay_rate_frame = self.mul_one_sub_decay.mul_scalar(
                frames[0], self.one_sub_decay
            )
            decay_rate_fusion_data = self.mul_decay_rate.mul_scalar(
                fusion_data, self.decay_rate
            )
            fusion_data = self.adds_decay.add(
                decay_rate_frame, decay_rate_fusion_data
            )

        return fusion_data, fusion_data

    def to_dumped_feat(self, feat):
        """Directly dump the output feature."""
        return feat

    def fuse_model(self):
        self.projection.fuse_model()


@OBJECT_REGISTRY.register
class ANCSpatialGRU(torch.nn.Module):
    """SpatialGRU.

    Adapted from https://github.com/wayveai/fiery/blob/master/fiery/layers/temporal.py.  # noqa
    Args:
        input_dim: Number of channels of input tensor.
        hidden_dim: Number of channels of latent state.
        num_layers: Number of gru layers stacked.
        h_w: Size of the image.
        bias: Bias or no bias in Convolution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        h_w: Sequence,
        bias: bool = True,
    ):
        super(ANCSpatialGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        # Convolutional recurrent model with z_t as an initial hidden state
        # and inputs the sample from the probabilistic model.
        # The architecture of the model is:
        # [Spatial GRU - [Bottleneck] x n_res_layers] x n_gru_blocks
        self.spatial_grus = []

        for i in range(self.num_layers):
            gru_in_channels = hidden_dim if i == 0 else input_dim
            self.spatial_grus.append(
                ANCSpatialGRUCell(
                    input_dim=gru_in_channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    h_w=h_w,
                    bias=bias,
                )
            )

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        # used to dump hidden feats.
        self.cat = nn.quantized.FloatFunctional()

    @staticmethod
    def parse_pre_feats(pre_feats: list):
        """Parse previous features during inference.

        Args:
            pre_feats: List of tensor with format of [s0, s1, ...],
                containing the hidden states for each layers.
        """
        return pre_feats  # list

    def forward_once(self, x: torch.Tensor, hidden_states: list = None):
        """Forward function for recurrent mode.

        Args:
            x: the feature of the current frame with shape
                of (B, C, H, W).
            hidden_states: List of Tensor. The hidden states of each
                layer.
        """
        if hidden_states is None:
            # x has shape (b, c, h, w)
            b = x.shape[0]
            hidden_states = self._init_hidden(b)
        else:
            hidden_states = _as_list(hidden_states)

        next_hidden_states = []
        for i in range(self.num_layers):
            x = self.spatial_grus[i].forward(x, hidden_states[i])
            next_hidden_states.append(x)
        return x, next_hidden_states

    def forward(self, frames: list):
        """Forward function.

        Args:
            frames: the origin features of frames.
        """
        hidden_states = None
        for cur_feat in frames[::-1]:
            fusion_data, hidden_states = self.forward_once(
                cur_feat, hidden_states
            )

        return fusion_data, hidden_states

    def to_dumped_feat(self, hidden_states):
        """Stack hidden_states on batch dim."""
        feat_to_dump = self.cat.cat(hidden_states, dim=0)
        # self.cat的scale可能和最后一层的hidden_states的scale不一致，导致部署的时候出现scale不一致问题
        return feat_to_dump

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.spatial_grus[i].init_hidden(batch_size))
        return init_states

    def fuse_model(self):
        for m in self.spatial_grus:
            m.fuse_model()
