import copy
import os
from typing import List, Optional

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from horizon_plugin_pytorch import quantization
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import (
    FixedScaleObserver,
    QuantStub,
    get_default_qat_qconfig,
)

import hat.utils.saved_tensor as savedtensor
from hat.core.traj_pred_utils import Affine2D
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager

__all__ = ["_get_clones", "inverse_sigmoid", "dict_select"]


def qint8_qconfig(freeze_scale=True):
    if freeze_scale:
        activation_qat_qkwargs = {
            "averaging_constant": 0.0,
            "dtype": qint8,
        }
    else:
        activation_qat_qkwargs = {
            "dtype": qint8,
        }
    return qconfig_manager.get_qconfig(
        activation_qat_observer="min_max",
        weight_qat_observer="min_max",
        activation_calibration_observer="mix",
        weight_calibration_observer="min_max",
        activation_qat_qkwargs=activation_qat_qkwargs,
        weight_qat_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "dtype": qint8,
            "ch_axis": 0,
        },
        activation_calibration_qkwargs={
            "dtype": qint8,
        },
        weight_calibration_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "dtype": qint8,
            "ch_axis": 0,
        },
    )


def qint16_qconfig(freeze_scale=True):
    if freeze_scale:
        activation_qat_qkwargs = {
            "averaging_constant": 0.0,
            "dtype": qint16,
        }
    else:
        activation_qat_qkwargs = {
            "dtype": qint16,
        }
    return qconfig_manager.get_qconfig(
        activation_qat_observer="min_max",
        weight_qat_observer="min_max",
        activation_calibration_observer="mix",
        weight_calibration_observer="min_max",
        activation_qat_qkwargs=activation_qat_qkwargs,
        weight_qat_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
            "dtype": qint8,
        },
        activation_calibration_qkwargs={
            "dtype": qint16,
        },
        weight_calibration_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
            "dtype": qint8,
        },
    )


def qat_out_qconfig():
    return qconfig_manager.get_qconfig(
        activation_fake_quant=None,
        weight_qat_observer="min_max",
        weight_calibration_observer="min_max",
        weight_qat_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
            "dtype": qint8,
        },
        weight_calibration_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
            "dtype": qint8,
        },
    )


QINT16_MAX = 32768.0
# 默认最高fps 为 30，可兼容30fps以下的情况
FPS_MAX = 30.0
# inverse sigmoid 空间的最大值，sigmoid(11) = 1.0
INV_SIGMOID_MAX = 11.0
# sigmoid 空间的最大值
SIGMOID_MAX = 1.0
# 规定的速度的最大值，单位为m/s，考虑到丢帧的情况，模型中两帧的位移量可能超预期，这里给出一定裕量
SPEED_MAX = 80.0


@OBJECT_REGISTRY.register
class Linear(nn.Sequential):
    """Linear layers implementation for E2E."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        conv = nn.Linear(
            in_channels,
            out_channels,
        )
        conv_list = [conv, norm_layer, act_layer]
        self.conv_list = [layer for layer in conv_list if layer is not None]
        super(Linear, self).__init__(*self.conv_list)
        st = bool(int(os.environ.get("HAT_USE_SAVEDTENSOR", "0")))
        if (
            st
            and norm_layer is not None
            and savedtensor.support_saved_tensor()
        ):
            self.st_fn = savedtensor.checkpoint_convbn_with_saved_tensor(
                conv, norm_layer, act_layer
            )
        else:
            self.st_fn = None

    def forward(self, x):
        cp = bool(int(os.environ.get("HAT_USE_CHECKPOINT", "0")))
        if cp and torch.is_tensor(x) and x.requires_grad:
            out = checkpoint.checkpoint(super().forward, x)
        elif self.st_fn is not None and self.training and torch.is_tensor(x):
            out = self.st_fn(x)
        else:
            out = super().forward(x)
        return out

    def fuse_model(self):
        if self.st_fn is not None:
            # remove saved tensor fn after fused module,
            # because there is only one conv remained after fused.
            self.st_fn = None

        if len(self.conv_list) > 1 and not isinstance(
            self.conv_list[-1], (nn.BatchNorm2d, nn.ReLU, nn.ReLU6)
        ):
            # not: conv2d+bn, conv2d+relu(6), conv2d+bn+relu(6)
            self.conv_list.pop()

        if len(self.conv_list) <= 1:
            # nn.Conv2d
            return

        fuse_list = ["0", "1", "2"]
        self.fuse_list = fuse_list[: len(self.conv_list)]
        torch.quantization.fuse_modules(
            self,
            self.fuse_list,
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


@OBJECT_REGISTRY.register
class MLP(nn.Module):
    """simple multi-layer perceptron (also called FFN).

    Args:
        input_dim: input dimension.
        hidden_dim: hidden dimension.
        output_dim: output dimension.
        num_layers: number of layers.
        is_output: whether the last layer is model output layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        is_output: bool = False,
        freeze_scale: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.is_output = is_output
        self.freeze_scale = freeze_scale
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()
        for i, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim])):
            if i < self.num_layers - 1:
                self.layers.append(
                    Linear(n, k, act_layer=nn.ReLU(inplace=True))
                )
            else:
                self.layers.append(Linear(n, k))

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def fuse_model(self):
        for m in self.layers:
            m.fuse_model()

    def set_qconfig(self):
        if not self.is_output:
            for layer in self.layers:
                layer.qconfig = qint16_qconfig(freeze_scale=self.freeze_scale)
        else:
            for layer in self.layers[:-1]:
                layer.qconfig = qint16_qconfig(freeze_scale=self.freeze_scale)
            # ! how to set qint32
            self.layers[-1].qconfig = qat_out_qconfig()


class FFN(nn.Module):
    """A simple implementation of FFN."""

    def __init__(self, channels, ffn_channels, dropout=0.1):
        super().__init__()
        self.linear1 = Linear(
            channels, ffn_channels, act_layer=nn.ReLU(inplace=True)
        )
        self.linear2 = Linear(ffn_channels, channels)
        self.skip_add = FF()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, identity):
        x = self.linear2(self.dropout1(self.linear1(x)))
        x = self.skip_add.add(self.dropout2(x), identity)  # 输出int8，后面接norm
        return x

    def fuse_model(self):
        self.linear1.fuse_model()

    def set_qconfig(self):
        self.qconfig = qint8_qconfig()


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention module.

    this is a hack implement, will be refactor by wenming.meng.
    """

    def __init__(self, dim, feat_shape, num_queries, num_head=8, x_y_max=1000):
        super().__init__()
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.dim = dim
        self.num_head = num_head
        self.group_base = dim // num_head
        self.softmax = nn.Softmax(dim=1)
        self.qk_mul = FF()
        self.qk_mean = FF()
        self.qkv_mul = FF()
        self.qkv_mean = FF()
        self.grid_sample = nn.ModuleList()
        feat_lvls = len(feat_shape)
        for _ in range(feat_lvls):
            self.grid_sample.append(
                hnn.GridSample(mode="bilinear", padding_mode="zeros")
            )

        self.generate_offsets = nn.ModuleList()
        self.add_offsets = nn.ModuleList()
        for _ in range(feat_lvls):
            self.generate_offsets.append(nn.Linear(dim, 2))
            self.add_offsets.append(FF())

        self.sampling_values_cat = FF()
        self.xy_add = nn.ModuleList()
        self.hw_mul = nn.ModuleList()
        for _ in range(feat_lvls):
            self.xy_add.append(FF())
            self.hw_mul.append(FF())

        shape_lvls = [
            torch.tensor(shape, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 1, x_y_max, 1)
            for shape in feat_shape
        ]
        shape = torch.cat(shape_lvls, dim=0).to(torch.float32)
        self.shape = nn.Parameter(shape, requires_grad=False)
        self.shape_max = max(feat_shape[0])

        self.quant = QuantStub(self.shape_max / QINT16_MAX)
        self.num_queries = num_queries

    def forward(self, q, srcs, reference_points, xy_tmp):

        sample_location = reference_points[
            :, :, :, 0:2
        ]  # sample location仅取x,y
        sampling_values = []
        xy_tmp = xy_tmp[:, :, : self.num_queries, :]
        shape = self.quant(self.shape[..., : self.num_queries, :])
        for lvl, src in enumerate(srcs):
            _, _, h_, w_ = src.size()
            sampling_grids = self.hw_mul[lvl].mul(
                sample_location, shape[lvl].unsqueeze(0)
            )
            sampling_grids = self.xy_add[lvl].add(sampling_grids, xy_tmp)
            offsets = self.generate_offsets[lvl](q)
            sampling_grids = self.add_offsets[lvl].add(sampling_grids, offsets)
            sampling_values_ = self.grid_sample[lvl](src, sampling_grids)
            sampling_values.append(sampling_values_)
        sampling_values = (
            self.sampling_values_cat.cat(sampling_values, dim=0)
            .squeeze(2)
            .permute(0, 2, 1)
        )

        q = q.squeeze(0)
        bs, n, c = q.size()
        level = sampling_values.size(0)
        q = (
            self.q(q)
            .permute(0, 2, 1)
            .reshape(bs, self.group_base, self.num_head, n)
        )  # q,k,全部int16
        k = (  # level 4 stride
            self.k(sampling_values)
            .permute(0, 2, 1)
            .reshape(level, self.group_base, self.num_head, n)
        )
        v = (
            self.v(sampling_values)
            .permute(0, 2, 1)
            .reshape(level, self.group_base, self.num_head, n)
        )

        attention = self.qk_mul.mul(q, k).permute(0, 2, 3, 1)  # int16
        attention = self.qk_mean.mean(attention, dim=-1)  # int16
        attention = attention.permute(3, 0, 1, 2)  # int16
        attention = self.softmax(attention).permute(1, 0, 2, 3)  # int16
        x = self.qkv_mul.mul(attention, v)
        x = self.qkv_mean.mean(x, dim=0)
        x = x.reshape(bs, c, n).permute(0, 2, 1)  # int16
        return x

    def fuse_model(self):
        self.q.fuse_model()
        self.k.fuse_model()
        self.v.fuse_model()

    def set_qconfig(self):
        self.q.qconfig = qint16_qconfig()
        self.k.qconfig = qint16_qconfig()
        self.v.qconfig = qint16_qconfig()
        self.qk_mul.qconfig = qint16_qconfig()
        self.qk_mean.qconfig = qint16_qconfig()
        self.softmax.qconfig = qint16_qconfig()
        self.qkv_mul.qconfig = qint16_qconfig()
        self.qkv_mean.qconfig = qint16_qconfig()
        self.sampling_values_cat.qconfig = qint16_qconfig()
        for layer in self.xy_add:
            layer.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": max(self.shape_max, self.num_queries)
                    / QINT16_MAX,
                },
            )
        for layer in self.hw_mul:
            layer.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": self.shape_max / QINT16_MAX,
                },
            )
        for layer in self.add_offsets:
            layer.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": max(self.shape_max, self.num_queries)
                    / QINT16_MAX,
                },
            )
        for layer in self.generate_offsets:
            layer.qconfig = qint16_qconfig()
        for layer in self.grid_sample:
            layer.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint8"
            )
        self.quant.qconfig = qint16_qconfig()


class SingleHeadSelfAttention(nn.Module):
    """single-head cross attention module.

    this is a hack implement, will be refactor by wenming.meng.
    """

    def __init__(self, dim):
        super().__init__()
        self.q = Linear(dim, 1)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim, act_layer=nn.ReLU(inplace=True))
        self.out = Linear(dim, dim)
        self.softmax = nn.Softmax(dim=1)
        self.qk_mul = FF()
        self.qk_sum = FF()
        self.qkv_mul = FF()

        self.active_mask_mulv = FF()
        self.active_mask_mulq = FF()
        self.active_mask_mulk = FF()

    def forward(self, q, k, v, active_mask=None):
        if active_mask is not None:
            # active_mask只乘以v的时候，每次推理的qk_sum值就跟填补的query有关，所以改成乘以Q或者乘以Q/K/V # noqa
            # 在selfAttension起始时候将填补序列置全零，重新训练后训练结果与填补方式/填补值等均无关# # noqa
            v = self.active_mask_mulv.mul(v, active_mask)  # [1, 1, nn, kk]
            k = self.active_mask_mulk.mul(k, active_mask)
            q = self.active_mask_mulq.mul(q, active_mask)
        q = self.q(q)  # int8
        k = self.k(k)  # int16
        v = self.v(v)  # int16
        q = q.permute(0, 2, 1, 3)
        q = self.softmax(q).permute(0, 2, 1, 3)  # int16
        qk = self.qk_mul.mul(k, q)  # int16
        qk = self.qk_sum.sum(qk, dim=2, keepdim=True)
        out = self.qkv_mul.mul(v, qk)  # int16
        out = self.out(out)  # int8
        return out

    def fuse_model(self):
        self.q.fuse_model()
        self.k.fuse_model()
        self.v.fuse_model()
        self.out.fuse_model()

    def set_qconfig(self):
        self.softmax.qconfig = qint16_qconfig()
        self.q.qconfig = qint16_qconfig()
        self.k.qconfig = qint16_qconfig()
        self.v.qconfig = qint16_qconfig()
        self.qk_mul.qconfig = qint16_qconfig()
        self.qk_sum.qconfig = qint16_qconfig()
        self.qkv_mul.qconfig = qint16_qconfig()
        self.out.qconfig = qint16_qconfig()

        self.active_mask_mulv.qconfig = qint16_qconfig()
        self.active_mask_mulq.qconfig = qint16_qconfig()
        self.active_mask_mulk.qconfig = qint16_qconfig()


class MultiHeadMulTemporalAttention(nn.Module):
    """multi-head cross attention module.

    this is a hack implement, will be refactor by wenming.meng.
    """

    def __init__(self, dim, num_head, per_head_channels):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.per_head_channels = per_head_channels
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.softmax = nn.Softmax(dim=1)
        self.qk_mul = FF()
        self.qk_mean = FF()
        self.qkv_mul = FF()
        self.qkv_mean = FF()
        self.mask_mul = FF()

    def forward(self, q, k, v, mask=None):
        bs, n, c = q.shape[1], q.shape[2], q.shape[3]
        temporal_len = k.shape[0]

        q = (
            self.q(q)
            .permute(0, 1, 3, 2)
            .reshape(bs, self.per_head_channels, self.num_head, n)
        )
        k = (
            self.k(k)
            .permute(0, 1, 3, 2)
            .reshape(temporal_len, self.per_head_channels, self.num_head, n)
        )
        v = (
            self.v(v)
            .permute(0, 1, 3, 2)
            .reshape(temporal_len, self.per_head_channels, self.num_head, n)
        )
        attention = self.qk_mul.mul(q, k).permute(0, 2, 3, 1)
        attention = self.qk_mean.mean(attention, dim=-1)
        attention = attention.permute(3, 0, 1, 2)
        attention = self.softmax(attention).permute(1, 0, 2, 3)
        if mask is not None:
            v = self.mask_mul.mul(v, mask)
        x = self.qkv_mul.mul(attention, v)
        x = self.qkv_mean.mean(x, dim=0)
        x = x.reshape(1, bs, c, n).permute(0, 1, 3, 2)

        return x

    def fuse_model(self):
        self.q.fuse_model()
        self.k.fuse_model()
        self.v.fuse_model()

    def set_qconfig(self):
        self.q.qconfig = qint16_qconfig()
        self.k.qconfig = qint16_qconfig()
        self.v.qconfig = qint16_qconfig()
        self.qk_mul.qconfig = qint16_qconfig()
        self.qk_mean.qconfig = qint16_qconfig()
        self.softmax.qconfig = qint16_qconfig()
        self.qkv_mul.qconfig = qint16_qconfig()
        self.qkv_mean.qconfig = qint16_qconfig()
        self.mask_mul.qconfig = qint16_qconfig()


class Embedding(nn.Module):
    """Return query embed as nn.Parameter."""

    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape))
        nn.init.uniform_(self.weight)
        self.q_stub = QuantStub()

    def forward(self):
        return self.q_stub(self.weight)

    def set_qconfig(self):
        self.q_stub.qconfig = qint16_qconfig()


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, shape):
        super().__init__()
        self.tp_embed = Embedding(shape)
        self.reset_parameters()
        self.add = FF()

    def reset_parameters(self):
        nn.init.uniform_(self.tp_embed.weight)

    def forward(self, x):
        max_len = x.size(1)
        pos_emb = self.tp_embed()[:max_len, :]
        return self.add.add(x, pos_emb.unsqueeze(0).unsqueeze(2))

    def set_qconfig(self):
        self.tp_embed.set_qconfig()
        self.add.qconfig = qint16_qconfig()


def _get_clones(module, N):
    """Clone nn.Module to nn.ModuleList."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    """Calculate inv sigmoid for torch.tensor."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def dict_select(d: dict, index, select_keys=None) -> dict:
    """Retuen a selected dict according index."""
    new_instances = {}
    for k, v in d.items():
        if select_keys is None:
            new_instances[k] = v[index]
        else:
            if k in select_keys:
                new_instances[k] = v[index]
    return new_instances


def dict_select_keep_dim(d: dict, index, select_keys=None) -> dict:
    """Retuen a selected dict according index and keep the dim of tensor."""
    index = index.squeeze(0).squeeze(0)
    if len(index.shape) == 0:
        index = index.unsqueeze(0)
    new_instances = {}

    def _select(v):
        if len(v.shape) >= 4:
            return v[:, :, index]
        else:
            return v[index]

    for k, v in d.items():
        if select_keys is None:
            new_instances[k] = _select(v)
        else:
            if k in select_keys:
                new_instances[k] = _select(v)
    return new_instances


def pad_tensors_to_target_shape(input_tensors, target_num=300, dim=0):
    target_tensors = []
    for input_tensor in input_tensors:
        assert isinstance(input_tensor, torch.Tensor)
        if input_tensor.shape[dim] >= target_num:
            target_tensor = input_tensor[:target_num]
        else:
            pad_num = target_num - input_tensor.shape[dim]
            pad_shape = list(input_tensor.shape)
            pad_shape[dim] = pad_num
            pad_data = torch.zeros(pad_shape).to(input_tensor.device)
            target_tensor = torch.cat([input_tensor, pad_data], dim=dim)
        target_tensors.append(target_tensor)
    active_mask = torch.ones(target_num).to(input_tensor.device)
    return target_tensors, active_mask


def update_his_coord(
    coord: torch.tensor, odo_info: torch.tensor, vcs_range: List
):
    """Update the history coordinate.

    Convert coordinate from the historical BEV coordinate system
    to the current BEV coordinate system.

    Args:
        coord: The historical coordinate in historical BEV
            coordinate system.
        odo_info: The historical odometry information.
        vcs_range: The range of vcs coordinate.
    """
    m_range_xy = [(vcs_range[2] - vcs_range[0]), (vcs_range[3] - vcs_range[1])]
    pred_y, pred_x = coord[:, 0], coord[:, 1]
    m_pred_vcs_x = (vcs_range[2] - pred_x * m_range_xy[0]).unsqueeze(1)
    m_pred_vcs_y = (vcs_range[3] - pred_y * m_range_xy[1]).unsqueeze(1)

    cos_inv_yaw = torch.cos(-odo_info[-1])
    sin_inv_yaw = torch.sin(-odo_info[-1])
    rotation_mat = (
        torch.tensor([cos_inv_yaw, -sin_inv_yaw, sin_inv_yaw, cos_inv_yaw])
        .reshape(2, 2)
        .float()
    )
    coordinate = torch.cat([m_pred_vcs_x, m_pred_vcs_y], 1)  # num_query 0
    wcs_xy = torch.matmul(coordinate, rotation_mat.to(coordinate))
    wcs_xy[:, 0] = wcs_xy[:, 0] + odo_info[0]
    wcs_xy[:, 1] = wcs_xy[:, 1] + odo_info[1]

    pred_x = (vcs_range[2] - wcs_xy[:, 0]) / m_range_xy[0]
    pred_y = (vcs_range[3] - wcs_xy[:, 1]) / m_range_xy[1]
    coord[:, 0] = pred_y[:]
    coord[:, 1] = pred_x[:]

    return coord


def update_his_velo(
    velo: torch.Tensor, odo_info: torch.Tensor, vcs_range: List
):
    """Update the history velocity.

    Convert velocity from the historical BEV coordinate system
    to the current BEV coordinate system.

    Args:
        velo: The historical velocity and acceleration in
            historical BEV coordinate system.
        odo_info: The historical odometry information.
        vcs_range: The range of vcs coordinate.
    """
    cos_inv_yaw = torch.cos(-odo_info[-1])
    sin_inv_yaw = torch.sin(-odo_info[-1])
    rotation_mat = (
        torch.tensor([cos_inv_yaw, -sin_inv_yaw, sin_inv_yaw, cos_inv_yaw])
        .reshape(2, 2)
        .float()
    )
    velo = torch.matmul(velo, rotation_mat.to(velo))
    velo_bev = -velo[:, [1, 0]]  # 转换为 [-vcs_y, -vcs_x]
    velo_bev[:, 0] /= vcs_range[3] - vcs_range[1]
    velo_bev[:, 1] /= vcs_range[2] - vcs_range[0]
    return velo_bev


def kalerman_filter(
    last_state: torch.Tensor,
    observer: torch.Tensor,
    K: torch.Tensor,
    time_delta: float,
):
    """
    Update the state of the object by kalerman filter.

    Our kalerman filter is a constant acceleration model, including 8 states:
    x, y, z, vx, vy, vz, ax, ay, az.
    We use the following equations to update the state of the object:
    x = x + v_x * time_delta + 0.5 * a_x * pow(time_delta, 2)
    y = y + v_y * time_delta + 0.5 * a_y * pow(time_delta, 2)
    v_x = v_x + a_x * time_delta
    v_y = v_y + a_y * time_delta
    w = w
    h = h
    a_x = a_x
    a_y = a_y
    Following transform matrix is built based on the above equations.

    Args:
        last_state: last state of the object, shape: (num_queries, 8).
        observer: current observed stage of the object, shape:
            (num_queries, 6), note current model output does
            not include acceleration, thus only 6 states.
        K: kalerman gain, shape (num_queries, 8, 6), got from model output.
        time_delta: time interval between last state and current state,
            unit is seconds.
    """
    time_delta = time_delta.item()
    transfunciton = np.array(
        [
            [1, 0, 0, 0, time_delta, 0, 0.5 * pow(time_delta, 2), 0],  # x
            [0, 1, 0, 0, 0, time_delta, 0, 0.5 * pow(time_delta, 2)],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0, 0, 0],  # h
            [0, 0, 0, 0, 1, 0, time_delta, 0],  # vx
            [0, 0, 0, 0, 0, 1, 0, time_delta],  # vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # ax
            [0, 0, 0, 0, 0, 0, 0, 1],  # ay
        ]
    )
    # 当前模型不输出加速度，因此last_state的加速度被强制置为0
    last_state[:, 6:] = 0
    transfunciton = torch.from_numpy(transfunciton).to(last_state)
    obserfunction = torch.zeros((6, 8))
    obserfunction[:6, :6] = torch.eye(6)
    obserfunction = obserfunction.to(last_state)  # 6 * 8s

    x_ = last_state @ transfunciton.T
    # 卡尔曼增益，8*6，模型直接输出
    K = (torch.sigmoid(K) - 0.5) + obserfunction.T / 2.0
    K_mat = torch.ones(8, 6)
    # 构建运动关联矩阵
    K_mat[2:4, :] = 0
    K_mat[2, 2] = 1.0
    K_mat[3, 3] = 1.0
    K_mat[0:4, 4:] = 0.0
    K_mat = K_mat.to(K)
    K = K * K_mat
    output = x_ + torch.bmm(
        (observer - x_ @ obserfunction.T).unsqueeze(1), K.transpose(1, 2)
    ).squeeze(1)
    return output


def get_lcf_odo_info(odo_info: torch.Tensor):
    """Transform history odometry infomation to current frame.

    Args:
        odo_info: odometry information in the global coordinate system,
            each row is single frame's [x, y, yaw].

    Returns:
        vcs_odo_info: odometry information in the local coordinate system,
            the origin is current frame's vcs origin.
    """
    device = odo_info.device
    odo_info = odo_info.cpu().detach().numpy()
    odo_xy = odo_info[:, :2]
    odo_yaw = odo_info[:, 2]
    # Calculate the odo coordiates in the current vcs's coordinate system.
    delta_x, delta_y = Affine2D.coord_translate(
        odo_xy[:, 0], odo_xy[:, 1], odo_xy[-1, 0], odo_xy[-1, 1]
    )
    vcs_x, vcs_y = Affine2D.coord_rotate(delta_x, delta_y, odo_yaw[-1])
    vcs_yaw = odo_yaw - odo_yaw[-1]
    vcs_odo_info = (
        torch.from_numpy(np.stack([vcs_x, vcs_y, vcs_yaw], axis=1))
        .float()
        .to(device)
    )
    return vcs_odo_info


def instance_convert_yaw(
    instances,
    decode_rot_setting,
    use_sigmoid=True,
):
    """Convert the yaw rate of instance to the format of [cos(yaw), sin(yaw)].

    Args:
        instances: Track intances.
        decode_rot_setting: The setting of PSC.
        use_sigmoid: Whether apply sigmoid for the inputs yaws.
    """
    assert "yaws" in instances
    if decode_rot_setting is None:
        return instances
    psc_rot_setting = decode_rot_setting["psc_rot"]
    vcs_yaw = decode_psc_rot(
        instances["yaws"],
        psc_rot_setting["coef_sin"],
        psc_rot_setting["coef_cos"],
        psc_rot_setting["N_steps_PSC_rot"],
        use_sigmoid=use_sigmoid,
    )
    instances["yaws"] = torch.cat(
        [
            torch.cos(vcs_yaw).unsqueeze(1),
            torch.sin(vcs_yaw).unsqueeze(1),
        ],
        axis=-1,
    )
    return instances


def e2e_instance_bev2vcs(
    instances: dict,
    vcs_range: List,
):
    """Transform the "boxes", "bev_loc_z", "hegihts" in e2e instance to vcs coord.

    Args:
        instances: object instances, Keys as below:
            boxes: optional, instance bev location and shape, shape: N * 4,
                in the format of [x, y, l, w].
            bev_loc_z: optional, the z height of instance, shape: N.
            heights: optional, the height of instance, shape: N.
        vcs_range: The range of vcs coordinate, in the format of
            (vcs_x_min, vcs_y_min, vcs_x_max, vcs_y_max, vcs_z_min, vcs_z_max)
    """
    assert len(vcs_range) == 6
    if "boxes" in instances:
        instances["boxes"] = coord_bev2vcs(instances["boxes"], vcs_range)
    if "bev_loc_z" in instances:
        instances["bev_loc_z"] *= vcs_range[5] - vcs_range[4]
    if "heights" in instances:
        # 对高度反 normalize
        instances["heights"] *= vcs_range[5] - vcs_range[4]

    return instances


def decode_psc_rot(
    bev3d_rot,
    coef_sin,
    coef_cos,
    N_steps_PSC_rot=3,
    use_sigmoid=False,
    rot_mod_threshold=0.0001,
) -> torch.Tensor:
    """Decode the heading angle by PSC.

    Args:
        bev3d_rot: N x N_steps_PSC_rot, the predicted angle in
            PSC format.
        coef_sin: The sin value of the angles predefined by dividing
            2*pi into N equal parts.
        coef_sin: The sin value of the predefined angle.
        N_steps_PSC_rot: Angle phase number for angle encoding.
        use_sigmoid: Whether the angle in sigmoid space. If yes,
            the angle need to be normalized.
        rot_mod_threshold: Numeric precision threshold.
    """
    if use_sigmoid:
        bev3d_rot = torch.add(
            torch.mul(bev3d_rot.sigmoid(), 2), -1
        )  # norm rot
    coef_sin = coef_sin.to(bev3d_rot)
    coef_cos = coef_cos.to(bev3d_rot)
    phase_sin = torch.sum(
        bev3d_rot[..., :N_steps_PSC_rot] * coef_sin,
        dim=-1,
        keepdim=False,
    )
    phase_cos = torch.sum(
        bev3d_rot[..., :N_steps_PSC_rot] * coef_cos,
        dim=-1,
        keepdim=False,
    )
    phase_mod = phase_cos ** 2 + phase_sin ** 2
    phase = -torch.atan2(phase_sin, phase_cos)
    phase[phase_mod < rot_mod_threshold] *= 0
    return phase


def get_coef_of_psc(N_steps_PSC_rot, return_tensor=False):
    """Get the coef of PSC according to the number of step.

    Args:
        N_steps_PSC_rot: Nums of steps to decode the heading angle by PSC.
        return_tensor: Whether return the coef in the format of tensor.
    """
    coef_sin = np.array(
        tuple(
            np.sin(np.array(2 * k * np.pi / N_steps_PSC_rot))
            for k in range(N_steps_PSC_rot)
        )
    )
    coef_cos = np.array(
        tuple(
            np.cos(np.array(2 * k * np.pi / N_steps_PSC_rot))
            for k in range(N_steps_PSC_rot)
        )
    )
    if return_tensor:
        coef_sin = torch.from_numpy(coef_sin).float()
        coef_cos = torch.from_numpy(coef_cos).float()

    return (coef_sin, coef_cos)


def coord_bev2vcs(pred_boxes, vcs_range):
    """Transfer the coordinate from bev to vcs.

    Args:
        pred_boxes: Tensor with N x 4, N is the targets num and
            each row represents [x, y, w, h] in bev coord.
        vcs_range: List, (bottom, right, top, left)

    Return:
        Tensor with N x 4, each row represents [vcs_x, vcs_y,
            width, length] in vcs coord.
    """
    vcs_wh = [
        vcs_range[3] - vcs_range[1],
        vcs_range[2] - vcs_range[0],
    ]
    pred_boxes[:, [0, 2]] *= vcs_wh[0]
    pred_boxes[:, [1, 3]] *= vcs_wh[1]
    pred_boxes[:, 0] = vcs_range[3] - pred_boxes[:, 0]
    pred_boxes[:, 1] = vcs_range[2] - pred_boxes[:, 1]
    return pred_boxes[:, [1, 0, 3, 2]]
