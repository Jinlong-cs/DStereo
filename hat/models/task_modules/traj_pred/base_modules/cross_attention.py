from typing import Callable, Optional

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.task_modules.traj_pred.base_modules.traj_qconfig import (
    qint8_freezescale_qconfig,
    qint16_freezescale_qconfig,
)
from hat.utils import qconfig_manager

__all__ = ["CrossAttention"]


class CrossAttention(nn.Module):
    """Cross Attention module.

    This extracts feature between modalities.
    """

    def __init__(
        self,
        q_in_channels: int,
        k_in_channels: int,
        hidden_units: int,
        num_q: int,
        num_k: int,
        num_attention_heads: int,
        need_scale: bool = False,
        use_layernorm: bool = False,
        qconfig_func: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            q_in_channels: the input channels of modality 1.
            k_in_channels: the input channels of modality 2.
            hidden_units: the number of hidden units in each attention
                heads.
            num_q: the number of features of modality 1.
            num_q: the number of features of modality 2.
            num_poly: the number of polylines.
            num_attention_heads: the number of attention heads.
            out_dim: the output dim of the fc
            need_scale: whether to consider scale factor
                during the calculation of attention. Defaults to False.
            use_layernorm: whether to use layernorm
        """
        super(CrossAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_q = num_q
        self.num_k = num_k
        self.num_heads = num_attention_heads
        self.all_head_size = hidden_units * num_attention_heads
        self.scale_factor = np.sqrt(hidden_units) if need_scale else 1.0
        self.sf_quant = QuantStub(scale=None)
        self.use_layernorm = use_layernorm
        self.qconfig_func = qconfig_func

        self.wq = nn.Conv2d(q_in_channels, self.all_head_size, kernel_size=1)
        self.wk = nn.Conv2d(k_in_channels, self.all_head_size, kernel_size=1)
        self.wv = nn.Conv2d(k_in_channels, self.all_head_size, kernel_size=1)
        self.q_op = nn.quantized.FloatFunctional()
        self.q_op_2 = nn.quantized.FloatFunctional()

        self.qTk_matmul = FF()
        self.v_attn_matmul = FF()
        self.softmax = nn.Softmax(dim=3)
        if self.use_layernorm:
            self.qk_norm = hnn.LayerNorm((self.num_k,))

    def forward(self, q_input, k_input, attn_mask=None):
        """Forward.

        Args:
            q_input (torch.Tensor, [num_obs, hidden_size, 1, num_q_feats]): the
                input features 1.
            k_input (torch.Tensor, [num_obs, hidden_size, 1, num_k_feats]): the
                input features 1.
            attn_mask (torch.Tensor, [batch_size, num_poly, num_poly]): the
                attention masks. It is a 0-1 matrix. It does not describe the
                polyline connections, but only identifies whether two polylines
                have the opportunity to interact with each other.

        Returns:
            context (torch.Tensor, [batch_size, all_head_size, 1, 1]): the
            feature between modalities.
        """
        # Calculate q, k, v: [batch_size, num_heads, hidden_units, num]
        kv_shape = [-1, self.num_heads, self.hidden_units, self.num_k]
        q_shape = [-1, self.num_heads, self.hidden_units, self.num_q]
        q_val = self.wq(q_input).reshape(q_shape)
        k_val = self.wk(k_input).reshape(kv_shape)
        v_val = self.wv(k_input).reshape(kv_shape)

        # Calculate self-attention scores.
        # shape: [batch_size, num_heads, 1, num_poly]
        trans_q_val = q_val.permute(0, 1, 3, 2)
        scores = self.qTk_matmul.matmul(trans_q_val, k_val, False, False)
        inv_sf = torch.tensor(
            [1 / self.scale_factor], device=scores.device, dtype=torch.float32
        )
        inv_sf = self.sf_quant(inv_sf)
        scores = self.q_op.mul(scores, inv_sf)
        if self.use_layernorm:
            attention = self.qk_norm(scores)
        else:
            attention = scores
        if attn_mask is not None:
            tmp_attn_mask = attn_mask[:, :, :, :].repeat(
                1, self.num_heads, 1, 1
            )
            attention = self.q_op_2.mul(attention, tmp_attn_mask)
        attention = self.softmax(attention)

        # Calculate features after attention.
        # shape: [batch_size, all_head_size, 1, num_q]
        context = self.v_attn_matmul.matmul(attention, v_val, False, True)
        context = context.permute(0, 1, 3, 2)
        context = context.reshape([-1, self.all_head_size, 1, self.num_q])
        return context

    def set_qconfig(self):
        if self.qconfig_func is None:
            self.qconfig = qconfig_manager.get_default_qat_qconfig()
        else:
            self.sf_quant.qconfig = qint16_freezescale_qconfig()
            self.wq.qconfig = qint8_freezescale_qconfig()
            self.wk.qconfig = qint8_freezescale_qconfig()
            self.wv.qconfig = qint8_freezescale_qconfig()
            self.q_op.qconfig = qint16_freezescale_qconfig()
            self.q_op_2.qconfig = qint16_freezescale_qconfig()
            self.qTk_matmul.qconfig = qint8_freezescale_qconfig()
            self.v_attn_matmul.qconfig = qint8_freezescale_qconfig()
            self.softmax.qconfig = qint8_freezescale_qconfig()


class CrossAttentionCut(nn.Module):
    """Cut Cross Attention module.

    裁剪操作包括:
    1. 将softmax(QK)变为softmax(Q)softmax(K)，降低softmax的计算维度
    2. 将scale操作从mul变为matmul，提高计算效率
    3. 在crossAttention的最后引入attention结果与query的add操作，实现级联

    This extracts feature between modalities.
    """

    def __init__(
        self,
        q_in_channels: int,
        k_in_channels: int,
        hidden_units: int,
        num_q: int,
        num_k: int,
        num_attention_heads: int,
        need_scale: bool = False,
        use_layernorm: bool = False,
        qconfig_func: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            q_in_channels: the input channels of modality 1.
            k_in_channels: the input channels of modality 2.
            hidden_units: the number of hidden units in each attention
                heads.
            num_q: the number of features of modality 1.
            num_q: the number of features of modality 2.
            num_poly: the number of polylines.
            num_attention_heads: the number of attention heads.
            out_dim: the output dim of the fc
            need_scale: whether to consider scale factor
                during the calculation of attention. Defaults to False.
            use_layernorm: whether to use layernorm
        """
        super(CrossAttentionCut, self).__init__()
        self.hidden_units = hidden_units
        self.num_q = num_q
        self.num_k = num_k
        self.num_heads = num_attention_heads
        self.all_head_size = hidden_units * num_attention_heads
        self.scale_factor = np.sqrt(hidden_units) if need_scale else 1.0
        self.sf_quant = QuantStub(scale=None)
        self.use_layernorm = use_layernorm
        self.qconfig_func = qconfig_func

        self.wq = nn.Conv2d(q_in_channels, self.all_head_size, kernel_size=1)
        self.wk = nn.Conv2d(k_in_channels, self.all_head_size, kernel_size=1)
        self.wv = nn.Conv2d(k_in_channels, self.all_head_size, kernel_size=1)
        self.q_op = nn.quantized.FloatFunctional()
        self.q_op_2 = nn.quantized.FloatFunctional()

        self.qTk_matmul = FF()
        self.v_attn_matmul = FF()

        # matmul
        self.score_matmul = FF()
        self.softmax_q = nn.Softmax(dim=3)
        self.softmax_k = nn.Softmax(dim=2)

        self.add_op = nn.quantized.FloatFunctional()

        if self.use_layernorm:
            self.qk_norm = hnn.LayerNorm((self.num_k,))

    def forward(self, q_input, k_input, attn_mask=None):
        """Forward.

        Args:
            q_input (torch.Tensor, [num_obs, hidden_size, 1, num_q_feats]): the
                input features 1.
            k_input (torch.Tensor, [num_obs, hidden_size, 1, num_k_feats]): the
                input features 1.
            attn_mask (torch.Tensor, [batch_size, num_poly, num_poly]): the
                attention masks. It is a 0-1 matrix. It does not describe the
                polyline connections, but only identifies whether two polylines
                have the opportunity to interact with each other.

        Returns:
            context (torch.Tensor, [batch_size, all_head_size, 1, 1]): the
            feature between modalities.
        """
        # Calculate q, k, v: [batch_size, num_heads, hidden_units, num]
        kv_shape = [-1, self.num_heads, self.hidden_units, self.num_k]
        q_shape = [-1, self.num_heads, self.hidden_units, self.num_q]
        q_val = self.wq(q_input).reshape(q_shape)
        k_val = self.wk(k_input).reshape(kv_shape)
        v_val = self.wv(k_input).reshape(kv_shape)

        # Calculate self-attention scores.
        # shape: [batch_size, num_heads, 1, num_poly]
        trans_q_val = q_val.permute(0, 1, 3, 2)

        trans_q_val = self.softmax_q(trans_q_val)
        k_val = self.softmax_k(k_val)

        scores = self.qTk_matmul.matmul(trans_q_val, k_val, False, False)

        # matmul
        sf = torch.tensor(
            [1 / self.scale_factor] * self.num_k,
            device=scores.device,
            dtype=torch.float32,
        )
        diag_sf = torch.diag(sf)
        diag_sf = self.sf_quant(diag_sf)
        scores = self.score_matmul.matmul(scores, diag_sf)

        if self.use_layernorm:
            attention = self.qk_norm(scores)
        else:
            attention = scores
        if attn_mask is not None:
            tmp_attn_mask = attn_mask[:, :, :, :].repeat(
                1, self.num_heads, 1, 1
            )
            attention = self.q_op_2.mul(attention, tmp_attn_mask)

        # Calculate features after attention.
        # shape: [batch_size, all_head_size, 1, num_q]
        context = self.v_attn_matmul.matmul(attention, v_val, False, True)
        context = context.permute(0, 1, 3, 2)
        context = context.reshape([-1, self.all_head_size, 1, self.num_q])

        context = self.add_op.add(context, q_input)
        return context

    def set_qconfig(self):
        if self.qconfig_func is None:
            self.qconfig = qconfig_manager.get_default_qat_qconfig()
        else:
            self.sf_quant.qconfig = qint8_freezescale_qconfig()
            self.wq.qconfig = qint8_freezescale_qconfig()
            self.wk.qconfig = qint8_freezescale_qconfig()
            self.wv.qconfig = qint8_freezescale_qconfig()
            self.q_op.qconfig = qint8_freezescale_qconfig()
            self.q_op_2.qconfig = qint16_freezescale_qconfig()
            self.qTk_matmul.qconfig = qint8_freezescale_qconfig()
            self.v_attn_matmul.qconfig = qint8_freezescale_qconfig()
            self.score_matmul.qconfig = qint8_freezescale_qconfig()
            self.softmax_q.qconfig = qint8_freezescale_qconfig()
            self.softmax_k.qconfig = qint8_freezescale_qconfig()
