# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Callable, Dict, Optional

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.task_modules.traj_pred.base_modules.cross_attention import (
    qint8_freezescale_qconfig,
    qint16_freezescale_qconfig,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager

__all__ = ["VectorNetBackbone"]


class SubGraphLayer(nn.Module):
    """The sub graph layer of VectorNet (Section 3.2 in the paper)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_vec: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        use_relu: bool = True,
        use_pool: bool = True,
        use_layernorm: bool = True,
        use_qint16: bool = False,
        qconfig_func: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels of the sub graph layer.
            out_channels: the output channels of the sub graph layer.
            num_vec: the number of vectors in a polyline.
            kernel_size: the kernal size.
            stride: the stride.
            padding: the padding.
            use_relu: whether to use relu. Defaults to True.
            use_pool: whether to use max pooling. Defaults to True. If the
                parameter is False, this layer degenerates into an simple
                MLP layer. Note that the first 1/2 of out channels
                are calculated by the MLP, the last 1/2 are derived from
                max-pooling. Therefore, if the user uses this layer as simple
                MLP layer, the `use_pool` should be False and `out_channels`
                should be twice the actual output dimension.
            use_layernorm: whether to use layernorm.
            use_qint16: use qint16 to quantize.
        """
        super(SubGraphLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vec = num_vec
        self.use_relu = use_relu
        self.use_pool = use_pool
        self.use_layernorm = use_layernorm
        self.use_qint16 = use_qint16
        self.qconfig_func = qconfig_func
        if self.use_pool:
            self.hidden_size = out_channels // 2
        else:
            self.hidden_size = out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if self.use_relu:
            self.relu = nn.ReLU6()

        if self.use_layernorm:
            self.norm = hnn.LayerNorm((self.hidden_size,))

        if self.use_pool:
            assert num_vec > 1, (
                "The number of vectors in a polyline should > 1 when using "
                "pooling."
            )
            self.max_pool = nn.MaxPool2d([1, num_vec - 1], stride=1)
            self.cat_op = nn.quantized.FloatFunctional()

    def forward(self, data):
        """Forward.

        Args:
            data (torch.tensor, [batch_size, num_feats, num_poly, num_vec]):
                the input features.

        Returns:
            out_feat (torch.tensor, [batch_size, out_channels, num_poly,
                num_vec]): the output features.
        """
        # MLP-norm: [batch_size, hidden_size, num_poly, num_vec]
        mlp_feat = self.conv(data)

        if self.use_relu:
            mlp_feat = self.relu(mlp_feat)

        if self.use_layernorm:
            mlp_feat = self.norm(mlp_feat.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )

        # Max pooling: [batch_size, hidden_size, num_poly, num_vec]
        if self.use_pool:
            pool_feat = []
            for i in range(self.num_vec):
                cat_feat = []
                if i > 0:
                    cat_feat.append(mlp_feat[:, :, :, :i])
                if i < self.num_vec - 1:
                    cat_feat.append(mlp_feat[:, :, :, (i + 1) :])
                tmp_feat = self.cat_op.cat(cat_feat, dim=-1)
                pool_feat.append(self.max_pool(tmp_feat))

            pool_feat = self.cat_op.cat(pool_feat, dim=-1)
            # Concatenate: [batch_size, out_channels, num_poly, num_vec]
            out_feat = self.cat_op.cat([mlp_feat, pool_feat], dim=1)
            return out_feat
        else:
            return mlp_feat

    def fuse_model(self):
        if self.use_relu:
            torch.quantization.fuse_modules(
                self,
                ["conv", "relu"],
                inplace=True,
                fuser_func=horizon.quantization.fuse_known_modules,
            )

    def set_qconfig(self):
        if self.use_qint16:
            if self.qconfig_func is None:
                self.qconfig = qconfig_manager.get_qconfig(
                    activation_qat_qkwargs={"dtype": qint16},
                    activation_calibration_qkwargs={"dtype": qint16},
                )
            else:
                self.qconfig = self.qconfig_func()
        else:
            self.qconfig = qconfig_manager.get_default_qat_qconfig()


class BasicSubGraph(nn.Module):
    """Subgraph Module of VectorNet (Section 3.2 in the paper).

    This graph extracts polyline-level feature.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_poly: int,
        num_vec: int,
        num_layers: int = 3,
    ):
        """Initialize method.

        Args:
            in_channels (int): the input channels of the sub graph layer.
            out_channels (int): the output channels of the sub graph layer.
            num_poly (int): the number of polylines.
            num_vec (int): the number of vectors in a polyline.
            num_layers (int, optional): the number of sub graph layers.
                Defaults to 3.
        """
        super(BasicSubGraph, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_poly = num_poly
        self.num_vec = num_vec
        self.num_layers = num_layers

        layers = []
        for _ in range(num_layers):
            layers.append(
                SubGraphLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_vec=num_vec,
                    use_layernorm=False,
                    use_pool=False,
                )
            )
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)
        self.max_pool = nn.MaxPool2d([1, self.num_vec], stride=1)

    def forward(self, data):
        """Forward.

        Args:
            data (torch.tensor, [batch_size, num_feats, num_poly, num_vec]):
                the input features.

        Return:
            out_feat (torch.tensor, [batch_size, out_channels, 1, num_poly]):
                the output features.
        """
        feats = data
        for layer in self.layers:
            feats = layer(feats)
        # Pooling: [batch_size, out_channels, 1, num_poly]
        out_feat = self.max_pool(feats)
        out_feat = out_feat.permute(0, 1, 3, 2)
        return out_feat

    def fuse_model(self):
        for module in self.layers:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in self.layers:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


class BasicGlobalGraph(nn.Module):
    """Global graph Module of VectorNet (Section 3.3 in the paper).

    As descripted in the original paper, this graph models higher-level
    interactions between polyline segments (whose features are extracted by
    SubGraph). This class is actually a simple self-attention module. It
    actually abandons intuitive geometric connections between polylines.
    The developers can also try other structures to model the interaction
    (try more explicitly).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_units: int,
        num_poly: int,
        num_attention_heads: int,
        need_scale: bool = False,
        use_layernorm: bool = True,
        qconfig_func: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels.
            hidden_units: the number of hidden units in each attention
                heads.
            num_poly: the number of polylines.
            num_attention_heads: the number of attention heads.
            need_scale: whether to consider scale factor during the
                calculation of attention.
        """
        super(BasicGlobalGraph, self).__init__()
        self.in_channels = in_channels
        self.hidden_units = hidden_units
        self.num_poly = num_poly
        self.num_heads = num_attention_heads
        self.all_head_size = hidden_units * num_attention_heads
        self.scale_factor = np.sqrt(hidden_units) if need_scale else 1.0
        self.sf_quant = QuantStub(scale=None)
        self.use_layernorm = use_layernorm
        self.qconfig_func = qconfig_func

        self.wq = nn.Conv2d(in_channels, self.all_head_size, kernel_size=1)
        self.wk = nn.Conv2d(in_channels, self.all_head_size, kernel_size=1)
        self.wv = nn.Conv2d(in_channels, self.all_head_size, kernel_size=1)
        self.q_op = nn.quantized.FloatFunctional()
        self.q_op_2 = nn.quantized.FloatFunctional()
        self.qTk_matmul = FF()
        self.v_attn_matmul = FF()

        if self.use_layernorm:
            self.qk_norm = hnn.LayerNorm((self.num_poly,))

    def forward(self, data, attn_mask=None):
        """Forward.

        Args:
            data (torch.Tensor, [batch_size, in_channels, 1, num_poly]): the
                input features. The feature of the prediction target should
                be located at index = 0 in the second dimension.
            attn_mask (torch.Tensor, [batch_size, num_poly, num_poly]): the
                attention masks. It is a 0-1 matrix. It does not describe the
                polyline connections, but only identifies whether two polylines
                have the opportunity to interact with each other.

        Returns:
            context (torch.Tensor, [batch_size, all_head_size, 1, 1]): the
                graph-level feature surrounding the predition target.
        """
        # Calculate q, k, v: [batch_size, num_heads, hidden_units, num_poly]
        kv_shape = [-1, self.num_heads, self.hidden_units, self.num_poly]
        q_shape = [-1, self.num_heads, self.hidden_units, 1]
        q_val = self.wq(data[:, :, :, 0:1]).reshape(q_shape)
        k_val = self.wk(data).reshape(kv_shape)
        v_val = self.wv(data).reshape(kv_shape)

        # Calculate self-attention scores.
        # shape: [batch_size, num_heads, 1, num_poly]
        # q_val = q_val[:, :, :, 0:1]  # the prediction target is at index = 0
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
            tmp_attn_mask = attn_mask[:, :, 0:1, :].repeat(
                1, self.num_heads, 1, 1
            )
            attention = self.q_op_2.mul(attention, tmp_attn_mask)

        # Calculate features after attention.
        # shape: [batch_size, 1, 1, all_head_size]
        context = self.v_attn_matmul.matmul(attention, v_val, False, True)
        context = context.reshape([-1, self.all_head_size, 1, 1])
        return context

    def fuse_model(self):
        for module in [self.wq, self.wk, self.wv]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        if self.qconfig_func is None:
            self.qconfig = qconfig_manager.get_default_qat_qconfig()
        else:
            self.sf_quant.qconfig = qint16_freezescale_qconfig()
            self.wq.qconfig = qint16_freezescale_qconfig(out_int8=True)
            self.wk.qconfig = qint16_freezescale_qconfig(out_int8=True)
            self.wv.qconfig = qint16_freezescale_qconfig(out_int8=True)
            self.q_op.qconfig = qint16_freezescale_qconfig()
            self.q_op_2.qconfig = qint16_freezescale_qconfig()
            self.qTk_matmul.qconfig = qint8_freezescale_qconfig()
            self.v_attn_matmul.qconfig = qint8_freezescale_qconfig()
            if self.use_layernorm:
                self.qk_norm.qconfig = qint16_freezescale_qconfig(
                    out_int8=True
                )


@OBJECT_REGISTRY.register
class VectorNetBackbone(nn.Module):
    """The VectorNet backbone.

    The structure is designed in the paper:
        VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized
        Representation.
    """

    def __init__(
        self,
        road_feat_channels: int,
        road_polyline_len: int,
        road_polyline_num: int,
        traj_feat_channels: int,
        traj_polyline_len: int,
        traj_polyline_num: int,
        road_feat_quant_scale: float = None,
        traj_feat_quant_scale: float = None,
        num_hidden_units: int = 128,
        num_attn_hidden_units: int = 128,
        num_sub_graph_layers: int = 3,
        num_attention_heads: int = 3,
        use_out_fc: bool = True,
        fc_out_channels: Optional[int] = None,
        output_all_feats: bool = False,
        freeze_grad: bool = False,
    ):
        """Initialize method.

        Args:
            road_feat_channels (int): the feature dimension of road polylines.
            road_polyline_len (int): the length of road polylines.
            road_polyline_num (int): the number of road polylines.
            traj_feat_channels (int): the feature dimension of obstacle
                trajectory polylines.
            traj_polyline_len (int): the length of trajectory polylines.
            traj_polyline_num (int): the number of trajectory polylines.
            road_feat_quant_scale (float): the quant scale of
                struct_road_feats.
            traj_feat_quant_scale (float): the quant scale of
                struct_traj_feats.
            num_hidden_units (int, optional): the hidden unit size of the
                sub graph feature extractor. Defaults to 128.
            num_attn_hidden_units (int, optional): the hidden unit size of the
                global graph attention modules. Defaults to 128.
            num_sub_graph_layers (int, optional): the number of the sub graph
                layers. Defaults to 3.
            num_attention_heads (int, optional): the number of the attention
                heads. Defaults to 3.
            use_out_fc (bool, optional): whether to add a FC layer after the
                global graph modules. Defaults to True.
            fc_out_channels (Optional[int], optional): the output channel of
                the final FC. If this parameter is None, the output channel
                will be set as `num_hidden_units`.
            output_all_feats (bool, optional): whether to return all feats.
                Defaults to False.
            freeze_grad: whether to freeze all gradients.
        """
        super(VectorNetBackbone, self).__init__()
        self.road_feat_channels = road_feat_channels
        self.road_polyline_len = road_polyline_len
        self.traj_feat_channels = traj_feat_channels
        self.traj_polyline_len = traj_polyline_len
        self.num_hidden_units = num_hidden_units
        self.num_sub_graph_layers = num_sub_graph_layers
        self.num_attention_heads = num_attention_heads
        self.use_out_fc = use_out_fc
        if fc_out_channels is None:
            fc_out_channels = num_hidden_units
        self.fc_out_channels = fc_out_channels
        self.q_op = nn.quantized.FloatFunctional()
        self.output_all_feats = output_all_feats

        # Sub graphs.
        self.road_sg = BasicSubGraph(
            in_channels=road_feat_channels,
            out_channels=num_hidden_units,
            num_poly=road_polyline_num,
            num_vec=road_polyline_len,
            num_layers=num_sub_graph_layers,
        )
        self.traj_sg = BasicSubGraph(
            in_channels=traj_feat_channels,
            out_channels=num_hidden_units,
            num_poly=traj_polyline_num,
            num_vec=traj_polyline_len,
            num_layers=num_sub_graph_layers,
        )
        # Global graphs.
        self.global_graph = BasicGlobalGraph(
            in_channels=num_hidden_units,
            hidden_units=num_attn_hidden_units,
            num_poly=road_polyline_num + traj_polyline_num,
            num_attention_heads=num_attention_heads,
            need_scale=True,
            use_layernorm=True,
        )
        self.all_ele = ["road_sg", "traj_sg", "global_graph"]
        # Output FC layer.
        # Note: this is not the official structure of the VectorNet (as shown
        #   in the paper).
        if self.use_out_fc:
            self.fc = SubGraphLayer(
                in_channels=num_attention_heads * num_attn_hidden_units,
                out_channels=self.fc_out_channels,
                num_vec=1,
                use_relu=False,
                use_pool=False,
                use_layernorm=False,
            )
            self.all_ele.append("fc")
        self.road_quant = QuantStub(scale=road_feat_quant_scale)
        self.traj_quant = QuantStub(scale=traj_feat_quant_scale)
        self.attn_quant = QuantStub(scale=None)

        if freeze_grad:
            for key in self.all_ele:
                if hasattr(self, key):
                    element = getattr(self, key)
                    for _, param in element.named_parameters():
                        param.requires_grad = False

    def forward(self, data: Dict):
        """Forward.

        Args:
            data (Dict): the model input dictionary with the following keys:
                "struct_road_feats" (torch.Tensor, [num_obs,
                    road_feat_channels, num_road_poly, road_polyline_len]):
                    the road polyline features.
                "struct_traj_feats" (torch.Tensor, [num_obs,
                    traj_feat_channels, num_traj_poly, traj_polyline_len]):
                    the trajectory polyline features.
                "attention_mask" (torch.Tensor, [num_obs, 1,
                    num_poly, num_poly]):
                    the attention mask to represent if two polylines can have
                    interactions. num_poly = num_road_poly + num_traj_poly.

        Returns:
            graph_feats (torch.Tensor, [num_obs, fc_out_channels, 1, 1]): the
                graph features.
        """
        road_input = self.road_quant(data["struct_road_feats"])
        traj_input = self.traj_quant(data["struct_traj_feats"])
        attention_mask = self.attn_quant(data["attention_mask"])

        # Sub graph feature extracting.
        road_feats = self.road_sg(road_input)
        traj_feats = self.traj_sg(traj_input)
        cat_feats = self.q_op.cat([traj_feats, road_feats], dim=-1)

        # Graph feature extracting.
        # - [batch_size, all_head_size, 1, 1]
        graph_feats = self.global_graph(cat_feats, attention_mask)
        # FC layer.
        if self.use_out_fc:
            graph_feats = self.fc(graph_feats)

        if self.output_all_feats:
            return graph_feats, road_feats, cat_feats, traj_feats
        else:
            return graph_feats

    def fuse_model(self):
        modules = [self.road_sg, self.traj_sg, self.global_graph]
        if self.use_out_fc:
            modules.append(self.fc)
        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        self.road_quant.qconfig = torch.quantization.QConfig(
            activation=horizon.quantization.fake_quantize.default_16bit_fake_quant,  # noqa
            weight=None,
        )
        self.traj_quant.qconfig = torch.quantization.QConfig(
            activation=horizon.quantization.fake_quantize.default_16bit_fake_quant,  # noqa
            weight=None,
        )
        modules = [self.road_sg, self.traj_sg, self.global_graph]
        if self.use_out_fc:
            modules.append(self.fc)
        for module in modules:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
