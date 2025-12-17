# Copyright (c) Horizon Robotics. All rights reserved.

import logging

import numpy as np
import torch
import torch.nn as nn
from spconv.pytorch import ops
from spconv.pytorch.conv import SparseConv3d, SubMConv3d
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.pool import SparseMaxPool3d

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__file__)

__all__ = ["SpMiddleModel"]


def conv3x3(
    in_planes,
    out_planes,
    stride=1,
    indice_key=None,
    bias=True,
    algo=ops.ConvAlgo.Native,
):

    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
        algo=algo,
    )


def conv1x1(
    in_planes,
    out_planes,
    stride=1,
    indice_key=None,
    bias=True,
    algo=ops.ConvAlgo.Native,
):

    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
        algo=algo,
    )


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}


def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer.

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer

    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


class SparseBasicBlock(SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
        **kwargs,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = {"type": "BN1d", "eps": 1e-3, "momentum": 0.01}

        bias = norm_cfg is not None
        algo = kwargs.get("algo", ops.ConvAlgo.Native)

        self.add = inplanes == planes

        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            indice_key=indice_key,
            bias=bias,
            algo=algo,
        )
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(
            planes, planes, indice_key=indice_key, bias=bias, algo=algo
        )
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.add:
            out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SparseResidualBottleNeck(SparseModule):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        padding=0,
        norm_cfg=None,
        indice_key_num=1,
        **kwargs,
    ):

        super(SparseResidualBottleNeck, self).__init__()

        self.expansion = 2

        if norm_cfg is None:
            norm_cfg = {"type": "BN1d", "eps": 1e-3, "momentum": 0.01}
        bias = norm_cfg is not None
        algo = kwargs.get("algo", ops.ConvAlgo.Native)

        expand_inplanes = (
            inplanes * self.expansion if stride == 1 else inplanes
        )

        self.add = inplanes == planes and stride == 1

        self.conv1 = conv1x1(
            inplanes,
            expand_inplanes,
            indice_key="res{}".format(indice_key_num),
            bias=bias,
            algo=algo,
        )
        self.bn1 = build_norm_layer(norm_cfg, expand_inplanes)[1]

        if stride > 1:
            self.conv2 = SparseConv3d(
                expand_inplanes,
                expand_inplanes,
                3,
                stride,
                padding=padding,
                bias=bias,
                algo=algo,
            )
        else:
            self.conv2 = conv3x3(
                expand_inplanes,
                expand_inplanes,
                indice_key="res{}".format(indice_key_num),
                bias=bias,
                algo=algo,
            )
        self.bn2 = build_norm_layer(norm_cfg, expand_inplanes)[1]

        if stride > 1:
            indice_key_num += 1
        self.conv3 = conv1x1(
            expand_inplanes,
            planes,
            indice_key="res{}".format(indice_key_num),
            bias=bias,
            algo=algo,
        )
        self.bn3 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv3(out)
        out = out.replace_feature(self.bn3(out.features))

        if self.add:
            out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


@OBJECT_REGISTRY.register_module
class SpMiddleModel(nn.Module):
    def __init__(
        self, model_cfg, num_input_features=128, norm_cfg=None, **kwargs
    ):
        """Construct 3D sparse conv backbone based on model_cfg.

        See AFDetV2 <https://arxiv.org/pdf/2112.09205.pdf> for detail.

        Args:
            model_cfg: config for building model.
            num_input_features: number of input voxel features.
            norm_cfg: config for normalization used in model.

        """
        super(SpMiddleModel, self).__init__()

        if norm_cfg is None:
            norm_cfg = {"type": "BN1d", "eps": 1e-3, "momentum": 0.01}
        self.norm_cfg = norm_cfg

        self.num_input_features = num_input_features
        use_BGG = kwargs.get("use_BGG", False)
        if use_BGG:
            print(
                "Using BatchGemmGather algorithm for spconv, make sure you are in inference mode!"  # noqa
            )
        self.algo = (
            ops.ConvAlgo.BatchGemmGather if use_BGG else ops.ConvAlgo.Native
        )

        model_configs = self._decode_model_cfg(model_cfg)
        self._build_model(model_configs)

        # perform feature transpose
        self.feature_transpose = kwargs.get("feature_transpose", False)

    def _decode_model_cfg(self, model_cfg):
        def _split_dimension(x):
            if len(x) > 1:
                return tuple(int(xx) for xx in x)
            else:
                return int(x)

        configs = []

        layers = model_cfg.split(";")
        self.name = layers[0][1:]
        for s in layers[1:-1]:
            config = {}
            left_brace_pos = s.find("(")
            layer_type = s[1:left_brace_pos]
            right_brace_pos = s.find(")", left_brace_pos + 1)

            params = s[left_brace_pos + 1 : right_brace_pos]
            params = params.split(", ")

            out_channels = int(params[0])
            kernel_size = "1" if len(params) < 2 else params[1]
            stride = "1" if len(params) < 3 else params[2]
            padding = "0" if len(params) < 4 else params[3]

            kernel_size = _split_dimension(kernel_size)
            stride = _split_dimension(stride)
            padding = _split_dimension(padding)

            params = {
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
            }

            # find times
            times_pos = s.find("*", right_brace_pos + 1)
            if times_pos > 0:
                times = int(s[times_pos + 2])
            else:
                times = 1

            config["type"] = layer_type
            config["params"] = params
            config["times"] = times
            configs.append(config)

        return configs

    def _build_model(self, model_configs):

        layer_dict = {
            "conv": SparseConv3d,
            "subm": SubMConv3d,
            "bloc": SparseBasicBlock,
            "rsbn": SparseResidualBottleNeck,
            "pool": SparseMaxPool3d,
        }

        self.middle_conv = SparseSequential()
        indice_key = 0
        last_layer_channel = self.num_input_features
        for i, config in enumerate(model_configs):
            layer_type = config["type"]
            params = config["params"]
            times = config["times"]
            layer = layer_dict[layer_type]

            if layer_type == "bloc":
                planes = params["out_channels"]
                inplanes = last_layer_channel
                for _ in range(times):
                    self.middle_conv.add(
                        layer(
                            inplanes,
                            planes,
                            norm_cfg=self.norm_cfg,
                            indice_key="res{}".format(indice_key),
                            algo=self.algo,
                        )
                    )
            elif layer_type == "rsbn":
                planes = params["out_channels"]
                stride = params["stride"]
                padding = params["padding"]
                inplanes = last_layer_channel
                for t in range(times):
                    if t > 0:
                        inplanes = planes
                    self.middle_conv.add(
                        layer(
                            inplanes,
                            planes,
                            stride,
                            padding=padding,
                            norm_cfg=self.norm_cfg,
                            indice_key_num=indice_key,
                            algo=self.algo,
                        )
                    )
                if stride > 1:
                    indice_key += 1
            elif layer_type == "pool":
                pool_size = list(params["kernel_size"])
                self.middle_conv.add(layer(pool_size))
            else:
                params["in_channels"] = last_layer_channel
                params["bias"] = False
                params["algo"] = self.algo
                if layer_type == "subm":
                    params["indice_key"] = "res{}".format(indice_key)
                if layer_type == "conv":
                    indice_key += 1
                for _ in range(times):
                    self.middle_conv.add(layer(**params))
                    self.middle_conv.add(
                        build_norm_layer(
                            self.norm_cfg, params["out_channels"]
                        )[1]
                    )
                    self.middle_conv.add(
                        nn.ReLU(
                            inplace=True if i < len(model_configs) else False
                        )
                    )

            last_layer_channel = params["out_channels"]

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape.tolist()[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        if self.feature_transpose:
            ret = torch.flip(ret.permute([0, 1, 3, 2]), [2, 3])

        return ret
