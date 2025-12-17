# Copyright (c) Horizon Robotics. All rights reserved.
from functools import partial
from typing import Dict, List

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.backbones.resnet import BottleNeck, ResNet
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["RewardModel", "PatialResNet"]


@OBJECT_REGISTRY.register
class RewardModel(nn.Module):
    """
    Take the raw features from backbone and downsample to customized \
    grid size, using logsigmoid acitvation to generate spatial \
    reward feature.

    Args:
        in_channels : input feature channels.
        scene_feat_size : extra feature size.
        grid_cell_size : grid conv kernel size.
        agg_sizes : grid feature size.
        encode_motion: whether to enable encode ego motion into feature.
        motion_size: ego motion channel dim, valie only when encode_motion
                     is true.
    """

    def __init__(
        self,
        in_feat_size: int,
        scene_feat_size: int,
        agg_size: List[int],
        grid_cell_size: List[int],
        encode_motion: bool = False,
        motion_size: int = 3,
    ):
        super(RewardModel, self).__init__()

        self.encode_motion = encode_motion

        # Layers for aggregating motion and scene features:
        self.conv_grid = ConvModule2d(
            in_feat_size,
            scene_feat_size,
            grid_cell_size,
            stride=grid_cell_size,
            padding=1,
            act_layer=nn.ReLU(inplace=True),
        )

        # Conv layers to aggregate context at each grid location
        agg_in_size = (
            scene_feat_size + motion_size
            if self.encode_motion
            else scene_feat_size
        )
        self.agg1_path = ConvModule2d(
            agg_in_size, agg_size[0], 1, act_layer=nn.ReLU(inplace=True)
        )
        self.agg2_path = ConvModule2d(
            agg_size[0], agg_size[1], 1, act_layer=nn.ReLU(inplace=True)
        )

        # Output layers:
        self.op_path = nn.Conv2d(agg_size[1], 1, 1)
        # Non-linearities:
        self.sigmoid = nn.Sigmoid()

        self.quant = QuantStub(scale=None)
        self.log_op = horizon.nn.HardLog()
        self.cat_op = nn.quantized.FloatFunctional()
        self.cat_op1 = nn.quantized.FloatFunctional()

    def forward(self, data: Dict):
        """Forward.

        Args:
            data (dict): the model input with the following required keys:
            feats: features generated from backbone.
            plan_ego_motion(optional): ego motion tensors.

        Return:
            results (Dict): features including:
            feats : features that encodes the extra states.
            reward: features that activate by nonlinear.
        """

        img_feats = data["feats"]
        img_feats = self.conv_grid(img_feats)

        if self.encode_motion:
            # Extract data
            state_vectors = data["plan_ego_motion"]
            state_vectors = self.quant(state_vectors)
            img_feats = self.cat_op1.cat((img_feats, state_vectors), dim=1)

        # # Output goal and path rewards
        out = self.op_path(self.agg2_path(self.agg1_path(img_feats)))

        reward = self.sigmoid(out)
        reward = torch.clamp(reward, min=1e-4)
        reward = self.log_op(reward)

        results = {
            "feats": img_feats,
            "reward": reward,
        }

        return results

    def fuse_model(self):
        modules = [self.conv_grid, self.agg1_path, self.agg2_path]
        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.op_path.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )
        self.sigmoid.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )
        self.log_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )


@OBJECT_REGISTRY.register
class PatialResNet(ResNet):
    """
    A simple module of resnet for planning task.

    Args:
        input_channels (int): channels of input feature.
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for BN layer.
        bias (bool): Whether to use bias in module.
    """

    def __init__(
        self,
        input_channels: int,
        bn_kwargs: dict,
        bias: bool = True,
        stride_change: bool = False,
    ):
        super(ResNet, self).__init__()
        unit = [3, 4, 6, 3]
        channels_list = [64, 64, 128, 256, 512]
        self.basic_block = partial(BottleNeck, stride_change=stride_change)
        self.expansion = 4
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.include_top = False
        self.in_channels = channels_list[0]

        self.mod1 = nn.Sequential(
            ConvModule2d(
                input_channels,
                channels_list[0],
                7,
                stride=2,
                padding=3,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channels_list[0], **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.mod2 = self._make_stage(channels_list[1], unit[0], 1)

    def forward(self, x):
        output = []
        # x = self.quant(x)
        for module in [self.mod1, self.mod2]:
            x = module(x)
            output.append(x)
        # x = self.dequant(x)
        return x

    def fuse_model(self):
        modules = [self.mod1, self.mod2]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()
