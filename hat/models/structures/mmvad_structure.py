# Copyright (c) Horizon Robotics, All rights reserved.

from typing import Mapping, Optional

import torch
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.quantization import DeQuantStub

from hat.models.frame_utils import pad_and_split_v2
from hat.registry import OBJECT_REGISTRY

__all__ = ["TriMMVADStructure"]


@OBJECT_REGISTRY.register
class TriMMVADStructure(nn.Module):
    """
        多模语音端点检测模型的结构.

    Args:
        vision_net: 视觉特征提取网络.
        trihead_net: 三个检测头的多模语音端点检测网络.
        loss: 损失函数.
        loss_weight: 三个检测头的损失权重.
        num_cached_frames: 时间维度上的窗长.
        replicate_type: 时间维度上的拼接方法.
    """

    def __init__(
        self,
        vision_net: nn.Module,
        trihead_net: nn.Module,
        loss: nn.Module,
        loss_weight: list,
        num_cached_frames: int,
        replicate_type: str,
    ):
        super(TriMMVADStructure, self).__init__()
        self.vision_net = vision_net
        self.trihead_net = trihead_net
        self.loss = loss
        self.loss_weight = loss_weight
        self.num_cached_frames = num_cached_frames
        self.replicate_type = replicate_type
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.a_quant = QuantStub()
        self.v_quant = QuantStub()
        self.dequant = DeQuantStub()

    def pre_vision_preprocess(self, x):
        shape = x.shape
        x_batch_flatten = torch.reshape(
            x, [shape[0] * shape[1], shape[2], shape[3], shape[4]]
        )
        x = x_batch_flatten
        return x

    def pre_mmvad_preprocess(self, ax, vx):
        batch_size, seq_len = ax.shape[0], ax.shape[1]
        vx_num_channel = vx.shape[1]
        vx = torch.reshape(vx, [batch_size, seq_len, vx_num_channel])
        vx = pad_and_split_v2(vx, self.num_cached_frames, self.replicate_type)
        vx = torch.reshape(vx, [batch_size * seq_len, vx_num_channel, 1, -1])
        ax = torch.reshape(ax, [batch_size, seq_len, -1])
        ax_num_channel = ax.shape[2]
        ax = pad_and_split_v2(ax, self.num_cached_frames, self.replicate_type)
        ax = torch.reshape(ax, [batch_size * seq_len, ax_num_channel, 1, -1])
        return ax, vx

    def forward(self, data: Mapping[str, Optional[torch.Tensor]]):
        images = data["images"]
        ax = data["audio"]
        masks = data["masks"]
        vx = self.pre_vision_preprocess(images)
        vx = self.vision_net(vx)  # TODO: vision模块也需要改成structure中做quant
        ax, vx = self.pre_mmvad_preprocess(ax, vx)
        ax = self.a_quant(ax)
        vx = self.v_quant(vx)
        av_out, a_out, v_out = self.trihead_net(ax, vx)
        av_out = self.dequant(av_out)
        a_out = self.dequant(a_out)
        v_out = self.dequant(v_out)
        label = data["labels"]
        av_out = av_out.reshape(-1, 2)
        a_out = a_out.reshape(-1, 2)
        v_out = v_out.reshape(-1, 2)
        label = label.reshape(-1, 1)
        av_loss = self.loss(av_out, label)
        av_loss = av_loss["focal_loss"]
        a_loss = self.loss(a_out, label)
        a_loss = a_loss["focal_loss"]
        v_loss = self.loss(v_out, label)
        v_loss = v_loss["focal_loss"]

        losses = (
            av_loss * self.loss_weight[0]
            + a_loss * self.loss_weight[1]
            + v_loss * self.loss_weight[2]
        ) / torch.sum(masks)

        return losses, av_loss, a_loss, v_loss, label, av_out, a_out, v_out

    def fuse_model(self):
        for module in [self.vision_net, self.trihead_net]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.vision_net, self.trihead_net]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        if self.loss is not None:
            self.loss.qconfig = None
