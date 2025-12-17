from typing import Dict, List, Optional

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["ActMultiModalityEncoder", "ActKPSEncoder", "ActRGBEncoder"]


@OBJECT_REGISTRY.register
class ActMultiModalityEncoder(nn.Module):
    """Encoder of multimodality input for action task, such as gesture.

    Args:
        network_kps: backbone module of kps branch.
        network_frames: backbone module of frames branch
        img_seq_len: length of img seqence.
            Defaults to 8.
        proj_channels: channels of projection layer.
            Defaults to None.
        bn_kwargs: params of bn.
            Defaults to None.
        add_motion_in_rgb_branch: whether to add motion in rgb-frames branch.
            Defaults to False.
        disable_quanti_input: whether quanti input.
            Defaults to False.
        enable_kps_gap: whether use gap layer for kps branch.
            Defaults to False.
        enable_frames_temporal_gap: whether use gap layer for frames branch.
            Defaults to False.
    """

    def __init__(
        self,
        network_kps: nn.Module,
        network_frames: nn.Module,
        img_seq_len: int = 8,
        proj_channels: List = None,
        bn_kwargs: Optional[Dict] = None,
        add_motion_in_rgb_branch: Optional[bool] = False,
        disable_quanti_input: Optional[bool] = False,
        enable_kps_gap: Optional[bool] = False,
        enable_frames_temporal_gap: Optional[bool] = False,
    ):

        super(ActMultiModalityEncoder, self).__init__()
        self.bn_kwargs = bn_kwargs or {}
        # for kps branch
        self.network_kps = network_kps
        self.enable_kps_gap = enable_kps_gap
        if self.enable_kps_gap:
            self.gap_layer_kp = nn.AvgPool2d(7)
        self.proj_layer_kps = self.get_projection_layer(
            proj_channels[0], modality="kps"
        )

        # for rgb branch
        self.network_frames = network_frames
        self.img_seq_len = img_seq_len
        self.gap_layer_frames = nn.AvgPool2d(4)
        self.proj_layer_frames = self.get_projection_layer(
            proj_channels[1], modality="frames"
        )
        self.gap_layer_frames = nn.AvgPool2d(4)
        self.enable_frames_temporal_gap = enable_frames_temporal_gap
        if self.enable_frames_temporal_gap:
            self.gap_layer_frames_temporal = nn.AvgPool2d((2, 1))
        # motion info
        self.add_motion_in_rgb_branch = add_motion_in_rgb_branch

        if self.add_motion_in_rgb_branch:
            self.disable_quanti_input = disable_quanti_input
            self.motion_quant = QuantStub()
            self.motion_concat = nn.quantized.FloatFunctional()

    def get_projection_layer(self, proj_channels, modality):
        kernel_size = (3, 1) if modality == "frames" else (1, 1)
        stride = (2, 1) if modality == "frames" else (1, 1)
        padding = (1, 0) if modality == "frames" else (0, 0)
        projection_layer = []
        for in_channels, out_channels in proj_channels:
            projection_layer.append(
                ConvModule2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    norm_layer=nn.BatchNorm2d(
                        out_channels,
                        **self.bn_kwargs,
                    ),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        return nn.Sequential(*projection_layer)

    def fuse_model(self):
        for mod in [
            self.network_kps,
            self.network_frames,
        ]:
            if mod is not None and hasattr(mod, "fuse_model"):
                mod.fuse_model()

        for module in [
            self.proj_layer_kps,
            self.proj_layer_frames,
        ]:
            for mod in module:
                if mod is not None and hasattr(mod, "fuse_model"):
                    mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [
            self.network_kps,
            self.proj_layer_kps,
            self.network_frames,
            self.proj_layer_frames,
        ]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

    def forward(self, data):
        # kps branch
        keypoints = data["clip_keypoints"]
        feature_kps = self.network_kps(keypoints)[-1]
        if self.enable_kps_gap:
            feature_kps = self.gap_layer_kps(feature_kps)
        feature_kps = self.proj_layer_kps(feature_kps)

        # rgb branch - frames
        frames = data["frames"]
        feature_frames = self.network_frames(frames)[-1]
        feature_frames = self.gap_layer_frames(feature_frames)
        if self.add_motion_in_rgb_branch:
            if self.disable_quanti_input:
                motion_input = data["box_center"]
            else:
                motion_input = self.motion_quant(data["box_center"])
            feature_frames = self.motion_concat.cat(
                (feature_frames, motion_input), dim=1
            )

        _, C, H, W = feature_frames.shape
        B = feature_kps.shape[0]
        feature_frames = feature_frames.view(-1, self.img_seq_len, C, H, W)
        feature_frames = feature_frames.permute(0, 2, 1, 3, 4)
        feature_frames = feature_frames.view(B, C, self.img_seq_len, -1)
        feature_frames = self.proj_layer_frames(feature_frames)
        if self.enable_frames_temporal_gap:
            feature_frames = self.gap_layer_frames_temporal(feature_frames)
        return feature_kps, feature_frames


@OBJECT_REGISTRY.register
class ActRGBEncoder(nn.Module):
    """
    The encoder for respective branches.

    Args:
        backbone: Backbone module.
        img_seq_len: The length of img branch.
        bn_kwargs: arguments of BN.
        add_motion_in_rgb_branch: Whether to add motion information
            to the rgb branch.
        disable_quanti_input: Whether to disable quanti input.
        mode: training mode: 'train'; split model: 'rgb' 'kps' 'head';
    """

    def __init__(
        self,
        backbone: nn.Module,
        img_seq_len: int = 8,
        bn_kwargs: Optional[Dict] = None,
        add_motion_in_rgb_branch: Optional[bool] = False,
        disable_quanti_input: Optional[bool] = False,
        mode: str = "train",
    ):

        super(ActRGBEncoder, self).__init__()
        self.bn_kwargs = bn_kwargs or {}

        self.backbone = backbone
        self.img_seq_len = img_seq_len
        self.gap_layer_frames = nn.AvgPool2d(4)
        self.add_motion_in_rgb_branch = add_motion_in_rgb_branch
        self.mode = mode

        if self.add_motion_in_rgb_branch:
            self.disable_quanti_input = disable_quanti_input
            self.motion_quant = QuantStub()
            self.motion_concat = nn.quantized.FloatFunctional()

    def fuse_model(self):
        for mod in [
            self.backbone,
        ]:
            if mod is not None and hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [
            self.backbone,
        ]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

    def forward(self, frames_data):

        if self.add_motion_in_rgb_branch:
            assert len(frames_data) == 2
            frames, motion_input = frames_data
        else:
            frames = frames_data[0]

        # rgb branch - frames
        feature_frames = self.backbone(frames)[-1]
        feature_frames = self.gap_layer_frames(feature_frames)
        if self.add_motion_in_rgb_branch:
            if self.disable_quanti_input:
                motion_input = motion_input
            else:
                motion_input = self.motion_quant(motion_input)
            feature_frames = self.motion_concat.cat(
                (feature_frames, motion_input), dim=1
            )

        if self.mode == "train":
            BT, C, H, W = feature_frames.shape
            B = int(BT / self.img_seq_len)
            feature_frames = feature_frames.view(
                B, self.img_seq_len, C, H, W
            )  # [64, 8, 128, 1, 1]
            feature_frames = feature_frames.permute(
                0, 2, 1, 3, 4
            )  # [64, 128, 8, 1, 1]
            feature_frames = feature_frames.view(
                B, C, self.img_seq_len, -1
            )  # [64, 128, 8, 1]
        return feature_frames


@OBJECT_REGISTRY.register
class ActKPSEncoder(nn.Module):
    """
    The encoder for respective branches.

    Args:
        backbone: Backbone module.
        bn_kwargs: arguments of BN.

    """

    def __init__(
        self,
        backbone: nn.Module,
        bn_kwargs: Optional[Dict] = None,
    ):

        super(ActKPSEncoder, self).__init__()
        self.backbone = backbone
        self.bn_kwargs = bn_kwargs or {}

    def fuse_model(self):
        for mod in [
            self.backbone,
        ]:
            if mod is not None and hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, 2]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

    def forward(self, keypoints):
        feature_kps = self.backbone(keypoints)[-1]

        return feature_kps
