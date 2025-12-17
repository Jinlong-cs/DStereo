from typing import Dict, Optional

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["ActMultiModalityHead", "StaMultiModalityHead"]


@OBJECT_REGISTRY.register
class ActMultiModalityHead(nn.Module):
    """Head network of  multimodality input for action task, such as gesture.

    Args:
        fusion_type: type of multimodality feature fusion.
            Defaults to "result_fusion".
        use_dropout: whether to use dropout in prediction layer.
            Defaults to False.
        in_channels_dict: dict of input channels.
            Defaults to None.
        num_classes: number of class.
            Defaults to 59.
        flat_output: whether to view the output tensor.
            Defaults to True.
    """

    def __init__(
        self,
        fusion_type: str = "result_fusion",
        use_dropout: Optional[bool] = False,
        in_channels_dict: Dict = None,
        num_classes: int = 59,
        flat_output: bool = True,
    ):

        super(ActMultiModalityHead, self).__init__()
        assert fusion_type in [
            "result_fusion",
            "feature_fusion",
            "cross_modal_fusion",
        ], f"not support fusion_type:{fusion_type}, \
              support type list is  \
             'result_fusion','feature_fusion','cross_modal_fusion'"
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.flat_output = flat_output
        self.dequant = DeQuantStub()

        self.pred_layer_kps = None
        self.pred_layer_frames = None
        self.pred_layer = None
        if self.fusion_type in ["result_fusion", "cross_modal_fusion"]:
            self.pred_layer_kps = self.get_prediction_layer(
                in_channels=in_channels_dict["kps"], num_classes=num_classes
            )
            self.pred_layer_frames = self.get_prediction_layer(
                in_channels=in_channels_dict["frames"], num_classes=num_classes
            )
        if self.fusion_type in ["feature_fusion", "cross_modal_fusion"]:
            self.feature_concat = nn.quantized.FloatFunctional()
            self.pred_layer = self.get_prediction_layer(
                in_channels=in_channels_dict["frames"]
                + in_channels_dict["kps"],
                num_classes=num_classes,
            )

    def get_prediction_layer(self, in_channels, num_classes):
        prediction_layer = []
        if self.use_dropout:
            prediction_layer.append(nn.Dropout(p=0.5))
        prediction_layer.append(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        return nn.Sequential(*prediction_layer)

    def fuse_model(self):
        for module in [
            self.pred_layer_kps,
            self.pred_layer_frames,
            self.pred_layer,
        ]:
            if module is not None:
                for mod in module:
                    if mod is not None and hasattr(mod, "fuse_model"):
                        mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [
            self.pred_layer_kps,
            self.pred_layer_frames,
            self.pred_layer,
        ]:
            if module is not None:
                module[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def forward(self, feature_kps, feature_frames):
        logit_output = [None, None, None]
        if self.fusion_type in ["result_fusion", "cross_modal_fusion"]:
            logit_kps = self.pred_layer_kps(feature_kps)
            logit_kps = self.dequant(logit_kps)
            if self.flat_output:
                logit_kps = logit_kps.view(-1, self.num_classes)
            logit_output[0] = logit_kps

            logit_frames = self.pred_layer_frames(feature_frames)
            logit_frames = self.dequant(logit_frames)
            if self.flat_output:
                logit_frames = logit_frames.view(-1, self.num_classes)
            logit_output[1] = logit_frames

        if self.fusion_type in ["feature_fusion", "cross_modal_fusion"]:
            feature_kps_frames = self.feature_concat.cat(
                (feature_kps, feature_frames), dim=1
            )
            logit_kps_frames = self.pred_layer(feature_kps_frames)
            logit_kps_frames = self.dequant(logit_kps_frames)
            if self.flat_output:
                logit_kps_frames = logit_kps_frames.view(-1, self.num_classes)
            logit_output[2] = logit_kps_frames
        return tuple(logit_output)


@OBJECT_REGISTRY.register
class StaMultiModalityHead(nn.Module):
    """
    Head network of multimodality input for action task, such as sta gesture.

    Args:
        fusion_type: type of multimodality feature fusion.
            Defaults to "result_fusion".
        use_dropout: whether to use dropout in prediction layer.
            Defaults to False.
        in_channels_dict: dict of input channels.
            Defaults to None.
        num_classes: number of class.
            Defaults to 71.
        flat_output: whether to view the output tensor.
            Defaults to True.
        bn_kwargs: arguments of BN.
        mode: training mode: 'train'; split model: 'rgb' 'kps' 'head';
    """

    def __init__(
        self,
        fusion_type: str = "result_fusion",
        use_dropout: Optional[bool] = False,
        in_channels_dict: Dict = None,
        num_classes: int = 71,
        flat_output: bool = True,
        bn_kwargs: Optional[Dict] = None,
        add_motion_in_rgb_branch: Optional[bool] = False,
        mode: str = "train",
    ):

        super(StaMultiModalityHead, self).__init__()
        assert fusion_type in [
            "result_fusion",
            "feature_fusion",
            "cross_modal_fusion",
        ], f"not support fusion_type:{fusion_type}, \
              support type list is  \
             'result_fusion','feature_fusion','cross_modal_fusion'"
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.flat_output = flat_output
        self.bn_kwargs = bn_kwargs or {}
        self.add_motion_in_rgb_branch = add_motion_in_rgb_branch
        self.mode = mode
        self.dequant = DeQuantStub()
        self.pred_layer_kps = None
        self.pred_layer_frames = None
        self.pred_layer = None
        if self.fusion_type in ["result_fusion", "cross_modal_fusion"]:
            self.pred_layer_kps = self.get_prediction_layer(
                in_channels=in_channels_dict["kps"], num_classes=num_classes
            )
            self.pred_layer_frames = self.get_prediction_layer(
                in_channels=in_channels_dict["frames"], num_classes=num_classes
            )
        if self.fusion_type in ["feature_fusion", "cross_modal_fusion"]:
            self.feature_concat = nn.quantized.FloatFunctional()
            self.pred_layer = self.get_prediction_layer(
                in_channels=in_channels_dict["frames"]
                + in_channels_dict["kps"],
                num_classes=num_classes,
            )

        self.frames_neck = self.get_frames_neck()
        self.kps_neck = self.get_kps_neck()

        if self.mode == "kps":
            self.frames_neck = None
            self.pred_layer_frames = None

        if self.mode == "rgb" or self.mode == "head":
            self.kps_neck = None
            self.pred_layer_kps = None

        self.quant = QuantStub(scale=1.0 / 128.0)

    def get_frames_neck(self):
        neck_layer = []
        if self.add_motion_in_rgb_branch:
            feature_dim = 160
        else:
            feature_dim = 128
        neck_layer.append(
            ConvModule2d(
                in_channels=feature_dim,
                out_channels=512,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
                bias=True,
                act_layer=nn.ReLU(inplace=True),
                norm_layer=nn.BatchNorm2d(512, **self.bn_kwargs),
            )
        )
        neck_layer.append(
            ConvModule2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
                bias=True,
                act_layer=nn.ReLU(inplace=True),
                norm_layer=nn.BatchNorm2d(512, **self.bn_kwargs),
            )
        )
        neck_layer.append(nn.AvgPool2d((2, 1), padding=(0, 0), stride=(1, 1)))
        return nn.Sequential(*neck_layer)

    def get_kps_neck(self):
        neck_layer = []
        neck_layer.append(
            ConvModule2d(
                in_channels=128,
                out_channels=512,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
                act_layer=nn.ReLU(inplace=True),
                norm_layer=nn.BatchNorm2d(512, **self.bn_kwargs),
            )
        )

        return nn.Sequential(*neck_layer)

    def get_prediction_layer(self, in_channels, num_classes):
        prediction_layer = []
        if self.use_dropout:
            prediction_layer.append(nn.Dropout(p=0.5))
        prediction_layer.append(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        return nn.Sequential(*prediction_layer)

    def fuse_model(self):
        for module in [
            self.pred_layer_kps,
            self.pred_layer_frames,
            self.pred_layer,
            self.frames_neck,
            self.kps_neck,
        ]:
            if module is not None:
                for mod in module:
                    if mod is not None and hasattr(mod, "fuse_model"):
                        mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [
            self.pred_layer_kps,
            self.pred_layer_frames,
        ]:
            if module is not None:
                module[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def forward(self, feature_list):

        if self.mode == "kps":
            feature_kps = feature_list[0]
            feature_kps = self.kps_neck(feature_kps)
        elif self.mode == "head":
            feature_frames = feature_list[0]
            feature_frames = self.quant(feature_frames)
            feature_frames = self.frames_neck(feature_frames)
        elif self.mode == "train":
            feature_kps, feature_frames = feature_list
            feature_kps = self.kps_neck(feature_kps)
            feature_frames = self.frames_neck(feature_frames)
        else:
            raise ValueError(self.mode)

        logit_output = []
        if self.fusion_type in ["result_fusion", "cross_modal_fusion"]:
            if self.mode in ["kps", "train"]:
                logit_kps = self.pred_layer_kps(feature_kps)
                logit_kps = self.dequant(logit_kps)
                if self.flat_output:
                    logit_kps = logit_kps.view(-1, self.num_classes)
                logit_output.append(logit_kps)

            if self.mode in ["train", "head"]:
                logit_frames = self.pred_layer_frames(feature_frames)
                logit_frames = self.dequant(logit_frames)
                if self.flat_output:
                    logit_frames = logit_frames.view(-1, self.num_classes)
                logit_output.append(logit_frames)

        if self.fusion_type in ["feature_fusion", "cross_modal_fusion"]:
            feature_kps_frames = self.feature_concat.cat(
                (feature_kps, feature_frames), dim=1
            )
            logit_kps_frames = self.pred_layer(feature_kps_frames)
            logit_kps_frames = self.dequant(logit_kps_frames)
            if self.flat_output:
                logit_kps_frames = logit_kps_frames.view(-1, self.num_classes)
            logit_output.append(logit_kps_frames)
        return tuple(logit_output)
