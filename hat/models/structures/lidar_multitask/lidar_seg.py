from typing import List, Optional, Sequence

import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

torch.fx.wrap("len")

__all__ = ["LidarSegmentor"]


@OBJECT_REGISTRY.register
class LidarSegmentor(nn.Module):
    """
    The basic structure of CenterPoint.

    Args:
        feature_map_shape: Feature map shape, in (W, H, 1) format.
        pre_process: pre_process module.
        reader: reader module.
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        targets: Target generator module.
        loss: Loss module.
        postprocess: Postprocess module.
        quant_begin_neck: Whether to quantize beginning from neck.
        is_deploy: Is deploy model or not.
    """

    def __init__(
        self,
        feature_map_shape: List[int],
        pre_process: Optional[nn.Module] = None,
        reader: Optional[nn.Module] = None,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        seg_decoder: Optional[nn.Module] = None,
        feat_upscale: int = 1,
        quant_begin_neck: bool = False,
        is_deploy: bool = False,
    ):
        super(LidarSegmentor, self).__init__()

        self.pre_process = pre_process
        self.reader = reader
        self.backbone = backbone
        self.neck = neck
        self.seg_decoder = seg_decoder
        self.quant_begin_neck = quant_begin_neck

        self.feature_map_shape = feature_map_shape
        self.is_deploy = is_deploy

        self.feat_upscale = feat_upscale
        if self.feat_upscale > 1:
            self.resize = hnn.Interpolate(
                scale_factor=self.feat_upscale,
                align_corners=None,
                recompute_scale_factor=True,
            )

    def forward(self, example):

        # train: [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #   -> voxel-feature -> neck && Head -> output -> Loss
        # eval: [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #   -> voxel-feature -> neck && Head -> output -> post_process
        # deploy: -------------------------------------------------------
        #    [features, coors] -> FeatureExtractor -> PillarScatter
        #       -> voxel-feature -> neck && Head -> output

        if self.pre_process:
            # use horizon pre_process
            features, coords = self.pre_process(
                example["points"], self.is_deploy or not self.training
            )
            data = dict(  # noqa C408
                features=features,
                coors=coords,
                num_points_in_voxel=None,
                batch_size=len(example["points"]),
                input_shape=self.feature_map_shape,
            )
        else:
            data = dict(  # noqa C408
                features=example["features"],
                coors=example["coors"],
                num_points_in_voxel=None,
                batch_size=1,
                input_shape=self.feature_map_shape,
            )

        input_features = self.reader(
            data["features"],
            data["coors"],
            data["num_points_in_voxel"],
            horizon_preprocess=True,
        )

        x = self.backbone(
            input_features,
            data["coors"],
            data["batch_size"],
            torch.tensor(self.feature_map_shape),
        )
        x = self.neck(x)

        if isinstance(x, Sequence):
            x = x[0]
        if self.feat_upscale > 1:
            x = self.resize(x)

        pred, result = self.seg_decoder(x, example)

        if self.is_deploy:
            return pred

        return pred, result

    def fuse_model(self):
        if not self.quant_begin_neck:
            # P2
            if self.reader:
                self.reader.fuse_model()

        for module in [self.neck, self.seg_decoder]:
            if module is not None:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.quant_begin_neck:
            # P1
            for module in [self.reader, self.backbone]:
                module.qconfig = None
        else:
            for module in [self.reader, self.backbone]:
                if module is not None:
                    if hasattr(module, "set_qconfig"):
                        module.set_qconfig()

        # P2
        for module in [self.neck, self.seg_decoder]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
