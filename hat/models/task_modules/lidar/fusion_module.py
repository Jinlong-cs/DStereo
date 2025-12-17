from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    QuantStub,
)
from torch import nn
from torch.quantization import QConfig

from hat.models.base_modules.conv_module import (
    ConvModule2d,
    ConvTransposeModule2d,
    ConvUpsample2d,
)
from hat.models.utils import _take_features
from hat.registry import OBJECT_REGISTRY

grid_qconfig = QConfig(
    activation=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=qinfo("qint16").min,
        quant_max=qinfo("qint16").max,
        dtype="qint16",
        saturate=True,
    ),
    weight=None,
)


@OBJECT_REGISTRY.register_module
class LidarCameraFusionModule(nn.Module):
    def __init__(
        self,
        lidar_feat_channels: int,
        camera_feat_channels: int,
        out_channel: int,
        bn_kwargs: Optional[Dict] = None,
        fusion_feat_shape: Optional[Tuple[int, int]] = None,
        lidar_vcs_range: Optional[Tuple[float, float, float, float]] = None,
        camera_vcs_range: Optional[Tuple[float, float, float, float]] = None,
        fusion_vcs_range: Optional[Tuple[float, float, float, float]] = None,
    ):
        """Lidar and camera fusion module.

        Args:
            lidar_feat_channels: lidar input feature's channel.
            camera_feat_channels: camera input feature's channel.
            out_channel: fusion module out feature's channel.
            bn_kwargs: batch norm layer's kwargs.
            fusion_feat_shape: the shape of feature after fusion.
            lidar_vcs_range: lidar feature coverage range in VCS.
                (x_min, y_min, x_max, y_max).
            camera_vcs_range: camera(BEV) feature coverage range in VCS.
                (x_min, y_min, x_max, y_max).
            fusion_vcs_range: after fusion lidar feature coverage range in VCS.
                (x_min, y_min, x_max, y_max).
        """
        super(LidarCameraFusionModule, self).__init__()
        if bn_kwargs is None:
            bn_kwargs = {"eps": 1e-3, "momentum": 0.01}
        self.cat = nn.quantized.FloatFunctional()
        self.lidar2fusion_grid = None
        self.cam2fusion_grid = None
        self.lidar_grid_quant = QuantStub()
        self.cam_grid_quant = QuantStub()

        if fusion_feat_shape is not None:
            assert fusion_vcs_range is not None
            if lidar_vcs_range is not None:
                self.lidar2fusion_grid = self._calculate_grid_coords(
                    lidar_vcs_range,
                    fusion_feat_shape,
                    fusion_vcs_range,
                )

            if camera_vcs_range is not None:
                self.cam2fusion_grid = self._calculate_grid_coords(
                    camera_vcs_range,
                    fusion_feat_shape,
                    fusion_vcs_range,
                )

        self.fusion_conv = ConvModule2d(
            lidar_feat_channels + camera_feat_channels,
            out_channel,
            kernel_size=3,
            padding=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(out_channel, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )

    def _calculate_grid_coords(
        self,
        src_vcs_range: Tuple[float, float, float, float],
        dst_grid_size: Tuple[int, int],
        dst_vcs_range: Tuple[float, float, float, float],
    ):
        """Calculate the grid coordinates for the grid_sample function.

        Args:
            src_vcs_range: the source feature coverage range in VCS.
                (x_min, y_min, x_max, y_max).
            dst_grid_size: the destination grid size. (height, width).
            dst_vcs_range: the destination feature coverage range in VCS.
                (x_min, y_min, x_max, y_max).
        """
        grid_dst_x = torch.flip(
            torch.linspace(
                dst_vcs_range[0], dst_vcs_range[2], dst_grid_size[0]
            ),
            dims=[0],
        )
        grid_dst_y = torch.linspace(
            dst_vcs_range[1], dst_vcs_range[3], dst_grid_size[1]
        )
        grid_dst_x, grid_dst_y = torch.meshgrid(grid_dst_x, grid_dst_y)

        # Map the destination grid coordinates to the source coordinate space
        grid_src_x = ((src_vcs_range[2] - grid_dst_x)) / (
            src_vcs_range[2] - src_vcs_range[0]
        ) * 2 - 1
        grid_src_y = ((grid_dst_y - src_vcs_range[1])) / (
            src_vcs_range[3] - src_vcs_range[1]
        ) * 2 - 1
        grid_src = torch.stack((grid_src_y, grid_src_x), dim=2).unsqueeze(0)

        return grid_src

    def forward(self, cam_feats, lidar_feats):
        if self.lidar2fusion_grid is not None:
            lidar2fusion_grid = self.lidar_grid_quant(
                self.lidar2fusion_grid.to(lidar_feats.device)
            )
            lidar_feats = F.grid_sample(
                lidar_feats,
                lidar2fusion_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

        if self.cam2fusion_grid is not None:
            cam2fusion_grid = self.cam_grid_quant(
                self.cam2fusion_grid.to(cam_feats.device)
            )
            cam_feats = F.grid_sample(
                cam_feats,
                cam2fusion_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

        x = self.cat.cat([cam_feats, lidar_feats], dim=1)

        x = self.fusion_conv(x)

        return x

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.lidar_grid_quant.qconfig = grid_qconfig
        self.cam_grid_quant.qconfig = grid_qconfig

    def fuse_model(self):
        for m in self.modules():
            if isinstance(
                m, (ConvModule2d, ConvUpsample2d, ConvTransposeModule2d)
            ):
                m.fuse_model()


@OBJECT_REGISTRY.register_module
class ExtractBEVFeatureForFusion(nn.Module):
    def __init__(
        self,
        cam_feature_name,
        cam_in_strides,
        cam_out_strides,
    ):
        """Lidar and camera fusion module.

        Args:
            cam_feature_name: camera feature's name.
            cam_in_strides: a list contains scales to selectd inputs.
            cam_out_strides: a list contains scales to selectd.
        """
        super(ExtractBEVFeatureForFusion, self).__init__()
        self.cam_feature_name = cam_feature_name
        self.cam_in_strides = cam_in_strides
        self.cam_out_strides = cam_out_strides

    def forward(self, cam_feats):
        input_cam_feats = (
            cam_feats[self.cam_feature_name][0]
            if isinstance(cam_feats, Mapping)
            else cam_feats
        )

        selected_cam_feat = _take_features(
            input_cam_feats, self.cam_in_strides, self.cam_out_strides
        )
        selected_cam_feat = selected_cam_feat[0]

        return selected_cam_feat
