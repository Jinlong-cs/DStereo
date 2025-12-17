# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Tuple

import torch
from torch import nn

from hat.core.hand3d.hand3d_func import posenc
from hat.registry import OBJECT_REGISTRY

try:
    from pytorch3d.renderer import (
        NormWeightedCompositor,
        PerspectiveCameras,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
    )
    from pytorch3d.structures import Pointclouds
except Exception:
    Pointclouds = None
    PerspectiveCameras = None
    PointsRasterizationSettings = None
    PointsRasterizer = None
    PointsRenderer = None
    NormWeightedCompositor = None

__all__ = ["PointNeRF"]


class MLP(nn.Module):
    def __init__(self, layer_sizes=None, bias=True):
        super().__init__()
        """Multi-layer perceptron model.

        Args:
            layer_sizes:
                A list of integers representing the sizes of each layer.
            bias: Whether to use bias terms in the linear layers.
                Defaults to True.
        """
        assert len(layer_sizes) > 2

        layers = [
            layer
            for i in range(len(layer_sizes) - 1)
            for layer in [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias),
                nn.ReLU(inplace=True)
                if i + 2 < len(layer_sizes)
                else nn.Identity(),
            ]
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)


@OBJECT_REGISTRY.register
class PointNeRF(nn.Module):
    def __init__(
        self,
        intput_shape: Tuple[int, int] = (256, 256),
        point_cloud_radius: float = 0.025,
        points_per_pixel: int = 50,
        text_feat_length: int = 128,
        position_encoding_length: int = 4,
        density_feat_length: int = 12,
    ):
        super().__init__()
        """Initializes the PointNeRF model.

        Args:
            intput_shape:
                The shape of the input tensor. Defaults to (256, 256).
            point_cloud_radius:
                Point cloud radius during rendering. Defaults to 0.025.
            points_per_pixel:
                Max projected point clouds number per pixel in
                rendered image. Defaults to 50.
            text_feat_length:
                Length of texture feature of texture head. Defaults to 128.
            position_encoding_length:
                Length of position encoding for points.
                Defaults to 4, means x + sin(2*i*x) + cos(2*i*x).
            density_feat_length:
                Length of density feature for color predict. Defaults to 12.
        """
        if PointsRasterizationSettings is not None:
            self.raster_settings = PointsRasterizationSettings(
                image_size=intput_shape,
                radius=point_cloud_radius,
                points_per_pixel=points_per_pixel,
            )
        else:
            self.raster_settings = None

        self.intput_shape = intput_shape
        self.text_feat_length = text_feat_length
        self.position_encoding_length = position_encoding_length
        self.density_feat_length = density_feat_length

        density_layer_sizes = [
            self.text_feat_length,
            64,
            64,
            64,
            4 + self.density_feat_length,
        ]
        color_layer_sizes = [
            self.text_feat_length
            + 3 * (2 * self.position_encoding_length + 1)
            + 3 * (2 * self.position_encoding_length + 1)
            + self.density_feat_length,
            64,
            64,
            64,
            3,
        ]
        self.density_model = MLP(density_layer_sizes)
        self.color_model = MLP(color_layer_sizes)

    def forward(self, pred_verts_sub, text_feat_xyz, intrinsic3x3):
        cameras = PerspectiveCameras(
            device=intrinsic3x3.device,
            focal_length=torch.diagonal(intrinsic3x3, dim1=1, dim2=2)[:, :2],
            principal_point=intrinsic3x3[:, :2, 2],
            R=torch.eye(3).unsqueeze(0).repeat(intrinsic3x3.shape[0], 1, 1),
            T=torch.zeros(intrinsic3x3.shape[0], 3),
            image_size=(self.intput_shape,),
        )

        pcl_cam_coords = cameras.transform_points(pred_verts_sub)
        origins = cameras.get_camera_center()
        ray_directions = pcl_cam_coords - origins[:, None, :]
        ray_directions = ray_directions / torch.norm(
            ray_directions, dim=-1, keepdim=True
        )

        # predict rgb color of every vert
        encoded_xyz = posenc(pred_verts_sub, self.position_encoding_length)
        encoded_dir = posenc(ray_directions, self.position_encoding_length)

        density_diffuse = self.density_model(text_feat_xyz)
        pred_verts_density, diffuse_color, density_encoding = torch.split(
            density_diffuse, [1, 3, self.density_feat_length], dim=-1
        )
        specular_color = self.color_model(
            torch.cat(
                [
                    text_feat_xyz,
                    encoded_xyz,
                    encoded_dir,
                    density_encoding,
                ],
                dim=-1,
            )
        )
        pred_verts_color = diffuse_color + specular_color

        # render point clouds
        rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        ).to(intrinsic3x3.device)
        pcl = Pointclouds(points=pred_verts_sub, features=pred_verts_color)

        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=NormWeightedCompositor(background_color=[0, 0, 0]),
        )
        render_images = renderer(pcl).permute(0, 3, 1, 2)

        return render_images
