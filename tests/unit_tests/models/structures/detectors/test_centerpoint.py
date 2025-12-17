import copy

import numpy as np
import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_centerpoint_task(mode):

    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]
    max_num_points = 20
    max_voxels = (300, 400)
    norm_cfg = None
    tasks = [
        dict(num_class=1, class_names=["car"]),
        dict(num_class=2, class_names=["truck", "construction_vehicle"]),
        dict(num_class=2, class_names=["bus", "trailer"]),
        dict(num_class=1, class_names=["barrier"]),
        dict(num_class=2, class_names=["motorcycle", "bicycle"]),
        dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
    ]
    common_heads = dict(
        reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
    )
    with_velocity = "vel" in common_heads.keys()

    def get_feature_map_size(point_cloud_range, voxel_size):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]
        ) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        return grid_size

    config = dict(
        type="CenterPointDetector",
        feature_map_shape=get_feature_map_size(point_cloud_range, voxel_size),
        pre_process=dict(
            type="CenterPointPreProcess",
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels_num=max_voxels,
            max_points_in_voxel=max_num_points,
            norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
            norm_dims=[0, 1, 2, 3],
        ),
        reader=dict(
            type="PillarFeatureNet",
            num_input_features=5,
            num_filters=(64,),
            with_distance=False,
            pool_size=(max_num_points, 1),
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            bn_kwargs=norm_cfg,
            quantize=True,
            use_4dim=True,
            use_conv=True,
            hw_reverse=True,
        ),
        backbone=dict(
            type="PointPillarScatter",
            num_input_features=64,
            use_horizon_pillar_scatter=True,
            quantize=True,
        ),
        neck=dict(
            type="SequentialBottleNeck",
            layer_nums=[3, 5, 5],
            ds_layer_strides=[2, 2, 2],
            ds_num_filters=[64, 128, 256],
            us_layer_strides=[0.5, 1, 2],
            us_num_filters=[128, 128, 128],
            num_input_features=64,
            bn_kwargs=norm_cfg,
            use_tconv=True,
            use_secnet=True,
            quantize=True,
            use_relu6=False,
        ),
        head=dict(
            type="CenterPointHead",
            in_channels=sum([128, 128, 128]),
            tasks=tasks,
            share_conv_channels=64,
            share_conv_num=1,
            common_heads=common_heads,
            head_conv_channels=64,
            init_bias=-2.19,
            final_kernel=3,
        ),
        targets=dict(
            type="CenterPointLidarTarget",
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            tasks=tasks,
            dense_reg=1,
            max_objs=500,
            gaussian_overlap=0.1,
            min_radius=2,
            out_size_factor=4,
            norm_bbox=True,
            with_velocity=with_velocity,
        ),
        loss=dict(
            type="CenterPointLoss",
            loss_cls=dict(type="GaussianFocalLoss", loss_weight=1.0),
            loss_bbox=dict(
                type="L1Loss",
                reduction="mean",
                loss_weight=0.25,
            ),
            with_velocity=with_velocity,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        ),
        postprocess=dict(
            type="CenterPointPostProcess",
            tasks=tasks,
            norm_bbox=True,
            bbox_coder=dict(
                type="CenterPointBBoxCoder",
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=100,
                score_threshold=0.1,
                out_size_factor=4,
                voxel_size=voxel_size[:2],
            ),
            # test_cfg
            max_pool_nms=False,
            score_threshold=0.1,
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            out_size_factor=4,
            nms_type="rotate",
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            box_size=9,
        ),
    )

    float_model = build_from_registry(config)
    qat_model = copy.deepcopy(float_model)

    input_data = dict(
        points=[torch.rand(3000, 5)],
        gt_boxes=[torch.randn(20, 9)],
        gt_classess=[torch.randint(1, 11, (20,))],
    )
    if mode == "val":
        float_model.eval()
        qat_model.eval()

    float_model(input_data)
    qat_test(qat_model, input_data, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
