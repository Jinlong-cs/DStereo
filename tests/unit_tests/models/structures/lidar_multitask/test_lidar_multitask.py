import copy

import numpy as np
import pytest
import torch

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_lidar_multitask_task(mode):

    point_cloud_range = [-3.2, -3.2, -5.0, 3.2, 3.2, 3.0]
    voxel_size = [0.1, 0.1, 8]
    max_num_points = 20
    max_voxels = (300, 400)
    norm_cfg = None
    bn_kwargs = dict(eps=2e-5, momentum=0.1)
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

    net_config = [
        [
            MixVarGENetConfig(
                in_channels=64,
                out_channels=64,
                head_op="mixvarge_f2",
                stack_ops=[],
                stack_factor=1,
                stride=1,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 2
        [
            MixVarGENetConfig(
                in_channels=64,
                out_channels=64,
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 4
        [
            MixVarGENetConfig(
                in_channels=64,
                out_channels=64,
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 8
        [
            MixVarGENetConfig(
                in_channels=64,
                out_channels=96,
                head_op="mixvarge_f2_gb16",
                stack_ops=[
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                ],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 16
        [
            MixVarGENetConfig(
                in_channels=96,
                out_channels=160,
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 32
    ]

    model = dict(
        type="LidarMultiTask",
        feature_map_shape=get_feature_map_size(point_cloud_range, voxel_size),
        pre_process=dict(
            type="CenterPointPreProcess",
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels_num=max_voxels,
            max_points_in_voxel=max_num_points,
            norm_range=[-3.2, -3.2, -5.0, 0.0, 3.2, 3.2, 3.0, 255.0],
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
        scatter=dict(
            type="PointPillarScatter",
            num_input_features=64,
            use_horizon_pillar_scatter=True,
            quantize=True,
        ),
        backbone=dict(
            type="MixVarGENet",
            net_config=net_config,
            disable_quanti_input=True,
            input_channels=64,
            input_sequence_length=1,
            num_classes=1000,
            bn_kwargs=bn_kwargs,
            include_top=False,
            bias=True,
            output_list=[0, 1, 2, 3, 4],
        ),
        neck=dict(
            type="Unet",
            in_strides=(2, 4, 8, 16, 32),
            out_strides=(4,),
            stride2channels=dict(
                {
                    2: 64,
                    4: 64,
                    8: 64,
                    16: 96,
                    32: 160,
                }
            ),
            out_stride2channels=dict(
                {
                    2: 128,
                    4: 128,
                    8: 128,
                    16: 128,
                    32: 160,
                }
            ),
            factor=2,
            group_base=8,
            bn_kwargs=bn_kwargs,
        ),
        lidar_decoders=[
            dict(
                type="LidarSegDecoder",
                name="seg",
                task_weight=80.0,
                task_feat_index=0,
                head=dict(
                    type="DepthwiseSeparableFCNHead",
                    input_index=0,
                    in_channels=128,
                    feat_channels=64,
                    num_classes=2,
                    dropout_ratio=0.1,
                    num_convs=2,
                    bn_kwargs=bn_kwargs,
                    int8_output=False,
                ),
                target=dict(
                    type="FCNTarget",
                ),
                loss=dict(
                    type="CrossEntropyLoss",
                    loss_name="seg",
                    reduction="mean",
                    ignore_index=-1,
                    use_sigmoid=False,
                    class_weight=[1.0, 10.0],
                ),
                decoder=dict(
                    type="FCNDecoder",
                    upsample_output_scale=4,
                    use_bce=False,
                    bg_cls=-1,
                ),
            ),
            dict(
                type="LidarDetDecoder",
                name="det",
                task_weight=1.0,
                task_feat_index=0,
                head=dict(
                    type="DepthwiseSeparableCenterPointHead",
                    in_channels=128,
                    tasks=tasks,
                    share_conv_channels=64,
                    share_conv_num=1,
                    common_heads=common_heads,
                    head_conv_channels=64,
                    init_bias=-2.19,
                    final_kernel=3,
                ),
                target=dict(
                    type="CenterPointLidarTarget",
                    grid_size=[64, 64, 1],
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
                    code_weights=[
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.2,
                        0.2,
                    ],
                ),
                decoder=dict(
                    type="CenterPointPostProcess",
                    tasks=tasks,
                    norm_bbox=True,
                    bbox_coder=dict(
                        type="CenterPointBBoxCoder",
                        pc_range=point_cloud_range[:2],
                        post_center_range=[-6.2, -6.2, -10.0, 6.2, 6.2, 10.0],
                        max_num=100,
                        score_threshold=0.1,
                        out_size_factor=4,
                        voxel_size=voxel_size[:2],
                    ),
                    # test_cfg
                    max_pool_nms=False,
                    score_threshold=0.1,
                    post_center_limit_range=[
                        -6.2,
                        -6.2,
                        -10.0,
                        6.2,
                        6.2,
                        10.0,
                    ],
                    min_radius=[4, 12, 10, 1, 0.85, 0.175],
                    out_size_factor=4,
                    nms_type="rotate",
                    pre_max_size=1000,
                    post_max_size=83,
                    nms_thr=0.2,
                    box_size=9,
                ),
            ),
        ],
    )
    float_model = build_from_registry(model)
    qat_model = copy.deepcopy(float_model)

    input_data = dict(
        points=[torch.rand(3000, 5)],
        gt_boxes=[torch.randn(20, 9)],
        gt_classess=[torch.randint(1, 11, (20,))],
        gt_seg_labels=torch.randint(-1, 2, (1, 64, 64)),
    )
    if mode == "val":
        float_model.eval()
        qat_model.eval()

    float_model(input_data)
    qat_test(qat_model, input_data, with_quantized=True)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
