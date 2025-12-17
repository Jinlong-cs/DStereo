import copy

import pytest
import torch
from horizon_plugin_pytorch.quantization import QTensor

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test_with_multi_inputs

bn_kwargs = dict(eps=2e-5, momentum=0.1)


def gen_data():
    feat = [torch.randn(1, 128, 16, 16)]
    gt_seg_labels = torch.randint(-1, 2, (1, 64, 64))
    gt_boxes = [torch.randn(20, 9)]
    gt_classess = [torch.randint(1, 11, (20,))]
    data = {
        "gt_seg_labels": gt_seg_labels,
        "gt_boxes": gt_boxes,
        "gt_classess": gt_classess,
    }
    return feat, data


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_lidar_seg_decocder(mode):

    config = dict(
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
    )
    decoder = build_from_registry(config)
    qat_decoder = copy.deepcopy(decoder)

    feat, data = gen_data()

    if mode == "val":
        decoder.eval()
        qat_decoder.eval()

    decoder(feat, data)
    q_feat = [QTensor(feat[0], scale=torch.tensor([0.78]), dtype="qint8")]
    qat_test_with_multi_inputs(
        qat_decoder, (q_feat, data), with_quantized=True
    )


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_lidar_det_decocder(mode):

    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]
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

    config = dict(
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
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=100,
                score_threshold=0.1,
                out_size_factor=4,
                voxel_size=voxel_size[:2],
            ),
            # test_cfg
            max_pool_nms=False,
            score_threshold=0.1,
            post_center_limit_range=[
                -61.2,
                -61.2,
                -10.0,
                61.2,
                61.2,
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
    )

    decoder = build_from_registry(config)
    qat_decoder = copy.deepcopy(decoder)

    feat, data = gen_data()

    if mode == "val":
        decoder.eval()
        qat_decoder.eval()

    decoder(feat, data)
    q_feat = [QTensor(feat[0], scale=torch.tensor([0.78]), dtype="qint8")]
    qat_test_with_multi_inputs(
        qat_decoder, (q_feat, data), with_quantized=True
    )
