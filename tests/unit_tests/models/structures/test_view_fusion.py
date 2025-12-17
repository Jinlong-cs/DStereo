import copy

import numpy as np
import pytest
import torch

import hat.data.datasets.nuscenes_dataset as NuscenesDataset
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test

bn_kwargs = dict(eps=2e-5, momentum=0.1)

bev_size = (51.2, 51.2, 0.8)
map_size = (15, 30, 0.15)
task_map_size = (15, 30, 0.15)

tasks = [
    dict(name="car", num_class=1, class_names=["car"]),
    dict(
        name="truck",
        num_class=2,
        class_names=["truck", "construction_vehicle"],
    ),
    dict(name="bus", num_class=2, class_names=["bus", "trailer"]),
    dict(name="barrier", num_class=1, class_names=["barrier"]),
    dict(name="bicycle", num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(
        name="pedestrian",
        num_class=2,
        class_names=["pedestrian", "traffic_cone"],
    ),
]

config = dict(
    type="ViewFusion",
    bev_feat_index=-1,
    bev_upscale=2.0,
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b0",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="FastSCNNNeck",
        in_channels=[112, 320],
        feat_channels=[64, 64],
        indexes=[-2, -1],
        bn_kwargs=bn_kwargs,
        scale_factor=2,
    ),
    view_transformer=dict(
        type="WrappingTransformer",
        bev_size=bev_size,
        num_views=6,
        grid_quant_scale=1 / 128,
        grid_size=(128, 128),
    ),
    bev_encoder=dict(
        type="BevEncoder",
        backbone=dict(
            type="efficientnet",
            bn_kwargs=bn_kwargs,
            model_type="b0",
            num_classes=1000,
            include_top=False,
            activation="relu",
            use_se_block=False,
            input_channels=64,
            quant_input=False,
        ),
        neck=dict(
            type="BiFPN",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[2, 4, 8, 16, 32],
            stride2channels=dict({2: 16, 4: 24, 8: 40, 16: 112, 32: 320}),
            out_channels=48,
            num_outs=5,
            stack=3,
            start_level=0,
            end_level=-1,
            fpn_name="bifpn_sum",
        ),
    ),
    bev_decoders=[
        dict(
            type="BevSegDecoder",
            name="bev_seg",
            use_bce=False,
            bev_size=bev_size,
            task_size=task_map_size,
            grid_quant_scale=1 / 128,
            task_weight=10.0,
            head=dict(
                type="DepthwiseSeparableFCNHead",
                input_index=0,
                in_channels=48,
                feat_channels=48,
                num_classes=10,
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
                use_sigmoid=True,
                class_weight=2.0,
            ),
            decoder=dict(
                type="FCNDecoder",
                upsample_output_scale=1,
                use_bce=False,
                bg_cls=-1,
            ),
        ),
        dict(
            type="BevDetDecoder",
            name="bev_det",
            task_weight=1.0,
            head=dict(
                type="DepthwiseSeparableCenterPointHead",
                in_channels=48,
                tasks=tasks,
                share_conv_channels=48,
                share_conv_num=1,
                common_heads=dict(
                    reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2),
                ),
                head_conv_channels=48,
                num_heatmap_convs=2,
                final_kernel=3,
            ),
            target=dict(
                type="CenterPointTarget",
                class_names=NuscenesDataset.CLASSES,
                tasks=tasks,
                gaussian_overlap=0.1,
                min_radius=3,
                out_size_factor=1,
                norm_bbox=True,
                max_num=500,
                bbox_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            loss_cls=dict(type="GaussianFocalLoss", loss_weight=1.0),
            loss_reg=dict(
                type="L1Loss",
                loss_weight=0.25,
            ),
            decoder=dict(
                type="CenterPointDecoder",
                class_names=NuscenesDataset.CLASSES,
                tasks=tasks,
                bev_size=bev_size,
                out_size_factor=1,
                score_threshold=0.1,
                use_max_pool=True,
                nms_type=[
                    "rotate",
                    "rotate",
                    "rotate",
                    "circle",
                    "rotate",
                    "rotate",
                ],
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                nms_threshold=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
                decode_to_ego=True,
            ),
        ),
    ],
)


def gen_data():
    imgs = torch.from_numpy(np.random.randn(6, 3, 512, 960)).float()
    ego2imgs = torch.from_numpy(np.random.randn(6, 4, 4)).float()

    bev_seg = torch.from_numpy(np.random.randn(1, 10, 512, 512)).float()
    bev_bboxes = [np.random.randn(10, 10), np.random.randn(11, 10)]
    data = {
        "img": imgs,
        "ego2img": ego2imgs,
        "bev_bboxes_labels": bev_bboxes,
        "bev_seg_indices": bev_seg,
    }
    return data


def gen_data_sequence():
    imgs = torch.from_numpy(np.random.randn(18, 3, 512, 960)).float()
    ego2imgs = torch.from_numpy(np.random.randn(18, 4, 4)).float()
    ego2global = [np.random.randn(3, 4, 4).astype(dtype=np.float32)]

    bev_seg = torch.from_numpy(np.random.randn(1, 10, 512, 512)).float()
    bev_bboxes = [np.random.randn(10, 10), np.random.randn(11, 10)]
    data = {
        "img": imgs,
        "ego2img": ego2imgs,
        "ego2global": ego2global,
        "bev_bboxes_labels": bev_bboxes,
        "bev_seg_indices": bev_seg,
    }
    return data


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_ipm_view_fusion(mode):
    bev_detector = build_from_registry(config)

    data = gen_data()

    if mode == "train":
        bev_detector(data)

        qat_test(bev_detector, data, with_quantized=False)

    if mode == "val" or mode == "test":
        bev_detector(data)


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_ipm4d_view_fusion(mode):
    config_4d = copy.deepcopy(config)
    config_4d["temporal_fusion"] = dict(
        type="AddTemporalFusion",
        in_channels=64,
        out_channels=64,
        num_seq=3,
        bev_size=bev_size,
        grid_size=(128, 128),
        num_encoder=2,
        num_project=1,
        grid_quant_scale=1 / 128,
    )

    bev_detector = build_from_registry(config_4d)

    data = gen_data_sequence()
    print(data["img"].shape)
    if mode == "train":
        bev_detector(data)

        qat_test(bev_detector, data, with_quantized=False)

    if mode == "val" or mode == "test":
        bev_detector(data)


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_gkt_view_fusion(mode):
    config["view_transformer"] = dict(
        type="GKTTransformer",
        bev_size=bev_size,
        grid_size=(128, 128),
        num_views=6,
        embed_dims=64,
        grid_quant_scale=1 / 128,
    )

    bev_detector = build_from_registry(config)

    data = gen_data()

    if mode == "train":
        bev_detector(data)

        qat_test(bev_detector, data, with_quantized=False)

    if mode == "val" or mode == "test":
        bev_detector(data)


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_lss_view_fusion(mode):
    config["view_transformer"] = dict(
        type="LSSTransformer",
        in_channels=64,
        feat_channels=64,
        z_range=(-10.0, 10.0),
        depth=60,
        num_points=10,
        bev_size=bev_size,
        grid_size=(128, 128),
        num_views=6,
        grid_quant_scale=1 / 128,
        depth_grid_quant_scale=1 / 128,
    )

    bev_detector = build_from_registry(config)

    data = gen_data()

    if mode == "train":
        bev_detector(data)

        qat_test(bev_detector, data, with_quantized=False)

    if mode == "val" or mode == "test":
        bev_detector(data)
