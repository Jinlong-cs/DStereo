import numpy as np
import pytest
import torch
from horizon_plugin_pytorch.quantization import QTensor

import hat.data.datasets.nuscenes_dataset as NuscenesDataset
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test_with_multi_inputs

bn_kwargs = dict(eps=2e-5, momentum=0.1)

bev_size = (51.2, 51.2, 0.8)
map_size = (15, 30, 0.15)
task_map_size = (15, 30, 0.15)


def gen_data():
    feat = [torch.from_numpy(np.random.randn(1, 48, 128, 128)).float()]
    bev_seg = torch.from_numpy(np.random.randn(1, 10, 512, 512)).float()
    bev_bboxes = [np.random.randn(10, 10), np.random.randn(11, 10)]
    data = {"bev_bboxes_labels": bev_bboxes, "bev_seg_indices": bev_seg}
    return feat, data


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_seg_decocde(mode):

    config = dict(
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
    )

    decoder = build_from_registry(config)

    feat, data = gen_data()

    if mode == "train":
        decoder(feat, data)
        q_feat = [QTensor(feat[0], scale=torch.tensor([0.78]), dtype="qint8")]
        qat_test_with_multi_inputs(
            decoder, (q_feat, data), with_quantized=False
        )

    if mode == "val" or mode == "test":
        decoder(feat, data)


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_det_decocde(mode):

    tasks = [
        dict(name="car", num_class=1, class_names=["car"]),
        dict(
            name="truck",
            num_class=2,
            class_names=["truck", "construction_vehicle"],
        ),
        dict(name="bus", num_class=2, class_names=["bus", "trailer"]),
        dict(name="barrier", num_class=1, class_names=["barrier"]),
        dict(
            name="bicycle", num_class=2, class_names=["motorcycle", "bicycle"]
        ),
        dict(
            name="pedestrian",
            num_class=2,
            class_names=["pedestrian", "traffic_cone"],
        ),
    ]

    config = dict(
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
    )

    decoder = build_from_registry(config)

    feat, data = gen_data()

    if mode == "train":
        decoder(feat, data)

        q_feat = [QTensor(feat[0], scale=torch.tensor([0.78]), dtype="qint8")]
        qat_test_with_multi_inputs(
            decoder, (q_feat, data), with_quantized=False
        )

    if mode == "val" or mode == "test":
        decoder(feat, data)
