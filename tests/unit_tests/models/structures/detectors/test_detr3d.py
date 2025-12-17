import numpy as np
import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test

bn_kwargs = dict(eps=2e-5, momentum=0.1)
bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
position_range = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)

num_query = 900
num_levels = 4
query_align = 128

config = dict(
    type="Detr3d",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    head=dict(
        type="PETRHead",
        num_query=num_query,
        query_align=query_align,
        in_channels=384,
        embed_dims=256,
        num_cls_fcs=2,
        num_reg_fcs=2,
        num_views=6,
        depth_num=64,
        depth_start=1,
        reg_out_channels=10,
        cls_out_channels=10,
        bev_range=bev_range,
        position_range=position_range,
        positional_encoding=dict(
            type="SinePositionalEncoding3D", num_feats=128, normalize=True
        ),
        transformer=dict(
            type="PETRTransformer",
            decoder=dict(
                type="PETRDecoder",
                num_layer=6,
                num_heads=8,
                embed_dims=256,
                dropout=0.1,
                feedforward_channels=2048,
            ),
        ),
        int8_output=False,
        dequant_output=True,
    ),
    target=dict(
        type="Detr3dTarget",
        cls_cost=dict(
            type="FocalLossCost",
            alpha=0.25,
            gamma=2.0,
            weight=2.0,
        ),
        reg_cost=dict(
            type="BBox3DL1Cost",
            weight=0.25,
        ),
        bev_range=bev_range,
    ),
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=10 + 1,
        loss_weight=2.0,
        alpha=0.25,
        gamma=2.0,
    ),
    loss_reg=dict(
        type="L1Loss",
        loss_weight=0.25,
    ),
    post_process=dict(
        type="Detr3dPostProcess",
        max_num=300,
        score_threshold=-1,
        bev_range=bev_range,
    ),
)


def gen_data():
    imgs = torch.from_numpy(np.random.randn(6, 3, 512, 960)).float()
    ego2imgs = torch.from_numpy(np.random.randn(6, 4, 4)).float()

    bev_bboxes = [np.random.rand(10, 10), np.random.rand(11, 10)]
    data = {
        "img": imgs,
        "ego2img": ego2imgs,
        "ego_bboxes_labels": bev_bboxes,
    }
    return data


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_detr3d_struct(mode):
    detector = build_from_registry(config)

    data = gen_data()

    if mode == "train":
        detector(data)

        qat_test(detector, data, with_quantized=False)

    if mode == "val" or mode == "test":
        detector(data)
