from collections import OrderedDict

import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test

NUM_LDMK = 4


@pytest.mark.parametrize(
    ["mode", "is_train"], [["train", True], ["val", False], ["deploy", False]]
)
def test_cls_coords(mode, is_train):
    config = dict(
        type="SmokeKpsModel",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            include_top=False,
            bn_kwargs={},
        ),
        cls_head=dict(
            type="SmokeKpsClsHead",
            in_channels=256,
            middle_dim=32,
            output_dim=3,
            is_train=is_train,
            loss_func=dict(
                type="SoftmaxCELoss",
                dim=1,
                reduction="mean",
            ),
        ),
        mode=mode,
    )
    ldmk_model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 3, 128, 128),
        "gt_visable": torch.rand(4, 1),
        "gt_vis_weight": torch.ones(4, 1),
        "gt_classes": torch.rand(4, 2),
        "gt_cls_weight": torch.ones(4, 2),
    }
    outputs = ldmk_model(x)
    if mode == "deploy":
        assert isinstance(outputs, list)
    elif mode == "train":
        assert isinstance(outputs, OrderedDict)
        assert "total_loss" in outputs and "cls_loss" in outputs
    elif mode == "val":
        assert isinstance(outputs, OrderedDict)
        assert "pr_visable" in outputs
        assert outputs["pr_visable"].shape == (4, 1)
    qat_test(ldmk_model, x, with_quantized=False)


@pytest.mark.parametrize(
    ["mode", "is_train"], [["train", True], ["val", False], ["deploy", False]]
)
def test_heatmap_coords(mode, is_train):
    config = dict(
        type="SmokeKpsModel",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            include_top=False,
            bn_kwargs={},
        ),
        decoder=dict(
            type="LdmkDecoder",
            in_channels=256,
            out_channels=128,
            in_stride=32,
            out_stride=4,
        ),
        vector_head=None,
        feat_stride=4,
        heatmap_head=dict(
            type="SmokeKpsHeatmapHead",
            in_channels=128,
            num_ldmk=NUM_LDMK,
            is_train=is_train,
            loss_func=dict(type="LdmkLoss", loss_type="l2"),
        ),
        mode=mode,
    )
    ldmk_model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 3, 128, 128),
        "gt_heatmap": torch.rand(4, 5, 32, 32),
        "gt_heatmap_weight": torch.ones(4, 5, 32, 32),
    }
    outputs = ldmk_model(x)
    if mode == "deploy":
        assert isinstance(outputs, list)
    elif mode == "train":
        assert isinstance(outputs, OrderedDict)
        assert "total_loss" in outputs and "heatmap_loss" in outputs
    elif mode == "val":
        assert isinstance(outputs, OrderedDict)
        assert "pr_heatmap" in outputs
        assert outputs["pr_heatmap"].shape == (4, 5, 32, 32)
    qat_test(ldmk_model, x, with_quantized=False)


@pytest.mark.parametrize(
    ["mode", "is_train"], [["train", True], ["val", False], ["deploy", False]]
)
def test_vector_coords(mode, is_train):
    config = dict(
        type="SmokeKpsModel",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            include_top=False,
            bn_kwargs={},
        ),
        decoder=dict(
            type="LdmkDecoder",
            in_channels=256,
            out_channels=128,
            in_stride=32,
            out_stride=4,
        ),
        feat_stride=4,
        vector_head=dict(
            type="SmokeKpsVectorHead",
            in_channels=128,
            num_ldmk=NUM_LDMK,
            band_width=1,
            vector_size=(32, 32),
            band_module_type="conv",
            is_train=is_train,
            loss_func=dict(type="LdmkLoss", loss_type="l2"),
        ),
        heatmap_head=None,
        mode=mode,
    )
    ldmk_model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 3, 128, 128),
        "gt_vector_x": torch.rand(4, NUM_LDMK, 32),
        "gt_vector_y": torch.rand(4, NUM_LDMK, 32),
        "gt_vector_weight_x": torch.ones(4, NUM_LDMK, 32),
        "gt_vector_weight_y": torch.ones(4, NUM_LDMK, 32),
    }
    outputs = ldmk_model(x)
    if mode == "deploy":
        assert isinstance(outputs, list)
        assert len(outputs) == 2
    elif mode == "train":
        assert isinstance(outputs, OrderedDict)
        assert "total_loss" in outputs and "vector_loss" in outputs
    elif mode == "val":
        assert isinstance(outputs, OrderedDict)
        assert "pr_vector_x" in outputs and "pr_vector_y" in outputs
        assert outputs["pr_vector_x"].shape == (4, NUM_LDMK, 1, 32)
        assert outputs["pr_vector_y"].shape == (4, NUM_LDMK, 32, 1)
    qat_test(ldmk_model, x, with_quantized=False)
