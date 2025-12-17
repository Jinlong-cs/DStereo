from collections import OrderedDict

import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test

NUM_LDMK = 68


@pytest.mark.parametrize(["mode"], [["train"], ["val"], ["deploy"]])
def test_ldmk_coords(mode):
    config = dict(
        type="LdmkModel",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            include_top=False,
            bn_kwargs={},
        ),
        coords_head=dict(
            type="LdmkCoordsHead",
            in_channels=256,
            kernel_size=4,
            num_ldmk=NUM_LDMK,
            loss_func=dict(type="LdmkLoss", loss_type="l2"),
        ),
        mode=mode,
    )
    ldmk_model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 3, 128, 128),
        "gt_ldmk": torch.rand(4, NUM_LDMK, 2),
        "gt_ldmk_weight": torch.ones(4, NUM_LDMK, 2),
    }
    if mode == "train":
        ldmk_model.coords_head.training = True
        outputs = ldmk_model(x)
        assert isinstance(outputs, OrderedDict)
        assert "total_loss" in outputs and "ldmk_loss" in outputs
    elif mode == "val":
        ldmk_model.coords_head.training = False
        outputs = ldmk_model(x)
        assert isinstance(outputs, OrderedDict)
        assert "pr_ldmk" in outputs
        assert outputs["pr_ldmk"].shape == (4, NUM_LDMK * 2, 1, 1)
    elif mode == "deploy":
        ldmk_model.coords_head.training = False
        outputs = ldmk_model(x)
        assert isinstance(outputs, list)
    qat_test(ldmk_model, x, with_quantized=False)


@pytest.mark.parametrize(["mode"], [["train"], ["val"], ["deploy"]])
def test_heatmap(mode):
    config = dict(
        type="LdmkModel",
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
        coords_head=None,
        feat_stride=4,
        heatmap_head=dict(
            type="LdmkHeatmapHead",
            in_channels=128,
            num_ldmk=NUM_LDMK,
            loss_func=dict(type="LdmkLoss", loss_type="l2"),
        ),
        mode=mode,
    )
    ldmk_model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 3, 128, 128),
        "gt_heatmap": torch.rand(4, NUM_LDMK, 32, 32),
        "gt_heatmap_weight": torch.ones(4, NUM_LDMK, 32, 32),
    }
    if mode == "train":
        ldmk_model.heatmap_head.training = True
        outputs = ldmk_model(x)
        assert isinstance(outputs, OrderedDict)
        assert "total_loss" in outputs and "heatmap_loss" in outputs
    elif mode == "val":
        ldmk_model.heatmap_head.training = False
        outputs = ldmk_model(x)
        assert isinstance(outputs, OrderedDict)
        assert "pr_heatmap" in outputs
        assert outputs["pr_heatmap"].shape == (4, NUM_LDMK, 32, 32)
    elif mode == "deploy":
        ldmk_model.heatmap_head.training = False
        outputs = ldmk_model(x)
        assert isinstance(outputs, list)
    qat_test(ldmk_model, x, with_quantized=False)


@pytest.mark.parametrize(["mode"], [["train"], ["val"], ["deploy"]])
def test_vector(mode):
    config = dict(
        type="LdmkModel",
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
        coords_head=None,
        feat_stride=4,
        vector_head=dict(
            type="LdmkVectorHead",
            in_channels=128,
            num_ldmk=NUM_LDMK,
            band_width=1,
            vector_size=(32, 32),
            band_module_type="conv",
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
    if mode == "train":
        ldmk_model.vector_head.training = True
        outputs = ldmk_model(x)
        assert isinstance(outputs, OrderedDict)
        assert "total_loss" in outputs and "vector_loss" in outputs
    elif mode == "val":
        ldmk_model.vector_head.training = False
        outputs = ldmk_model(x)
        assert isinstance(outputs, OrderedDict)
        assert "pr_vector_x" in outputs and "pr_vector_y" in outputs
        assert outputs["pr_vector_x"].shape == (4, NUM_LDMK, 1, 32)
        assert outputs["pr_vector_y"].shape == (4, NUM_LDMK, 32, 1)
    elif mode == "deploy":
        ldmk_model.vector_head.training = False
        outputs = ldmk_model(x)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
    qat_test(ldmk_model, x, with_quantized=False)
