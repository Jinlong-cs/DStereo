import pytest
import torch

from hat.metrics.smoke_kps_metric import (
    SmokeKpsNME,
    SoftmaxAccuracy,
    SoftmaxPrecision,
    SoftmaxRecall,
)

NUM_LDMK = 4
DATA = {
    "gt_ldmk": torch.rand((8, NUM_LDMK, 2)),
    "gt_visable": torch.ones((8, 1)),
    "gt_classes": torch.ones((8, 1)),
    "ldmk_mask": torch.ones(8, NUM_LDMK),
    "gt_ldmk_attr": torch.ones(8, NUM_LDMK),
    "pr_ldmk": torch.rand((8, NUM_LDMK, 2)),
    "pr_heatmap": torch.rand((8, NUM_LDMK + 1, 32, 32)),
    "pr_vector_x": torch.rand((8, NUM_LDMK, 1, 32)),
    "pr_vector_y": torch.rand((8, NUM_LDMK, 32, 1)),
    "pr_visable": torch.rand((8, 1)),
    "pr_classes": torch.rand((8, 2)),
}


@pytest.mark.parametrize(
    ["norm_type", "mode", "is_ignore_idx", "decoding_method"],
    [
        ["norm12", "heatmap", "True", "diff"],
        ["norm12", "heatmap", "True", "shift"],
        ["norm12", "heatmap", "True", "taylor"],
        ["norm1234", "heatmap", "True", "diff"],
        ["norm1234", "heatmap", "True", "shift"],
        ["norm1234", "heatmap", "True", "taylor"],
    ],
)
def test_nme(norm_type, mode, is_ignore_idx, decoding_method):
    metric = SmokeKpsNME(
        NUM_LDMK,
        "NME",
        norm_type,
        mode,
        is_ignore_idx,
        decoding_method,
        feat_stride=4,
    )
    metric.update(DATA)
    _, value = metric.get()
    assert value > 0


@pytest.mark.parametrize(["cls_type"], [["classes"], ["visable"]])
def test_accuracy(cls_type):
    metric = SoftmaxAccuracy("ACC", cls_type)
    metric.update(DATA)
    _, value = metric.get()
    assert value > 0


@pytest.mark.parametrize(
    ["cls_type", "cls_id"], [["classes", 1], ["visable", 0]]
)
def test_recall(cls_type, cls_id):
    metric = SoftmaxRecall("Rec", cls_type, cls_id)
    metric.update(DATA)
    _, value = metric.get()
    assert value > 0


@pytest.mark.parametrize(
    ["cls_type", "cls_id"], [["classes", 1], ["visable", 0]]
)
def test_precision(cls_type, cls_id):
    metric = SoftmaxPrecision("Pre", cls_type, cls_id)
    metric.update(DATA)
    _, value = metric.get()
    assert value > 0
