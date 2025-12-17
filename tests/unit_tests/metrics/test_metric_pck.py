import pytest
import torch

from hat.metrics.metric_keypoints import MeanKeypointDist, PCKMetric


@pytest.mark.parametrize(
    ["use_ldmk", "decode_method"],
    [
        [True, None],
        [False, "diff_sign"],
        [True, "averaged"],
    ],
)
def test_pck(use_ldmk, decode_method):
    metric1 = PCKMetric(
        alpha=0.1,
        feat_stride=4,
        img_shape=(128, 128),
        decode_mode=decode_method,
    )
    ldmk = torch.randn(8, 12, 2) * 128
    if use_ldmk:
        data = {
            "gt_ldmk": torch.randn(8, 12, 2) * 128,
            "pr_ldmk": ldmk,
            "gt_ldmk_attr": torch.ones(8, 12),
        }
    else:
        data = {
            "gt_ldmk": torch.randn(8, 12, 2) * 128,
            "pr_heatmap": torch.randn(8, 12, 32, 32),
            "gt_ldmk_attr": torch.ones(8, 12),
        }
    metric1.update(data)
    name, result = metric1.get()
    assert result is not None


@pytest.mark.parametrize(
    ["use_ldmk", "decode_method"],
    [
        [True, None],
        [False, "diff_sign"],
        [True, "averaged"],
    ],
)
def test_mean_keypoint_dist(use_ldmk, decode_method):
    metric1 = MeanKeypointDist(
        feat_stride=4,
        decode_mode=decode_method,
    )
    ldmk = torch.randn(8, 12, 2) * 128
    if use_ldmk:
        data = {
            "gt_ldmk": torch.randn(8, 12, 2) * 128,
            "pr_ldmk": ldmk,
            "gt_ldmk_attr": torch.ones(8, 12),
        }
    else:
        data = {
            "gt_ldmk": torch.randn(8, 12, 2) * 128,
            "pr_heatmap": torch.randn(8, 12, 32, 32),
            "gt_ldmk_attr": torch.ones(8, 12),
        }
    metric1.update(data)
    name, result = metric1.get()
    assert result is not None
