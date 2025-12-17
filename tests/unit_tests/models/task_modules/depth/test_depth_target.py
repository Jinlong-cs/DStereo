import copy

import numpy as np
import pytest

from hat.models.task_modules.depth import DepthMultiTargets, DepthValueTarget
from tests.utils import gen_fake_depth_data


@pytest.mark.parametrize(
    ["out_stride"],
    [
        pytest.param([1, 2, 4]),
    ],
)
def test_depth_decoder(out_stride):
    w, h = 896, 896

    preds, labels = gen_fake_depth_data(
        w, h, strides=out_stride, label_dim=2, n=8, label_name="gt_depth"
    )
    labels["origin_img"] = np.random.randint(0, 255, labels["gt_depth"].shape)
    target_modules = DepthValueTarget(
        label_name="gt_depth",
        with_smooth=True,
        with_virtual_normal=True,
    )
    gt_with_input_size = labels
    for i, downsample_scale in enumerate(out_stride[1:]):
        depth_multi_targets = DepthMultiTargets(
            low_depth=0.01,
            high_depth=150.0,
            downsample_scale=downsample_scale,
            label_name="gt_depth",
            pred_names="pred_depth",
            depth_type="Cartesian",
            target_modules=target_modules,
            gt_scale=1.0,
        )
        gt_with_input_size_i = copy.deepcopy(gt_with_input_size)
        ret = depth_multi_targets(
            gt_with_input_size_i, {"pred_depth": preds[i + 1]}
        )
        h, w = ret[0]["target"][0]["gt_depth"].shape[2:]
        h_with_input_size, w_with_input_size = gt_with_input_size[
            "gt_depth"
        ].shape[2:]
        assert h * downsample_scale == h_with_input_size
        assert w * downsample_scale == w_with_input_size
        assert ret[0]["target"][0]["gt_depth"].shape == ret[0]["pred"][0].shape


@pytest.mark.parametrize(
    ["out_stride"],
    [
        pytest.param([2]),
    ],
)
def test_depth_value_target(out_stride):
    w, h = 896, 896
    preds, label = gen_fake_depth_data(
        w, h, strides=out_stride, label_dim=2, n=8, label_name="gt_depth"
    )
    preds = preds * 3
    label["origin_img"] = np.random.randint(0, 255, label["gt_depth"].shape)
    target_modules = DepthValueTarget(
        label_name="gt_depth",
        with_smooth=True,
        with_virtual_normal=True,
    )
    target_modules(label, preds)


if __name__ == "__main__":
    test_depth_decoder([1, 2, 4])
