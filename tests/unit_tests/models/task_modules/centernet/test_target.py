# Copyright (c) Horizon Robotics. All rights reserved.
import collections

import pytest

from hat.models.task_modules.centernet import CenterNetTarget
from tests.unit_tests.models.task_modules.centernet.utils import (
    gen_fake_label_data_dict,
)
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    ["in_strides", "num_classes", "width", "height"],
    [
        pytest.param(4, 6, 704, 576),
        pytest.param(16, 6, 704, 576),
    ],
)
def test_centernet_head_decoder(in_strides, num_classes, width, height):
    wh_pred_channel = 2
    offset_pred_channel = 2
    heatmap_pred_channel = num_classes
    mp_heat_pred_channel = num_classes
    feats_dict = collections.OrderedDict()
    for pred_name, pred_channel in zip(
        ["heatmap_pred", "wh_pred", "offset_pred", "mp_heat_pred"],
        [
            heatmap_pred_channel,
            wh_pred_channel,
            offset_pred_channel,
            mp_heat_pred_channel,
        ],
    ):
        feats, _, _ = gen_fake_feats(
            width,
            height,
            strides=[in_strides],
            n=1,
            feat_channels=pred_channel,
            channels_strides=None,
        )
        feats_dict[pred_name] = feats[0]
    centernet_target = CenterNetTarget(
        num_classes=num_classes,
        strides2levels={4: 0, 8: 1, 16: 2, 32: 3, 64: 4},
        in_strides=in_strides,
        img_shape=(height, width),
        use_ignore=True,
        gaussian_radius_alpha=0.5,
    )
    label = gen_fake_label_data_dict(height, width)
    target_result, avg_factor = centernet_target(label, feats_dict)
    for property_name, channel_num in zip(
        [
            "center_heatmap_target",
            "heatmap_hard_mask",
            "wh_target",
            "offset_target",
            "wh_offset_target_weight",
            "bbox_target",
            "ig_bboxes_mask",
        ],
        [
            num_classes,
            num_classes,
            wh_pred_channel,
            offset_pred_channel,
            1,
            4,
            num_classes,
        ],
    ):
        assert target_result[property_name].size(1) == channel_num
        assert target_result[property_name].size(2) == height // in_strides
        assert target_result[property_name].size(3) == width // in_strides
    assert target_result["valid_classes_list"][0] == [1, 2, 3]
    assert avg_factor >= 1


if __name__ == "__main__":
    pytest.main(["-s", __file__])
