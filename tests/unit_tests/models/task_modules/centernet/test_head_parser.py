# Copyright (c) Horizon Robotics. All rights reserved.
import collections

import pytest

from hat.models.task_modules.centernet import CenterNetHeadParser
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    ["num_classes", "in_strides", "width", "height"],
    [
        pytest.param(6, 4, 704, 576),
        pytest.param(6, 8, 704, 576),
    ],
)
def test_centernet_head_parser(num_classes, in_strides, width, height):
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
    centernet_head_parser = CenterNetHeadParser()
    update = centernet_head_parser(feats_dict)
    assert update["bbox_pred"].size(1) == 4
    assert update["bbox_pred"].size(2) == height // in_strides
    assert update["bbox_pred"].size(3) == width // in_strides


if __name__ == "__main__":
    pytest.main(["-s", __file__])
