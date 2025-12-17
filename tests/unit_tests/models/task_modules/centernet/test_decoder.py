# Copyright (c) Horizon Robotics. All rights reserved.
import collections

import pytest

from hat.models.task_modules.centernet import CenterNetDecoder
from hat.utils.package_helper import check_packages_available
from tests.unit_tests.models.task_modules.centernet.utils import (
    gen_fake_img_meta_data_dict,
)
from tests.utils import gen_fake_feats


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["in_strides", "num_classes", "width", "height"],
    [
        pytest.param(4, 6, 704, 576),
        pytest.param(8, 6, 704, 576),
    ],
)
def test_centernet_head_decoder(in_strides, num_classes, width, height):
    test_cfg_topk = 100
    wh_pred_channel = 2
    offset_pred_channel = 2
    heatmap_pred_channel = num_classes
    mp_heat_pred_channel = num_classes
    img_meta = gen_fake_img_meta_data_dict(height, width)
    channels_strides = None
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
            channels_strides=channels_strides,
        )
        feats_dict[pred_name] = feats[0]
    centernet_head_decoder = CenterNetDecoder(
        num_classes=num_classes, img_shape=((height, width))
    )
    update = centernet_head_decoder(feats_dict, img_meta)
    assert update["pred_bboxes"][0].size(0) <= test_cfg_topk
    assert update["pred_bboxes"][0].size(1) == num_classes


if __name__ == "__main__":
    pytest.main(["-s", __file__])
