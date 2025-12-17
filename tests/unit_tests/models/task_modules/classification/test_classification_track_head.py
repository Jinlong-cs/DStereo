# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.classification.classfication_head import (
    ClassificationTrackHead,
)
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "in_strides",
    ],
    [
        pytest.param([4, 8, 16, 32, 64]),
        pytest.param([2, 4, 8, 16, 32]),
    ],
)
def test_classification_track_head(in_strides):
    w, h = 64, 128
    feats, _, _ = gen_fake_feats(w, h, in_strides)

    track_head = ClassificationTrackHead()
    out = track_head(feats)
    assert out.shape == feats[-1].shape

    # test qat
    feats_qat = [qtensor_test(feat) for feat in feats]
    qat_test(track_head, feats_qat, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
