import pytest

from hat.models.task_modules.person_detection.postprocess import (
    PersonDetFilterV2,
)
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "input_key",
        "num_anchor",
        "threshold",
        "num_classes",
    ],
    [
        pytest.param(
            "rpn_head_out",
            2,
            0.05,
            4,
        ),
    ],
)
def test_rpn_det_filter(input_key, num_anchor, threshold, num_classes):
    w, h = 128, 128
    feat_channels = (4 + num_classes) * num_anchor
    feats, _, _ = gen_fake_feats(w, h, [4], feat_channels=feat_channels)
    rpn_head_results = {"rpn_head_out": feats}

    filter = PersonDetFilterV2(
        input_key=input_key,
        num_anchor=num_anchor,
        threshold=threshold,
    )
    outs = filter(rpn_head_results)

    assert "filter_coord" in outs
    assert "filter_scores" in outs
    assert "filter_bboxes" in outs
    assert outs["filter_coord"].shape[1] == 2
    assert outs["filter_scores"].shape[1] == num_anchor * num_classes
    assert outs["filter_bboxes"].shape[1] == 4 * num_anchor


if __name__ == "__main__":
    pytest.main(["-s", __file__])
