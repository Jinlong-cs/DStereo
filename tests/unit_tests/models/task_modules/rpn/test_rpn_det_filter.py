import pytest

from hat.models.task_modules.rpn.rpn_head import RPNDetFilter
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "threshold",
        "num_anchor",
        "feat_strides",
        "num_classes",
    ],
    [
        pytest.param(0.25, 1, [4], 1),
        pytest.param(0.5, 2, [8], 4),
    ],
)
def test_rpn_det_filter(threshold, num_anchor, feat_strides, num_classes):
    w, h = 64, 128
    feat_channels = (4 + num_classes) * num_anchor
    feats, _, _ = gen_fake_feats(
        w, h, feat_strides, feat_channels=feat_channels
    )
    rpn_head_results = {"head_predict_rpn_head_out": feats}

    filter = RPNDetFilter(threshold=threshold, num_anchor=num_anchor)
    outs = filter(None, rpn_head_results)

    assert "filter_coord" in outs
    assert "filter_scores" in outs
    assert "filter_bboxes" in outs
    assert outs["filter_coord"].shape[1] == 2
    assert outs["filter_scores"].shape[1] == num_classes * num_anchor
    assert outs["filter_bboxes"].shape[1] == 4 * num_anchor


if __name__ == "__main__":
    pytest.main(["-s", __file__])
