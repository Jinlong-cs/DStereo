import pytest

from hat.models.task_modules.traffic_sign.postprocess import (
    RPNTrafficSignFilter,
)
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "input_key",
        "threshold",
        "anchor_args",
    ],
    [
        pytest.param(
            "rpn_head_out",
            0.05,
            dict(
                feat_strides=[4],
                anchor_wh_groups=[
                    [
                        [50, 16],
                        [55, 44],
                        [98, 65],
                        [50, 30],
                        [32, 32],
                        [40, 42],
                    ],
                ],
                num_fg_classes=2,
                exclude_background=True,
            ),
        ),
    ],
)
def test_rpn_det_filter(input_key, threshold, anchor_args):
    w, h = 128, 128
    num_anchor = len(anchor_args["anchor_wh_groups"])
    num_classes = anchor_args["num_fg_classes"]
    feat_channels = (4 + num_classes) * num_anchor
    feats, _, _ = gen_fake_feats(
        w, h, anchor_args["feat_strides"], feat_channels=feat_channels
    )
    rpn_head_results = {"rpn_head_out": feats}

    filter = RPNTrafficSignFilter(
        threshold=threshold,
        anchor_args=anchor_args,
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
