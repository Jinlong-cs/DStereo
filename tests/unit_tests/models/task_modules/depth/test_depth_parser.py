import pytest

from hat.models.task_modules.depth import DepthHeadParserWithScale
from tests.utils import gen_fake_depth_data


@pytest.mark.parametrize(
    ["out_stride"],
    [
        pytest.param([1, 2, 4]),
        pytest.param([4, 8, 16]),
    ],
)
def test_parser(out_stride):
    w, h = 896, 896

    preds, _ = gen_fake_depth_data(
        w, h, strides=out_stride, label_dim=2, n=8, label_name="gt_depth"
    )
    head_out_names = {"depth": "pred_depth"}
    head_parser_strides = {head_out_names["depth"]: out_stride}
    depth_head_parser = DepthHeadParserWithScale(
        out_strides=head_parser_strides,
    )
    resize_preds = depth_head_parser({"pred_depth": preds})
    for _, preds in resize_preds.items():
        for pred in preds:
            assert pred.shape[2:] == (
                h,
                w,
            ), f"pred: {pred.shape}, target: {h}, {w}"
