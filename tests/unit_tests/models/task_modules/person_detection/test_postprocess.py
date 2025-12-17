import pytest
import torch
from horizon_plugin_pytorch.nn import AnchorGenerator

from hat.models.task_modules.person_detection.postprocess import (
    PersonDetAnchorPostProcess,
)
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "input_key",
        "num_classes",
        "threshold",
        "img_hw",
        "num_anchor",
        "stride",
    ],
    [
        pytest.param(
            "rpn_head_out",
            4,
            0.5,
            [160, 160],
            2,
            [4],
        ),
    ],
)
def test_rpn_det_filter(
    input_key, num_classes, threshold, img_hw, num_anchor, stride
):
    feat_channels = (4 + num_classes) * num_anchor
    feats, _, _ = gen_fake_feats(
        img_hw[1], img_hw[0], stride, feat_channels=feat_channels
    )
    rpn_head_results = {"rpn_head_out": feats}

    filter = PersonDetAnchorPostProcess(
        num_classes=num_classes,
        class_offsets=[0],
        use_clippings=True,
        image_hw=img_hw,
        nms_iou_threshold=threshold,
        pre_nms_top_k=2000,
        post_nms_top_k=100,
        nms_margin=0.0,
        input_key=input_key,
        box_filter_threshold=0.05,
        nms_padding_mode="pad_zero",
        bbox_min_hw=[1, 1],
        input_shift=6,
    )
    anchor = generate_anchor(num_classes, img_hw, num_anchor, stride)
    outs = filter(anchor, rpn_head_results)

    assert "pred_boxes_out" in outs
    assert "pred_boxes" in outs
    assert "pred_scores" in outs
    assert "pred_cls" in outs
    assert outs["pred_boxes_out"][0].shape[1] == 4


def generate_anchor(num_classes, img_hw, num_anchor, stride):
    feat_channels = (4 + num_classes) * num_anchor
    h = int(img_hw[1] / stride[0])
    w = int(img_hw[0] / stride[0])
    head_out = [torch.zeros(1, feat_channels, h, w)]
    anchorgen = AnchorGenerator(
        feat_strides=stride,
        anchor_wh_groups=[[[40, 92], [92, 92]]],
        legacy_bbox=False,
    )
    anchor = anchorgen.forward(head_out)
    return anchor


if __name__ == "__main__":
    pytest.main(["-s", __file__])
