import pytest
import torch
from horizon_plugin_pytorch.nn import AnchorGenerator

from hat.registry import build_from_registry

try:
    import hatbc
except ImportError:
    hatbc = None


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.parametrize(
    ["anchor_wh_groups", "post_nms_top_k"],
    [
        pytest.param([[[16, 16], [24, 24], [52, 52]]], 100),
    ],
)
def test_attr_dpp(anchor_wh_groups, post_nms_top_k):
    config = dict(
        type="TLAttrDetPostProcess",
        num_classes=1,
        class_offsets=[0],
        use_clippings=True,
        image_hw=[96, 96],
        nms_iou_threshold=0.2,
        pre_nms_top_k=500,
        post_nms_top_k=post_nms_top_k,
        nms_margin=0.0,
        input_key="rpn_head_out",
        box_filter_threshold=0.01,
        nms_padding_mode="rollover",
        bbox_min_hw=(1, 1),
        attr_list=[["type", 13], ["color", 3]],
        node_name="anchor_pred",
    )
    model = build_from_registry(config)
    model.eval()

    head_out = dict(
        rpn_head_out=[
            dict(
                det_feature=torch.rand(1, 15, 24, 24),
                attr_feature_list=[
                    torch.rand(1, 39, 24, 24),
                    torch.rand(1, 9, 24, 24),
                ],
            ),
        ]
    )
    anchor_generator = AnchorGenerator(
        feat_strides=[4],
        anchor_wh_groups=anchor_wh_groups,
        legacy_bbox=True,
    )
    dummpy_data = torch.randn(1, 16, 24, 24)
    mlvl_anchors = anchor_generator([dummpy_data])[-1]

    y = model([mlvl_anchors], head_out, torch.Tensor([96, 96]).view(1, 2))
    assert isinstance(y, dict)
    assert y["pred_boxes_out"][0].shape == (post_nms_top_k, 4)
    assert y["pred_scores"][0].shape[0] == post_nms_top_k


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.parametrize(
    ["strides"],
    [
        pytest.param([1]),
    ],
)
def test_rpn_traffic_light_filter(strides):
    config = dict(
        type="RPNTrafficLightFilter",
        threshold=0.1,
        strides=strides,
        idx_range=None,
        for_compile=True,
        node_name="anchor_pred",
    )
    model = build_from_registry(config)
    model.eval()

    head_out = dict(
        rpn_cls_pred=[torch.rand(1, 3, 128, 128)],
        rpn_reg_pred=[torch.rand(1, 12, 128, 128)],
        rpn_attr_pred_dict=dict(
            color=[torch.rand(1, 9, 128, 128)],
            type=[torch.rand(1, 39, 128, 128)],
        ),
    )

    y = model(None, head_out, None)
    assert isinstance(y, dict)
    assert "filter_coord" in y.keys() and y["filter_coord"].shape[-1] == 2
    assert "filter_scores" in y.keys() and y["filter_scores"].shape[-1] == 3
    assert y["filter_coord"].shape[0] == y["filter_scores"].shape[0]
    assert y["filter_coord"].shape[0] == y["filter_bboxes"].shape[0]


if __name__ == "__main__":
    pytest.main(["-s", __file__])
