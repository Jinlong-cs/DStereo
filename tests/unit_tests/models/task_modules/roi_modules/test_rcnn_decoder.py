from typing import Dict, List

import torch

from hat.core.data_struct.base_struct import DetBoxes2D
from hat.models.base_modules.bbox_decoder import XYWHBBoxDecoder
from hat.models.task_modules.roi_modules import RCNNDecoder


def test_rcnn_decoder():
    classnames = [
        "full_visible",
        "occluded",
        "heavily_occluded",
        "invisible",
    ]

    rcnn_decoder = RCNNDecoder(
        bbox_decoder=XYWHBBoxDecoder(legacy_bbox=True),
        cls_act_type="identity",
        cls_name_mapping={
            i + 1: cls_name for i, cls_name in enumerate(classnames)
        },
    )

    fake_rois_data = [torch.randn([10, 4])]
    fake_head_out = dict(
        rcnn_cls_pred=torch.randn([10, 1, 1, 1]),
        rcnn_reg_pred=torch.randn([10, 4, 1, 1]),
    )

    pred_ret = rcnn_decoder(fake_rois_data, fake_head_out)
    assert isinstance(pred_ret, Dict)
    assert "pred_boxes" in pred_ret
    assert isinstance(pred_ret["pred_boxes"], List)
    assert len(pred_ret["pred_boxes"]) == 1
    assert isinstance(pred_ret["pred_boxes"][0], DetBoxes2D)
    assert len(pred_ret["pred_boxes"][0]) == 10
