from typing import Dict, List

import torch

from hat.core.data_struct.base_struct import MultipleBoxes2D
from hat.models.task_modules.roi_modules import HeatmapBox2dDecoder


def test_heatmap_bbox2d_decoder():

    ground_line_decoder = HeatmapBox2dDecoder(score_threshold=0.3)

    fake_rois_data = [torch.randn([10, 4])]
    fake_head_out = dict(
        rcnn_cls_pred=torch.randn([10, 1, 8, 8]),
        rcnn_reg_pred=torch.randn([10, 4, 8, 8]),
    )

    pred_ret = ground_line_decoder(fake_rois_data, fake_head_out)
    assert isinstance(pred_ret, Dict)
    assert "pred_boxes" in pred_ret
    assert isinstance(pred_ret["pred_boxes"], List)
    assert len(pred_ret["pred_boxes"]) == 1
    assert isinstance(pred_ret["pred_boxes"][0], MultipleBoxes2D)
    assert len(pred_ret["pred_boxes"][0]) == 10
