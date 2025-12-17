from typing import Dict, List

import torch

from hat.core.data_struct.base_struct import Lines2D
from hat.models.task_modules.roi_modules import GroundLinePointDecoder


def test_ground_line_point_decoder():

    ground_line_decoder = GroundLinePointDecoder(roi_expand_param=1.0)

    fake_rois_data = [torch.randn([10, 4])]
    fake_head_out = dict(
        rcnn_cls_pred=torch.randn([10, 1, 1, 1]),
        rcnn_reg_pred=torch.randn([10, 2, 1, 1]),
    )

    pred_ret = ground_line_decoder(fake_rois_data, fake_head_out)
    assert isinstance(pred_ret, Dict)
    assert "pred_gdl" in pred_ret
    assert isinstance(pred_ret["pred_gdl"], List)
    assert len(pred_ret["pred_gdl"]) == 1
    assert isinstance(pred_ret["pred_gdl"][0], Lines2D)
    assert len(pred_ret["pred_gdl"][0]) == 10
