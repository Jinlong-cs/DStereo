from typing import Dict, List

import torch

from hat.models.task_modules.roi_modules import KpsDecoder


def test_kps_decoder():

    kps_decoder = KpsDecoder(num_kps=2, pos_distance=1.0, roi_expand_param=1.0)

    fake_rois_data = [torch.randn([10, 4])]
    fake_head_out = dict(
        kps_rcnn_cls_pred=torch.randn([10, 2, 8, 8]),
        kps_rcnn_reg_pred=torch.randn([10, 4, 8, 8]),
    )

    pred_ret = kps_decoder(fake_rois_data, fake_head_out)
    assert isinstance(pred_ret, Dict)
    assert "pred_kps" in pred_ret
    assert isinstance(pred_ret["pred_kps"], List)
    assert len(pred_ret["pred_kps"]) == 1
    assert len(pred_ret["pred_kps"][0]) == 10
