from typing import Dict, List

import torch

from hat.core.data_struct.base_struct import ClsLabels
from hat.models.task_modules.roi_modules import SoftmaxRoIClsDecoder


def test_softmax_roi_cls_decoder():

    classnames = [
        "full_visible",
        "occluded",
        "heavily_occluded",
        "invisible",
    ]

    cls_decoder = SoftmaxRoIClsDecoder(
        cls_name_mapping={i: name for i, name in enumerate(classnames)}
    )

    fake_rois_data = [torch.randn([10, 4])]
    fake_head_out = dict(rcnn_cls_pred=torch.randn([10, 8, 1, 1]))

    pred_ret = cls_decoder(fake_rois_data, fake_head_out)
    assert isinstance(pred_ret, Dict)
    assert "pred_cls" in pred_ret
    assert isinstance(pred_ret["pred_cls"], List)
    assert len(pred_ret["pred_cls"]) == 1
    assert isinstance(pred_ret["pred_cls"][0], ClsLabels)
    assert len(pred_ret["pred_cls"][0]) == 10
