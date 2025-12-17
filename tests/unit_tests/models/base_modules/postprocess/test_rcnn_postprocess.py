import re
from typing import Dict

import horizon_plugin_pytorch
import pytest
import torch

version_info = re.split("[.+]", horizon_plugin_pytorch.__version__)


@pytest.mark.skipif(
    not (
        (int(version_info[0]) >= 1)
        and (int(version_info[1]) >= 2)
        and (int(version_info[2]) >= 3)
    ),
    reason="RcnnPostProcess is required, \
        Please install horizon_plugin_pytorch >= 1.2.3",
)
def test_rcnn_post_process():
    from hat.models.base_modules.postprocess import RCNNPostProcess

    rcnn_post_process = RCNNPostProcess(
        input_score_key="rcnn_cls_pred",
        input_deltas_key="rcnn_reg_pred",
        num_fg_classes=1,
        image_hw=(352, 640),
        nms_threshold=0.5,
        box_filter_threshold=0.7,
        post_nms_top_k=10,
        delta_mean=[0.0, 0.0, 0.0, 0.0],
        delta_std=[1.0, 1.0, 1.0, 1.0],
    )

    fake_rois_data = [torch.randn([10, 4])]
    fake_head_out = dict(
        rcnn_cls_pred=torch.randn((10, 2, 1, 1)),
        rcnn_reg_pred=torch.randn((10, 8, 1, 1)),
    )

    pred_ret = rcnn_post_process(fake_rois_data, fake_head_out)
    assert isinstance(pred_ret, Dict)
    assert "rpp_pred" in pred_ret
    assert isinstance(pred_ret["rpp_pred"], torch.Tensor)
    assert pred_ret["rpp_pred"].shape[0] == 1
    assert pred_ret["rpp_pred"].shape[1] == 10


if __name__ == "__main__":
    pytest.main(["-s", __file__])
