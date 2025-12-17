from typing import Dict, List

import torch

from hat.core.data_struct.base_struct import DetBoxes3D
from hat.models.task_modules.roi_modules import ROI3DDecoder


def test_rcnn_decoder():
    roi3d_decoder = ROI3DDecoder(
        focal_length_default=1114.3466796875,
        scale_wh=(0.5, 0.5),
        undistort_depth_uv=True,
        image_hw=(640, 1024),
    )

    calib = torch.zeros((1, 3, 4))
    distCoeffs = torch.zeros((1, 8))

    fake_rois_data = [torch.randn([10, 4])]

    fake_head_out = dict(
        cls_pred=torch.randn([10, 1, 8, 8]),
        offset_2d_pred=torch.randn([10, 2, 8, 8]),
        offset_3d_pred=torch.randn([10, 2, 8, 8]),
        depth_u_pred=torch.randn([10, 1, 8, 8]),
        depth_v_pred=torch.randn([10, 1, 8, 8]),
        dims_pred=torch.randn([10, 3, 1, 1]),
        rot_pred=torch.randn([10, 2, 1, 1]),
        iou_pred=torch.randn([10, 1, 1, 1]),
    )

    pred_ret = roi3d_decoder(fake_rois_data, fake_head_out, calib, distCoeffs)

    assert isinstance(pred_ret, Dict)
    assert "pred_roi_3d" in pred_ret
    assert isinstance(pred_ret["pred_roi_3d"], List)
    assert len(pred_ret["pred_roi_3d"]) == 1
    assert isinstance(pred_ret["pred_roi_3d"][0], DetBoxes3D)
    assert len(pred_ret["pred_roi_3d"][0]) == 10
