import numpy as np
import torch

from hat.core.data_struct.base_struct import DetBoxes2D3D
from hat.models.task_modules.real3d import HeatMap3DDecoder


def test_heatmap_3d_decoder():
    stride = 4
    image_hw = (640, 1024)
    decoder = HeatMap3DDecoder(
        focal_length_default=1114.3466796875,
        scale_wh=(0.5, 0.5),
        center=np.array([2048, 1280]) // 2,
        down_stride=stride,
        use_bev_nms=True,
        nms_iou_thresh=0.5,
        image_hw=image_hw,
        undistort_depth_uv=True,
    )

    h, w = image_hw[0] // stride, image_hw[1] // stride
    calib = torch.zeros((1, 3, 4))
    dist_coeffs = torch.zeros((1, 8))

    fake_head_out = dict(
        hm=torch.randn([1, 1, h, w]),
        dep_u=torch.randn([1, 1, h, w]),
        dep_v=torch.randn([1, 1, h, w]),
        rot=torch.randn([1, 2, h, w]),
        dim=torch.randn([1, 3, h, w]),
        loc_offset=torch.randn([1, 2, h, w]),
        wh=torch.randn([1, 2, h, w]),
    )

    pred_ret = decoder(fake_head_out, calib, dist_coeffs)

    assert isinstance(pred_ret, list)
    assert len(pred_ret) == 1
    assert isinstance(pred_ret[0], DetBoxes2D3D)
