from collections import OrderedDict

import numpy as np
import torch

from hat.core.virtual_camera import CylindricalCamera, FisheyeCamera


def gen_fake_mtf_real3d_label_pred_data(h, w, strides, head_channels):
    preds = OrderedDict()
    for name, out_channel in head_channels.items():
        preds[name] = []
        for stride in strides:
            assert (w % stride) == 0
            assert (h % stride) == 0
            stride_h = h // stride
            stride_w = w // stride
            preds[name].append(
                torch.randn((2, out_channel[-1], stride_h, stride_w))
            )

    label = {}
    label["img_name"] = "dummy.jpg"
    label["img_id"] = 0
    virtual_camera = CylindricalCamera.init_cam_param_by_matrix(
        image_size=[w, h],  # W * H
        camera_matrix=np.array(
            [[w // 2, 0, w // 2], [0, h // 2, h // 2], [0, 0, 1]]
        ),
        is_virtual=True,
    )
    source_camera = FisheyeCamera()
    label["virtual_cam"] = [virtual_camera, virtual_camera]
    label["source_cam"] = [source_camera, source_camera]

    label["layout"] = ["chw", "chw"]
    label["gt_bboxes"] = [torch.zeros((1, 4)), torch.zeros((1, 4))]
    label["gt_classes"] = [torch.zeros((1,)), torch.zeros((1,))]
    label["gt_bboxes_3d"] = [torch.zeros((1, 7)), torch.zeros((1, 7))]
    label["gt_classes_3d"] = [torch.zeros((1,)), torch.zeros((1,))]
    label["centers2d_prj"] = [torch.zeros((1, 2)), torch.zeros((1, 2))]
    label["depths"] = [torch.ones((1,)), torch.ones((1,))]

    return label, preds
