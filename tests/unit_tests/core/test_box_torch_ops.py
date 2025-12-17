import os

import numpy as np
import pytest
import torch

from hat.core.box_torch_ops import (
    boxes_aligned_iou3d_gpu,
    boxes_iou3d_gpu_horizon,
)


def test_boxes_iou3d_gpu_horizon():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    boxes_a = torch.rand(20, 7) * torch.tensor(
        [10, 10, 3, 4, 4, 4, np.pi], dtype=torch.float32
    ) + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi], dtype=torch.float32)
    boxes_b = torch.rand(20, 7) * torch.tensor(
        [10, 10, 3, 4, 4, 4, np.pi], dtype=torch.float32
    ) + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi], dtype=torch.float32)

    iou3d, iou_bev = boxes_iou3d_gpu_horizon(boxes_a.cuda(), boxes_b.cuda())

    assert iou3d.shape == (20, 20)
    assert iou_bev.shape == (20, 20)


@pytest.mark.parametrize(
    ["rect"],
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_boxes_aligned_iou3d_gpu(rect: bool):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    boxes_a = torch.rand(20, 7) * torch.tensor(
        [10, 10, 3, 4, 4, 4, np.pi], dtype=torch.float32
    ) + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi], dtype=torch.float32)
    boxes_b = torch.rand(20, 7) * torch.tensor(
        [10, 10, 3, 4, 4, 4, np.pi], dtype=torch.float32
    ) + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi], dtype=torch.float32)

    iou3d, iou_bev = boxes_aligned_iou3d_gpu(
        boxes_a.cuda(), boxes_b.cuda(), rect=rect
    )

    assert iou3d.shape == (20, 20)
    assert iou_bev.shape == (20, 20)
