import numpy as np
import pytest
import torch

from hat.models.task_modules.lidar.lidar_loss import (
    AfdetLidarLoss,
    AfsegLidarLoss,
)
from hat.utils.package_helper import check_packages_available


def gen_fake_data_pred():
    return dict(
        hm=torch.zeros(1, 3, 128, 240, dtype=torch.float32),
        dim=torch.zeros(1, 3, 128, 240, dtype=torch.float32),
        height=torch.zeros(1, 1, 128, 240, dtype=torch.float32),
        rot=torch.zeros(1, 2, 128, 240, dtype=torch.float32),
        reg=torch.zeros(1, 2, 128, 240, dtype=torch.float32),
    )


def gen_fake_data_target():

    anno_box = torch.rand(1, 8) * torch.tensor(
        [10, 10, 3, 4, 4, 4, np.pi, 0], dtype=torch.float32
    ) + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi, 0], dtype=torch.float32)
    gt_boxes_task = torch.zeros((1, 8), dtype=torch.float32)

    return dict(
        metadata=range(1),
        hm=torch.zeros(1, 3, 128, 240, dtype=torch.float32),
        ind=torch.ones((1, 1), dtype=torch.int64),
        ind_reg=torch.ones((1, 1), dtype=torch.int64),
        mask_reg=torch.ones((1, 1), dtype=torch.int64),
        reg=torch.ones((1, 1), dtype=torch.int64),
        cat=torch.ones((1, 1), dtype=torch.int64),
        mask=torch.ones((1, 1), dtype=torch.int64),
        anno_box=anno_box,
        gt_boxes_tasks=gt_boxes_task,
    )


def gen_fake_data_pred_seg():
    return [
        dict(
            map_seg_hm=torch.zeros(1, 2, 128, 128, dtype=torch.float32),
        )
    ]


def gen_fake_data_target_seg():

    return dict(
        metadata=range(1),
        map_seg_hm=torch.zeros(1, 2, 128, 128, dtype=torch.float32),
        seg_loss_mask=torch.zeros((1, 2, 128, 128), dtype=torch.int64),
    )


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_AfdetLidarLoss():

    model = AfdetLidarLoss(
        weight=2,
        weight_iou=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )

    example = gen_fake_data_target()
    preds_dict = gen_fake_data_pred()

    result_dict = model(example, preds_dict, None)

    assert type(result_dict["loss"]) is torch.Tensor
    assert type(result_dict["loss_loc"]) is torch.Tensor
    assert type(result_dict["loss_hm"]) is torch.Tensor


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_AfsegLidarLoss():

    tasks = [
        dict(
            num_class=2,
            class_names=[
                0,
                1,
            ],
        ),
    ]
    model = AfsegLidarLoss(
        tasks=tasks,
        weight=2,
        use_bd=True,
    )

    example = gen_fake_data_target_seg()
    preds_dict = gen_fake_data_pred_seg()

    result_dict = model(example, preds_dict, None)

    assert type(result_dict["loss"]) is torch.Tensor
    assert type(result_dict["loss_hm"]) is torch.Tensor
    assert type(result_dict["loss_bd"]) is torch.Tensor
