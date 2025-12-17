import torch

from hat.models.task_modules.dddv import DepthTarget


def test_depth_target():

    gt_depth_name = "gt_depth"
    gt_depth = (
        torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        .view((1, 1, 4, 4))
        .repeat((1, 1, 4, 4))
    )  # (1,1,16,16)

    depth_target = DepthTarget(gt_depth_name, downsample_scale=4)

    label_dict, pred_dict = dict(), dict()
    label_dict[gt_depth_name] = gt_depth

    label_dict = depth_target(label_dict, pred_dict)

    result = torch.tensor(
        [
            [2, 2, 2, 2],
            [2, 1, 1, 1],
            [2, 1, 1, 1],
            [2, 1, 1, 1],
        ],
        dtype=torch.float32,
    ).view((1, 1, 4, 4))

    assert torch.equal(label_dict[gt_depth_name], result)
