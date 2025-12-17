import torch

from hat.models.task_modules.centerpoint.bbox_coders import (
    CenterPointBBoxCoder,
)


def test_centerpoint_bbox_coder():
    bbox_coder = CenterPointBBoxCoder(
        pc_range=[-51.2, -51.2],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=100,
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=[0.2, 0.2],
    )

    heatmap = torch.rand(1, 2, 32, 32)
    rot_sine = torch.rand(1, 1, 32, 32)
    rot_cosine = torch.rand(1, 1, 32, 32)
    height = torch.rand(1, 1, 32, 32)
    reg = torch.rand(1, 2, 32, 32)
    dim = torch.rand(1, 3, 32, 32)
    vel = torch.rand(1, 2, 32, 32)

    temp = bbox_coder.decode(
        heatmap,
        rot_sine,
        rot_cosine,
        height,
        dim,
        vel,
        reg=reg,
        task_id=0,
    )
    assert temp is not None
