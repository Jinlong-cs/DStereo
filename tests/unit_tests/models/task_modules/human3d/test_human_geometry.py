import torch

from hat.models.task_modules.human3d.utils.geometry import (
    estimate_translation,
    perspective_projection,
)

BATCH_SIZE = 4
NUM_LDMK = 24


def test_perspective_projection():
    joints_3d = torch.randn((BATCH_SIZE, NUM_LDMK, 3))
    rotation = torch.eye(3).unsqueeze(0).expand(BATCH_SIZE, -1, -1)
    translation = torch.randn((BATCH_SIZE, 3))
    focal_length = torch.ones((BATCH_SIZE,)) * 5000.0
    camera_center = torch.zeros((BATCH_SIZE, 2))
    joints_2d = perspective_projection(
        joints_3d, rotation, translation, focal_length, camera_center
    )
    assert joints_2d.size() == (BATCH_SIZE, 24, 2)


def test_estimate_translation():
    joints_3d = torch.randn((BATCH_SIZE, NUM_LDMK, 3))
    joints_2d = torch.randn((BATCH_SIZE, NUM_LDMK, 2))
    joints_2d_conf = torch.randn((BATCH_SIZE, NUM_LDMK, 1))
    trans = estimate_translation(joints_3d, joints_2d, joints_2d_conf)
    assert trans.size() == (BATCH_SIZE, 3)
