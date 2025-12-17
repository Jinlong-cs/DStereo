import torch

from hat.core.face3d.lbs import lbs, rot6d_to_rotmat
from tests.unit_tests.models.task_modules.face3d.test_flame import NUM_VERTICES

BATCH_SIZE = 4
NUM_VETICES = 5023
NUM_JOINTS = 5

NUM_SHAPE = 150
NUM_EXP = 50
NUM_POSE = 36


def test_lbs():
    betas = torch.rand(BATCH_SIZE, NUM_SHAPE)
    pose = torch.rand(BATCH_SIZE, NUM_JOINTS * 3)
    v_template = torch.rand(NUM_VERTICES, 3)
    v_template = v_template.unsqueeze(0).expand(BATCH_SIZE, -1, -1)
    shapedirs = torch.rand(NUM_VERTICES, 3, NUM_SHAPE)
    posedirs = torch.rand(NUM_POSE, NUM_VERTICES * 3)
    J_regressor = torch.rand(NUM_JOINTS, NUM_VERTICES)
    parents = torch.tensor([-1, 0, 1, 1, 1], dtype=torch.int64)
    lbs_weights = torch.rand(NUM_VERTICES, NUM_JOINTS)

    verts, joints = lbs(
        betas,
        pose,
        v_template,
        shapedirs,
        posedirs,
        J_regressor,
        parents,
        lbs_weights,
        pose2rot=True,
    )
    assert isinstance(verts, torch.Tensor)
    assert isinstance(joints, torch.Tensor)
    assert verts.shape == (BATCH_SIZE, NUM_VERTICES, 3)
    assert joints.shape == (BATCH_SIZE, NUM_JOINTS, 3)


def test_rot6d_to_rotmat():
    rot_6d = torch.randn((BATCH_SIZE, 6))
    rot_mat = rot6d_to_rotmat(rot_6d)
    assert rot_mat.size() == (BATCH_SIZE, 3, 3)
