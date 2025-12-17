import torch

try:
    import smplx  # noqa

    SMPLX = True
except ImportError:
    SMPLX = False

from hat.models.task_modules.human3d.smpl import ExtentedSMPL


def test_smpl():
    if not SMPLX:
        return

    smpl = ExtentedSMPL(
        joint_regressor_train_extra="./tmp_orig_data/human3d/spin_params/data/J_regressor_extra.npy",  # noqa
        model_path="./tmp_orig_data/human3d/spin_params/data/smpl",
        batch_size=1,
        create_transl=False,
    )

    global_orient = torch.ones((1, 3))
    body_pose = torch.ones((1, 69))
    betas = torch.ones((1, 10))

    smpl_output = smpl(
        global_orient=global_orient, body_pose=body_pose, betas=betas
    )
    assert "betas" in smpl_output
    assert "body_pose" in smpl_output
    assert "global_orient" in smpl_output
    assert "joints" in smpl_output
    assert "vertices" in smpl_output
