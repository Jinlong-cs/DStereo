import torch

try:
    import smplx  # noqa

    SMPLX = True
except ImportError:
    SMPLX = False

from hat.models.backbones.resnet import ResNet50
from hat.models.structures.human3d import HMR, SPIN
from hat.models.task_modules.human3d.head import HMRHead
from hat.models.task_modules.human3d.smpl import ExtentedSMPL


def test_spin():
    if not SMPLX:
        return
    hmr = HMR(
        backbone=ResNet50(1000, {}, include_top=False),
        head=HMRHead(
            smpl_mean_params="./tmp_orig_data/human3d/spin_params/data/smpl_mean_params.npz"  # noqa
        ),
    )
    smpl = ExtentedSMPL(
        joint_regressor_train_extra="./tmp_orig_data/human3d/spin_params/data/J_regressor_extra.npy",  # noqa
        model_path="./tmp_orig_data/human3d/spin_params/data/smpl",
        batch_size=1,
        create_transl=False,
    )
    spin = SPIN(
        hmr,
        "val",
        smpl,
        None,
        None,
        224,
        None,
        run_smplify=False,
    )
    inputs = torch.randn((1, 3, 224, 224))
    gt_ldmk = torch.randn((1, 24, 2))
    gt_pose = torch.randn((1, 72))
    gt_betas = torch.randn((1, 10))
    gt_joints = torch.randn((1, 24, 3))
    has_smpl = torch.zeros((1, 1))
    has_pose_3d = torch.zeros((1, 1))
    is_flipped = torch.zeros((1, 1))
    rot_angle = torch.zeros((1, 1))
    dataset_name = ["test"]
    indices = torch.zeros((1, 1))
    data = {
        "img": inputs,
        "gt_ldmk": gt_ldmk,
        "gt_smpl_pose": gt_pose,
        "gt_smpl_betas": gt_betas,
        "gt_ldmk_3d": gt_joints,
        "has_smpl": has_smpl,
        "has_ldmk_3d": has_pose_3d,
        "is_flipped": is_flipped,
        "rot_angle": rot_angle,
        "dataset_name": dataset_name,
        "sample_index": indices,
    }
    outputs = spin(data)
    assert "pred_pose" in outputs
    assert "pred_betas" in outputs
    assert "pred_camera" in outputs
    assert outputs["pred_pose"].size() == (1, 144)
    assert outputs["pred_betas"].size() == (1, 11)
    assert outputs["pred_camera"].size() == (1, 3)
