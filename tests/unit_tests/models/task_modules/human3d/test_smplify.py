import torch

try:
    import smplx  # noqa

    SMPLX = True
except ImportError:
    SMPLX = False
from hat.models.task_modules.human3d.smpl import ExtentedSMPL
from hat.models.task_modules.human3d.smplify import SMPLify


def test_smplify():
    if not SMPLX:
        print("please install smplx first")
        return

    smplify = SMPLify(
        prior_folder="./tmp_orig_data/human3d/spin_params/data",
        smpl_model=ExtentedSMPL(
            model_path="./tmp_orig_data/human3d/spin_params/data/smpl",
            batch_size=1,
            create_transl=False,
            age="kid",
            kid_template_path="./tmp_orig_data/human3d/spin_params/data/smpl_kid_template.npy",  # noqa
            joint_regressor_train_extra="./tmp_orig_data/human3d/spin_params/data/J_regressor_extra.npy",  # noqa
        ),
        loss_weights={
            "pose_prior_weight": 4.78,
            "shape_prior_weight": 5,
            "angle_prior_weight": 15.2,
        },
        device="cpu",
    )

    cam_t = torch.ones((1, 3))
    pose = torch.ones((1, 72))
    betas = torch.ones((1, 11))
    kp = torch.randn((1, 24, 2))
    conf = torch.randn((1, 24))

    (
        vertices,
        joints,
        pose,
        betas,
        camera_translation,
        reprojection_loss,
    ) = smplify(
        init_pose=pose,
        init_betas=betas,
        init_cam_t=cam_t,
        camera_center=torch.tensor([112.0, 112.0]),
        keypoints_2d=kp,
        keypoints_2d_conf=conf,
    )
    assert vertices.size() == (1, 6890, 3)
    assert joints.size() == (1, 24, 3)
    assert pose.size() == (1, 72)
    assert betas.size() == (1, 11)
    assert camera_translation.size() == (1, 3)
    assert reprojection_loss.size() == (1, 24)
