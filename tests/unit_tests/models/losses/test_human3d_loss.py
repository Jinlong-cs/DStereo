import torch

from hat.models.losses.human3d import Human3dLoss


def test_human3d_loss():
    human3d_loss = Human3dLoss(
        loss_weights={
            "shape_loss_weight": 0 * 60,
            "keypoint_loss_weight": 5 * 60,
            "keypoint3d_loss_weight": 5 * 60,
            "pose_loss_weight": 1.0 * 60,
            "beta_loss_weight": 0.1 * 60,
            "cam_loss_weight": 1.0 * 60,
        }
    )
    data = {}
    data["pred"] = {
        "pr_cam": torch.randn((1, 3)),
        "pr_pose": torch.randn((1, 24, 3, 3)),
        "pr_betas": torch.randn((1, 11)),
        "pr_verts": torch.randn((1, 6890, 3)),
        "pr_keypoint": torch.randn((1, 24, 2)),
        "pr_keypoint_3d": torch.randn((1, 24, 3)),
    }
    data["label"] = {
        "gt_pose": torch.randn((1, 72)),
        "gt_betas": torch.randn((1, 11)),
        "gt_verts": torch.randn((1, 6890, 3)),
        "gt_keypoint": torch.randn((1, 24, 2)),
        "gt_keypoint_3d": torch.randn((1, 24, 3)),
    }
    keypoint2d_conf = torch.ones((1, 24))
    keypoint3d_conf = torch.ones((1, 24))
    has_pose_3d = torch.ones(1)
    valid_fit = torch.ones(1)
    loss = human3d_loss(
        data, keypoint2d_conf, keypoint3d_conf, has_pose_3d, valid_fit
    )
    assert "loss_keypoints_3d" in loss
    assert "loss_keypoints" in loss
    assert "loss_shape" in loss
    assert "loss_regr_betas" in loss
    assert "loss_regr_pose" in loss
    assert "loss_regr_cam" in loss
