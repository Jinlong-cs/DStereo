import torch

from hat.models.losses.dddv_losses import (
    ConsistencyCrossEntropyLoss,
    DepthConfidenceLoss,
    DepthLoss,
    PoseResflowLoss,
)
from tests.utils import gen_fake_torch_randint_data


def test_depth_loss():
    torch.manual_seed(0)
    target_loss = 70.6035
    depth_loss = DepthLoss(low=0.001, high=150.0, loss_weight=1.0)
    pred_depths = gen_fake_torch_randint_data(
        (1, 1, 512, 960), low=0, high=150
    ).float()
    gt_depth = gen_fake_torch_randint_data(
        (1, 1, 512, 960), low=0, high=150
    ).float()
    loss = depth_loss(pred_depths, gt_depth)

    assert torch.abs(loss - target_loss) < 1e-4


def test_depth_confidence_loss():
    torch.manual_seed(0)
    target_l1_loss, target_confidence_loss = 70.7074, 73.9805
    depth_loss = DepthConfidenceLoss(low=0.001, high=150.0, loss_weight=1.0)
    pred_depths = gen_fake_torch_randint_data(
        (1, 2, 512, 960), low=0, high=150
    ).float()
    gt_depth = gen_fake_torch_randint_data(
        (1, 1, 512, 960), low=0, high=150
    ).float()
    loss = depth_loss(pred_depths, gt_depth)

    assert torch.abs(loss["depth_l1_loss"] - target_l1_loss) < 1e-4
    assert (
        torch.abs(loss["depth_confidence_loss"] - target_confidence_loss)
        < 1e-4
    )


def test_pose_resflow_loss():
    loss = PoseResflowLoss(
        scale_list=[2, 4, 8, 16],
        input_size=[512, 960],
        loss_weight=1.0,
        collect_vis=True,
    )
    pred_depths = gen_fake_torch_randint_data(
        (1, 1, 512 // 4, 960 // 4), low=0, high=150
    ).float()
    axisangle = [
        gen_fake_torch_randint_data((1, 3, 1, 1), low=0, high=255).float()
        / 255
        for i in range(2)
    ]
    translation = [
        gen_fake_torch_randint_data((1, 3, 1, 1), low=0, high=255).float()
        / 255
        for i in range(2)
    ]
    residual_flow = [
        gen_fake_torch_randint_data(
            (1, 3, 512 // 4, 960 // 4), low=0, high=255
        ).float()
        / 255
        for i in range(2)
    ]

    # target_dict
    color_imgs = [
        gen_fake_torch_randint_data((1, 3, 512, 960), low=0, high=255).float()
        / 255
        for i in range(3)
    ]
    intrinsics = (
        gen_fake_torch_randint_data((1, 3, 3), low=0, high=255).float() / 255
    )
    obj_mask = gen_fake_torch_randint_data(
        (1, 1, 512, 960), low=0, high=2
    ).float()
    result = loss(
        pred_depths,
        axisangle,
        translation,
        residual_flow,
        color_imgs,
        intrinsics,
        obj_mask,
    )

    assert "depth_pose_loss" in result
    assert "warp_imgs_vis" in result
    assert "recon_vis" in result
    assert "flow_vis" in result
    assert "depth_vis" in result


def test_consistency_ce_loss():
    torch.manual_seed(0)
    num_classes = 9
    loss_name = "consistency_ce"
    target_loss = 60.0237
    loss = ConsistencyCrossEntropyLoss(loss_name=loss_name, loss_weight=1.0)
    data1 = gen_fake_torch_randint_data(
        (1, num_classes, 512, 960), low=0, high=150
    ).float()
    data2 = gen_fake_torch_randint_data(
        (1, num_classes, 512, 960), low=0, high=150
    ).float()

    result = loss(data1, data2)

    assert loss_name in result
    assert torch.abs(result[loss_name] - target_loss) < 1e-4


def get_fake_bev3d_data():
    pred = {
        "bev3d_hm": 0.98 * torch.ones(1, 3, 256, 256),
        "bev3d_dim": 0.88 * torch.ones(1, 3, 256, 256),
        "bev3d_rot": 0.76 * torch.ones(1, 2, 256, 256),
        "bev3d_ct_offset": 0.5 * torch.ones(1, 2, 256, 256),
        "bev3d_loc_z": 0.05 * torch.ones(1, 1, 256, 256),
    }
    target = {
        "gt_bev_3d": {
            "bev3d_hm": 0.98 * torch.ones(1, 3, 256, 256),
            "bev3d_dim": 0.88 * torch.ones(1, 3, 256, 256),
            "bev3d_rot": 0.75 * torch.ones(1, 2, 256, 256),
            "bev3d_ct_offset": 0.5 * torch.ones(1, 2, 256, 256),
            "bev3d_loc_z": 0.05 * torch.ones(1, 1, 256, 256),
            "bev3d_weight_hm": torch.rand(1, 1, 256, 256),
            "bev3d_point_pos_mask": torch.ones(1, 1, 256, 256),
        }
    }
    return pred, target
