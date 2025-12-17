import torch

from hat.models.losses.e2e_loss import E2EDynamicLoss


def get_fake_e2e_data():
    pred = [
        {
            "output_for_losses": {
                "frome_0_output_layer_0": {
                    "pred_output": {
                        "pred_logits": torch.randn(300, 2),
                        "pred_boxes": torch.rand(300, 4),
                        "pred_yaws": torch.ones(300, 2),
                        "pred_zheights": torch.ones(300, 2),
                        "pred_track_scores": torch.rand(300, 1),
                        "pred_velocities": torch.randn(300, 3),
                        "pred_TrajRegs": torch.randn(1, 5, 300, 60, 2),
                        "pred_TrajLogits": torch.randn(1, 5, 300, 1),
                    },
                    "gt_instances": {
                        "obj_idxes": torch.ones(100, dtype=torch.int64) * -99,
                        "labels": torch.zeros(100, dtype=torch.int64),
                        "boxes": torch.rand((100, 4), dtype=torch.float32),
                        "yaws": torch.rand((100, 2), dtype=torch.float32),
                        "bev_loc_z": torch.rand(100, dtype=torch.float32),
                        "heights": torch.rand(100, dtype=torch.float32),
                        "scores": torch.rand(100, dtype=torch.float32),
                        "velocities": torch.rand(
                            (100, 3), dtype=torch.float32
                        ),
                        "gt_traj_regs": torch.rand(
                            (100, 60, 2), dtype=torch.float32
                        )
                        * 2.0
                        - 1.0,
                        "gt_traj_masks": (
                            torch.randn((100, 60, 2), dtype=torch.float32)
                            > 0.0
                        ).to(torch.float32),
                        "gt_traj_modals": torch.randint(0, 5, (100,)).long(),
                    },
                    "match_indices": (
                        torch.randint(1, 100, (100,)),
                        torch.randint(1, 100, (100,)),
                    ),
                    "matched_gt_idxes": torch.randint(1, 300, (300,)),
                    "pre_matched_indices": (
                        torch.randint(1, 100, (100,)),
                        torch.randint(1, 100, (100,)),
                    ),
                },
                "aux_outputs_for_loss": {
                    "frome_0_aux_output_layer_0": {
                        "pred_output": {
                            "pred_logits": torch.rand(300, 2),
                            "pred_boxes": torch.rand(300, 4),
                            "pred_yaws": torch.ones(300, 2),
                            "pred_zheights": torch.ones(300, 2),
                        },
                        "gt_instances": {
                            "obj_idxes": torch.ones(100, dtype=torch.int64)
                            * -99,
                            "labels": torch.zeros(100, dtype=torch.int64),
                            "boxes": torch.rand((100, 4), dtype=torch.float32),
                            "yaws": torch.rand((100, 2), dtype=torch.float32),
                            "bev_loc_z": torch.rand(100, dtype=torch.float32),
                            "heights": torch.rand(100, dtype=torch.float32),
                            "scores": torch.rand(100, dtype=torch.float32),
                        },
                        "match_indices": (
                            torch.randint(1, 100, (100,)),
                            torch.randint(1, 100, (100,)),
                        ),
                    },
                },
            },
            "clip_num_samples": 100,
        }
    ]
    for tmp in pred:
        match_gt_idxes = tmp["output_for_losses"]["frome_0_output_layer_0"][
            "matched_gt_idxes"
        ]
        tmp["output_for_losses"]["frome_0_output_layer_0"]["pred_output"][
            "matched_gt_idxes"
        ] = match_gt_idxes
        tmp["output_for_losses"]["aux_outputs_for_loss"][
            "frome_0_aux_output_layer_0"
        ]["pred_output"]["matched_gt_idxes"] = match_gt_idxes
    return pred


def test_e2e_loss(calc_velocity_loss=True, calc_trajectory_loss=True):
    torch.manual_seed(0)
    loss_weights = {
        "loss_ce": 2,
        "loss_xy": 5,
        "loss_wh": 2.5,
        "loss_ciou": 1,
        "loss_zheight": 1,
        "loss_yaw": 1,
        "loss_velocity": 0.2,
        "loss_velo_yaw_consistency": 0.1,
        "loss_trajprob": 5,
        "loss_trajreg": 5,
        "loss_track_score": 2,
    }

    map_label = {
        "Car": 0,
        "Cyclist": 1,
        "Pedestrian": 2,
    }
    loss = E2EDynamicLoss(
        map_label=map_label,
        loss_weights=loss_weights,
        calc_velocity_loss=calc_velocity_loss,
        calc_trajectory_loss=calc_trajectory_loss,
    )
    pred = get_fake_e2e_data()
    result = loss(pred)
    loss_name = list(loss_weights.keys())
    assert loss_name[0] in result

    target_loss = {
        "loss_ce": 629.3589,
        "loss_xy": 641.5439,
        "loss_wh": 1031.9504,
        "loss_ciou": 216.6832,
        "loss_zheight": 283.7530,
        "loss_yaw": 196.3133,
        "loss_velocity": 0.2968,
        "loss_velo_yaw_consistency": 0.0010,
        "trackloss_ce": 70.7374,
        "loss_trajprob": 10.6326,
        "loss_trajreg": 4.4991,
    }
    for key in result.keys():
        errstr = "{} unmatch!".format(key)
        assert (result[key] - target_loss[key]).abs() < 1e-4, errstr
