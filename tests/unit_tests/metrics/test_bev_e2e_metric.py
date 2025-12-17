import numpy as np
import torch

from hat.metrics.bev.bev_e2e import BEVE2Eval
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import (
    e2e_instance_bev2vcs,
)


def get_fake_data():
    batch_gt = (
        {
            "pack_names": ["UT126_20221109_D_20221109-152311_964"],
            "timestamp": torch.tensor([[1667978591933]]),
            "fake_dataset_flag": [False],
            "motr_targets": [
                {
                    "bev_tracking": [
                        {
                            "obj_idxes": torch.tensor([348, 27]),
                            "labels": torch.tensor([0, 0]).long(),
                            "boxes": torch.tensor(
                                [
                                    [0.4455, 0.3828, 0.0121, 0.0316],
                                    [0.4855, 0.3828, 0.0121, 0.0316],
                                ]
                            ),
                            "yaws": torch.tensor(
                                [[0.9691, 0.2465], [0.9691, 0.2465]]
                            ),
                            "bev_loc_z": torch.tensor([0.8, 0.9]),
                            "heights": torch.tensor([1.8, 1.6]),
                            "scores": torch.tensor([0.9, 1.0]),
                            "velocities": torch.tensor(
                                [
                                    [12.2819, 2.6928, -0.0431],
                                    [12.2819, 2.6928, -0.0431],
                                ]
                            ),
                            "gt_traj_regs": torch.tensor(
                                [[[1.2263, 0.2714]], [[1.2263, 0.2714]]]
                            ),
                            "gt_traj_masks": torch.tensor(
                                [[[1, 1.0]], [[1, 1.0]]]
                            ),
                            "gt_traj_modals": torch.tensor([4, 3]).long(),
                        }
                    ],
                    "trajectory_pred": [
                        {
                            "ego_hisArrs": torch.tensor(
                                [
                                    [8.4062e-01, -3.3482e-03, 4.4022e-03],
                                    [2.1004e00, 1.9442e-04, 1.0818e-02],
                                    [3.3508e00, 1.6360e-02, 1.7668e-02],
                                    [4.6021e00, 4.3406e-02, 2.5954e-02],
                                    [5.8484e00, 8.4887e-02, 3.5511e-02],
                                    [7.0920e00, 1.3493e-01, 4.4790e-02],
                                    [8.3265e00, 1.9273e-01, 5.3736e-02],
                                    [9.5644e00, 2.6910e-01, 6.2842e-02],
                                    [1.0798e01, 3.5505e-01, 7.1840e-02],
                                    [1.2024e01, 4.4289e-01, 8.0812e-02],
                                    [1.3243e01, 5.3798e-01, 8.9702e-02],
                                    [1.4465e01, 6.4895e-01, 9.8441e-02],
                                ]
                            ),
                            "ids_list": np.array([28, 27]),
                            "valid_ids": np.array([348]),
                            "valid_classes": {28: 0, 27: 0},
                            "gt_traj_list": {
                                348: np.array(
                                    [
                                        [1.22633172, 0.27143016],
                                        [2.46373256, 0.54734931],
                                        [3.69993086, 0.81279977],
                                        [4.93519838, 1.0761227],
                                    ]
                                )
                            },
                            "gt_mask_list": {
                                348: np.array(
                                    [
                                        [1.0, 1.0],
                                        [1.0, 1.0],
                                        [1.0, 1.0],
                                        [1.0, 1.0],
                                    ]
                                )
                            },
                        }
                    ],
                }
            ],
        },
        "e2e_dynamic_detection",
    )

    output_bev = {
        "boxes": torch.tensor(
            [
                [0.4444, 0.3809, 0.0123, 0.0343],
                [0.4244, 0.3809, 0.0123, 0.0343],
            ]
        ),
        "bev_loc_z": torch.tensor([1.0611, 1.0611]),
        "appear_time": torch.tensor([0, 0]).long(),
        "disappear_time": torch.tensor([0, 0]).long(),
        "heights": torch.tensor([1.7061, 1.7061]),
        "labels": torch.tensor([0, 0]).long(),
        "obj_idxes": torch.tensor([1, 2]).long(),
        "scores": torch.tensor([0.8, 0.7]),
        "yaws": torch.tensor([[-7.2245, 1.7077], [-7.2245, 1.7077]]),
        "velocities": torch.tensor([[-7.0, 1.4, -1.5], [-7.0, 1.4, -1.5]]),
        "TrajRegs": torch.tensor(
            [
                [
                    [
                        [-9.21248078e-01, 1.18127279e-01],
                        [-1.87642813e00, 1.93336472e-01],
                        [-2.84342313e00, 3.11195910e-01],
                        [-3.79324079e00, 3.73930037e-01],
                    ]
                ],
                [
                    [
                        [-9.21248078e-01, 1.18127279e-01],
                        [-1.87642813e00, 1.93336472e-01],
                        [-2.84342313e00, 3.11195910e-01],
                        [-3.79324079e00, 3.73930037e-01],
                    ]
                ],
            ]
        ),
        "TrajScores": torch.tensor(
            [[0.21197549, 0.17071402], [0.21197549, 0.17071402]]
        ),
    }
    output_vcs = e2e_instance_bev2vcs(
        instances=output_bev,
        vcs_range=(-31.2, -76.8, 102.4, 76.8, -3, 5),
    )

    return batch_gt, {
        "e2e_dynamic": [{"bev_stage2_e2e_dynamic_head_predict": [output_vcs]}]
    }


def test_beve2e_metric(tmpdir):
    # tmpdir="tmp_output"
    beve2e_eval = BEVE2Eval(
        [0],
        0.2,
        0.1,
        100,
        metrics=["dx", "dxp", "dy", "dyp", "dxyp", "drot"],
        classes=["car", "bicycle", "pedestrian"],
        vcs_range=(-31.2, -76.8, 102.4, 76.8, -3, 5),
        time_delta=0.1,
        e2e_eval_tracking=False,
        e2e_eval_trajectory=True,
        e2e_submit_evs=False,
        e2e_predefined_classes=("car", "bicycle", "pedestrian"),
        prcurv_save_path=tmpdir,
        save_path=tmpdir,
        enable_ignore=False,
        eval_mode="let_iou",
        let_iou_param={"p_t": 0.35, "min_t": 4.0, "max_t": 8.0},
    )

    batch_gt, batch_output = get_fake_data()
    beve2e_eval.reset()
    beve2e_eval.update(batch_gt, batch_output)
    val = beve2e_eval.compute()
    assert len(val) == 3
    assert abs(val[2][0][0]["Recall"] - 0.5) < 1e-6  # recall
    assert abs(val[2][0][0]["Precision"] - 0.5) < 1e-6  # precision
