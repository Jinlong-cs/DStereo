import numpy as np
import torch

from hat.models.task_modules.landmark import HumanPoseLabelFromMatch


def test_human_pose_encoder():
    gluon_target_label = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                [
                    [-1, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0],
                ],
            ]
        ]
    )

    gluon_target_label_weight = np.array(
        [
            [
                [
                    [1.5, 1.5, 1.5, 1.5],
                    [1.5, 1.5, 1.5, 1.5],
                    [1.5, 1.5, 1.5, 1.5],
                    [1.5, 1.5, 1.5, 1.5],
                ],
                [
                    [1.5, 1.5, 1.5, 1.5],
                    [1.5, 1.5, 1.5, 1.5],
                    [1.5, 1.5, 1.5, 1.5],
                    [1.5, 1.5, 1.5, 1.5],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ]
    )

    gluon_target_offset = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, -0.22277232, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.8663366, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, -0.02475251, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, -0.66831696, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ]
    )

    gluon_target_offset_weight = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.5, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.5, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ]
    )

    np.set_printoptions(threshold=np.inf)
    label_encoder = HumanPoseLabelFromMatch(
        4,
        4,
        3,
        0.1,
        (0, 3),
        "smooth_L1",
    )
    boxes = torch.tensor([[[0, 0, 100, 100]]], dtype=torch.float32)
    gt_boxes = torch.tensor(
        [[[0, 0, 100, 100, 1, 23, 34, 1, 25, 69, 2, 57, 89, 2]]],
        dtype=torch.float32,
    )
    match_pos_flag = torch.randint(1, 2, (1, 1))
    match_gt_id = torch.randint(0, 1, (1, 1))
    label = label_encoder(boxes, gt_boxes, match_pos_flag, match_gt_id)
    hat_target_label = label["ldmk_cls_label"][0].numpy()
    hat_target_label_weight = label["ldmk_cls_label_weight"][0].numpy()
    hat_target_offset = label["ldmk_reg_label"][0].numpy()
    hat_target_offset_weight = label["ldmk_reg_label_weight"][0].numpy()
    assert ((hat_target_label - gluon_target_label) < 1e-5).all()
    assert ((hat_target_label_weight - gluon_target_label_weight) < 1e-5).all()
    assert ((hat_target_offset - gluon_target_offset) < 1e-5).all()
    assert (
        (hat_target_offset_weight - gluon_target_offset_weight) < 1e-5
    ).all()
    assert len(label) == 4
    assert "ldmk_cls_label" in label
    assert "ldmk_cls_label_weight" in label
    assert "ldmk_reg_label" in label
    assert "ldmk_reg_label_weight" in label
