import torch

from hat.metrics.confusion_matrix import ConfusionMatrixBEV3D


def test_confusion_matrix_for_bev3d():
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
    }
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor(
                [[[1, 1, 0], [3, 3, 0], [7, 7, 0], [10, 10, 0]]]
            ),
            "vcs_cls_": torch.Tensor([[1, 1, 2, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0, 0]]),
            "vcs_dim_": torch.Tensor(
                [[[1, 1, 1], [7, 7, 7], [2, 2, 2], [2, 2, 2]]]
            ),
            "vcs_ignore_": torch.tensor([[0, 0, 0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5, 0.8]]),
        },
    }
    det = {
        "bev3d_dim": torch.Tensor(
            [[[1, 1, 1], [7, 7, 7], [2, 2, 2], [2, 2, 2]]]
        ),
        "bev3d_rot": torch.Tensor([[0, 0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 0, 2, 2]]),
        "bev3d_score": torch.Tensor([[1, 1, 1, 1]]),
        "bev3d_ct": torch.Tensor([[[1, 1], [3, 3], [7, 7], [20, 20]]]),
    }
    bev3d_confusion = ConfusionMatrixBEV3D(
        name="Bev3DVehicleCategoryConfusionMatrix",
        num_classes=3,
        score_threshold=0.1,
        iou_threshold=0.1,
        gt_max_depth=110,
        eval_vcs_range=(-30.0, -75.0, 100.0, 75.0),
        enable_ignore=True,
        ego_ignore_range=(-0.6, -0.5, 2.0, 0.5),
        id2label=id2label,
        pred_keys=(
            "bev3d_ct",
            "bev3d_cls_id",
            "bev3d_score",
            "bev3d_rot",
            "bev3d_dim",
            "bev3d_loc_z",
        ),
        confusion_save_path=None,
        save_score_thr=0.1,
        gt_id_key="vcs_cls_",
        pred_id_key="bev3d_cls_id",
    )
    bev3d_confusion.update(gt, det)
    assert bev3d_confusion.confusion_matrix[1][0] == 1
