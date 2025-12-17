import os
import shutil

import numpy as np
import torch

from hat.core.box3d_utils import bev3d_let_nms, bev3d_multiclass_nms, bev3d_nms
from hat.metrics.bev_3d import BEVDetEval, BEVDetTagEval
from hat.metrics.metric_3dv_utils import (
    bev3d_bbox_tag_eval,
    collect_data,
    let_iou_2d,
    rotate_iou,
)


def get_fake_gt_and_pred(enable_ignore=True):
    """Generate fake gt and pred for unitest.

    Args:
        enable_ignore [bool]: whether to enable gt_ignore.

    Returns:
        gt [dict]: [gt det's attribute]
        det [dict]: [pred dets]
    """
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.randn((2, 300, 3)),
            "vcs_cls_": torch.randint(0, 2, (2, 300)),
            "vcs_rot_z_": torch.rand((2, 300)),
            "vcs_dim_": torch.rand((2, 300, 3)),
            "vcs_visible_": torch.rand((2, 300)),
        },
    }

    det = {
        "bev3d_dim": torch.rand((2, 100, 3)),
        "bev3d_rot": torch.randn((2, 100)),
        "bev3d_loc_z": torch.rand((2, 100)),
        "bev3d_cls_id": torch.randint(0, 2, (2, 100)),
        "bev3d_score": torch.rand((2, 100)),
        "bev3d_ct": torch.randn((2, 100, 2)),
    }

    if enable_ignore:
        gt["annos_bev_3d"].update(
            {"vcs_ignore_": torch.randint(0, 2, (2, 300))}
        )

    return gt, det


def get_certain_gt_and_pred(enable_ignore=True):
    """Generate certain fake gt and pred for unitest.

    Args:
        enable_ignore [bool]: whether to enable gt_ignore.

    Returns:
        gt [dict]: [gt det's attribute]
        det [dict]: [pred dets]
    """
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[1, 1, 0], [3, 3, 0], [5, 5, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }

    det = {
        "bev3d_dim": torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
        "bev3d_rot": torch.Tensor([[0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 1]]),
        "bev3d_score": torch.Tensor([[1, 1, 1]]),
        "bev3d_ct": torch.Tensor([[[1, 1], [3, 3], [5, 5]]]),
    }

    if enable_ignore:
        gt["annos_bev_3d"].update(
            {"vcs_ignore_": torch.tensor([[0, 0, 1]], dtype=bool)}
        )

    return gt, det


def test_bev3d_eval_init():
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    bev_det_eval = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        id2label=id2label,
    )
    assert bev_det_eval.metrics == (
        "dx",
        "dxp",
        "dy",
        "dyp",
        "dxyp",
        "drot",
    )
    assert len(bev_det_eval.eval_category_ids) > 0


def test_bev3d_tag_eval_init():
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    taggers = {
        "YAW --": {
            "gt_key": "vcs_rot_z_",
            "pred_key": "bev3d_rot",
            "mapper": {
                "YAW --": [[90 - 22.5, 90 + 22.5], [270 - 22.5, 270 + 22.5]]
            },
            "tagger_fn": "tag_by_range",
            "transforms": lambda rad: np.rad2deg(rad) % 360.0,
        },
    }
    bev_det_eval = BEVDetTagEval(
        metric_save_dir="./bev3d_tag_eval",
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        id2label=id2label,
        base_taggers=taggers,
    )
    assert bev_det_eval.metrics == (
        "dx",
        "dxp",
        "dy",
        "dyp",
        "dxyp",
        "drot",
    )
    assert len(bev_det_eval.eval_category_ids) > 0
    assert bev_det_eval.taggers
    assert 1 in bev_det_eval.taggers
    assert "YAW --" in bev_det_eval.taggers[1]


def test_bev3d_eval_vcs():
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[1, 1, 0], [3, 3, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1]]]),
            "vcs_ignore_": torch.tensor([[0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }
    gt_inside_ego_range = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[0.1, 0.1, 0], [0.1, 0.1, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1]]]),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }
    gt_outside_eval_vcs_range = {
        "timestamp": torch.tensor(
            [[162490000100], [162490000101]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[5, 5, 0], [6, 6, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1]]]),
            "vcs_ignore_": torch.tensor([[0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }
    # valid det
    det = {
        "bev3d_dim": torch.Tensor([[[1, 1, 1], [1, 1, 1]]]),
        "bev3d_rot": torch.Tensor([[0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1]]),
        "bev3d_score": torch.Tensor([[1, 1]]),
        "bev3d_ct": torch.Tensor([[[1, 1], [3, 3]]]),
    }
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    # with eval_vcs_range
    bev_det_w_eval_vcs = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        depth_intervals=(1, 3),
        enable_ignore=True,
        eval_vcs_range=(-4, -4, 4, 4),  # (bottom, right, top, left)
        id2label=id2label,
    )
    bev_det_w_eval_vcs.update(gt, det)
    bev_det_w_eval_vcs.update(gt_outside_eval_vcs_range, det)
    values_w_eval_vcs = bev_det_w_eval_vcs.get()
    assert round(values_w_eval_vcs[1][1][0], 2) == 1.00
    assert round(values_w_eval_vcs[1][1][1], 2) == 0.50

    # without eval_vcs_range
    bev_det_wo_eval_vcs = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        enable_ignore=True,
        id2label=id2label,
    )
    bev_det_wo_eval_vcs.update(gt, det)
    bev_det_wo_eval_vcs.update(gt_outside_eval_vcs_range, det)
    values_wo_eval_vcs = bev_det_wo_eval_vcs.get()
    assert round(values_wo_eval_vcs[1][1][0], 2) == 0.50
    assert round(values_wo_eval_vcs[1][1][1], 2) == 0.50

    # with ego_ignore_range
    bev_det_w_ego_ignore = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        enable_ignore=True,
        ego_ignore_range=(-0.6, -0.5, 2.0, 0.5),  # (bottom, right, top, left)
        id2label=id2label,
    )
    bev_det_w_ego_ignore.update(gt, det)
    bev_det_w_ego_ignore.update(gt_inside_ego_range, det)
    values_w_ego_ignore = bev_det_w_ego_ignore.get()
    assert round(values_w_ego_ignore[1][1][0], 2) == 1.00
    assert round(values_w_ego_ignore[1][1][1], 2) == 0.50


def test_bev3d_update_get():
    torch.manual_seed(0)
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    bev_det_eval = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        id2label=id2label,
    )
    gt, det = get_fake_gt_and_pred()
    bev_det_eval.update(gt, det)
    names, values = bev_det_eval.get()
    assert values[0][0] == "recall"
    assert 0 <= values[1][0] <= 1.0


def test_bev3d_visibility():
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    bev_det_eval_w_ignore = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        visibility_intervals=(0, 0.5, 1),
        id2label=id2label,
    )
    bev_det_eval_wo_ignore = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        visibility_intervals=(0, 0.5, 1),
        enable_ignore=False,
        id2label=id2label,
    )
    gt, det = get_certain_gt_and_pred()
    bev_det_eval_w_ignore.update(gt, det)
    bev_det_eval_wo_ignore.update(gt, det)
    _, values = bev_det_eval_w_ignore.get()
    assert values[0][0] == "recall"
    assert 0 <= values[1][0] <= 1.0
    assert (
        getattr(bev_det_eval_w_ignore, "category_1_(0,0.5)_bev_iou_seg_tp")[0]
        == 1.0
    )
    assert (
        getattr(bev_det_eval_w_ignore, "category_1_(0,1)_bev_iou_seg_tp")[0]
        == 2.0
    )
    assert (
        getattr(bev_det_eval_wo_ignore, "category_1_(0,1)_bev_iou_seg_tp")[0]
        == 3.0
    )


def test_bev3d_gt_ignore():
    # ignore_gt case1
    gt_w_ignore1 = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[1, 1, 0], [3, 3, 0], [5, 5, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
            "vcs_ignore_": torch.tensor([[0, 0, 1]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }
    # ignore_gt case2
    gt_w_ignore2 = {
        "timestamp": torch.tensor(
            [[162490000010], [162490000011]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[1, 1, 0], [3, 3, 0], [5, 5, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
            "vcs_ignore_": torch.tensor([[0, 0, 1]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }
    gt_wo_ignore = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[1, 1, 0], [3, 3, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1]]]),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }
    # valid det
    det = {
        "bev3d_dim": torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
        "bev3d_rot": torch.Tensor([[0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 1]]),
        "bev3d_score": torch.Tensor([[1, 1, 1]]),
        "bev3d_ct": torch.Tensor([[[1, 1], [3, 3], [5, 5]]]),
    }
    # empty det
    empty_det = {
        "bev3d_dim": torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
        "bev3d_rot": torch.Tensor([[0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 1]]),
        "bev3d_score": torch.Tensor([[0.001, 0.001, 0.001]]),
        "bev3d_ct": torch.Tensor([[[1, 1], [3, 3], [5, 5]]]),
    }
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }

    bev_det_eval = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        enable_ignore=True,
        id2label=id2label,
    )
    bev_det_eval.update(gt_w_ignore1, det)
    bev_det_eval.update(gt_w_ignore2, empty_det)
    _, values_w_ignore = bev_det_eval.get()
    assert round(values_w_ignore[1][0], 2) == 0.50
    assert round(values_w_ignore[1][1], 2) == 1.00

    bev_det_eval = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        enable_ignore=False,
        id2label=id2label,
    )
    bev_det_eval.update(gt_wo_ignore, det)
    _, values_wo_ignore = bev_det_eval.get()
    assert round(values_wo_ignore[1][0], 2) == 1.00
    assert round(values_wo_ignore[1][1], 2) == 0.67


def get_rotate_boxes_sample(case):
    """Return different rotate box sample for test.

    Args:
        case (str): the different case type, e.g.
            rotate: the normal rotate boxes.
            same: boxes are exactly the same.
            collinear: edges on two boxes are collinear.
            scale: input boxes in different scale level

    """
    if case == "rotate":
        """case:1"""
        rbbox1 = [[0, 0, 4, 1.5, -np.pi / 4]]
        rbbox2 = [[0, 0, 4, 2, 0]]
        expected_iou = 0.4308
        """case:2"""
        # rbbox1 = [[1.54, -3.57, 4, 1.25, -np.pi / 3]]
        # rbbox2 = [[2.33, 1.65, 3.6, 2.01, 0.736]]
        # expected_iou = 0.
        """case:3"""
        # rbbox1 = [[1, 1.2, 4, 1.5, -np.pi / 4]]
        # rbbox2 = [[1, 1.2, 4, 1.5, 0]]
        # expected_iou = 0.3608
        """case:4"""
        # rbbox1 = [[34.8982048, -4.01927567,  4.15128033,
        # 1.70010416, 0.0371456]]
        # rbbox2 = [[34.84272, -3.9025896,  4.104194,
        # 1.6739343,  -0.06501262]]
        # expected_iou = 0.8301
    elif case == "same":
        rbbox1 = [[0, 0, 4, 2, np.pi / 5]]
        rbbox2 = [[0, 0, 4, 2, np.pi / 5]]
        expected_iou = 1.0
    elif case == "collinear":
        rbbox1 = [[0.0, 0.0, 2.0, 2.0, 0.0]]
        rbbox2 = [[0.0, 1.0, 2.0, 2.0, 0.0]]
        expected_iou = 0.333
    elif case == "scale":
        rbbox1 = [[38, 120, 1.3, 20, 50]]
        rbbox2 = [[38, 120, 1.3, 20, 50]]
        expected_iou = 1.0
    else:
        raise NotImplementedError
    return (rbbox1, rbbox2, expected_iou)


def test_rotate_iou_on_vcs():
    """Test rotate iou on vcs coordinate.

    rbbox format: [x, y, l, w, yaw]:
        the vcs coordinate: x(front), y(left), z(up).
        l, w (meter): the box length and width.
        yaw (rad): the yaw angle around z-axis,
        NOTE: the yaw should be clockwise w.r.t. x-axis.
    """
    # Prepare 4 corner cases for test
    test_cases = ["rotate", "same", "collinear", "scale"]

    for case in test_cases:
        rbbox1, rbbox2, iou_exp = get_rotate_boxes_sample(case)
        rbbox1 = np.array(rbbox1, dtype=np.float32)
        rbbox2 = np.array(rbbox2, dtype=np.float32)
        iou = rotate_iou(rbbox1, rbbox2, criterion=-1)
        assert abs(iou[0][0] - iou_exp) <= 1e-2


LET_NMS_PARAMS = dict(
    vehicle=[
        dict(
            p_t=0.1,
            min_t=2.0,
            max_t=7.0,
            radius=3.0,
            e_loc_threshold=0.3,
        ),
    ],
    cyclist=[
        dict(
            p_t=0.2,
            min_t=4.0,
            max_t=6.0,
            radius=1.5,
            e_loc_threshold=0.2,
        ),
    ],
    pedestrian=[
        dict(
            p_t=0.2,
            min_t=4.0,
            max_t=6.0,
            radius=0.5,
            e_loc_threshold=0.2,
        ),
    ],
    vrumerge=[
        dict(
            p_t=0.2,
            min_t=4.0,
            max_t=6.0,
            radius=1.5,
            e_loc_threshold=0.2,
        ),
        dict(
            p_t=0.2,
            min_t=4.0,
            max_t=6.0,
            radius=0.5,
            e_loc_threshold=0.2,
        ),
    ],
)


def gen_fake_bbox(task):
    pred_bboxes = torch.tensor(
        [
            [10, 10, 3, 3, 20],
            [15, 15, 3, 3, 20],
            [11, 11, 3, 3, 20],
            [3, -3, 3, 3, 20],
            [4, -4, 3, 3, 20],
            [6.5, -6.4, 3, 3, 20],
        ]
    )
    scores = torch.tensor([0.51, 0.52, 0.69, 0.8, 0.82, 0.93])
    if task == "vehicle":
        left_exp = np.array([0, 1, 1, 0, 1, 1], dtype=np.bool_)
    elif task == "cyclist":
        left_exp = np.array([0, 1, 1, 1, 0, 1], dtype=np.bool_)
    elif task == "pedestrian":
        left_exp = np.array([0, 1, 1, 1, 0, 1], dtype=np.bool_)
    return pred_bboxes, scores, left_exp


def test_bev3d_nms():
    pred_bboxes = torch.tensor(
        [
            [10, 10, 3, 3, 20],
            [15, 15, 3, 3, 20],
            [11, 11, 3, 3, 20],
            [3, -3, 3, 3, 20],
            [4, -4, 3, 3, 20],
            [6.5, -6.4, 3, 3, 20],
        ]
    )
    scores = torch.tensor([0.51, 0.52, 0.69, 0.8, 0.82, 0.93])
    thresh = 0.2
    left_exp = np.array([0, 1, 1, 0, 1, 1], dtype=np.bool_)
    left = bev3d_nms(
        boxes=pred_bboxes,
        scores=scores,
        thresh=thresh,
    )
    assert (left == left_exp).all()

    thresh = 0.4
    left_exp = np.array([1, 1, 1, 1, 1, 1], dtype=np.bool_)
    left = bev3d_nms(
        boxes=pred_bboxes,
        scores=scores,
        thresh=thresh,
    )
    assert (left == left_exp).all()


def test_bev3d_let_nms():
    tasks = ["vehicle", "cyclist", "pedestrian"]
    for task in tasks:
        pred, scores, left_exp = gen_fake_bbox(task)
        cls_ids = torch.zeros_like(scores, dtype=torch.int64)
        let_nms_param = LET_NMS_PARAMS.get(task)
        left = bev3d_let_nms(
            boxes=pred,
            scores=scores,
            let_nms_param=let_nms_param,
            cls_ids=cls_ids,
        )
        assert (left == left_exp).all()

    pred_bboxes = torch.tensor(
        [
            [9, 9, 3, 3, 20],
            [10, 10, 3, 3, 20],
            [15, 15, 3, 3, 20],
            [11, 11, 3, 3, 20],
            [3, -3, 3, 3, 20],
            [4, -4, 3, 3, 20],
            [6.5, -6.4, 3, 3, 20],
        ]
    )
    scores = torch.tensor([0.5, 0.51, 0.52, 0.69, 0.8, 0.82, 0.93])
    cls_ids = torch.tensor([1, 0, 1, 0, 0, 1, 1])
    left_exp = np.array([1, 0, 1, 1, 1, 0, 1], dtype=np.bool_)
    let_nms_param = LET_NMS_PARAMS.get("vrumerge")

    # not agnostic nms
    left = bev3d_let_nms(
        boxes=pred_bboxes,
        scores=scores,
        let_nms_param=let_nms_param,
        cls_ids=cls_ids,
        agnostic=False,
    )
    assert (left == left_exp).all()

    # agnostic nms
    left_exp = np.array([0, 0, 1, 1, 1, 0, 1], dtype=np.bool_)
    left = bev3d_let_nms(
        boxes=pred_bboxes,
        scores=scores,
        let_nms_param=let_nms_param,
        cls_ids=cls_ids,
        agnostic=True,
    )
    assert (left == left_exp).all()


def test_bev3d_multiclass_nms():

    labels_results = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    scores_nms_results = torch.tensor(
        [
            0.9300,
            0.8200,
            0.6900,
            0.5200,
            0.9300,
            0.8200,
            0.6900,
            0.5200,
            0.9300,
            0.8200,
            0.6900,
            0.5200,
        ]
    )
    bboxes_nms_results = torch.tensor(
        [
            [6.5000, -6.4000, 3.0000, 3.0000, 20.0000],
            [4.0000, -4.0000, 3.0000, 3.0000, 20.0000],
            [11.0000, 11.0000, 3.0000, 3.0000, 20.0000],
            [15.0000, 15.0000, 3.0000, 3.0000, 20.0000],
        ]
        * 3
    )

    tasks = ["vehicle", "cyclist", "pedestrian"]
    bboxes = []
    total_labels = []
    total_scores = []
    for ti, task in enumerate(tasks):
        pred, scores, _ = gen_fake_bbox(task)
        bboxes.append(pred)
        total_labels.extend([ti] * pred.shape[0])
        total_scores.append(scores)

    total_bboxes = torch.cat(bboxes, dim=0)
    total_scores = torch.cat(total_scores, dim=0)
    total_labels = torch.tensor(total_labels)
    scores_for_nms = torch.nn.functional.one_hot(total_labels, len(tasks)).to(
        dtype=total_scores.dtype
    )
    scores_for_nms[
        torch.arange(total_labels.shape[0]), total_labels
    ] = total_scores

    bboxes_nms, scores_nms, labels_nms = bev3d_multiclass_nms(
        boxes=total_bboxes,
        boxes_for_nms=total_bboxes.clone(),
        scores=scores_for_nms,
        score_thresh=0.1,
        nms_thresh=0.2,
    )

    assert (bboxes_nms == bboxes_nms_results).all()
    assert (scores_nms == scores_nms_results).all()
    assert (labels_nms == labels_results).all()


def test_evaluation_include_all():
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[1, 1, 0], [3, 3, 0], [7, 7, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1, 2]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1], [2, 2, 2]]]),
            "vcs_ignore_": torch.tensor([[0, 0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }
    det = {
        "bev3d_dim": torch.Tensor([[[1, 1, 1], [7, 7, 7], [2, 2, 2]]]),
        "bev3d_rot": torch.Tensor([[0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 2]]),
        "bev3d_score": torch.Tensor([[1, 1, 1]]),
        "bev3d_ct": torch.Tensor([[[1, 1], [3, 3], [7, 7]]]),
    }
    bev_det_eval = BEVDetEval(
        eval_category_ids=(1, 2, "all"),
        score_threshold=0.5,
        iou_threshold=0.8,
        gt_max_depth=100,
        depth_intervals=(0, 50),
        enable_ignore=True,
        id2label=id2label,
    )
    bev_det_eval.update(gt, det)
    _, eval_res = bev_det_eval.get()
    assert round(eval_res[1][0], 2) == 0.67
    assert round(eval_res[1][1], 2) == 0.67


def test_occlusion_attribute():
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[1, 1, 0], [3, 3, 0], [7, 7, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1], [2, 2, 2]]]),
            "vcs_ignore_": torch.tensor([[0, 0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
            "vcs_occlusion_": torch.Tensor([[0, 2, 1]]),
        },
    }
    det = {
        "bev3d_dim": torch.Tensor([[[1, 1, 1], [1, 1, 1], [2, 2, 2]]]),
        "bev3d_rot": torch.Tensor([[0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 1]]),
        "bev3d_score": torch.Tensor([[1, 1, 1]]),
        "bev3d_ct": torch.Tensor([[[1, 1], [3, 3], [7, 7]]]),
        "bev3d_occlusion_id": torch.Tensor([[0, 2, 2]]),
    }
    bev_det_eval = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.5,
        iou_threshold=0.8,
        gt_max_depth=100,
        metrics=("dx", "dxp", "dy", "dyp", "dxyp", "drot", "occlusion"),
        depth_intervals=(0, 50),
        enable_ignore=True,
        eval_occlusion=True,
        occlusion_ignore_id=-99.0,
    )
    bev_det_eval.update(gt, det)
    _, _, metirc_dict_all = bev_det_eval.compute()
    assert (
        round(
            metirc_dict_all[1]["depthwise_metric"]["occlusion"]["(0,50)"],
            2,
        )
        == 0.67
    )


def test_bev3d_metrics_let_iou():
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor([[[1, 1, 0], [-3, 3, 0], [7, -7, 0]]]),
            "vcs_cls_": torch.Tensor([[1, 1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0]]),
            "vcs_dim_": torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
            "vcs_ignore_": torch.tensor([[0, 0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5]]),
        },
    }
    det = {
        "bev3d_dim": torch.Tensor(
            [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]]
        ),
        "bev3d_rot": torch.Tensor([[0, 0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 1, 1]]),
        "bev3d_score": torch.Tensor([[1, 1, 1, 1]]),
        "bev3d_ct": torch.Tensor([[[2, 2], [-3, 3], [8, -8], [2.1, 2.1]]]),
    }
    bev_det_eval = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.5,
        iou_threshold=0.8,
        gt_max_depth=100,
        metrics=("dx", "dxp", "dy", "dyp", "dxyp", "drot"),
        depth_intervals=(0, 50),
        enable_ignore=True,
        eval_mode="let_iou",
        let_iou_param={"p_t": 0.35, "min_t": 4.0, "max_t": 8.0},
    )
    bev_det_eval.update(gt, det)
    _, pr, _ = bev_det_eval.compute()
    assert round(pr[0], 2) == 1.0
    assert round(pr[1], 2) == 0.75

    bev_det_eval = BEVDetEval(
        eval_category_ids=(1,),
        score_threshold=0.5,
        iou_threshold=0.8,
        gt_max_depth=100,
        metrics=("dx", "dxp", "dy", "dyp", "dxyp", "drot"),
        depth_intervals=(0, 50),
        enable_ignore=True,
        eval_mode="bev_iou",
    )
    bev_det_eval.update(gt, det)
    _, pr, _ = bev_det_eval.compute()
    assert round(pr[0], 2) == 0.33
    assert round(pr[1], 2) == 0.25


def test_let_iou_on_vcs():
    gt_bbox = np.array(
        [
            [1, 1, 1, 1, 0],
            [2, 2, 1, 1, 0],
            [3, 3, 1, 1, 0],
            [10, 10, 1, 1, 0],
            [-1, 1, 1, 1, 0],
        ],
        dtype="float32",
    )
    pred_bbox = np.array([[1, 1, 1, 1, 0]], dtype="float32")
    excepted_let_iou = np.array([[1, 1, 1, 0, 0]], dtype="float32")
    except_al = np.array(
        [[1, 0.64644665, 0.2928931, 0, 0.64644665]], dtype="float32"
    )
    let_iou, al = let_iou_2d(
        pred_bbox, gt_bbox, p_t=0.35, max_t=8.0, min_t=4.0
    )
    assert float((excepted_let_iou - let_iou).max() < 1e-2)
    assert float((except_al - al).max() < 1e-2)


def test_bev3d_bbox_tag_eval():
    torch.manual_seed(0)
    eval_category_ids = (1,)
    gt, det = get_fake_gt_and_pred()
    gt_group_by_cid, det_group_by_cid, timestamps = collect_data(
        gt,
        det,
        eval_category_ids,
        eval_category_ids,
        False,
        False,
        "annos_bev_3d",
    )
    base_taggers = {
        "YAW --": {
            "gt_key": "vcs_rot_z_",
            "pred_key": "bev3d_rot",
            "mapper": {
                "YAW --": [[90 - 22.5, 90 + 22.5], [270 - 22.5, 270 + 22.5]]
            },
            "tagger_fn": "tag_by_range",
            "transforms": lambda rad: np.rad2deg(rad) % 360.0,
        },
    }
    for cid in eval_category_ids:
        gt = {
            "timestamps": timestamps,
            "annotations": gt_group_by_cid[cid],
        }
        det = det_group_by_cid[cid]
        eval_input = [
            det,
            gt,
            (20, 50, 70),
            0.1,
            0.1,
            100,
            None,
            True,
            ["(0,1)"],
            None,
            False,
            -99,
        ]
        matched_res = bev3d_bbox_tag_eval(
            *eval_input,
            "bev_iou",
            base_taggers=base_taggers,
            taggers={"YAW --": {"base_tags": ["YAW --"]}}
        )
        assert "detail" in matched_res


def test_bev3d_tag_eval_vcs():
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "pack_dir": ["1", "w"],
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor(
                [[[1, 1, 0], [3, 3, 0]], [[1, 1, 0], [3, 3, 0]]]
            ),
            "vcs_cls_": torch.Tensor([[1, 1], [1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0], [0, 0]]),
            "vcs_dim_": torch.Tensor(
                [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
            ),
            "vcs_ignore_": torch.tensor([[0, 0], [0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7], [0.3, 0.7]]),
        },
    }
    gt_inside_ego_range = {
        "timestamp": torch.tensor(
            [[162480000000], [162480000001]], dtype=torch.float64
        ),  # noqa
        "pack_dir": ["1", "w"],  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor(
                [
                    [[0.1, 0.1, 0], [0.1, 0.1, 0]],
                    [[0.1, 0.1, 0], [0.1, 0.1, 0]],
                ]
            ),
            "vcs_cls_": torch.Tensor([[1, 1], [1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0], [0, 0]]),
            "vcs_dim_": torch.Tensor(
                [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
            ),
            "vcs_visible_": torch.Tensor([[0.3, 0.7], [0.3, 0.7]]),
        },
    }
    gt_outside_eval_vcs_range = {
        "timestamp": torch.tensor(
            [[162490000100], [162490000101]], dtype=torch.float64
        ),  # noqa
        "pack_dir": ["1", "w"],  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor(
                [[[5, 5, 0], [6, 6, 0]], [[5, 5, 0], [6, 6, 0]]]
            ),
            "vcs_cls_": torch.Tensor([[1, 1], [1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0], [0, 0]]),
            "vcs_dim_": torch.Tensor(
                [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
            ),
            "vcs_ignore_": torch.tensor([[0, 0], [0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5], [0.3, 0.7, 0.5]]),
        },
    }
    # valid det
    det = {
        "bev3d_dim": torch.Tensor(
            [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
        ),
        "bev3d_rot": torch.Tensor([[0, 0], [0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0], [0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1], [1, 1]]),
        "bev3d_score": torch.Tensor([[1, 1], [1, 1]]),
        "bev3d_ct": torch.Tensor([[[1, 1], [3, 3]], [[1, 1], [3, 3]]]),
    }
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    # with eval_vcs_range
    taggers = {
        "YAW --": {
            "gt_key": "vcs_rot_z_",
            "pred_key": "bev3d_rot",
            "mapper": {
                "YAW --": [[90 - 22.5, 90 + 22.5], [270 - 22.5, 270 + 22.5]]
            },
            "tagger_fn": "tag_by_range",
            "transforms": lambda rad: np.rad2deg(rad) % 360.0,
        },
    }
    bev_det_w_eval_vcs = BEVDetTagEval(
        metric_save_dir="./bev3d_tag_eval_w_range",
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        depth_intervals=(1, 3),
        enable_ignore=True,
        eval_vcs_range=(-4, -4, 4, 4),  # (bottom, right, top, left)
        id2label=id2label,
        base_taggers=taggers,
    )
    bev_det_w_eval_vcs.reset()
    bev_det_w_eval_vcs.update(gt, det)
    bev_det_w_eval_vcs.update(gt_outside_eval_vcs_range, det)
    values_w_eval_vcs = bev_det_w_eval_vcs.get()
    assert round(values_w_eval_vcs[1][1][0], 2) == 1.00
    assert round(values_w_eval_vcs[1][1][1], 2) == 0.50
    shutil.rmtree("./bev3d_tag_eval_w_range")

    # without eval_vcs_range
    bev_det_wo_eval_vcs = BEVDetTagEval(
        metric_save_dir="./bev3d_tag_eval_wo_range",
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        enable_ignore=True,
        id2label=id2label,
        base_taggers=taggers,
    )
    bev_det_wo_eval_vcs.reset()
    bev_det_wo_eval_vcs.update(gt, det)
    bev_det_wo_eval_vcs.update(gt_outside_eval_vcs_range, det)
    values_wo_eval_vcs = bev_det_wo_eval_vcs.get()
    assert round(values_wo_eval_vcs[1][1][0], 2) == 0.50
    assert round(values_wo_eval_vcs[1][1][1], 2) == 0.50
    shutil.rmtree("./bev3d_tag_eval_wo_range")

    # with ego_ignore_range
    bev_det_w_ego_ignore = BEVDetTagEval(
        metric_save_dir="./bev3d_tag_eval_ego_range",
        eval_category_ids=(1,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        enable_ignore=True,
        ego_ignore_range=(-0.6, -0.5, 2.0, 0.5),  # (bottom, right, top, left)
        id2label=id2label,
        base_taggers=taggers,
    )
    bev_det_w_ego_ignore.reset()
    bev_det_w_ego_ignore.update(gt, det)
    bev_det_w_ego_ignore.update(gt_inside_ego_range, det)
    values_w_ego_ignore = bev_det_w_ego_ignore.get()
    assert round(values_w_ego_ignore[1][1][0], 2) == 1.00
    assert round(values_w_ego_ignore[1][1][1], 2) == 0.50
    shutil.rmtree("./bev3d_tag_eval_ego_range")


def test_tag_eval_occlusion_attribute():
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "pack_dir": ["1", "w"],
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor(
                [
                    [[1, 1, 0], [3, 3, 0], [7, 7, 0]],
                    [[1, 1, 0], [3, 3, 0], [7, 7, 0]],
                ]
            ),
            "vcs_cls_": torch.Tensor([[1, 1, 1], [1, 1, 1]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0], [0, 0, 0]]),
            "vcs_dim_": torch.Tensor(
                [
                    [[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                    [[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                ]
            ),
            "vcs_ignore_": torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5], [0.3, 0.7, 0.5]]),
            "vcs_occlusion_": torch.Tensor([[0, 2, 1], [0, 2, 1]]),
        },
    }
    det = {
        "bev3d_dim": torch.Tensor(
            [
                [[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                [[1, 1, 1], [1, 1, 1], [2, 2, 2]],
            ]
        ),
        "bev3d_rot": torch.Tensor([[0, 0, 0], [0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0], [0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 1], [1, 1, 1]]),
        "bev3d_score": torch.Tensor([[1, 1, 1], [1, 1, 1]]),
        "bev3d_ct": torch.Tensor(
            [[[1, 1], [3, 3], [7, 7]], [[1, 1], [3, 3], [7, 7]]]
        ),
        "bev3d_occlusion_id": torch.Tensor([[0, 2, 2], [0, 2, 2]]),
    }
    taggers = {
        "YAW --": {
            "gt_key": "vcs_rot_z_",
            "pred_key": "bev3d_rot",
            "mapper": {
                "YAW --": [[90 - 22.5, 90 + 22.5], [270 - 22.5, 270 + 22.5]]
            },
            "tagger_fn": "tag_by_range",
            "transforms": lambda rad: np.rad2deg(rad) % 360.0,
        },
    }
    bev_det_eval = BEVDetTagEval(
        metric_save_dir="./bev3d_tag_eval_occl",
        eval_category_ids=(1,),
        score_threshold=0.5,
        iou_threshold=0.8,
        gt_max_depth=100,
        metrics=("dx", "dxp", "dy", "dyp", "dxyp", "drot", "occlusion"),
        depth_intervals=(0, 50),
        enable_ignore=True,
        eval_occlusion=True,
        occlusion_ignore_id=-99.0,
        base_taggers=taggers,
    )
    bev_det_eval.reset()
    bev_det_eval.update(gt, det)
    _, _, metirc_dict_all = bev_det_eval.compute()
    assert (
        round(
            metirc_dict_all[1]["tagwise_metric"]["occl"]["(0,50)"],
            2,
        )
        == 0.67
    )
    assert "YAW --" in metirc_dict_all[1]["tagwise_metric"]["occl"]
    shutil.rmtree("./bev3d_tag_eval_occl")


def test_tag_eval_evaluation_include_all():
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    gt = {
        "timestamp": torch.tensor(
            [[162490000000], [162490000001]], dtype=torch.float64
        ),  # noqa
        "pack_dir": ["1", "w"],  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor(
                [
                    [[1, 1, 0], [3, 3, 0], [7, 7, 0]],
                    [[1, 1, 0], [3, 3, 0], [7, 7, 0]],
                ]
            ),
            "vcs_cls_": torch.Tensor([[1, 1, 2], [1, 1, 2]]),
            "vcs_rot_z_": torch.Tensor([[0, 0, 0], [0, 0, 0]]),
            "vcs_dim_": torch.Tensor(
                [
                    [[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                    [[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                ]
            ),
            "vcs_ignore_": torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=bool),
            "vcs_visible_": torch.Tensor([[0.3, 0.7, 0.5], [0.3, 0.7, 0.5]]),
        },
    }
    det = {
        "bev3d_dim": torch.Tensor(
            [
                [[1, 1, 1], [7, 7, 7], [2, 2, 2]],
                [[1, 1, 1], [7, 7, 7], [2, 2, 2]],
            ]
        ),
        "bev3d_rot": torch.Tensor([[0, 0, 0], [0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0], [0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 2], [1, 1, 2]]),
        "bev3d_score": torch.Tensor([[1, 1, 1], [1, 1, 1]]),
        "bev3d_ct": torch.Tensor(
            [[[1, 1], [3, 3], [7, 7]], [[1, 1], [3, 3], [7, 7]]]
        ),
    }
    taggers = {
        "YAW --": {
            "gt_key": "vcs_rot_z_",
            "pred_key": "bev3d_rot",
            "mapper": {
                "YAW --": [[90 - 22.5, 90 + 22.5], [270 - 22.5, 270 + 22.5]]
            },
            "tagger_fn": "tag_by_range",
            "transforms": lambda rad: np.rad2deg(rad) % 360.0,
        },
    }
    bev_det_eval = BEVDetTagEval(
        metric_save_dir="./bev3d_tag_eval_all",
        eval_category_ids=(1, 2, "all"),
        score_threshold=0.5,
        iou_threshold=0.8,
        gt_max_depth=100,
        depth_intervals=(0, 50),
        enable_ignore=True,
        id2label=id2label,
        base_taggers=taggers,
    )
    bev_det_eval.reset()
    bev_det_eval.update(gt, det)
    _, eval_res = bev_det_eval.get()
    assert round(eval_res[1][0], 2) == 0.67
    assert round(eval_res[1][1], 2) == 0.67
    assert os.path.exists(
        "./bev3d_tag_eval_all/results/matched_dict_bev_iou_interval0-1_cidall_rankall.npy"  # noqa
    )
    assert os.path.exists(
        "./bev3d_tag_eval_all/tables/BEV3D-classall[]-bev_iou.html"  # noqa
    )
    shutil.rmtree("./bev3d_tag_eval_all")


def test_tag_eval_evaluation_temporal_all():
    id2label = {
        0: "label0",
        1: "label1",
        2: "label2",
        3: "label3",
    }
    gt = {
        "timestamp": torch.tensor(
            [[162490000000, 162490000001]], dtype=torch.float64
        ),  # noqa
        "pack_dir": ["1"],  # noqa
        "annos_bev_3d": {
            # [batch, num_objs, channel]
            "vcs_loc_": torch.Tensor(
                [
                    [
                        [[1, 1, 0], [3, 3, 0], [7, 7, 0]],
                        [[1, 1, 0], [3, 3, 0], [7, 7, 0]],
                    ]
                ]
            ),
            "vcs_cls_": torch.Tensor([[[1, 1, 2], [1, 1, 2]]]),
            "vcs_rot_z_": torch.Tensor([[[0, 0, 0], [0, 0, 0]]]),
            "vcs_dim_": torch.Tensor(
                [
                    [
                        [[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                        [[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                    ],
                ]
            ),
            "vcs_ignore_": torch.tensor([[[0, 0, 0], [0, 0, 0]]], dtype=bool),
            "vcs_visible_": torch.Tensor([[[0.3, 0.7, 0.5], [0.3, 0.7, 0.5]]]),
        },
    }
    det = {
        "bev3d_dim": torch.Tensor(
            [
                [[1, 1, 1], [7, 7, 7], [2, 2, 2]],
                [[1, 1, 1], [7, 7, 7], [2, 2, 2]],
            ]
        ),
        "bev3d_rot": torch.Tensor([[0, 0, 0], [0, 0, 0]]),
        "bev3d_loc_z": torch.Tensor([[0, 0, 0], [0, 0, 0]]),
        "bev3d_cls_id": torch.Tensor([[1, 1, 2], [1, 1, 2]]),
        "bev3d_score": torch.Tensor([[1, 1, 1], [1, 1, 1]]),
        "bev3d_ct": torch.Tensor(
            [[[1, 1], [3, 3], [7, 7]], [[1, 1], [3, 3], [7, 7]]]
        ),
    }
    taggers = {
        "YAW --": {
            "gt_key": "vcs_rot_z_",
            "pred_key": "bev3d_rot",
            "mapper": {
                "YAW --": [[90 - 22.5, 90 + 22.5], [270 - 22.5, 270 + 22.5]]
            },
            "tagger_fn": "tag_by_range",
            "transforms": lambda rad: np.rad2deg(rad) % 360.0,
        },
    }
    bev_det_eval = BEVDetTagEval(
        metric_save_dir="./bev3d_tag_eval_temporal_all",
        eval_category_ids=(1, 2, "all"),
        score_threshold=0.5,
        iou_threshold=0.8,
        gt_max_depth=100,
        depth_intervals=(0, 50),
        enable_ignore=True,
        id2label=id2label,
        base_taggers=taggers,
    )
    bev_det_eval.reset()
    bev_det_eval.update(gt, det)
    _, eval_res = bev_det_eval.get()
    assert round(eval_res[1][0], 2) == 0.67
    assert round(eval_res[1][1], 2) == 0.67
    assert os.path.exists(
        "./bev3d_tag_eval_temporal_all/results/matched_dict_bev_iou_interval0-1_cidall_rankall.npy"  # noqa
    )
    assert os.path.exists(
        "./bev3d_tag_eval_temporal_all/tables/BEV3D-classall[]-bev_iou.html"  # noqa
    )
    shutil.rmtree("./bev3d_tag_eval_temporal_all")
