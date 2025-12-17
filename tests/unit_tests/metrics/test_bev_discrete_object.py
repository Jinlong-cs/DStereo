import pytest
import torch

from hat.metrics.bev.bev_discobj_eval import ANCBEVDiscreteObjectEval

try:
    import aidisdk
except ImportError:
    aidisdk = None


def get_fake_discobj():
    """Generate fake gt and pred for unitest.

    Returns:
        gt (Dict): A dict contains GT bbox info.
        det (List): A list contains predcit bbox info, e.g., bbox
            center, size, score, yaw, category.
    """
    timestamp = torch.tensor(
        [[162490000000], [162490000001]], dtype=torch.float64
    ).cuda()
    annos_bev_discobj = {
        # [batch, num_objs, channel]
        "vcs_discobj_loc": torch.randn((2, 300, 2)).cuda(),
        "vcs_discobj_cls": torch.randint(0, 2, (2, 300)).cuda(),
        "vcs_discobj_yaw": torch.rand((2, 300)).cuda(),
        "vcs_discobj_wh": torch.rand((2, 300, 2)).cuda(),
        "vcs_discobj_ignore": torch.rand((2, 300)).cuda(),
    }
    gt = dict(timestamp=timestamp, annos_bev_discobj=annos_bev_discobj)

    pred_ct = torch.randn((2, 100, 2)).cuda()
    pred_wh = torch.rand((2, 100, 2)).cuda()
    pred_score = torch.rand((2, 100)).cuda()
    pred_yaw = torch.randn((2, 100)).cuda()
    pred_bev_discobj_cls_id = torch.randint(0, 2, (2, 100)).cuda()
    det = dict(
        pred_ct=pred_ct,
        pred_wh=pred_wh,
        pred_score=pred_score,
        pred_yaw=pred_yaw,
        pred_bev_discobj_cls_id=pred_bev_discobj_cls_id,
    )

    return gt, det


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
def test_bev_discrete_obj_eval_init():
    bev_det_eval = ANCBEVDiscreteObjectEval(
        id2label={0: "Stopline"},
        eval_category_ids=(0,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        depth_intervals={"Stopline": (-20, -10, 0, 10, 20, 30, 40, 50, 70)},
        eval_vcs_range={"Stopline": (-20.0, -8.0, 50.0, 8.0)},
        name="BEVDiscreteObjectEval",
    )
    assert bev_det_eval.metrics == ("dx", "dy", "dxy", "dw", "dh", "drot")
    assert len(bev_det_eval.eval_category_ids) > 0
    assert len(bev_det_eval.depth_intervals) > 0


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
def test_bev_discrete_obj_update_get():
    torch.manual_seed(10)
    bev_det_eval = ANCBEVDiscreteObjectEval(
        id2label={0: "Stopline"},
        eval_category_ids=(0,),
        score_threshold=0.2,
        iou_threshold=0.2,
        gt_max_depth=100,
        yaw_amplitude=180,
        depth_intervals={"Stopline": (-20, -10, 0, 10, 20, 30, 40, 50, 70)},
        eval_vcs_range={"Stopline": (-20.0, -8.0, 50.0, 8.0)},
        compute_foreground_prec=True,
        name="BEVDiscreteObjectEval",
        annos_key="annos_bev_discobj",
    )
    gt, det = get_fake_discobj()
    bev_det_eval.update(gt, det)

    _, values = bev_det_eval.get()
    assert "BEVDiscreteObjectEval: all AP" in values.summary
    assert values.summary["BEVDiscreteObjectEval: all AP"] == 0.121
