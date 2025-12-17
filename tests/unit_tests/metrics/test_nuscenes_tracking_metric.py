import json
import os.path as osp

import pytest

from hat.metrics.tracking.nuscenes_tracking_metric import (
    NUSC,
    NuscenesTrackingMetric,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not (HAT_BUCKET_EXISTS and NUSC), reason="need context")
def testNuscenesTrackingMetric(tmpdir):
    data_root = osp.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/tracking/nuscenes"
    )
    work_dir = tmpdir
    pred_path = osp.join(data_root, "nusc_pred.json")
    gt_path = osp.join(data_root, "nusc_gt.json")
    meta_path = osp.join(data_root, "nusc_meta.json")
    cfg_path = osp.join(data_root, "tracking_nips_2019.json")
    gt = NuscenesTrackingMetric.nusc_submission_to_frames(gt_path, meta_path)
    pred = NuscenesTrackingMetric.nusc_submission_to_frames(
        pred_path, meta_path
    )
    with open(cfg_path) as f:
        config = json.load(f)
    nusc_eval = NuscenesTrackingMetric(
        save_dir=work_dir,
        gt=gt,
        pred=pred,
        eval_classes=config["tracking_names"],
        eval_ranges=config["class_range"],
        dist_fcn=config["dist_fcn"],
        dist_th_tp=config["dist_th_tp"],
        min_recall=config["min_recall"],
        max_boxes_per_sample=config["max_boxes_per_sample"],
        num_thresholds=config["num_thresholds"],
        num_workers=0,
        metric_worst=config["metric_worst"],
        verbose=False,
    )
    result = nusc_eval.main(render_curves=False)
    assert result["amota"] == 1
