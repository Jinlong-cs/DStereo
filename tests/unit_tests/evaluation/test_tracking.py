import json
import os.path as osp
import pickle

import cv2
import pytest
import yaml

from hat.evaluation.tracking import evaluate
from hat.metrics.tracking.nuscenes_tracking_metric import (
    NUSC,
    NuscenesTrackingMetric,
)
from hat.visualize.tracking.draw_samples import (
    draw_sample,
    list_samples_order_by_tp,
)
from tests import (
    AIDI_PUBLIC_DATA_BUCKET_EXISTS,
    AIDI_PUBLIC_DATA_BUCKET_PATH,
    HAT_BUCKET_EXISTS,
    HAT_BUCKET_PATH,
)


@pytest.mark.skipif(
    not (HAT_BUCKET_EXISTS and NUSC and AIDI_PUBLIC_DATA_BUCKET_EXISTS),
    reason="need context",
)
def testNuscenesTrackingMetric(tmpdir):
    # raw minitest data
    data_root = osp.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/tracking/nuscenes"
    )
    img_dir = osp.join(
        AIDI_PUBLIC_DATA_BUCKET_PATH, "nuScenes/origin/samples/"
    )
    work_dir = tmpdir
    pred_path = osp.join(data_root, "nusc_pred.json")
    gt_path = osp.join(data_root, "nusc_gt.json")
    meta_path = osp.join(data_root, "nusc_meta.json")
    cfg_path = osp.join(data_root, "tracking_nips_2019.json")
    # inputs for evaluation
    gt_file = osp.join(work_dir, "mini_gt.pkl")
    pred_file = osp.join(work_dir, "mini_pred.pkl")
    config_file = osp.join(work_dir, "setting.yaml")
    # eval
    all_results = osp.join(work_dir, "result.pkl")
    gt = NuscenesTrackingMetric.nusc_submission_to_frames(gt_path, meta_path)
    pred = NuscenesTrackingMetric.nusc_submission_to_frames(
        pred_path, meta_path
    )
    with open(cfg_path, "r") as f:
        config = json.load(f)
        config["eval_classes"] = config.pop("tracking_names")
        config["eval_ranges"] = config.pop("class_range")
        config.pop("pretty_tracking_names")
        config.pop("tracking_colors")
        config["eval_metric_type"] = "Nuscenes"
        config["eval_vis_cfg"] = {
            "vis_score_threshold": 0.2,
            "vis_image_layout": {
                0: [0, 0],
                1: [0, 1],
                2: [0, 2],
                3: [1, 0],
                4: [1, 1],
                5: [1, 2],
                "bird_eye_view": [2, 1],
            },
            "vis_camera": [0, 1, 2, 3, 4, 5],
            "bird_eye_size": [600, 900],
            "vcs_range": [-50, -50, 50, 50],
        }
        filters = []
        for k1, v1 in [
            ["y", [-3, 3]],
            ["y", [[-5, -3], [3, 5]]],
            ["x", [-80, 200]],
        ]:
            k2 = "x"
            for v2 in (
                [[0, 5], [5, 10]]
                + [[i * 10, (i + 1) * 10] for i in range(1, 10)]
                + [[100, 120], [120, 150], [150, 200], [0, 200]]
                + [[-5, 0], [-10, -5]]
                + [[-(i + 1) * 10, -i * 10] for i in range(1, 10)]
                + [[-120, -100], [-120, 0], [-120, 200]]
            ):
                f = {k1: v1}
                f[k2] = v2
                filters.append(f)
        config["eval_filters"] = filters
        config["num_workers"] = 6
        config = {"eval_type": [config]}
    with open(gt_file, "wb+") as f:
        pickle.dump(gt, f)
    with open(pred_file, "wb+") as f:
        pickle.dump(pred, f)
    with open(config_file, "w+") as f:
        yaml.dump(config, f)

    summary = evaluate(gt_file, pred_file, config_file, work_dir)

    # test draw sample
    def get_multi_img(data):
        img_dict = {}
        img_meta = data["img_meta"]
        for key, meta in img_meta.items():
            img_path = osp.join(img_dir, meta.image.url)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img_dict[key] = image
        return img_dict

    for func in [
        list_samples_order_by_tp,
    ]:
        samples = func(all_results)
        for sample in samples[:2]:
            draw_sample(
                get_multi_img(sample),
                sample,
                save_local_path=work_dir,
            )

    assert abs(summary["AMOTA"] - 1) < 1e-3
    assert abs(summary["AMOTP"] - 0) < 1e-3
