import json
import os

import pytest
import yaml

from hat.evaluation import pose
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH
from . import HORIZON_DRIVING_DATASET_AVAILABLE


@pytest.mark.skipif(
    True,
    reason="aidi platform not support yet",
)
@pytest.mark.skipif(
    not HORIZON_DRIVING_DATASET_AVAILABLE,
    reason="need horizon_driving_dataset",
)
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_pose_evaluation(tmpdir):
    dataset_fpath = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/2p5d_pose/dataset"
    )
    prediction_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/2p5d_pose/2p5d_pose_eval_result.pkl",
    )
    setting_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/2p5d_pose/2p5d_pose.yaml",
    )

    cfg = yaml.load(open(setting_fpath))
    if not os.path.exists(cfg["gt_pose_path"]):
        # update config for ci
        cfg["gt_pose_path"] = cfg["gt_pose_path"].replace(
            "/horizon-bucket/HDLTAlgorithm", HAT_BUCKET_PATH
        )
        setting_fpath = os.path.join(tmpdir, "2p5d_pose.yaml")
        with open(setting_fpath, "w") as fid:
            yaml.dump(cfg, fid)

    result = pose.evaluate(
        dataset_fpath, prediction_fpath, setting_fpath, tmpdir
    )
    assert isinstance(result, dict)
    result_json = os.path.join(tmpdir, "result.json")
    result_in_file = json.load(open(result_json, "r"))
    assert len(result_in_file.keys()) == 1
