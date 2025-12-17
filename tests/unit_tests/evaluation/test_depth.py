import json
import os
import pickle

import cv2
import numpy as np
import numpy.testing as npt
import pytest

from hat.evaluation import depth
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_depth_evaluation(tmpdir):
    dataset_fpath = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/2p5d_depth/dataset"
    )
    prediction_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/2p5d_depth/depth_pose_resflow.tar",
    )
    setting_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/2p5d_depth/depth.yaml",
    )
    result = depth.evaluate(
        dataset_fpath, prediction_fpath, setting_fpath, tmpdir
    )
    assert isinstance(result, dict)
    result_json = os.path.join(tmpdir, "result.json")
    result_in_file = json.load(open(result_json, "r"))
    npt.assert_almost_equal(
        result_in_file["absrel"], [[0.2111], [0.0179]], decimal=4
    )


@pytest.mark.skipif(True, reason="not support yet")
def test_depth_visualize(tmpdir):
    image_fpath = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/2p5d_depth/dataset"
    )
    sample_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/2p5d_depth/2p5d_depth_eval_result.pkl",
    )
    eval_data = pickle.load(open(sample_fpath, "rb"))
    sample = eval_data["outputs"]
    image = cv2.imread(
        os.path.join("{}/".format(image_fpath), sample["img_name"][0])
    )
    result = depth.visualize(image, sample)
    cv2.imwrite(
        os.path.join("{}/".format(tmpdir), sample["img_name"][0]),
        result,
    )
    assert isinstance(result, np.ndarray)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_load_depth_png():
    depth_png = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/2p5d_depth/dataset/1617260548714.png",
    )
    result = depth.load_depth_png(depth_png, 2 ** 8)
    assert isinstance(result, np.ndarray)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_eval_depth_preprocess():
    pre = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/2p5d_depth/dataset/1617260548714.png",
    )
    gt = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/2p5d_depth/dataset/1617260548714.png",
    )
    pre = depth.load_depth_png(pre, 2 ** 8)
    gt = depth.load_depth_png(gt, 2 ** 8)
    gt, pre = depth.eval_depth_preprocess(pre, 10, gt, (2048, 3840, 0, 0))
    assert isinstance(gt, np.ndarray)
    assert isinstance(pre, np.ndarray)
