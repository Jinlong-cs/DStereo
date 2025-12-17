import os
import pickle

import cv2
import numpy as np
import pytest

from hat.evaluation import real3d
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_real3d_evaluation(tmpdir):
    dataset_fpath = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/real3d/dataset"
    )
    prediction_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/real3d/real3d_fov120.tar",
    )
    setting_fpath = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/real3d/real3d.yaml"
    )
    result = real3d.evaluate(
        dataset_fpath, prediction_fpath, setting_fpath, tmpdir
    )
    assert isinstance(result, dict)


@pytest.mark.skipif(True, reason="not support yet")
def test_real3d_visualize(tmpdir):
    image_fpath = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/real3d/dataset"
    )
    sample_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/real3d/real3d_eval_result.pkl",
    )
    eval_data = pickle.load(open(sample_fpath, "rb"))
    batch = eval_data["batch"]
    sample = eval_data["outputs"]
    image = cv2.imread(
        os.path.join("{}/".format(image_fpath), batch["image_name"][0])
    )
    result = real3d.visualize(image, sample)
    cv2.imwrite(
        os.path.join("{}/".format(tmpdir), batch["image_name"][0]),
        result,
    )
    assert isinstance(result, np.ndarray)
