import os
import pickle

import cv2
import numpy as np
import pytest

from hat.evaluation import bev_seg
from tests import HAT_BUCKET_PATH


@pytest.mark.skipif(True, reason="cost time")
def test_bev_seg_evaluation(tmpdir):
    dataset_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/bev_seg_unit_test_data/data_gt/images",
    )
    prediction_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/bev_seg_unit_test_data/data_pred/bev_seg.tar",  # noqa
    )
    setting_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/bev_seg_unit_test_data/bev_seg.yaml",
    )
    result = bev_seg.evaluate(
        dataset_fpath, prediction_fpath, setting_fpath, tmpdir
    )
    assert isinstance(result, dict)


@pytest.mark.skipif(True, reason="not support yet")
def test_bev_seg_visualize(tmpdir):
    image_fpath = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/evaluation_data/bev_seg/dataset"
    )
    sample_fpath = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/evaluation_data/bev_seg/bev_seg_eval_result.pkl",
    )
    eval_data = pickle.load(open(sample_fpath, "rb"))
    sample = eval_data["outputs"]
    image = cv2.imread(
        os.path.join("{}/".format(image_fpath), sample["img_name"][0])
    )
    result = bev_seg.visualize(image, sample)
    cv2.imwrite(
        os.path.join("{}/".format(tmpdir), sample["img_name"][0]),
        result,
    )
    assert isinstance(result, np.ndarray)
