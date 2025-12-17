# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
import pickle

import numpy as np

PATH_REC_READER_AVAIABLE = os.path.exists("/horizon-bucket/HDLTAlgorithm/")


PATH_DATA = (
    "./tmp_orig_data/action/gesture/hat_test_data/"
    if not PATH_REC_READER_AVAIABLE
    else "/horizon-bucket/HDLTAlgorithm/data/orig_data/action/gesture/hat_test_data/"  # noqa
)


def assert_func(pred_data, gt_data):
    for key in gt_data.keys():
        np.testing.assert_almost_equal(
            gt_data[key],
            pred_data[key],
            err_msg=f"key:{key}, gt: {gt_data[key]}, pred:{pred_data[key]}",
        )


def get_data(*filenames):
    output = []
    for filename in filenames:
        logging.info(
            f"gesture: load data - {os.path.join(PATH_DATA, filename)}"
        )
        logging.info(f"gesture: data name - {filename}")
        logging.info(f"gesture: PATH_DATA - {PATH_DATA}")
        with open(os.path.join(PATH_DATA, filename), "rb+") as f:
            output.append(pickle.load(f))
    return output
