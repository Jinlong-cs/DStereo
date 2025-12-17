# Copyright (c) Horizon Robotics. All rights reserved.

import os
import pickle
import time

import pytest

from hat.models.task_modules.traj_pred.post_process.sgnet_post_process import (  # noqa: E501
    SGNetPostProcessor,
)
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH


@pytest.mark.skipif(not SD_AlGORITHM_BUCKET_EXISTS, reason="require SD bucket")
def test_sgnet_post_processor():
    pickle_dir = "11_perception_prediction/02_user/zhanbo01.li/train_data/HAT_Unit_Test/postprocess/sgnet_out.pkl"  # noqa: E501
    res_save_dir = "11_perception_prediction/02_user/zhanbo01.li/train_data/HAT_Unit_Test/postprocess"  # noqa: E501
    res_save_dir = os.path.join(SD_AlGORITHM_BUCKET_PATH, res_save_dir)
    batch_size = 16
    enc_step = 3
    dec_step = 8
    K = 1
    dim = 4
    dir_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    pickled_model_output = os.path.join(SD_AlGORITHM_BUCKET_PATH, pickle_dir)
    if not os.path.isfile(pickle_dir):
        return
    with open(pickled_model_output, "rb") as f:
        raw_output = pickle.load(f)
    # Test sampling.
    pp_tools = SGNetPostProcessor(
        bbox_type="cxcywh",
        normalize_type="zero-one",
        save_res=True,
        save_res_path=res_save_dir,
        dir_name=dir_name,
    )
    pp_output = pp_tools(raw_output)
    pp_keys = [
        "pred_traj",
        "input_traj",
        "all_goal_trajs",
        "target_traj",
        "pred_cxcympb",
        "input_cxcympb",
        "goal_cxcympb",
        "target_cxcympb",
        "position",
        "height",
        "vcs_vel",
        "global_vel",
    ]
    for key in pp_keys:
        assert key in pp_output

    assert pp_output["pred_traj"].shape == (batch_size, K, dec_step, dim)
    assert pp_output["input_traj"].shape == (batch_size, 1, enc_step, dim)
    assert pp_output["all_goal_trajs"].shape == (batch_size, 1, dec_step, dim)
    assert pp_output["target_traj"].shape == (batch_size, 1, dec_step, dim)
    assert pp_output["position"].shape == (batch_size, 1, enc_step, 2)
    assert pp_output["height"].shape == (batch_size, 1, enc_step, 1)
    assert pp_output["vcs_vel"].shape == (batch_size, 1, enc_step, 2)
    assert pp_output["global_vel"].shape == (batch_size, 1, enc_step, 2)
