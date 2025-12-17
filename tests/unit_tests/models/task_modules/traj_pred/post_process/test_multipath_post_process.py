# Copyright (c) Horizon Robotics. All rights reserved.

import os
import pickle

import numpy as np
import pytest
import torch

from hat.models.task_modules.traj_pred.post_process.multipath_post_process import (  # noqa: E501
    MultipathPostProcessor,
    TrackValidHeadPostProcessor,
)
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_multipath_post_processor():
    pickled_model_output = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{HAT_UNITTEST_PREFIX}/outputs/test_loss_data_new.pkl",
    )
    with open(pickled_model_output, "rb") as f:
        raw_output = pickle.load(f)
        model_output = {
            key.split("head_")[1] if "head_" in key else key: value
            for key, value in raw_output.items()
        }

    data = {}
    data["batch_valid_class"] = torch.zeros((model_output["means"].shape[0]))

    sample_times = 5

    # Test sampling.
    pp_tools = MultipathPostProcessor(
        filter_reverse=True,
        sample_traj=True,
        num_samples_per_anchor=sample_times,
        use_traj_nms=False,
        nms_params=None,
        use_adaptor=False,
    )
    pp_output = pp_tools(model_output, data)
    pp_keys = [
        "cur_head_mask",
        "trajectories",
        "means",
        "scale_trils",
        "gts",
        "log_anchors_probs",
        "masks",
    ]
    for key in pp_keys:
        assert key in pp_output
    num_obj, num_anchors, traj_len, _ = model_output["means"].shape
    assert list(pp_output["trajectories"].shape) == [
        num_obj,
        num_anchors,
        sample_times,
        traj_len,
        2,
    ]

    # Test trajectory NMS.
    nms_num_trajs = 10
    pp_tools = MultipathPostProcessor(
        filter_reverse=True,
        sample_traj=False,
        use_traj_nms=True,
        nms_params=dict(
            nms_threshold=0.05,
            nms_prob_threshold=0.01,
            nms_num_trajs=nms_num_trajs,
            change_probs=False,
        ),
        use_adaptor=False,
    )
    pp_output = pp_tools(model_output, data)
    for key in pp_keys:
        assert key in pp_output
    num_obj, num_anchors, traj_len, _ = model_output["means"].shape
    assert list(pp_output["trajectories"].shape) == [
        num_obj,
        nms_num_trajs,
        1,
        traj_len,
        2,
    ]

    # Test Blend Trajectory
    nms_num_trajs = 10
    pp_tools = MultipathPostProcessor(
        filter_reverse=True,
        sample_traj=False,
        use_traj_nms=True,
        nms_params=dict(
            nms_threshold=0.05,
            nms_prob_threshold=0.01,
            nms_num_trajs=nms_num_trajs,
            change_probs=False,
        ),
        use_adaptor=False,
        blend_traj_by_history=True,
    )
    data["track_ids"] = [-42.0, 0.0, 1.0, 2.0]
    data["valid_track_ids"] = [[-42.0, 0.0, 1.0]]
    data["context_states"] = np.random.rand(32, 4, 2)

    res = pp_tools(model_output, data)

    assert "blend_track_ids" in res.keys()


@pytest.mark.parametrize(
    "output, data",
    [
        pytest.param(
            {
                "masks": torch.ones(32, 1, 1, 1).long(),
                "model_is_training": False,
                "track_valid_cls": torch.rand(32, 3, 1, 1),
            },
            {
                "batch_valid_class": torch.ones(32),
            },
        ),
    ],
)
def test_track_valid_head_post_processor(output, data):
    post_process_func = TrackValidHeadPostProcessor(
        valid_head_threshold=[0.5, 0.5, 0.5],
    )
    ret = post_process_func(output, data)
    assert "track_valid_cls_binary" in ret
    assert "track_valid_cls_metric" in ret
