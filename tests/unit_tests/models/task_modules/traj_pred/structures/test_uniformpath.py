# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest

from hat.registry import build_from_registry
from projects.prediction.configs.color_table import (
    gen_roadmap_with_vl_color_table,
)
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_PATH
from tests.unit_tests.core.test_traj_pred_utils import (  # noqa: E501
    gen_autourban_example_data_loader,
)


def gen_example_uniformpath_model(anchor_num, color_table):
    # Build model.
    model = dict(
        type="Uniformpath",
        heads={"Uniformpath": None},
        anchor_num=anchor_num,
        post_process=dict(
            type="UniformpathPostProcessor",
            ego_track_id=-42,
        ),
        table=color_table,
    )
    model = build_from_registry(model)
    return model


@pytest.mark.skipif(True, reason="requiring New Dataset")
def test_uniformpath_structure():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tdt_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "datasets/auto_urban_8days/hdfs_save/csv_trans/concat_tdt/",  # noqa: E501
    )
    tdt_yaml_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "datasets/auto_urban_8days/hdfs_save/csv_trans/concat_tdt/tdt_meta.yaml",  # noqa: E501
    )
    img_dir = os.path.join(SD_AlGORITHM_BUCKET_PATH, "mengyuan")

    # Build dataloader.
    train_dataloader = gen_autourban_example_data_loader(
        tdt_dir,
        tdt_yaml_dir,
        img_dir,
        assign_trajs_for_filtered_obs=True,
        enable_high_freq=True,
    )

    # Build model.
    anchor_num = 124
    color_table = gen_roadmap_with_vl_color_table()
    model = gen_example_uniformpath_model(anchor_num, color_table)

    # Forward and check the outputs.
    dl_iter = iter(train_dataloader)
    sample = next(dl_iter)
    model_output = model(sample)

    for i in sample.keys():
        assert i in model_output
    add_keys = []
    add_head_keys = [
        "gts",
        "probabilities",
        "log_anchors_probs",
        "means",
        "masks",
    ]

    for head_name in model.head_name_list:
        add_keys += [head_name + "_" + key for key in add_head_keys]
    for i in add_keys:
        assert i in model_output

    num_obj, traj_len, _ = sample["filtered_obs_gt"].shape
    num_anchors = 124
    for head_name in model.head_name_list:
        assert list(model_output[head_name + "_probabilities"].shape) == [
            num_obj,
            num_anchors,
        ]
        assert list(model_output[head_name + "_means"].shape) == [
            num_obj,
            num_anchors,
            traj_len,
            2,
        ]
