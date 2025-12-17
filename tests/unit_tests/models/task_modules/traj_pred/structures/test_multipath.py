# Copyright (c) Horizon Robotics. All rights reserved.

import os
from collections import OrderedDict

import numpy as np
import pytest

from hat.models.backbones.vargnetv2 import (
    VargNetV2,
    get_vargnetv2_stride2channels,
)
from hat.models.losses.traj_pred_loss import SoftMultipathLoss
from hat.models.task_modules.traj_pred.heads import BasicAnchorBasedDecoder
from hat.models.task_modules.traj_pred.necks import MultiPathNeck
from hat.models.task_modules.traj_pred.structures import MultipathV2
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_PATH
from tests.unit_tests.core.test_traj_pred_utils import (  # noqa: E501
    gen_autourban_example_data_loader,
)


def gen_example_roadmap_color_table():
    # Get color table.
    label_colors = {
        "others": [0, 0, 0],
        "roadedge": [0, 0, 255],
        "roadarrow": [47, 79, 79],
        "solid_lanes": [200, 200, 200],
        "stopline": [192, 0, 64],
        "crosswalk": [255, 127, 80],
    }
    all_seg_labels = list(label_colors.keys())
    color_table = {}
    rand_table_before = (np.random.randint(-128, 127, size=128) / 128).tolist()
    rand_table_after = (
        np.random.randint(-128, 127, size=128 - len(all_seg_labels)) / 128
    ).tolist()

    color_table["r_val"] = tuple(
        rand_table_before
        + [(label_colors[label][0] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    color_table["g_val"] = tuple(
        rand_table_before
        + [(label_colors[label][1] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    color_table["b_val"] = tuple(
        rand_table_before
        + [(label_colors[label][2] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    return color_table


def gen_example_multipath_model_v2(
    anchor_file,
    anchor_num,
    color_table,
    losses=None,
    post_process=None,
    post_process_dict=None,
):
    # Build model.
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    alpha = 0.5
    data_shape = (7, 512, 512)
    input_channels = data_shape[0]
    roi_input_size = 16
    roi_output_size = 8
    out_stride = 32
    if post_process_dict is None:
        post_process_dict = {}

    model_heads = OrderedDict(
        traj_head=BasicAnchorBasedDecoder(
            in_channels=dict(
                rroi_feat=16,
                state_vector_feat=3,
            ),
            anchor_cfg=dict(
                anchor_file=anchor_file,
                anchor_method="kmeans",
                anchor_num=anchor_num,
            ),
            n_hidden_layers=[1024, 1024],
            use_momentum=True,
            log_std_clamp_min=-2,
            log_std_clamp_max=2,
            is_int_infer_model=False,
            loss=SoftMultipathLoss(
                head_weight=1,
                reg_loss_scale=1,
                is_soft_cls=True,
                is_soft_reg=True,
                num_soft_trajs=3,
                soft_mode="by_l2",
            ),
            post_process=post_process_dict["traj_head"]
            if "traj_head" in post_process_dict
            else None,
        ),
    )

    model_necks = OrderedDict(
        multipath_neck=MultiPathNeck(
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[out_stride],
            stride2channels=get_vargnetv2_stride2channels(alpha),
            roi_input_patch=[roi_input_size, roi_input_size],
            roi_output_patch=[roi_output_size, roi_output_size],
            conv_channels=[16, 16, 16, 16],
            use_depthwise_as_avg=True,
            is_int_infer_model=False,
        ),
    )

    model = MultipathV2(
        data_shape=data_shape,
        backbone=VargNetV2(
            num_classes=1000,
            bn_kwargs=bn_kwargs,
            alpha=alpha,
            group_base=8,
            factor=2,
            bias=True,
            disable_quanti_input=True,
            include_top=False,
            flat_output=True,
            input_channels=input_channels,
            head_factor=2,
        ),
        necks=model_necks,
        heads=model_heads,
        table=color_table,
        map_augmentation=True,
        drivable_area_table=None,
        use_drivable_area=False,
        post_process=post_process,
        losses=losses,
    )
    return model


@pytest.mark.skipif(True, reason="requiring New Dataset")
def test_multipath_structure(version=2):
    if version == 1:
        raise ValueError(f"Multipath version {version} has been removed.")
    elif version == 2:
        model_gen_func = gen_example_multipath_model_v2
    else:
        raise ValueError(f"Unsupported model version {version}.")

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
        enable_high_freq=True,
        enable_behav_trans=False,
    )
    # Build model.
    anchor_file = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "anchors/ZGC_anchors_kmeans_dirnorm_add_lc48_remove_straight70_fixbug.pkl",  # noqa: E501
    )
    anchor_num = 106
    color_table = gen_example_roadmap_color_table()
    model = model_gen_func(anchor_file, anchor_num, color_table)
    model = model.cuda()
    model.fuse_model()
    model.set_qconfig()

    # Forward and check the outputs.
    dl_iter = iter(train_dataloader)
    sample = next(dl_iter)
    model_output = model(sample)

    for i in sample.keys():
        assert i in model_output

    if version == 1:
        add_keys = []
        add_head_keys = [
            "gts",
            "anchors",
            "probabilities",
            "log_anchors_probs",
            "means",
            "scale_trils",
            "track_ids",
            "masks",
            "resize_ratio",
        ]
        for head_name in model.head_name_list:
            add_keys += [head_name + "_" + key for key in add_head_keys]
        traj_output_heads = model.head_name_list
    else:
        add_keys = [
            "traj_head_anchor_probs",
            "traj_head_anchor_mean_var",
            "traj_head_means",
            "traj_head_scale_trils",
            "traj_head_probabilities",
            "traj_head_log_anchors_probs",
        ]
        traj_output_heads = ["traj_head"]

    for i in add_keys:
        assert i in model_output

    num_obj, traj_len, _ = sample["future_trajectories"].shape
    for head_name in traj_output_heads:
        assert list(model_output[head_name + "_probabilities"].shape) == [
            num_obj,
            anchor_num,
        ]
        assert list(model_output[head_name + "_means"].shape) == [
            num_obj,
            anchor_num,
            traj_len,
            2,
        ]
        assert list(model_output[head_name + "_scale_trils"].shape) == [
            num_obj,
            anchor_num,
            traj_len,
            2,
            2,
        ]
