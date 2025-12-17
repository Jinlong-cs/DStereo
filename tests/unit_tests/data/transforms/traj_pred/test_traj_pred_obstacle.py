# Copyright (c) Horizon Robotics. All rights reserved.
# The unit test for trajectory prediction transforms about obstacles.

import os
import sys

import numpy as np
import pytest

from hat.data.collates.traj_pred_collates import (
    gt_path_func_for_data_pipeline_v3,
)
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.data.transforms.traj_pred.traj_pred_anchor import SampleNaviTrajAnchor
from hat.data.transforms.traj_pred.traj_pred_obstacle import (
    FilterObstacles,
    FilterObstaclesByFuture,
    GenFutureTrackids,
    GenHighFreqTraj,
    GenStatesAndMask,
    GetNavinetmapInfo,
    GetObstacleSafeArea,
    GetSeqDataFrameMask,
    VectorNetTrajExtractor,
)
from hat.utils.package_helper import check_packages_available
from projects.prediction.configs.anchor_type_split import (
    ANCHOR_TYPE_CLASSIFY_FUNC,
)
from projects.prediction.configs.processed_dataset import (
    DENSETNT_UNITTEST_PREFIX,
    HAT_UNITTEST_PREFIX,
)
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH
from tests.unit_tests.core.test_traj_pred_utils import (  # noqa: E501
    gen_data_loader,
    gen_data_loader_v2,
    gen_example_transforms,
)

pd_available = check_packages_available("pandas==1.1.0", raise_exception=False)


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_obstacle_filter():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)

    sample = dataset[0]

    basic_trans, _, _, _ = gen_example_transforms()
    for trans_func in basic_trans:
        sample = trans_func(sample)

    ped_cyc_type_id = [2, 18]
    track_id_list = sample["track_ids"]
    # Test ObsFilter.
    obs_filter = FilterObstacles(
        is_training=True,
        is_multiagent=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode="vehicles",
        valid_distance=[
            [130.0, -50.0, 50.0, -50.0],
            [30.0, 0.0, 10.0, -10.0],
            [130.0, -50.0, 20.0, -20.0],
        ],
        num_frame_thr=2,
        bounce_thr=np.pi / 8,
        speed_drift_thr=[35.0, 3.25, 20.0],
        veh_lateral_drift_thr=1.5,
        static_thr=[0.1, 0.2, 0.2],
        if_train_filter_bounce=[True, True],
        if_val_filter_bounce=[False, False],
        classify_by_shape=False,
        ped_shape_thr=1,
    )
    sample = obs_filter(sample)
    assert "track_ids" in sample
    assert "valid_track_ids" in sample
    assert "drift_ids" in sample
    assert sample["track_ids"] == track_id_list
    assert set(sample["valid_track_ids"]).issubset(set(track_id_list))
    # In this dataset, the track_id of all the pedestrain > 100000,
    # and the leaving mode of this transform is `vehicle` only.
    # Therefore, the `valid_track_ids` will not contain any pedestrain.
    for t_id in track_id_list:
        if t_id >= 100000:
            assert t_id not in sample["valid_track_ids"]


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_filter_for_valid_head():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)
    sample = dataset[0]

    basic_trans, _, _, _ = gen_example_transforms()
    for trans_func in basic_trans:
        sample = trans_func(sample)

    gen_states = GenStatesAndMask(
        enable_incomplete_gts=True,
    )
    sample = gen_states(sample)
    ped_cyc_type_id = [2, 18]
    track_id_list = sample["track_ids"]
    # Test ObsFilter.
    obs_filter = FilterObstaclesByFuture(
        is_multiagent=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode="vehicles",
        num_frame_thr=[6, 2, 4],
        bounce_thr=np.pi / 8,
        speed_drift_thr=[35.0, 3.25, 20.0],
        veh_lateral_drift_thr=1.5,
        static_thr=[0.1, 0.2, 0.2],
        if_filter_bounce=[True, True],
        classify_by_shape=False,
        ped_shape_thr=1,
        enable_incomplete_gts=True,
        key_name="valid_future_track_ids",
    )
    sample = obs_filter(sample)
    assert "track_ids" in sample
    assert "valid_future_track_ids" in sample
    assert "valid_future_track_ids_stat_dict" in sample
    assert sample["track_ids"] == track_id_list
    assert set(sample["valid_future_track_ids"]).issubset(set(track_id_list))
    # In this dataset, the track_id of all the pedestrain > 100000,
    # and the leaving mode of this transform is `vehicle` only.
    # Therefore, the `valid_track_ids` will not contain any pedestrain.
    for t_id in track_id_list:
        if t_id >= 100000:
            assert t_id not in sample["valid_future_track_ids"]


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_gen_state_and_masks():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)
    sample = dataset[0]
    gen_track_id = GenFutureTrackids()
    sample = gen_track_id(sample)

    # Test GetSeqDataFrameMask
    get_seq_frame_id = GetSeqDataFrameMask(freq_ratio=1)
    sample = get_seq_frame_id(sample)
    assert "lctx_mask" in sample

    # Test GenStatesAndMask.
    seq_length = (6 + 6) * 2
    context_frames = 6 * 2
    state_cols = ["x", "y", "width", "length"]
    gen_states = GenStatesAndMask(
        state_col_name=state_cols,
        enable_incomplete_gts=False,
    )
    sample = gen_states(sample)
    assert "states" in sample
    assert "masks" in sample
    assert "context_states" in sample
    assert "context_masks" in sample
    assert "classification" in sample
    track_id_list = sample["track_ids"]
    num_obs = len(track_id_list)
    assert sample["states"].shape == (
        num_obs,
        seq_length - context_frames,
        len(state_cols),
    )
    assert sample["context_states"].shape == (
        num_obs,
        context_frames,
        len(state_cols),
    )
    assert sample["masks"].shape == (num_obs, seq_length - context_frames)
    assert sample["context_masks"].shape == (num_obs, context_frames)
    assert sample["classification"].shape == (num_obs,)


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_get_seq_data_frame_mask():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)
    sample = dataset[0]

    # Test GenStatesAndMask.
    freq_ratio = 4
    gen_data_frames = GetSeqDataFrameMask(
        freq_ratio=8,
    )
    sample = gen_data_frames(sample)
    assert "lctx_mask" in sample
    assert "ctx_mask" in sample
    assert "fut_mask" in sample
    assert "ctx_frame_id" in sample
    assert "fut_frame_id" in sample
    if freq_ratio > 1:
        assert "sampled_ctx_mask" in sample
        assert "sampled_fut_mask" in sample
        assert "sampled_ctx_frame_id" in sample
        assert "sampled_fut_frame_id" in sample


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_gen_high_freq_traj():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)

    sample = dataset[0]

    ped_cyc_type_id = [2, 18]
    get_sampled_frames = False

    basic_trans, _, _, _ = gen_example_transforms()
    for trans_func in basic_trans:
        sample = trans_func(sample)

    obs_filter = FilterObstacles(
        is_training=True,
        is_multiagent=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode="vehicles",
        valid_distance=[
            [50, -30, 50, -50],
            [30, 0, 10, -10],
            [50, -30, 20, -20],
        ],
        num_frame_thr=2,
        bounce_thr=np.pi / 8,
        speed_drift_thr=[35.0, 3.25, 20.0],
        veh_lateral_drift_thr=1.5,
        static_thr=[0.1, 0.2, 0.2],
        if_train_filter_bounce=[True, True],
        if_val_filter_bounce=[False, False],
        classify_by_shape=False,
        ped_shape_thr=1,
    )

    gen_high_freq_frames = GenHighFreqTraj(
        get_sampled_frames=get_sampled_frames,
        enable_incomplete_gts=True,
        is_training=True,
    )
    sample = obs_filter(sample)
    sample = gen_high_freq_frames(sample)

    assert "high_freq_ctx_trajectories" in sample
    assert "high_freq_ctx_masks" in sample


@pytest.mark.skipif(True, reason="requiring New Dataset")
def test_get_traj_pred_info():
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
    data_token2path_mapping = {
        "DG201_20210401": "BEV_data_0624/AutoZGC_v5",
        "DG201_20210401_viz": "BEV_data_0624/AutoZGC_v5",
    }

    # Build data loader.
    ds_kwargs = {
        "max_stamp_thr": 550,
        "path_prefix": "DG201_",
        "check_map_exist_func": None,
    }
    dataset = gen_data_loader(
        tdt_dir, tdt_yaml_dir, dataset_name="autourban", **ds_kwargs
    )
    sample = dataset[0]

    basic_trans, obstacle_trans, map_trans, _ = gen_example_transforms(
        data_token2path_mapping, img_dir
    )
    all_transforms = basic_trans + obstacle_trans + map_trans
    for trans in all_transforms:
        sample = trans(sample)
    assert "valid_class" in sample
    assert "valid_img_coords" in sample
    assert "valid_masks" in sample
    assert "future_trajectories" in sample
    assert "resize_ratio" in sample
    assert (
        len(sample["valid_class"])
        == len(sample["valid_track_ids"])
        == len(sample["valid_img_coords"])
        == len(sample["valid_masks"])
        == len(sample["future_trajectories"])
        == len(sample["resize_ratio"])
    )
    assert np.max(sample["rendered_obs"]) <= 1
    assert np.min(sample["rendered_obs"]) >= -1
    return sample


@pytest.mark.skipif(True, reason="requiring New Dataset")
def test_get_obstacle_safe_area():
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
    data_token2path_mapping = {
        "DG201_20210401": "BEV_data_0624/AutoZGC_v5",
        "DG201_20210401_viz": "BEV_data_0624/AutoZGC_v5",
    }

    # Build data loader.
    ds_kwargs = {
        "max_stamp_thr": 550,
        "path_prefix": "DG201_",
        "check_map_exist_func": None,
    }
    dataset = gen_data_loader(
        tdt_dir, tdt_yaml_dir, dataset_name="autourban", **ds_kwargs
    )
    sample = dataset[0]

    basic_trans, obstacle_trans, map_trans, _ = gen_example_transforms(
        data_token2path_mapping, img_dir
    )
    all_transforms = basic_trans + obstacle_trans + map_trans
    for trans in all_transforms:
        sample = trans(sample)

    assert "safe_area" not in sample
    get_safe_area = GetObstacleSafeArea(traj_len=2)
    sample = get_safe_area(sample)
    assert "safe_area" in sample


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS or "3.8" in sys.version or not pd_available,
    reason="requiring SD_Algorithm bucket",
)
def test_vectornet_traj_extractor():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)

    sample = dataset[0]
    basic_trans, obstacle_trans, _, _ = gen_example_transforms()
    all_transforms = basic_trans + obstacle_trans

    for trans in all_transforms:
        sample = trans(sample)

    map_origin_params = [72.4, 51.2, 0.2]
    source_freq = 2
    target_freq = 2
    ego_track_id = -42
    max_obs_num = 32
    local_ele_seg_thr = 100

    # traj feats: start_x, start_y, end_x, end_y, time_stamp, pid, one-hot
    # for type then scale them to [-5, 5]
    traj_optional_feats = ["timestamp", "obstacle_class"]  # "obstacle_class"
    num_obs_type = 3
    traj_feat_dim = 5
    traj_feat_scale = [0.04, 0.04, 0.04, 0.04, 3]
    if "obstacle_class" in traj_optional_feats:
        traj_feat_dim += num_obs_type
        traj_feat_scale += [5 for i in range(num_obs_type)]

    traj_seg = VectorNetTrajExtractor(
        map_origin_params=map_origin_params,
        source_freq=source_freq,
        target_freq=target_freq,
        ego_track_id=ego_track_id,
        max_obs_num=max_obs_num,
        local_ele_seg_thr=local_ele_seg_thr,
        image_coordinates="bev",
        reverse=True,
        scale=traj_feat_scale,
        traj_feat_dim=traj_feat_dim,
        num_obs_type=num_obs_type,
        traj_optional_feats=traj_optional_feats,
        valid_track_ids_key="history_valid_track_ids",
    )
    sample = traj_seg(sample)
    assert "struct_traj_feats" in sample
    traj_feats = sample["struct_traj_feats"]
    num_obs = len(sample["history_valid_track_ids"])
    assert traj_feats.shape[:-1] == (num_obs, max_obs_num, target_freq * 1.5)


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS or "3.8" in sys.version or not pd_available,
    reason="requiring SD_Algorithm bucket",
)
def test_get_navinetmap_info():
    root_pkl_path = os.path.join(
        DENSETNT_UNITTEST_PREFIX, "densetnt/val_his6_tiny.pkl"
    )
    dataset = gen_data_loader_v2(root_pkl_path)

    sample = dataset[0]
    basic_trans, obstacle_trans, _, _ = gen_example_transforms()
    all_transforms = basic_trans + obstacle_trans
    for trans in all_transforms:
        sample = trans(sample)

    bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
    anchor_num = 124
    traj_len = 12
    transform_pre = SampleNaviTrajAnchor(
        bev_origin_x=bev_origin_x,
        bev_origin_y=bev_origin_y,
        img_resolution=img_resolution,
        num_anchors=anchor_num,
        traj_len=traj_len,
        navi_file_dir=SD_AlGORITHM_BUCKET_PATH,
        navi_info_path_mapping={},
        navi_file_path_func=gt_path_func_for_data_pipeline_v3,
        anchor_classify_func=ANCHOR_TYPE_CLASSIFY_FUNC["classical"],
    )
    transform_cur = GetNavinetmapInfo(
        use_extend_state_vectors=True,
    )
    sample = transform_pre(sample)
    sample = transform_cur(sample)
    assert "valid_block_marks_dict" in sample
    assert "valid_stopline_dis_dict" in sample
    state_vectors = sample["state_vectors"]
    assert state_vectors.shape[1] == 5
    valid_block_marks_dict = sample["valid_block_marks_dict"]
    valid_stopline_dis_dict = sample["valid_stopline_dis_dict"]
    valid_track_ids = set(sample["valid_track_ids"])
    track_ids_from_mark_dict = set(valid_block_marks_dict.keys())
    track_ids_from_dis_dict = set(valid_stopline_dis_dict.keys())
    assert valid_track_ids == track_ids_from_mark_dict
    assert valid_track_ids == track_ids_from_dis_dict
