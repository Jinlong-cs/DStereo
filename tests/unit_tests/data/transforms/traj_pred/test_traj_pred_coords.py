# Copyright (c) Horizon Robotics. All rights reserved.
# The unit test for trajectory prediction transforms about coodinates.

import os

import numpy as np
import pytest

from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.data.transforms.traj_pred.traj_pred_coords import (
    DetectCurve,
    EgoCentricGt,
    GenBoundingBox,
    GenSeqCenter,
    LateralSmoothing,
    PhyToBEV,
    PhyToImage,
)
from hat.data.transforms.traj_pred.traj_pred_obstacle import (
    GenFutureTrackids,
    GetSeqDataFrameMask,
    RemapObsCls,
    SelectYawArray,
)
from hat.utils.package_helper import check_packages_available
from projects.prediction.configs.processed_dataset import (
    DENSETNT_UNITTEST_PREFIX,
    HAT_UNITTEST_PREFIX,
)
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH
from tests.unit_tests.core.test_traj_pred_utils import (  # noqa: E501
    gen_data_loader,
    gen_data_loader_v2,
    gen_densetnt_transforms,
    gen_example_transforms,
)

pd_available = check_packages_available("pandas==1.1.0", raise_exception=False)


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_traj_pred_coordinate_transforms():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)

    sample = dataset[0]
    # Test GenSeqCenter.
    assert "seq_center" not in sample
    gen_seq_center = GenSeqCenter(
        anchor_frame="last_context", context_frames=4, augmentation=False
    )
    sample = gen_seq_center(sample)
    assert "seq_center" in sample

    # Test GenBoundingBox.
    # -- test generating obstacle bounding boxes.
    gen_bbox = GenBoundingBox(
        gen_ego_bbox=False, clockwise=True, min_shape=1, const_z_value=1.5
    )
    obs_update_cols = gen_bbox.OBS_UPDATE_COLS
    sample["seq_df"][obs_update_cols] = np.nan
    sample = gen_bbox(sample)
    assert not np.any(np.isnan(sample["seq_df"][obs_update_cols].values))
    # -- test generating ego bounding boxes.
    gen_bbox = GenBoundingBox(
        gen_ego_bbox=True, clockwise=True, min_shape=1, const_z_value=1.5
    )
    ego_update_cols = gen_bbox.EGO_UPDATE_COLS
    sample["seq_df"][ego_update_cols] = np.nan
    sample = gen_bbox(sample)
    assert not np.any(np.isnan(sample["seq_df"][ego_update_cols].values))

    # Test PhyToImage.
    ego_cols = gen_bbox.EGO_COLS
    obs_cols = gen_bbox.OBS_COLS
    all_cols_to_trans = (
        ego_cols + obs_cols + ego_update_cols[0:8] + obs_update_cols[0:8]
    )
    for col in all_cols_to_trans:
        assert col in sample["seq_df"].columns
        assert "img_" + col not in sample["seq_df"].columns
    img_h, img_w, img_resolution = 512, 512, 0.2
    phy_to_image = PhyToImage(
        map_height=img_h, map_width=img_w, resolution=img_resolution
    )
    sample = phy_to_image(sample)
    for col in all_cols_to_trans:
        assert "img_" + col in sample["seq_df"].columns
    # -- if disable augmentation in GenSeqCenter, the sequence center is
    #    the coordinates of the ego vehicle at the last context frame.
    #    After translating to image coordinates, it will be located at
    #    the center of the image.
    df = sample["seq_df"]
    ego_df = df[df["track_id"] == -BaseTrajDataset.ANSWER]
    ego_df = ego_df[ego_df["frame_id"] == sample["last_context_frame_id"]]
    assert ego_df["img_x"].values[0] == img_h / 2
    assert ego_df["img_y"].values[0] == img_w / 2
    assert ego_df["img_yaw"].values[0] == 0

    # Test PhyToBEV. The sequence center will not locate at the map
    # center.
    bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
    phy_to_bev = PhyToBEV(
        bev_origin_x=bev_origin_x,
        bev_origin_y=bev_origin_y,
        resolution=img_resolution,
        reverse=True,
    )
    sample = phy_to_bev(sample)
    df = sample["seq_df"]
    ego_df = df[df["track_id"] == -BaseTrajDataset.ANSWER]
    ego_df = ego_df[ego_df["frame_id"] == sample["last_context_frame_id"]]
    assert ego_df["img_x"].values[0] == bev_origin_x / img_resolution
    assert ego_df["img_y"].values[0] == bev_origin_y / img_resolution
    assert ego_df["img_yaw"].values[0] == np.pi


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_centric_coordinate_transforms():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)

    veh_type_id = 1
    ped_cyc_type_id = [2, 18]
    sample = dataset[0]

    # Test EgoCentricGt.
    all_transforms = [
        GenFutureTrackids(),
        RemapObsCls(
            veh_type_id=veh_type_id,
            ped_cyc_type_id=ped_cyc_type_id,
            detect_peds_by_shape=False,
            ped_shape_thr=1,
        ),
        GetSeqDataFrameMask(
            freq_ratio=1,
        ),
        SelectYawArray(
            yaw_select_type=[1, 1, 1],
        ),
        EgoCentricGt(),
    ]

    for trans in all_transforms:
        sample = trans(sample)

    assert "object_centric_x" in sample["seq_df"].columns
    assert "object_centric_y" in sample["seq_df"].columns

    seq_df = sample["seq_df"]
    lcf_mask = sample["lctx_mask"]
    track_yaw_dict = sample["track_yaw_dict"]
    df_vals = seq_df.values
    df_cols = seq_df.columns
    col_track_id = df_cols.get_loc("track_id")
    col_centric_x = df_cols.get_loc("object_centric_x")
    col_centric_y = df_cols.get_loc("object_centric_y")
    lcf_track_ids = df_vals[lcf_mask, col_track_id]
    # All the centric coordinates in the last context frame should
    # be (zero, zero). the coordinates in the other frames cannot
    # be nan.
    for track_id in lcf_track_ids:
        track_id_mask = seq_df["track_id"] == track_id
        track_id_lcf_mask = track_id_mask & lcf_mask
        agent_yaw = track_yaw_dict[track_id]
        if agent_yaw is None:
            continue
        assert df_vals[track_id_lcf_mask, col_centric_x] == 0
        assert df_vals[track_id_lcf_mask, col_centric_y] == 0
        assert not np.any(
            np.isnan(df_vals[track_id_mask, col_centric_x].astype("float64"))
        )
        assert not np.any(
            np.isnan(df_vals[track_id_mask, col_centric_y].astype("float64"))
        )


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_lateral_smoothing():
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
    dataset = gen_data_loader(tdt_dir, tdt_yaml_dir)

    sample = dataset[0]
    old_df = sample["seq_df"].copy()

    # Test LateralSmoothing.
    lateral_smooth = LateralSmoothing(squashing_scale=10)
    sample = lateral_smooth(sample)
    new_df = sample["seq_df"].copy()
    last_context_frame_id = sample["last_context_frame_id"]
    # -- The coordinates in the last context frame of the two dataframes
    # should be aligned.
    old_lctx_df = old_df[old_df["frame_id"] == last_context_frame_id]
    new_lctx_df = new_df[new_df["frame_id"] == last_context_frame_id]
    old_lctx_xy = old_lctx_df[["x", "y"]].values
    new_lctx_xy = new_lctx_df[["x", "y"]].values
    assert np.max(np.abs(old_lctx_xy - new_lctx_xy)) < 1e-4
    # -- The lateral differences in the new dataframe should be smaller
    # than the old dataframe.
    old_ego_df = old_df[old_df["track_id"] == -BaseTrajDataset.ANSWER]
    new_ego_df = new_df[new_df["track_id"] == -BaseTrajDataset.ANSWER]
    old_ego_xy = old_ego_df[["x", "y"]].values
    old_ego_yaw = old_ego_df["yaw"].values
    new_ego_xy = new_ego_df[["x", "y"]].values
    new_ego_yaw = new_ego_df["yaw"].values
    old_ego_diff_xy = np.diff(old_ego_xy, axis=0)
    old_ego_diff = np.sqrt(
        old_ego_diff_xy[:, 0] ** 2 + old_ego_diff_xy[:, 1] ** 2
    )
    old_ego_diff_lateral = old_ego_diff * np.sin(old_ego_yaw[:-1])
    new_ego_diff_xy = np.diff(new_ego_xy, axis=0)
    new_ego_diff = np.sqrt(
        new_ego_diff_xy[:, 0] ** 2 + new_ego_diff_xy[:, 1] ** 2
    )
    new_ego_diff_lateral = new_ego_diff * np.sin(new_ego_yaw[:-1])
    assert np.mean(old_ego_diff_lateral) > np.mean(new_ego_diff_lateral)


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_detect_curve():
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
    dataset = gen_data_loader(tdt_dir, tdt_yaml_dir)

    sample = dataset[0]
    curve_thr = 0.5
    detect_curve = DetectCurve(
        seq_length=dataset.datasets[0].seq_length,
        threshold=curve_thr,
        only_detect_ego=False,
    )
    sample = detect_curve(sample)
    assert "is_curve" in sample
    # Straight line.
    assert not DetectCurve.detect_curve_by_residual(
        np.arange(0, 6), np.arange(0, 6), curve_thr
    )
    # Circle.
    assert DetectCurve.detect_curve_by_residual(
        np.cos(np.arange(0, 6)), np.sin(np.arange(0, 6)), curve_thr
    )
    # Polyline.
    assert DetectCurve.detect_curve_by_residual(
        np.array([0, 1, 2, 2, 1, 0]), np.array([0, 1, 2, 3, 4, 5]), curve_thr
    )


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_gen_obstacles_goals():
    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)
    sample = dataset[0]

    basic_trans, obstacle_trans, _, _ = gen_example_transforms()
    densetnt_trans = gen_densetnt_transforms()

    all_transforms = basic_trans + obstacle_trans + densetnt_trans

    for trans in all_transforms:
        sample = trans(sample)

    assert "goal_coords" in sample
    assert "nearest_goal_idxs" in sample
    assert "nearest_road_idxs" in sample
    goal_coords = sample["goal_coords"]
    nearest_goal_idxs = sample["nearest_goal_idxs"]
    nearest_road_idxs = sample["nearest_road_idxs"]
    end_points = sample["end_points"]
    num_obs = len(sample["valid_track_ids"])
    assert goal_coords.shape[0] == num_obs
    assert nearest_goal_idxs.shape == (num_obs,)
    assert nearest_road_idxs.shape == (num_obs,)
    assert end_points.shape == (num_obs, 2)
