# Copyright (c) Horizon Robotics. All rights reserved.
# The unit test for trajectory prediction transforms about coodinates.

import os
from distutils.version import LooseVersion

import numpy as np
import pandas
import pytest

from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.data.transforms.planning.plan_traj_tdt import (
    TDTEgoCentricGt,
    TDTFilterObstacles,
    TDTGenStatesAndMask,
    TDTGetBEVLocalMapByTimestamp,
    TDTGetTrajPredObjectsInfo,
    TDTOccupancyMapRender,
)
from hat.data.transforms.traj_pred import (
    GenBoundingBox,
    GenSeqCenter,
    GetLcfTimeStamp,
    PhyToBEV,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH
from tests.unit_tests.data.datasets.test_planning_dataset import (
    generate_dummy_dataset,
)

pandas_version = pandas.__version__
PANDAS_VERSION_MATCH = LooseVersion(pandas_version) == LooseVersion("1.1.0")


def gen_dummy_path_func(
    image_dir, bev_bucket_path, plate, date_token, timestamp, image_suffix
):
    bev_map_path = os.path.join(
        image_dir,
        bev_bucket_path[0],
        bev_bucket_path[1],
        timestamp + image_suffix,
    )
    return bev_map_path


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_plan_traj_tdt_transforms():
    csv_path = os.path.join(
        HAT_BUCKET_PATH,
        "users/bikun.wang/unit_test/concat_train.csv",
    )

    dataset = generate_dummy_dataset(csv_path)
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

    # Test TDTEgoCentricGt.
    ego_centric_gt = TDTEgoCentricGt()
    sample = ego_centric_gt(sample)
    assert "object_centric_x" in sample["seq_df"].columns
    assert "object_centric_y" in sample["seq_df"].columns

    # Test TDTGetLcfTimeStamp.
    get_lcf_stamp = GetLcfTimeStamp()
    sample = get_lcf_stamp(sample)
    assert "lcf_timestamp" in sample

    # Test TDTFilterObstacles.
    obs_filter = TDTFilterObstacles(
        is_training=True,
        is_multiagent=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=[2, 18],
        leaving_mode="all",
        valid_distance=50,
        static_thr=1,  # i.e., 2 m/s
        num_frame_thr=2,
        bounce_thr=np.pi / 8,
        ped_static_thr=0.2,  # i.e, 0.4 m/s
        ped_drift_thr=1.5,  # i.e., 3 m/s
        use_given_lcf_yaw=True,
        use_given_obs_yaw=False,
        classify_by_shape=False,
        ped_shape_thr=1,
    )
    sample = obs_filter(sample)
    assert "track_ids" in sample
    assert "valid_track_ids" in sample
    assert "drift_ids" in sample

    # Test TDTGenStatesAndMask.
    gen_states = TDTGenStatesAndMask(
        seq_length=16,
        context_frames=4,
        seq_period=1,
        enable_incomplete_gts=True,
    )
    sample = gen_states(sample)
    assert "states" in sample
    assert "masks" in sample
    assert "context_states" in sample
    assert "context_masks" in sample
    assert "classification" in sample
    track_id_list = sample["track_ids"]
    num_obs = len(track_id_list)
    assert sample["states"].shape == (num_obs, 12, 2)
    assert sample["context_states"].shape == (num_obs, 4, 2)

    # Test Render TDTOccupancyMapRender.
    og_renderer = TDTOccupancyMapRender(
        map_height=img_h,
        map_width=img_w,
        context_frames=4,
        render_ego=False,
        render_fut=True,
    )
    sample = og_renderer(sample)
    assert "rendered_obs" in sample
    assert "rendered_obs_fut" in sample
    assert sample["rendered_obs"].shape == (img_h, img_w, 4)
    assert sample["rendered_obs_fut"].shape == (img_h, img_w, 12)

    # Test TDTGetTrajPredObjectsInfo
    get_traj_info = TDTGetTrajPredObjectsInfo(
        scene_img_height=img_h,
        scene_img_width=img_w,
        ego_track_id=-BaseTrajDataset.ANSWER,
        use_given_obs_yaw=False,
        peds_use_given_obs_yaw=True,
        veh_type_id=1,
        ped_cyc_type_id=[2, 18],
        detect_peds_by_shape=False,
        ped_shape_thr=1,
        use_state_vectors=True,
        use_instant_state_vectors=True,
        if_clip_state_vectors=True,
    )
    sample = get_traj_info(sample)

    assert "valid_class" in sample
    assert "valid_track_ids" in sample
    assert "valid_img_coords" in sample
    assert "valid_masks" in sample
    assert "valid_class" in sample
    assert "future_trajectories" in sample
    assert "resize_ratio" in sample
    assert len(sample["valid_track_ids"]) == len(sample["future_trajectories"])

    # Test TDTGetBEVLocalMapByTimestamp
    data_token2path_mapping = {}
    data_token2path_mapping["3079_20230303"] = [
        "users/bikun.wang/unit_test",
        "bevgt_noa",
    ]

    get_bev = TDTGetBEVLocalMapByTimestamp(
        image_dir=HAT_BUCKET_PATH,
        image_suffix=".png",
        map_height=img_h,
        map_width=img_w,
        data_token2path_mapping=data_token2path_mapping,
        map_path_func=gen_dummy_path_func,
    )

    sample = get_bev(sample)

    assert "road_map" in sample
    assert sample["road_map"].shape == (img_h, img_w, 1)
    assert np.any(sample["road_map"] > 0)
