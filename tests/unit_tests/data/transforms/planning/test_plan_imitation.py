# Copyright (c) Horizon Robotics. All rights reserved.
# The unit test for trajectory prediction transforms about coodinates.
import os
from distutils.version import LooseVersion

import numpy as np
import pandas
import pytest

from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.data.transforms.planning.plan_imitation import (
    PlanAgentMotion,
    PlanAgentsGridGenerate,
    PlanEgoGridGenerate,
    PlanEgoMotion,
    PlanEgoState,
    PlanGaussianRandom,
    PlanGridAction,
    PlanPerturbation,
)
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
def test_plan_imitation_transforms():
    csv_path = os.path.join(
        HAT_BUCKET_PATH,
        "users/bikun.wang/unit_test/concat_train.csv",
    )
    # Test TDTGetBEVLocalMapByTimestamp
    data_token2path_mapping = {}
    data_token2path_mapping["3079_20230303"] = [
        "users/bikun.wang/unit_test",
        "bevgt_noa",
    ]

    dataset = generate_dummy_dataset(csv_path)
    sample = dataset[0]
    bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
    img_h, img_w = 512, 512

    pre_transforms = [
        GenSeqCenter(context_frames=4, augmentation=False),
        GenBoundingBox(
            gen_ego_bbox=False,
            clockwise=True,
            min_shape=1,
            const_z_value=1.5,
        ),
        GenBoundingBox(
            gen_ego_bbox=True,
            clockwise=True,
            min_shape=1,
            const_z_value=1.5,
        ),
        PhyToBEV(
            bev_origin_x=bev_origin_x,
            bev_origin_y=bev_origin_y,
            resolution=img_resolution,
            reverse=True,
        ),
        TDTEgoCentricGt(),
        GetLcfTimeStamp(),
        TDTFilterObstacles(
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
        ),
        TDTGenStatesAndMask(
            seq_length=16,
            context_frames=4,
            seq_period=1,
            enable_incomplete_gts=True,
        ),
        TDTOccupancyMapRender(
            map_height=img_h,
            map_width=img_w,
            context_frames=4,
            render_ego=False,
            render_fut=True,
        ),
        TDTOccupancyMapRender(
            map_height=img_h,
            map_width=img_w,
            context_frames=4,
            render_ego=True,
            render_fut=False,
        ),
        TDTGetBEVLocalMapByTimestamp(
            image_dir=HAT_BUCKET_PATH,
            image_suffix=".png",
            map_height=img_h,
            map_width=img_w,
            data_token2path_mapping=data_token2path_mapping,
            map_path_func=gen_dummy_path_func,
        ),
        TDTGetTrajPredObjectsInfo(
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
        ),
    ]

    for trans in pre_transforms:
        sample = trans(sample)

    # Test Render PlanGaussianRandom.
    gauss_random = PlanGaussianRandom(std=[1.5, 0.5, np.pi / 10])
    sample = gauss_random(sample)
    assert "perturb_center" in sample

    # Test PlanEgoMotion
    ego_motion = PlanEgoMotion(
        context_frames=4,
        seq_length=16,
    )
    sample = ego_motion(sample)
    assert "plan_navi_traj" in sample
    assert "plan_his_state" in sample
    assert "plan_fut_state" in sample
    assert sample["plan_his_state"].shape == (4, 6)
    assert sample["plan_fut_state"].shape == (12, 6)
    assert (
        sample["plan_navi_traj"].shape[0] >= sample["plan_fut_state"].shape[0]
    )

    ego_state = PlanEgoState(
        grid_dim=65,
        ego_track_id=-42,
    )
    sample = ego_state(sample)
    assert "plan_ego_motion" in sample
    assert sample["plan_ego_motion"].shape == (2, 65, 65)

    # Test PlanAgentMotion
    agent_motion = PlanAgentMotion(
        ego_track_id=-BaseTrajDataset.ANSWER,
        ego_center=(362, 256),
        res=-0.2,
    )
    sample = agent_motion(sample)
    assert "plan_agents_traj_vcs" in sample
    assert "plan_agents_his_traj_vcs" in sample
    assert "plan_agents_state_vector" in sample
    assert -BaseTrajDataset.ANSWER in sample["valid_track_ids"]
    agents_len = len(sample["valid_track_ids"]) - 1
    assert len(sample["plan_agents_traj_vcs"]) == agents_len
    assert len(sample["plan_agents_his_traj_vcs"]) == agents_len
    assert len(sample["plan_agents_state_vector"]) == agents_len

    # Test PlanPerturbation
    perturbation = PlanPerturbation(
        perturb_prob=1,
        ego_center=(362, 256),
        map_height=img_h,
        map_width=img_w,
        context_frames=4,
        seq_length=16,
        res=img_resolution,
        min_displacement=3,
        min_acc=-0.3,
        max_acc=0.15,
        min_steer=-0.2,
        max_steer=0.2,
    )
    sample = perturbation(sample)

    assert sample["rendered_obs"].shape == (img_h, img_w, 4)
    assert sample["rendered_obs_fut"].shape == (img_h, img_w, 12)
    assert sample["road_map"].shape == (img_h, img_w, 1)
    assert sample["plan_fut_state"].shape == (12, 6)
    assert (
        sample["plan_navi_traj"].shape[0] >= sample["plan_fut_state"].shape[0]
    )

    # Test PlanEgoGridGenerate
    grid_dim = 65
    grid_horizon = 50
    grid_extent = [-51.2, 51.2, -30, 72.4]
    grid_action_dim = 9

    grid_generate = PlanEgoGridGenerate(
        interpolate_freq=10,
        grid_dim=grid_dim,
        horizon=grid_horizon,
        grid_extent=grid_extent,
    )
    sample = grid_generate(sample)
    assert "plan_svf_path" in sample
    assert "plan_svf_goal" in sample
    assert "plan_waypts_expert" in sample
    assert "plan_grid_idcs" in sample
    assert sample["plan_svf_path"].shape == (1, grid_dim, grid_dim)
    assert sample["plan_svf_goal"].shape == (1, grid_dim, grid_dim)
    assert sample["plan_waypts_expert"].shape == (grid_horizon, 2)
    assert sample["plan_grid_idcs"].shape == (grid_horizon, 2)

    # Test PlanAgentsGridGenerate
    agents_grid_gen = PlanAgentsGridGenerate(
        interpolate_freq=10,
        grid_dim=grid_dim,
        horizon=grid_horizon,
        grid_extent=grid_extent,
    )
    sample = agents_grid_gen(sample)
    assert "plan_agents_svf" in sample
    assert "plan_agents_state_vector" in sample
    assert "plan_agents_svf_hist" in sample
    assert "plan_agents_waypts" in sample
    assert "plan_agents_grid_idcs" in sample

    # Test PlanGridAction
    grid_action = PlanGridAction(
        grid_dim=grid_dim,
        horizon=grid_horizon,
        action_num=grid_action_dim,
    )
    sample = grid_action(sample)
    assert "plan_bc_targets" in sample
    assert sample["plan_bc_targets"].shape == (
        grid_horizon,
        grid_action_dim,
        grid_dim,
        grid_dim,
    )
