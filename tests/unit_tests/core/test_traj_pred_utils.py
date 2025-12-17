# Copyright (c) Horizon Robotics. All rights reserved.

import os
import pickle
from glob import glob

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
import yaml

from hat.core.traj_pred_typing import TrajGroupIndex
from hat.core.traj_pred_utils import TdtCoordHelper as TCH
from hat.data.collates.traj_pred_collates import (
    collate_multipath,
    collate_uniformpath_viz,
    gt_path_func_for_data_pipeline_v3,
)
from hat.data.datasets.traj_pred_dataset import (
    BaseTrajDataset,
    ConcatTdtDataset,
)
from hat.data.transforms.traj_pred import (
    BehaviorSampling,
    EgoCentricGt,
    FilterObstacles,
    FilterObstaclesByFuture,
    GenBoundingBox,
    GenFutureTrackids,
    GenHighFreqTraj,
    GenObsShape,
    GenObstaclesGoalsV2,
    GenSeqCenter,
    GenStatesAndMask,
    GetBehavPredObjectsInfo,
    GetBEVHomography,
    GetBEVLocalMapByTimestamp,
    GetLcfTimeStamp,
    GetSeqDataFrameMask,
    GetTrajPredObjectsInfo,
    MapAugmentation,
    OccupancyMapRender,
    PhyToBEV,
    RemapObsCls,
    SelectYawArray,
    VectorNetStructuredMapServer,
    VectorNetTrajExtractor,
)
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH


def gen_data_loader_v2(root_pkl_path):
    with open(root_pkl_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def gen_data_loader(tdt_dir, tdt_yaml_dir, dataset_name="base", **kwargs):
    # Extract the meta info from the json file.
    with open(tdt_yaml_dir, "r") as f:  # noqa: E501
        tdt_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)
    train_csv_info, all_val_files = None, None
    if dataset_name == "autourban_navi":
        train_csv_info = tdt_yaml["autourban_navi"]["train_files"]
        all_val_files = os.path.join(tdt_dir, "concat_*.csv")
    else:
        train_csv_info = tdt_yaml["autourban"]["validation_files"]
        all_val_files = os.path.join(tdt_dir, "val_*.csv")
    traj_group_indices = [TrajGroupIndex(*i) for i in list(train_csv_info)]

    all_val_files = sorted(glob(all_val_files))
    files = []
    for file, traj_index in zip(all_val_files, traj_group_indices):
        file_name = file.split("/")[-1]
        files.append({"df": file_name, "traj_indices": [traj_index]})

    # Build a dataset instance.
    ego_veh_type_id = 1
    dataset = ConcatTdtDataset(
        prefix=tdt_dir,
        files=files[0:1],
        dataset_name=dataset_name,
        seq_length=16,
        seq_period=1,
        sample_step=1,
        context_frames=4,
        veh_type_id=ego_veh_type_id,
        transforms=None,
        ego_as_obs=True,
        **kwargs
    )
    return dataset


def gen_example_transforms(
    data_token2path_mapping=None,
    img_dir=None,
    freq_ratio=1,
    enable_high_freq=False,
    assign_trajs_for_filtered_obs=False,
    enable_multi_task=False,
):
    # Parameters.
    leaving_mode = "all"
    img_h, img_w = 512, 512
    bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
    x_ratio = bev_origin_x / img_resolution / img_h
    y_ratio = bev_origin_y / img_resolution / img_w
    context_frames = 4
    bounce_thr = np.pi / 8
    veh_type_id = 1
    ped_cyc_type_id = [2, 18]
    roi_input_size = 16
    roi_output_size = 8
    roi_spatial_scale = int(img_h / roi_input_size)
    detect_peds_by_shape = False
    ped_shape_thr = 1
    use_state_vectors = True
    use_behav_state_vectors = False
    use_instant_state_vectors = True
    num_behav_state_vectors = 7
    if_clip_state_vectors = True

    basic_trans = [
        GenSeqCenter(
            anchor_frame="last_context",
            context_frames=context_frames,
            augmentation=True,
        ),
        GetLcfTimeStamp(),
        GenFutureTrackids(),
        RemapObsCls(
            veh_type_id=veh_type_id,
            ped_cyc_type_id=ped_cyc_type_id,
            detect_peds_by_shape=False,
            ped_shape_thr=1,
        ),
        GenBoundingBox(
            gen_ego_bbox=False, clockwise=True, min_shape=1, const_z_value=1.5
        ),
        GenBoundingBox(
            gen_ego_bbox=True, clockwise=True, min_shape=1, const_z_value=1.5
        ),
        GenObsShape(),
        PhyToBEV(
            bev_origin_x=bev_origin_x,
            bev_origin_y=bev_origin_y,
            resolution=img_resolution,
            reverse=True,
        ),
        GetSeqDataFrameMask(
            freq_ratio=freq_ratio,
        ),
        SelectYawArray(
            yaw_select_type=[2, 7, 2],
            enable_none=False,
        ),
        EgoCentricGt(),
    ]
    obstacle_trans = [
        FilterObstacles(
            is_training=True,
            is_multiagent=True,
            ego_track_id=-BaseTrajDataset.ANSWER,
            ped_cyc_type_id=ped_cyc_type_id,
            leaving_mode=leaving_mode,
            valid_distance=[
                [130.0, -50.0, 50.0, -50.0],
                [30.0, 0.0, 10.0, -10.0],
                [130.0, -50.0, 20.0, -20.0],
            ],
            num_frame_thr=2,
            bounce_thr=bounce_thr,
            speed_drift_thr=[35.0, 3.25, 20.0],
            veh_lateral_drift_thr=1.5,
            static_thr=[0.0, 0.2, 0.2],
            if_train_filter_bounce=[True, True],
            if_val_filter_bounce=[False, False],
            classify_by_shape=detect_peds_by_shape,
            ped_shape_thr=ped_shape_thr,
        ),
        GenStatesAndMask(
            enable_incomplete_gts=True,
            is_training=True,
        ),
        FilterObstaclesByFuture(
            is_multiagent=True,
            ego_track_id=-BaseTrajDataset.ANSWER,
            ped_cyc_type_id=ped_cyc_type_id,
            leaving_mode=leaving_mode,
            num_frame_thr=[6, 2, 4],
            bounce_thr=bounce_thr,
            speed_drift_thr=[35.0, 3.25, 20.0],
            veh_lateral_drift_thr=1.5,
            static_thr=[0.0, 0.2, 0.2],
            if_filter_bounce=[True, True],
            classify_by_shape=detect_peds_by_shape,
            ped_shape_thr=ped_shape_thr,
            enable_incomplete_gts=True,
            key_name="valid_future_track_ids",
        ),
        GetTrajPredObjectsInfo(
            scene_img_height=img_h,
            scene_img_width=img_w,
            ego_track_id=-BaseTrajDataset.ANSWER,
            ped_cyc_type_id=ped_cyc_type_id,
            use_state_vectors=use_state_vectors,
            use_instant_state_vectors=use_instant_state_vectors,
            if_clip_state_vectors=if_clip_state_vectors,
            assign_trajs_for_filtered_obs=assign_trajs_for_filtered_obs,
            reverse=True,
            filter_invalid_end=True,
        ),
    ]
    behavior_trans = [
        BehaviorSampling(
            is_training=True,
            ego_track_id=-BaseTrajDataset.ANSWER,
            down_sampling=True,
            down_ratio=0.1,
            valid_track_ids_key="valid_track_ids",
        ),
        GetBehavPredObjectsInfo(
            ego_track_id=-BaseTrajDataset.ANSWER,
            use_behav_state_vectors=use_behav_state_vectors,
            num_behav_state_vectors=num_behav_state_vectors,
            use_obs_interaction=False,
            use_self_boxes=False,
            use_self_states=True,
            if_norm=False,
            norm_scale=5,
            mode="scene",
        ),
    ]
    if enable_high_freq:
        obstacle_trans.append(
            GenHighFreqTraj(
                get_sampled_frames=False,
                enable_incomplete_gts=True,
                is_training=True,
            )
        )
    get_map = GetBEVLocalMapByTimestamp(
        image_dir=img_dir,
        image_suffix=".png",
        map_height=img_h,
        map_width=img_w,
        data_token2path_mapping=data_token2path_mapping,
    )
    if enable_multi_task:
        get_map = GetBEVLocalMapByTimestamp(
            image_dir=img_dir,
            image_suffix=".png",
            map_height=img_h,
            map_width=img_w,
            data_token2path_mapping=data_token2path_mapping,
            map_path_func=gt_path_func_for_data_pipeline_v3,
        )
    map_trans = [
        OccupancyMapRender(
            map_height=img_h,
            map_width=img_w,
            render_ego=True,
            normalize=True,
        ),
        get_map,
        GetBEVHomography(
            src_shape=roi_input_size,
            dst_shape=roi_output_size,
            x_origin_ratio=x_ratio,
            y_origin_ratio=y_ratio,
            roi_spatial_scale=roi_spatial_scale,
            swap_xy=True,
            input_key="valid_img_coords",
            output_key="img_homographys",
        ),
        GetBEVHomography(
            src_shape=roi_input_size,
            dst_shape=roi_output_size,
            x_origin_ratio=x_ratio,
            y_origin_ratio=y_ratio,
            roi_spatial_scale=roi_spatial_scale,
            swap_xy=True,
            input_key="valid_img_coords",
            output_key="img_homographys",
        ),
        MapAugmentation(
            src_resolution=0.2,
            dst_resolution=img_resolution,
            map_origin_x=bev_origin_x,
            map_origin_y=bev_origin_y,
            swap_xy=True,
        ),
    ]
    return basic_trans, obstacle_trans, map_trans, behavior_trans


def gen_densetnt_transforms():
    pkl_fps = 2
    target_freq = 10
    max_obs_num = 32
    image_coordinates = "bev"  # in ["bev", "img"]
    img_h, img_w = 512, 512
    bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
    phy_img_h, phy_img_w = img_h * img_resolution, img_w * img_resolution
    if image_coordinates == "bev":
        map_origin_params = [bev_origin_x, bev_origin_y, img_resolution]
        coords_reverse = True
    elif image_coordinates == "img":
        map_origin_params = [phy_img_h, phy_img_w, img_resolution]
        coords_reverse = False
    local_ele_seg_thr = 100
    polyline_seg_len = 10
    max_num_ele_seg = 128

    all_supported_elements = [
        "stopline",
        "crosswalk",
        "solid_lane",
        "roadedge",
        "virtuallanelines",
    ]
    sample_mode_of_element = {
        "stopline": "uniform",
        "crosswalk": "uniform",
        "solid_lane": "uniform",
        "roadedge": "uniform",
        "virtuallanelines": "uniform",
    }
    # available SAMPLE_MODE for polyline: uniform/original
    # available SAMPLE_MODE for polygon: uniform
    all_elements = [
        "stopline",
        "crosswalk",
        "solid_lane",
        "roadedge",
    ]
    # sacle of feats to [-5,5]: "default":0.04, "turn_dir":5,
    # "pre_pre_point":0.04, "element_type":5
    polyline_optional_feats = ["turn_dir", "pre_pre_point", "element_type"]
    # road feats: start_x, start_y, end_x, end_y, one-hot
    road_feat_dim = 4
    road_feat_scale = [0.04, 0.04, 0.04, 0.04]
    if len(polyline_optional_feats) > 0:
        for feats in polyline_optional_feats:
            if feats == "turn_dir":
                road_feat_dim += 1
                road_feat_scale += [5]
            if feats == "pre_pre_point":
                road_feat_dim += 2
                road_feat_scale += [0.04, 0.04]
            if feats == "element_type":
                road_feat_dim += len(all_supported_elements)
                road_feat_scale += [
                    5 for i in range(len(all_supported_elements))
                ]

    # -- Parameters about traj polyline segment features.
    # all traj_optional_feats = ["timestamp", "pid", "obstacle_class"]
    # scale of feats to [-5, 5]: "timestamp":3, "pid":0.15, "obstacle_class":5
    traj_optional_feats = [
        "timestamp",
        "pid",
        "obstacle_class",
        "obs_width",
        "obs_length",
    ]
    num_obs_type = 3
    traj_feat_dim = 4
    traj_feat_scale = [1, 1, 1, 1]
    if len(traj_optional_feats) > 0:
        for feats in traj_optional_feats:
            if feats == "timestamp":
                traj_feat_dim += 1
                traj_feat_scale += [5]
            if feats == "pid":
                traj_feat_dim += 1
                traj_feat_scale += [1]
            if feats == "obstacle_class":
                traj_feat_dim += num_obs_type
                traj_feat_scale += [5 for _ in range(num_obs_type)]
            if feats == "obs_width":
                traj_feat_dim += 1
                traj_feat_scale += [1]
            if feats == "obs_length":
                traj_feat_dim += 1
                traj_feat_scale += [1]
    if_itp_traj = True

    # ---- GenObstaclesGoals
    # --------道路元素
    include_road_ele = False
    # 决定了两侧总共要采集多少点,0表示两侧都不采集点,1表示采集单侧点(默认右侧),
    # >=2表示采集两侧点(以每个 polyline 的点为中点,在垂直于 polyline 的方向上,左侧右侧分别采若干点),
    # 例如3表示左侧1个,右侧2个,4表示左右侧各2个:
    num_points_bothsides = 1
    assert (
        include_road_ele | num_points_bothsides > 0
    ), "include_road_ele is True or num_points_bothsides>0"
    # 如果使用密集采点(include_besides=True),那么 dense_goals_dis 会决定垂线上采点之间的间隔，单位 m
    dense_goals_dis = 1.7
    # dense_goals_dis=1
    # 每个道路元素的vector被分为几段
    road_ele_divide_num = 1
    assert road_ele_divide_num >= 1, "road_ele_divide_num must >=1"
    # 每个polyline上每连续多少点取1个点
    goal_interval = 2
    assert (
        goal_interval >= 1 and goal_interval <= polyline_seg_len
    ), "goal_interval should be an int between 0 and 10"
    # 使用差分轨迹
    use_diff_trajs = True

    # --------安全区
    use_safe_area = True
    expand_traj_method = "CV"
    use_his_traj = False  # 安全区是否要包括历史轨迹
    expand_base = 4
    expand_ratio = 0.2
    edge_divide_num = 3  # 每条安全区边被分为几段
    vertical_divide_num = 5  # 每条安全区垂线被分为几段

    # --------采样点总数
    num_goals = 2048
    # 缩放采样点坐标的数值
    goal_coords_scale = [max(road_feat_scale[:4])]

    # ---------去重算法
    hashv = 1.0

    # 补充采样点坐标
    pad_coords = np.array([-40.0, 0.0])

    densetnt_trans = [
        VectorNetStructuredMapServer(
            map_origin_params=map_origin_params,
            element_keys=all_elements,
            sample_mode=sample_mode_of_element,
            curve_threshold=1.08,
            polyline_seg_len=polyline_seg_len,
            polyline_optional_feats=polyline_optional_feats,
            local_ele_seg_thr=local_ele_seg_thr,
            max_num_ele_seg=max_num_ele_seg,
            image_coordinates=image_coordinates,
            shuffle=False,  # True
            scale=road_feat_scale,
            valid_img_coords_key="valid_img_coords",
        ),
        VectorNetTrajExtractor(
            map_origin_params=map_origin_params,
            source_freq=pkl_fps,
            target_freq=target_freq,
            ego_track_id=-42,
            max_obs_num=max_obs_num,
            local_ele_seg_thr=local_ele_seg_thr,
            image_coordinates=image_coordinates,
            reverse=coords_reverse,
            scale=traj_feat_scale,
            traj_feat_dim=traj_feat_dim,
            num_obs_type=num_obs_type,
            traj_optional_feats=traj_optional_feats,
            valid_track_ids_key="valid_track_ids",
            if_itp_traj=if_itp_traj,
        ),
    ]

    # transform for densetnt
    densetnt_trans += [
        GenObstaclesGoalsV2(
            num_goals=num_goals,
            include_road_ele=include_road_ele,
            road_ele_divide_num=road_ele_divide_num,
            dense_goals_dis=dense_goals_dis,
            use_safe_area=use_safe_area,
            expand_traj_method=expand_traj_method,
            use_his_traj=use_his_traj,
            expand_base=expand_base,
            expand_ratio=expand_ratio,
            edge_divide_num=edge_divide_num,
            vertical_divide_num=vertical_divide_num,
            hashv=hashv,
            goal_interval=goal_interval,
            goal_coords_scale=goal_coords_scale,
            use_diff_trajs=use_diff_trajs,
            pad_coords=pad_coords,
            map_origin_params=map_origin_params,
            element_keys=all_elements,
            image_coordinates=image_coordinates,
        ),
    ]

    return densetnt_trans


def gen_autourban_example_data_loader(
    tdt_dir,
    tdt_yaml_dir,
    img_dir,
    assign_trajs_for_filtered_obs=False,
    enable_high_freq=False,
    enable_behav_trans=False,
):
    data_token2path_mapping = {
        "DG201_20210401": "BEV_data_0624/AutoZGC_v5",
        "DG201_20210401_viz": "BEV_data_0624/AutoZGC_v5",
        "H3165_20220421": [
            "11_perception_prediction/03_pack_package/wenke.wang/TRAJ_DATA",
            "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
        ],
    }

    # Build data loader.
    ds_kwargs = {
        "max_stamp_thr": 550,
        "path_prefix": "DG201_",
        "check_map_exist_func": None,
    }
    if enable_behav_trans:
        ds_kwargs["path_prefix"] = "H3165_"
        dataset = gen_data_loader(
            tdt_dir, tdt_yaml_dir, dataset_name="autourban_navi", **ds_kwargs
        )
    else:
        dataset = gen_data_loader(
            tdt_dir, tdt_yaml_dir, dataset_name="autourban", **ds_kwargs
        )

    (
        basic_trans,
        obstacle_trans,
        map_trans,
        behavior_trans,
    ) = gen_example_transforms(
        data_token2path_mapping,
        img_dir,
        enable_high_freq=enable_high_freq,
        assign_trajs_for_filtered_obs=assign_trajs_for_filtered_obs,
        enable_multi_task=enable_behav_trans,
    )
    all_transforms = basic_trans + obstacle_trans + map_trans
    if enable_behav_trans:
        all_transforms += behavior_trans

    composed_transform = transforms.Compose(all_transforms)
    dataset.datasets[0].transforms = composed_transform
    batch_size = 16
    train_num_workers = 0
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collate_multipath
        if not assign_trajs_for_filtered_obs
        else collate_uniformpath_viz,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_num_workers,
    )
    return dataloader


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_coord_helper():
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

    context_frames = 4
    dataset = gen_data_loader(tdt_dir, tdt_yaml_dir)
    sample = dataset[0]
    df = sample["seq_df"]
    seq_index = sample["seq_index"]
    last_context_frame_id = sample["last_context_frame_id"]

    # Test getting sequence center.
    seq_center = TCH.get_seq_center(
        df, seq_index, "last_context", context_frames, last_context_frame_id
    )
    lctx_df = df[df["frame_id"] == last_context_frame_id]
    assert seq_center.pos_x == lctx_df["pos_x"].values[0]
    assert seq_center.pos_y == lctx_df["pos_y"].values[0]
    assert seq_center.yaw == lctx_df["yaw"].values[0]

    # Test coordinate translation.
    obs_x = df["x"].values
    obs_y = df["y"].values
    global_xy = np.array([obs_x, obs_y]).T
    img_offset = np.array([0, 0])
    # -- phy <-> img
    img_xy = TCH.global_phy_to_local_img(
        global_xy, seq_center, img_offset, 0.1
    )
    global_img_global_xy = TCH.local_img_to_global_phy(
        img_xy, seq_center, img_offset, 0.1
    )
    assert np.max(np.abs(global_xy - global_img_global_xy)) < 1e-4
    # -- phy <-> local
    local_xy = TCH.global_phy_to_local_phy(global_xy, seq_center)
    global_local_global_xy = TCH.local_phy_to_global_phy(local_xy, seq_center)
    assert np.max(np.abs(global_xy - global_local_global_xy)) < 1e-4
    # -- phy <-> centric
    ego_df = df[df["track_id"] == -BaseTrajDataset.ANSWER]
    ego_x = ego_df["x"].values
    ego_y = ego_df["y"].values
    ego_yaw = ego_df[["obs_yaw"]].values
    ego_centeric_center = np.array([seq_center.pos_x, seq_center.pos_y])
    centric_xy = TCH.global_phy_to_agent_centric_phy(
        global_xy, ego_centeric_center, seq_center.yaw
    )
    global_centric_global_xy = TCH.agent_centric_phy_to_global_phy(
        centric_xy, ego_centeric_center, seq_center.yaw
    )
    assert np.max(np.abs(global_xy - global_centric_global_xy)) < 1e-4

    # Test seq_df translation.
    img_df = TCH.global_phy_df_to_local_img_df(df, seq_center, img_offset, 0.1)
    phy_img_phy_df = TCH.local_img_df_to_global_phy_df(
        img_df, seq_center, img_offset, 0.1
    )
    np.max(np.abs(df["pos_x"].values - phy_img_phy_df["pos_x"].values))

    # Test get bounding box corners.
    corners_x, corners_y = TCH.get_bbox_corners(
        ego_x[0], ego_y[0], ego_yaw[0], 4, 1.5
    )
    corners_x_diff = np.max(
        np.abs(corners_x - ego_df[["x0", "x1", "x2", "x3"]].values[0])
    )
    corners_y_diff = np.max(
        np.abs(corners_y - ego_df[["y0", "y1", "y2", "y3"]].values[0])
    )
    assert corners_x_diff < 1e-4
    assert corners_y_diff < 1e-4
