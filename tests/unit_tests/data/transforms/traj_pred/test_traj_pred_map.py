# Copyright (c) Horizon Robotics. All rights reserved.
# The unit test for trajectory prediction transforms about maps.

import os
import pickle
import sys
from distutils.version import LooseVersion

import numpy as np
import pandas
import pytest

from hat.data.datasets.traj_pred_dataset import (  # noqa: E501
    PickledTdtDatasetV2,
)
from hat.data.transforms.traj_pred.traj_pred_coords import (
    GenBoundingBox,
    GenSeqCenter,
    PhyToImage,
)
from hat.data.transforms.traj_pred.traj_pred_map import (
    GetBEVHomography,
    GetBEVLocalMapByTimestamp,
    GetBevVectorizedMap,
    NuScenesRenderedMap,
    OccupancyMapRender,
    VectorNetStructuredMapServer,
)
from hat.data.transforms.traj_pred.traj_pred_obstacle import (
    GetLcfTimeStamp,
    GetSeqDataFrameMask,
)
from projects.prediction.configs.processed_dataset import (
    HAT_UNITTEST_PREFIX,
    NUSCENES_DATASET,
    TRAJ_BEHAV_SD_ROOT,
)
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH
from tests.unit_tests.core.test_traj_pred_utils import (  # noqa: E501
    gen_data_loader,
    gen_example_transforms,
)

pandas_version = pandas.__version__
PANDAS_VERSION_MATCH = LooseVersion(pandas_version) == LooseVersion("1.1.0")


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS,
    reason="requiring SD_Algorithm bucket",
)
@pytest.mark.skipif(not PANDAS_VERSION_MATCH, reason="requiring pandas==1.1.0")
def test_get_vec_road_map():
    pickle_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{HAT_UNITTEST_PREFIX}/datasets/PickledTdtDatasetV2",
        "val_583.pkl",  # noqa: E501
    )

    dataset_pkl = PickledTdtDatasetV2(
        pkl_path=pickle_dir,
        transforms=None,
    )
    for ds in dataset_pkl.loaded_ds.datasets:
        ds.navi_save_mode = "local"
        ds.use_struct_road_info = False
    sample = dataset_pkl[0]
    assert "struct_road" in sample
    size = 512
    expand_size = 3
    bev_origin_x = 72.4
    bev_origin_y = 51.2
    img_resolution = 0.2
    get_bev = GetBevVectorizedMap(
        size=size,
        expand_size=expand_size,
        bev_origin_x=bev_origin_x,
        bev_origin_y=bev_origin_y,
        img_resolution=img_resolution,
    )
    sample = get_bev(sample)
    assert "road_map" in sample
    image = sample["road_map"]

    assert np.max(image) <= 255
    assert np.min(image) >= 0


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_occupancy_map_render():
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

    # Prepare necessary items for occupancy map renderer.
    img_h, img_w, img_resolution = 512, 512, 0.2
    context_frames = 4
    all_transforms = [
        GenSeqCenter(
            anchor_frame="last_context",
            context_frames=context_frames,
            augmentation=False,
        ),
        GenBoundingBox(
            gen_ego_bbox=False, clockwise=True, min_shape=1, const_z_value=1.5
        ),
        PhyToImage(
            map_height=img_h, map_width=img_w, resolution=img_resolution
        ),
        GetSeqDataFrameMask(
            freq_ratio=1,
        ),
    ]
    for trans in all_transforms:
        sample = trans(sample)

    # Render occupancy maps.
    og_renderer = OccupancyMapRender(
        map_height=img_h,
        map_width=img_w,
        render_ego=True,
    )
    sample = og_renderer(sample)
    assert "rendered_obs" in sample
    og = sample["rendered_obs"]
    assert og.shape == (img_h, img_w, context_frames)

    # Check whether the obstacles are rendered.
    df = sample["seq_df"]
    df_vals = df.values
    df_cols = df.columns
    track_id_col = df_cols.get_loc("track_id")
    img_x_col = df_cols.get_loc("img_x")
    img_y_col = df_cols.get_loc("img_y")
    lcf_mask = df["frame_id"] == sample["last_context_frame_id"]
    all_track_ids = np.unique(df_vals[lcf_mask, track_id_col])
    for t_id in all_track_ids:
        track_mask = df["track_id"] == t_id
        lcf_track_mask = lcf_mask & track_mask
        lcf_img_x = int(df_vals[lcf_track_mask, img_x_col])
        lcf_img_y = int(df_vals[lcf_track_mask, img_y_col])
        if (
            lcf_img_x >= 0
            and lcf_img_x < img_h
            and lcf_img_y >= 0
            and lcf_img_y < img_w
        ):
            assert og[lcf_img_x, lcf_img_y, -1] == 255


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_get_bev_map():
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
    ds_kwargs = {
        "max_stamp_thr": 550,
        "path_prefix": "DG201_",
        "check_map_exist_func": None,
    }
    dataset = gen_data_loader(
        tdt_dir, tdt_yaml_dir, dataset_name="autourban", **ds_kwargs
    )
    sample = dataset[0]
    # Prepare necessary keys.
    get_lcf_stamp = GetLcfTimeStamp()
    sample = get_lcf_stamp(sample)

    # Get local map.
    img_h, img_w = 512, 512
    img_dir = os.path.join(SD_AlGORITHM_BUCKET_PATH, "mengyuan")
    data_token2path_mapping = {
        "DG201_20210401": "BEV_data_0624/AutoZGC_v5",
        "DG201_20210401_viz": "BEV_data_0624/AutoZGC_v5",
    }
    get_bev = GetBEVLocalMapByTimestamp(
        image_dir=img_dir,
        image_suffix=".png",
        map_height=img_h,
        map_width=img_w,
        data_token2path_mapping=data_token2path_mapping,
        is_viz=False,
    )
    sample = get_bev(sample)
    assert "road_map" in sample
    image = sample["road_map"]
    # The number of the road_map is the label of diffferent road
    # elements. 0~5 respectively means: 'others', 'roadedge', 'roadarrow',
    # 'solid_lanes', 'stopline, 'crosswalk'
    assert np.max(image) <= 5
    assert np.min(image) >= 0


def test_get_homography():
    img_h, img_w = 512, 512
    src_shape, dst_shape = 16, 16
    bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
    x_ratio = bev_origin_x / img_resolution / img_h
    y_ratio = bev_origin_y / img_resolution / img_w
    get_homography = GetBEVHomography(
        src_shape,
        dst_shape,
        x_origin_ratio=x_ratio,
        y_origin_ratio=y_ratio,
        roi_spatial_scale=int(img_h / src_shape),
    )

    sample = {
        "valid_img_coords": [
            np.array([256.0, 256.0, 0.0]),
            np.array([362.0, 256.0, 0.0]),
            np.array([454.515, 314.94, 1.565]),
        ]
    }
    sample = get_homography(sample)
    assert "img_homographys" in sample
    img_homo = sample["img_homographys"]
    assert len(img_homo) == len(sample["valid_img_coords"])


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_get_nuscenes_rendered_map():
    nusc_map_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{NUSCENES_DATASET}/maps/nusc_cnn_map/0/",
    )
    offline_server = NuScenesRenderedMap(
        height=512,
        width=512,
        query_resolution=0.2,
        map_path=nusc_map_dir,
        base_resolution=0.1,
        pre_load_map=True,
        basemap_suffix=".png",
    )
    online_server = NuScenesRenderedMap(
        height=512,
        width=512,
        query_resolution=0.2,
        map_path=nusc_map_dir,
        base_resolution=0.1,
        pre_load_map=False,
        basemap_suffix=".png",
    )
    offline_map = offline_server.get(520, 1170, 0)
    online_map = online_server.get(520, 1170, 0)
    assert np.mean(np.abs((offline_map - online_map))) < 0.01


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS or "3.8" in sys.version,
    reason="requiring SD_Algorithm bucket",
)
def test_get_vectornet_structual_map():
    pkl_file = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/all_scene_dataset/test_84322_all_scene_50ctxframes_dist_restriction.pkl",  # noqa: E501
    )
    with open(pkl_file, "rb") as f:
        dataset = pickle.load(f)

    sample = dataset[0]
    basic_trans, obstacle_trans, _, _ = gen_example_transforms()
    all_transforms = basic_trans + obstacle_trans
    for trans in all_transforms:
        sample = trans(sample)

    map_origin_params = [72.4, 51.2, 0.2]
    polyline_seg_len = 10
    max_num_ele_seg = 256
    all_elements = ["solid_lane", "virtuallanelines", "roadedge", "crosswalk"]
    polyline_optional_feats = [
        "turn_dir",
        "pid",
        "pid_pred_succ",
        "pre_pre_point",
    ]
    sample_mode_of_element = {
        "stopline": "uniform",
        "crosswalk": "uniform",
        "solid_lane": "original",
        "roadedge": "original",
        "virtuallanelines": "uniform",
    }
    map_server = VectorNetStructuredMapServer(
        map_origin_params=map_origin_params,
        element_keys=all_elements,
        sample_mode=sample_mode_of_element,
        curve_threshold=0.5,
        polyline_seg_len=polyline_seg_len,
        polyline_optional_feats=polyline_optional_feats,
        local_ele_seg_thr=100,  # m
        max_num_ele_seg=max_num_ele_seg,
        image_coordinates="bev",
        shuffle=True,
    )
    sample = map_server(sample)

    assert "struct_road_feats" in sample
    road_ele_feats = sample["struct_road_feats"]
    num_obs = len(sample["valid_track_ids"])
    assert road_ele_feats.shape[:-1] == (
        num_obs,
        max_num_ele_seg,
        polyline_seg_len - 1,
    )
