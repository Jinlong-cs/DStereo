import os
import random
import shutil

import numpy as np
import pytest
import timeout_decorator

from hat.data.datasets.auto_3dv import (
    ANCAuto3DV,
    ANCBev3DDatasetRec,
    HDELoader,
    NamedIndexDatasetV2,
    generate_sync_info,
)
from hat.data.transforms.auto_3dv import ANCCollect3DV, ANCConvertReal3dTo3DV
from hat.registry import build_from_registry
from tests import (
    HAT_BUCKET_EXISTS,
    HAT_BUCKET_PATH,
    SD_AlGORITHM_BUCKET_EXISTS,
)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_named_index_dataset_v2():
    bucket_path = HAT_BUCKET_PATH
    rec_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/wenming.meng/NamedIndexDatasetV2/pipeline_test_front_img.rec",  # noqa
    )
    dataset = NamedIndexDatasetV2(
        imgrec_path_list=[rec_path], read_by_name=True
    )

    img = dataset[
        "DG201_20210401_D/20210401-150149_735/camera_front/1617260548614.jpg"
    ]
    assert len(dataset.lst_lmdb) == 1
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_auto3dv_from_image():
    bucket_path = HAT_BUCKET_PATH
    camera_view_names = [
        "camera_front",
        "camera_front_left",
        "camera_front_right",
        "camera_rear_left",
        "camera_rear_right",
        "camera_rear",
    ]
    # NOTE: per-view's shape[h,w] should correspond to sub_dirs

    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }
    intrinsics = np.random.randint(0, 255, (3, 3), dtype="uint8").astype(
        "float32"
    )
    distortcoef = np.random.randint(0, 255, 4, dtype="uint8").astype("float32")
    config = dict(
        type="ANCAuto3DVFromImage",
        intrinsics=intrinsics,
        distortcoef=distortcoef,
        per_view_shape=per_view_shape,
        camera_view_names=camera_view_names,
        img_load_size=(960, 540),
        data_path=os.path.join(
            bucket_path,
            "unit_test_data/J5FSD/AutoZGC_v5/DG201_20210311_D/20210311-153028_183/camera_front",  # noqa
        ),
        transforms=[
            dict(
                type="ANCResize3DV",
                size=(540, 960),
            ),
            dict(
                type="ANCCrop3DV",
                height=512,
                width=960,
                top=None,
                left=None,  # random crop
            ),
            dict(type="ANCToTensor3DV", with_color_imgs=False),
        ],
    )

    dataset = build_from_registry(config)
    random_idx = random.randint(0, len(dataset) - 1)
    data = dataset.__getitem__(random_idx)

    # assert 'timestamp' in data
    assert "imgs" in data


@timeout_decorator.timeout(60)
def _copy_file(src, dst):
    shutil.copyfile(src, dst)


@pytest.mark.skipif(True, reason="miss test calib path!")
def test_Bev3DDatasetRec():
    bucket_path = HAT_BUCKET_PATH

    img_rec_path_part = "unit_test_data/J5FSD/detection_3D/4GD36/v05/20210702/"
    per_view_recs = [
        "data_4GD36_20210702_v05__no_front__.rec",
        "data_4GD36_20210702_v05__front_0820__.rec",
    ]
    img_rec_path = [
        os.path.join(bucket_path, img_rec_path_part, per_view_rec)
        for per_view_rec in per_view_recs
    ]
    homo_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/tools/split_jiawei02/homo_path/30C1G/30C1G_20210423_20210527/",  # noqa
    )

    category_id_dict = {
        1: -99,  # Pedestrian -> ignore
        2: 0,  # Car        -> Car
        3: -99,  # Cyclist    -> ignore
        4: 0,  # Bus        -> Car
        5: 0,  # Truck      -> Car
        6: 0,  # SpecialCar -> Car
        7: 0,  # Blur       -> Car
        8: -99,  # Other    -> ignore
    }

    collect = ANCConvertReal3dTo3DV(category_id_dict=category_id_dict)

    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }

    sub_dirs = [
        "front",
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
        "rear",
    ]
    homo_transforms = {
        "camera_front": {
            "Resize": (540, 960),
            "Crop": (0, 0, 540, 960),
            "ResizeHomo": 0.25,
        },
        "camera_front_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
            "ResizeHomo": 0.25,
        },
        "camera_front_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
            "ResizeHomo": 0.25,
        },
        "camera_rear_left": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
            "ResizeHomo": 0.25,
        },
        "camera_rear_right": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
            "ResizeHomo": 0.25,
        },
        "camera_rear": {
            "Resize": (640, 1024),
            "Crop": (0, 0, 640, 1024),
            "ResizeHomo": 0.25,
        },
    }
    dataset = ANCBev3DDatasetRec(
        img_data_path=img_rec_path,
        num_classes=3,
        per_view_shape=per_view_shape,
        sub_dirs=sub_dirs,
        views=6,
        transforms=collect,
        homo_gen=dict(
            homo_path=homo_path,
            calib_path=None,
            spatial_resolution=(0.2, 0.2),
            vcs_range=(-30.0, -51.2, 72.4, 51.2),
            camera_view_names=per_view_shape.keys(),
            task_camera_view_names=per_view_shape.keys(),
            per_view_shape=per_view_shape,
            homo_transforms=homo_transforms,
        ),
    )
    data = dataset[0]
    assert "homography" in data
    assert "image_name" in data
    assert "image_id" in data
    assert "gt_bev_3d" in data
    assert "timestamp" in data


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_auto3dv():
    bucket_path = HAT_BUCKET_PATH

    sync_file = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/xiangyu.li/AutoZGC_v5/bev3d/refactor_test.txt",  # noqa
    )
    sync_file_lmdb = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/xiangyu.li/AutoZGC_v5/bev3d/refactor_test",  # noqa
    )
    bev_3d_lmdb_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/xiangyu.li/AutoZGC_v5/bev3d/bev3d_lmdb",  # noqa
    )

    root = os.path.join(bucket_path, "unit_test_data/J5FSD/AutoZGC_v5")
    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }
    collect = ANCCollect3DV(
        load_data_types=[
            "gt_depth",
            "gt_bev_seg",
            "timestamp",
            "obj_mask",
            "pack_dir",
            "img_paths",
        ],
        img_idxs=[0, 1],
    )

    # from sync file
    dataset = ANCAuto3DV(
        root,
        camera_view_names=["camera_front"],
        per_view_shape=per_view_shape,
        sync_file=sync_file,
        transforms=collect,
    )
    data = dataset[0]
    assert "pil_imgs" in data
    assert "gt_bev_seg" in data
    assert "timestamp" in data
    assert "obj_mask" in data

    # from sync file lmdb
    dataset = ANCAuto3DV(
        root,
        camera_view_names=["camera_front"],
        per_view_shape=per_view_shape,
        sync_file=sync_file,
        sync_file_lmdb=sync_file_lmdb,
        bev_3d_lmdb_path=bev_3d_lmdb_path,
        transforms=collect,
    )
    data = dataset[0]
    assert "pil_imgs" in data
    assert "gt_bev_seg" in data
    assert "timestamp" in data
    assert "obj_mask" in data
    assert "pack_dir" in data
    assert "img_paths" in data

    dataset_multi_sync_file = ANCAuto3DV(
        root,
        camera_view_names=["camera_front"],
        per_view_shape=per_view_shape,
        sample_interval=[1, 1],
        sync_file_lmdb=[sync_file_lmdb, sync_file_lmdb],
        bev_3d_lmdb_path=bev_3d_lmdb_path,
        transforms=collect,
    )
    assert len(dataset_multi_sync_file) == 2 * len(dataset)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_generate_sync_info(tmpdir):
    bucket_path = HAT_BUCKET_PATH
    pack_list_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/wenming.meng/AutoZGC_v5/bev/test_pack_list.txt",  # noqa
    )
    dataset_root = os.path.join(bucket_path, "unit_test_data/J5FSD/AutoZGC_v5")
    sync_info_path = os.path.join(tmpdir, "test.txt")
    lmbd_path = os.path.join(tmpdir, "test")
    generate_sync_info(
        pack_list_path,
        dataset_root,
        sync_info_path,
        [0, -1, -2],
        100,
        True,
        lmbd_path,
    )


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HDLTAlgorithm bucket")
def test_discobj_auto3dv():
    bucket_path = HAT_BUCKET_PATH

    unit_test_data = (
        "unit_test_data/J5FSD/users/ben.hu/hde_discobj_uttest_data"
    )
    sync_file = os.path.join(
        bucket_path,
        unit_test_data,
        "train_FSD_Site_H3165_20220220_selected_list_refactor.txt",
    )
    sync_file_lmdb = os.path.join(
        bucket_path,
        unit_test_data,
        "train_FSD_Site_H3165_20220220_selected_list_refactor",
    )
    bev_discrete_obj_lmdb_path = os.path.join(
        bucket_path, unit_test_data, "train_FSD_Site_H3165_20220220_bevmap"
    )
    seg_data_path = os.path.join(
        bucket_path,
        unit_test_data,
        "train_FSD_Site_H3165_20220220_segmap_6v.rec",
    )

    per_view_recs = [
        "train_FSD_Site_H3165_20220220_5v0223.rec",
        "train_FSD_Site_H3165_20220220_front0820.rec",
    ]
    img_data_path = [
        os.path.join(bucket_path, unit_test_data, per_view_rec)
        for per_view_rec in per_view_recs
    ]
    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }
    collect = ANCCollect3DV(
        load_data_types=[
            "gt_seg",
            "gt_bev_discrete_obj",
            "timestamp",
            "img_paths",
        ],
        img_idxs=[0],
    )

    # from sync file lmdb
    dataset = ANCAuto3DV(
        root=None,
        camera_view_names=["camera_front"],
        per_view_shape=per_view_shape,
        img_data_path=img_data_path,
        sync_file=sync_file,
        sync_file_lmdb=sync_file_lmdb,
        seg_data_path=seg_data_path,
        bev_discrete_obj_lmdb_path=bev_discrete_obj_lmdb_path,
        transforms=collect,
    )
    data = dataset[0]
    assert "pil_imgs" in data
    assert "gt_bev_discrete_raw" in data
    assert "timestamp" in data
    assert "img_paths" in data


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="need SD_Algorithm bucket"
)
def test_HDELoader():

    site_path = "/horizon-bucket/SD_Algorithm/12_perception_bev_hde/03_pack_package/site_data/"  # noqa
    loader = HDELoader(site_path)

    key = "FSD_Site_UT263_20220924/Site_117_09908_40_13547_0/UT263_20220924_113435/camera_front/1663990611301"  # noqa

    loader.check_liosam_pose(key)
    liosam_pose = loader.load_liosam_pose(key)

    assert isinstance(liosam_pose, np.ndarray)

    loader.check_wheel_pose(key)
    wheel_pose = loader.load_wheel_pose(key)
    assert isinstance(wheel_pose, np.ndarray)

    # test load park img
    park_path = "/horizon-bucket/SD_Algorithm/12_perception_bev_hde/03_pack_package/park_data/"  # noqa
    loader = HDELoader(park_path)
    key = "UT0Q9_20230215_D/20230215-122546_955/camera_front_right/1676435147306.jpg"  # noqa
    assert loader.check_img(key)
    img = loader.load_img(key)
    # or split key
    assert loader.check_img(
        "UT0Q9_20230215_D/20230215-122546_955",
        "camera_front_right/1676435147306.jpg",  # noqa
    )
    img = loader.load_img(
        "UT0Q9_20230215_D/20230215-122546_955",
        "camera_front_right/1676435147306.jpg",  # noqa
    )
    assert isinstance(img, np.ndarray)

    # test load site img
    site_path = "/horizon-bucket/SD_Algorithm/12_perception_bev_hde/03_pack_package/site_data/"  # noqa
    loader = HDELoader(site_path)
    key = "FSD_Site_NC109_20221212/Site_121_34973_31_45059_1/NC109_20221212_182822/camera_front/1670840902300.jpg"  # noqa
    assert loader.check_img(key)
    img = loader.load_img(key)

    # or split key
    key = "Site_121_34973_31_45059_1/NC109_20221212_182822/camera_front/1670840902300.jpg"  # noqa
    assert loader.check_img("FSD_Site_NC109_20221212", key)
    img = loader.load_img("FSD_Site_NC109_20221212", key)

    assert isinstance(img, np.ndarray)
