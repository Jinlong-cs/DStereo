import os

import pytest

from hat.data.datasets.elevation_dataset import (
    Elevation,
    ElevationFromImage,
    ElevationRec,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_elevation():
    bucket_path = HAT_BUCKET_PATH
    root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/AutoZGC_v3_test/",  # noqa
    )
    ann_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/DHO/AutoZGC_v3/",  # noqa
    )
    list_file = "4GD36_20210715_D/20210715-134733_121.txt"
    cfg_file = "4GD36_20210715_D/4GD36_20210715_D_part.txt"
    source = "FRONT_camera_infos_oft.yaml"
    load_all_cfg = True
    img_shape = (1080, 1920)
    gt_source = "img"
    load_data_types = [
        "mask",
        "obj_mask",
        "ground_mask",
        "gt_gamma",
        "gt_height",
        "gt_depth",
        "intrinsics",
        "ground_norm",
        "ground_homo",
        "camera_high",
        "timestamp",
        "rotation",
        "transition",
    ]
    dataset = Elevation(
        root=root,
        ann_root=ann_root,
        list_file=list_file,
        cfg_file=cfg_file,
        transforms=None,
        source=source,
        gt_source=gt_source,
        load_data_types=load_data_types,
        img_shape=img_shape,
        load_all_cfg=load_all_cfg,
    )
    for i in range(0):
        data = dataset[i]
        assert "gt_depth" in data
        assert "gt_height" in data
        assert "gt_gamma" in data
        assert "obj_mask" in data
        assert "ground_mask" in data
        assert "mask" in data
        assert "timestamp" in data
        assert "intrinsics" in data
        assert "ground_norm" in data
        assert "camera_high" in data
        assert "ground_homo" in data
        assert "rotation" in data
        assert "transition" in data


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
@pytest.mark.parametrize(
    ["img_rec_path", "seg_rec_path", "gt_rec_path"],
    [
        pytest.param(
            os.path.join(
                HAT_BUCKET_PATH,
                "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/DHO/lists/tmp/for_train/version_2/rec_files/train_image.rec",  # noqa
            ),
            os.path.join(
                HAT_BUCKET_PATH,
                "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/DHO/lists/tmp/for_train/version_2/rec_files/train_seg.rec",  # noqa
            ),
            os.path.join(
                HAT_BUCKET_PATH,
                "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/DHO/lists/tmp/for_train/version_2/rec_files/train_dense_ele.rec",  # noqa
            ),
        ),
        pytest.param(None, None, None),
    ],
)
def test_elevation_rec(img_rec_path, seg_rec_path, gt_rec_path):
    bucket_path = HAT_BUCKET_PATH
    root = os.path.join(bucket_path, "unit_test_data/J5FSD/FSD_Urban_v4")
    ann_root = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/DHO/lists/tmp/for_train/version_2/",  # noqa
    )
    list_file = "train.txt"
    ele_lmdb_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/DHO/lists/tmp/for_train/version_2/train_dense_ground",  # noqa
    )
    intrinsics_lmdb_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/DHO/lists/tmp/for_train/version_2/train_dense_ground_intri",  # noqa
    )
    pack2idx_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/jiafeng.xie/ParallaxNet/DHO/lists/tmp/for_train/version_2/train.yaml",  # noqa
    )
    img_shape = (2160, 3840, 3)
    gt_shape = (1080, 1920, 3)
    load_data_types = [
        "mask",
        "obj_mask",
        "ground_mask",
        "gt_gamma",
        "gt_height",
        "gt_depth",
        "intrinsics",
        "ground_norm",
        "ground_homo",
        "camera_high",
        "timestamp",
        "rotation",
        "transition",
    ]
    dataset = ElevationRec(
        root=root,
        ann_root=ann_root,
        list_file=list_file,
        img_rec_path=img_rec_path,
        seg_rec_path=seg_rec_path,
        gt_rec_path=gt_rec_path,
        pack2idx_path=pack2idx_path,
        ele_lmdb_path=ele_lmdb_path,
        intrinsics_lmdb_path=intrinsics_lmdb_path,
        transforms=None,
        load_data_types=load_data_types,
        img_shape=img_shape,
        gt_shape=gt_shape,
    )
    for i in range(5):
        data = dataset[i]
        assert "gt_depth" in data
        assert "gt_height" in data
        assert "gt_gamma" in data
        assert "obj_mask" in data
        assert "ground_mask" in data
        assert "mask" in data
        assert "timestamp" in data
        assert "intrinsics" in data
        assert "ground_norm" in data
        assert "camera_high" in data
        assert "ground_homo" in data
        assert "rotation" in data
        assert "transition" in data


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_ele_from_img():
    bucket_path = HAT_BUCKET_PATH
    test_image_dir = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/FSD_Urban_v6/DG201_20210710_D/20210710-110501_523/camera_front",  # noqa
    )
    camera_view_names = ["camera_front"]
    att_json_path = "attribute.json"
    resize_size = (540, 960)
    dataset = ElevationFromImage(
        data_path=test_image_dir,
        camera_view_names=camera_view_names,
        att_json_path=att_json_path,
        img_load_size=resize_size[::-1],
    )
    for i in range(5):
        data = dataset[i]
        assert "pil_imgs" in data
        assert "img_name" in data
