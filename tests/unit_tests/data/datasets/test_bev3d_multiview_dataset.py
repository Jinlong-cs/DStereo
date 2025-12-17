import json
import os

import pytest

from hat.data.datasets.bev3d_multiview_dataset import (
    MultiViewImgDataset,
    MultiViewRecDataset,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH  # noqa


def get_common_data(prefix="camera_"):
    camera_view_names = [
        "camera_front_left",
        "camera_front_right",
        "camera_rear_left",
        "camera_rear_right",
        "camera_rear",
    ]
    view_names = [name.replace("camera_", "") for name in camera_view_names]
    camera_view_names = [
        name.replace("camera_", prefix) for name in camera_view_names
    ]
    per_view_shape = {
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
    }
    per_view_shape = {
        view.replace("camera_", prefix): shape
        for view, shape in per_view_shape.items()
    }
    vcs_range = (-70.0, -50.0, 30.0, 50.0)
    ipm_output_size = (256, 256)  # (height, witdh)
    spatial_resolution = (
        abs(vcs_range[2] - vcs_range[0]) / ipm_output_size[0],
        abs(vcs_range[3] - vcs_range[1]) / ipm_output_size[1],
    )  # (height, witdh)
    homo_cfg = dict(
        homo_path=None,
        calib_path=None,
        spatial_resolution=spatial_resolution,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        norm_homo=True,
        use_distorted_offset=True,
        homo_transforms={
            view: {"Resize": (640, 960)} for view in camera_view_names
        },
    )
    return view_names, per_view_shape, homo_cfg


@pytest.mark.skipif(True, reason="need update")
def test_multiview_rec_dataset():
    bucket_path = HAT_BUCKET_PATH
    print(bucket_path)
    rec_path = os.path.join(
        bucket_path,
        "users/jianglei.huang/multiview_dataset/multiview_test.rec",
    )
    dataset = MultiViewRecDataset(rec_path, rec_path + ".idx")
    imgs_meta = dataset[2]
    anno = imgs_meta["meta"]
    assert "timestamp" in anno
    assert "img_orders" in anno
    assert "objects" in anno

    imgs = imgs_meta["imgs"]
    assert len(imgs) == len(anno["img_orders"])
    assert len(imgs[0].shape) == 3

    # test with homo_cfg
    view_names, per_view_shape, homo_cfg = get_common_data()
    dataset = MultiViewRecDataset(
        rec_path,
        rec_path + ".idx",
        camera_view_names=view_names,
        view_shapes=per_view_shape,
        homo_cfg=homo_cfg,
    )
    imgs_meta = dataset[2]
    assert "homography" in imgs_meta
    assert "homo_offset" in imgs_meta


@pytest.mark.skipif(True, reason="need update")
@pytest.mark.parametrize("raw_img_mode", [True, False])
def test_multiview_img_dataset(raw_img_mode):
    bucket_path = HAT_BUCKET_PATH
    data_root = os.path.join(
        bucket_path, "users/yilin.xiong/data/example/bev_det"
    )
    img_dir = os.path.join(data_root, "data")
    anno_json_file = os.path.join(data_root, "mini_gt.json")
    calib_path = os.path.join(data_root, "calibration")

    view_names, per_view_shape, homo_cfg = get_common_data(prefix="")

    dataset = MultiViewImgDataset(
        img_dir,
        camera_list=view_names,
        view_shapes=per_view_shape,
        anno_json_file=anno_json_file if not raw_img_mode else None,
        calib_path=calib_path if raw_img_mode else None,
        homo_cfg=homo_cfg,
        nums_to_read=5,
    )
    imgs_meta = dataset[0]
    meta = json.loads(imgs_meta["meta"])
    assert "timestamp" in meta
    assert "img_orders" in meta
    assert "view_anno" in meta
    assert "calib" in meta["view_anno"]["rear"]["meta"]
    assert "homography" in imgs_meta
    assert "homo_offset" in imgs_meta

    for i in range(len(view_names)):
        assert view_names[i] == meta["img_orders"][i]
