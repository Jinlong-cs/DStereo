import copy
import os
import random

import numpy as np
import pytest
import torch
import torchvision

from hat.data.datasets.bev3d_multiview_dataset import MultiViewRecDataset
from hat.data.transforms.mvt4d_transforms import (
    BBoxRotation,
    MultiScaleDepthMapGenerator,
    ResizeCropFlipImage,
    Sparse4DAdaptor,
)
from hat.registry import OBJECT_REGISTRY, build_from_cfg

try:
    import mmcv

    _MMCV_IMPORTED = True
except ImportError:
    _MMCV_IMPORTED = False
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH
from tests.utils import check, check_shape, check_type


def get_common_data():
    camera_view_names = [
        "camera_front_left",
        "camera_front_right",
        "camera_rear_left",
        "camera_rear_right",
        "camera_rear",
    ]
    view_names = [name.replace("camera_", "") for name in camera_view_names]
    per_view_shape = {
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
    }

    # category_id_map
    category_id_dict = {
        1: 0,  # Pedestrian
        2: 1,  # Car        -> Car
        3: 2,  # Cyclist
        4: 1,  # Bus        -> Car
        5: 1,  # Truck      -> Car
        6: 1,  # SpecialCar -> Car
        7: 1,  # Blur       -> Car
        8: -99,  # Other    -> ignore
    }  # 'Dontcare' -> Ignore

    # normalize
    img_norm_cfg = dict(
        mean=[128.0, 128.0, 128.0], std=[128.0, 128.0, 128.0], to_rgb=False
    )

    point_cloud_range = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]

    # resize crop
    resize_target_dim = (640, 960)

    # pad output
    pad_output_dim = (640, 960)

    transforms_ = [
        dict(
            type="GetCalibParams",
            camera_view_names=view_names,
            view_shapes=per_view_shape,
        ),
        dict(
            type="MultiViewRecPadView",
            camera_view_names=view_names,
            view_shapes=per_view_shape,
        ),
        dict(
            type="MultiViewRecTransform",
            category_id_dict=category_id_dict,
            camera_view_names=view_names,
        ),
        dict(
            type="MultiViewFlipResizeCrop",
            resize_target_dim=resize_target_dim,
            horizontal_flip_ratio=0.0,
            keep_ratio=True,
        ),
        dict(
            type="MultiViewNormalize",
            img_norm_cfg=img_norm_cfg,
        ),
        dict(
            type="MultiViewPadImage",
            size=pad_output_dim,
        ),
        dict(
            type="MVT4DImgFormat",
        ),
        dict(
            type="MultiViewRangeFliter",
            point_cloud_range=point_cloud_range,
        ),
        dict(
            type="MultiViewCollect3D",
            keep_keys=("img", "gt_labels_3d", "gt_bboxes_3d", "img_metas"),
            img_metas_keys=(
                "img_shape",
                "T_vcs2img",
                "timestamp",
                "views_pad",
            ),
        ),
    ]

    return view_names, per_view_shape, transforms_


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
@pytest.mark.skipif(not _MMCV_IMPORTED, reason="need mmcv is imported")
@pytest.mark.skipif(True, reason="need update")
def test_multiview_rec_transforms():

    assert mmcv.__version__

    bucket_path = HAT_BUCKET_PATH
    print(bucket_path)
    rec_path = os.path.join(
        bucket_path,
        "users/jianglei.huang/multiview_dataset/multiview_test.rec",
    )

    view_names, per_view_shape, transforms = get_common_data()
    transforms_ = []
    for trans in transforms:
        transforms_.append(
            build_from_cfg(OBJECT_REGISTRY, trans),
        )
    transforms_ = torchvision.transforms.Compose(transforms_)  # noqa
    dataset = MultiViewRecDataset(
        rec_path=rec_path,
        rec_idx_file=rec_path + ".idx",
        camera_view_names=view_names,
        view_shapes=per_view_shape,
        to_rgb=True,
        decode_img=True,
        transforms=transforms_,
        homo_cfg=None,
    )
    data = dataset[2]

    check(data["img"], check_type, instance=torch.Tensor)
    check(data["img"], check_shape, shape=(5, 3, 640, 960))
    check(data["gt_labels_3d"], check_type, instance=torch.Tensor)
    check(data["gt_labels_3d"], check_shape, shape=(300,))
    check(data["gt_bboxes_3d"], check_type, instance=torch.Tensor)
    check(data["gt_bboxes_3d"], check_shape, shape=(300, 7))
    check(data["img_metas"]["img_shape"], check_type, instance=np.ndarray)
    assert np.array_equal(
        data["img_metas"]["img_shape"],
        np.array(
            [
                [640, 960],
            ]
            * 5
        ),
    )
    check(data["img_metas"]["T_vcs2img"], check_type, instance=np.ndarray)
    check(data["img_metas"]["T_vcs2img"], check_shape, shape=(5, 3, 4))
    check(data["img_metas"]["timestamp"], check_type, instance=np.float64)


def test_multi_scale_depth_map_generator():
    depth_map_generator = MultiScaleDepthMapGenerator(
        downsample=[1, 2], max_depth=60, project_key="lidar2img"
    )
    num_cams = 6
    data = dict(
        points=np.arange(100).reshape(20, 5),
        img_shape=[(20, 30)] * num_cams,
        lidar2img=[np.random.uniform(size=(4, 4)) for _ in range(num_cams)],
    )
    output = depth_map_generator(data)
    num_scale = len(depth_map_generator.downsample)
    assert num_scale == len(output["gt_depth"])
    for i, stride in enumerate(depth_map_generator.downsample):
        assert output["gt_depth"][i].shape == (
            num_cams,
            data["img_shape"][i][0] // stride,
            data["img_shape"][i][1] // stride,
        )


def test_resize_crop_flip_image():
    data_aug_configs = dict(
        output_img_hw=(30, 20),
        origin_img_hw=(60, 40),
        resize_range=(0.4, 0.6),
        rotation_range=(10.0, 10.0),
        rand_flip=True,
        crop_vertical_range=(0.0, 1.0),
    )
    trans = ResizeCropFlipImage(
        transform_matrix_key=("lidar2img", "cam_intrinsic"),
        data_aug_configs=data_aug_configs,
        test_mode=False,
    )
    num_cams = 6
    input = dict(
        imgs=[
            np.ones(shape=(*data_aug_configs["origin_img_hw"], 3))
            for _ in range(num_cams)
        ],
        lidar2img=[np.random.uniform(size=(4, 4)) for _ in range(num_cams)],
        cam_intrinsic=[
            np.random.uniform(size=(3, 3)) for _ in range(num_cams)
        ],
    )
    output = trans(input)
    assert (
        output["img_shape"] == [data_aug_configs["output_img_hw"]] * num_cams
    )


def test_bbox_rotation():
    rotation = BBoxRotation(
        transform_matrix_key=("lidar2img",),
        global_key=("lidar2global",),
        rotation_3d_range=None,
    )
    num_box = 10
    num_cams = 6
    for state_dims in (7, 9, 10):
        angle = random.random() * 10
        lidar2img = [np.random.uniform(size=(4, 4)) for _ in range(num_cams)]
        lidar2global = np.random.uniform(size=(4, 4))
        data = dict(
            gt_bboxes_3d=np.random.uniform(size=(num_box, state_dims)),
            lidar2img=lidar2img,
            lidar2global=lidar2global,
            aug_configs=dict(rotate_3d=angle),
        )
        output_1 = rotation(copy.deepcopy(data))
        output_2 = copy.deepcopy(output_1)
        output_2["aug_configs"]["rotate_3d"] *= -1
        output_2 = rotation(output_2)
        for i in range(num_cams):
            assert np.all(
                data["lidar2img"][i] - output_2["lidar2img"][i] < 1e-4
            )
        assert np.all(data["lidar2global"] - output_2["lidar2global"] < 1e-4)
        assert np.all(data["gt_bboxes_3d"] - output_2["gt_bboxes_3d"] < 1e-4)


def test_sparse4d_adaptor():
    adaptor = Sparse4DAdaptor(
        projection_key="lidar2img",
        img_shape_key="img_shape",
        ego_pose_key="lidar2global",
        cam_intrinsic_key="cam_intrinsic",
    )
    num_cams = 6
    data = dict(
        lidar2img=[np.eye(4)] * num_cams,
        img_shape=[(64, 64)] * num_cams,
        lidar2global=np.eye(4),
        cam_intrinsic=[np.eye(3)] * num_cams,
    )
    output = adaptor(data)
    assert output["projection_mat"].shape == (num_cams, 4, 4)
    assert output["image_wh"].shape == (num_cams, 2)
    assert output["T_global_inv"].shape == (4, 4)
    assert output["T_global"].shape == (4, 4)
    assert output["cam_intrinsic"].shape == (num_cams, 3, 3)
    assert output["focal"].shape == (num_cams,)
    data = dict(
        lidar2img=[np.eye(4)] * num_cams,
    )
    output = adaptor(data)
    assert output["projection_mat"].shape == (num_cams, 4, 4)
    assert "image_wh" not in output
    assert "T_global_inv" not in output
    assert "T_global" not in output
    assert "cam_intrinsic" not in output
    assert "focal" not in output


if __name__ == "__main__":
    pytest.main(["-s", __file__])
