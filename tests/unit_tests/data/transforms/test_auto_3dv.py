# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import os

import cv2
import numpy as np
import pytest
import torch
import torchvision

from hat.core.colormap import get_colormap_6c_for_bev
from hat.data.datasets.auto_3dv import ANCAuto3DV
from hat.data.transforms.auto_3dv import (
    ANCApplyMaskOnImg,
    ANCBevSegTargetGenerator,
    ANCClassRemap,
    ANCCollect3DV,
    ANCConvertPackDataTo3DV,
    ANCCrop3DV,
    ANCE2EDynamicTargetGenerator,
    ANCHomoAdaption,
    ANCMotionFlowGenerator,
    ANCNV12Transform3DV,
    ANCPad3DV,
    ANCPrepareDataBEV,
    ANCPrepareDepthPose,
    ANCPrepareTempoDataBEV,
    ANCPrepareTempoDataE2EDynamic,
    ANCResize3DV,
    ANCSelectDataByIdx,
    ANCStackData,
    ANCTemporalHomo,
    ANCToTensor3DV,
    ANCVisualizeIpm,
)
from hat.models.task_modules.bev.spatial_transfomer import SpatialTransfomer
from hat.registry import build_from_registry
from hat.utils.apply_func import _as_list, img_array2tensor, img_tensor2array
from hat.utils.package_helper import check_packages_available
from tests import (
    HAT_BUCKET_EXISTS,
    HAT_BUCKET_PATH,
    SD_AlGORITHM_BUCKET_EXISTS,
)
from tests.utils import (
    check,
    check_range,
    check_shape,
    check_type,
    gen_fake_np_data,
    gen_fake_pil_data,
)

try:
    import hat_sim
except ImportError:
    hat_sim = None


def generate_fake_data(h=1080, w=1920, views=8, frames=3):
    front_bool_idx = [False] * views
    front_bool_idx[0] = True

    data = dict()
    data["pil_imgs"] = [
        [
            gen_fake_pil_data(
                (h, w, 3), mode="RGB", dtype="uint8", low=0, high=255
            )
            for i in range(views)
        ]
        for j in range(frames)
    ]
    data["color_imgs"] = [
        [
            gen_fake_pil_data(
                (h, w, 3), mode="RGB", dtype="uint8", low=0, high=255
            )
            for i in range(views)
        ]
        for j in range(frames)
    ]

    data["gt_seg"] = [
        [gen_fake_pil_data((h, w, 3), mode="I") for i in range(views)]
        for j in range(frames)
    ]
    data["gt_depth"] = [
        gen_fake_pil_data((h, w), mode="F", dtype="float32")
        for i in range(views)
    ]
    data["gt_bev_seg"] = gen_fake_pil_data((h, w), mode="I")
    data["obj_mask"] = gen_fake_pil_data((h, w), mode="I")
    data["front_mask"] = gen_fake_pil_data((h, w), mode="I")

    data["intrinsics"] = gen_fake_np_data((3, 3), dtype="float32")
    data["distortcoef"] = gen_fake_np_data((8), dtype="float32")
    data["homography"] = torch.from_numpy(
        gen_fake_np_data((views, 3, 3), dtype="float32")
    )
    data["front_bool_idx"] = np.array(front_bool_idx)
    data["timestamp"] = gen_fake_np_data((1), dtype="float32")

    data["ud_coord"] = gen_fake_pil_data((h, w), mode="F", dtype="float32")
    return data


def test_undistortion():
    # generate fake data
    Undistortion_3dv = build_from_registry(dict(type="Undistortion"))
    h, w = 1080, 1920
    size = (h, w)
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames)

    undistorted_data = Undistortion_3dv(data)

    check(undistorted_data["color_imgs"], check_shape, shape=size)
    check(undistorted_data["gt_seg"], check_shape, shape=size)
    check(undistorted_data["gt_depth"], check_shape, shape=size)
    check(undistorted_data["obj_mask"], check_shape, shape=size)
    check(undistorted_data["front_mask"], check_shape, shape=size)


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    [
        "size",
        "interpolation",
    ],
    [
        pytest.param((540, 960), "nearest"),
        pytest.param((540, 960), "bilinear"),
    ],
)
def test_resize3dv(size, interpolation):
    # generate fake data
    h, w = 1080, 1920
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    resize_3dv = ANCResize3DV(size=[size] * views, interpolation=interpolation)

    resize_data = resize_3dv(data)

    check(resize_data["pil_imgs"], check_shape, shape=size)
    check(resize_data["color_imgs"], check_shape, shape=size)
    check(resize_data["gt_seg"], check_shape, shape=size)
    check(resize_data["gt_depth"], check_shape, shape=size)
    check(resize_data["obj_mask"], check_shape, shape=size)
    check(resize_data["front_mask"], check_shape, shape=size)

    check(resize_data["gt_bev_seg"], check_shape, shape=(h, w))
    check(resize_data["intrinsics"], check_shape, shape=(3, 3))
    check(resize_data["distortcoef"], check_shape, shape=(8,))
    check(resize_data["homography"], check_shape, shape=(views, 3, 3))
    check(resize_data["front_bool_idx"], check_shape, shape=(views,))
    check(resize_data["timestamp"], check_shape, shape=(1,))


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["height", "width", "top", "left"],
    [
        pytest.param(512, 960, None, None),
        pytest.param(512, 960, 28, None),
        pytest.param(512, 960, 0, None),
        pytest.param(512, 960, None, 0),
        pytest.param(512, 960, None, 28),
        pytest.param(512, 960, None, [28]),
        pytest.param(512, 960, [0], None),
    ],
)
def test_crop3dv(height, width, top, left):
    # generate fake data

    h, w = 1080, 1920
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    data["size"] = [(h, w)] * views
    top = _as_list(top) * views
    left = _as_list(left) * views

    crop_3dv = ANCCrop3DV([height] * views, [width] * views, top, left)
    crop_data = crop_3dv(data)

    check(crop_data["pil_imgs"], check_shape, shape=(height, width))
    check(crop_data["color_imgs"], check_shape, shape=(height, width))
    check(crop_data["gt_seg"], check_shape, shape=(height, width))
    check(crop_data["gt_depth"], check_shape, shape=(height, width))
    check(crop_data["obj_mask"], check_shape, shape=(height, width))
    check(crop_data["front_mask"], check_shape, shape=(height, width))

    check(crop_data["gt_bev_seg"], check_shape, shape=(h, w))
    check(crop_data["intrinsics"], check_shape, shape=(3, 3))
    check(crop_data["distortcoef"], check_shape, shape=(8,))
    check(crop_data["homography"], check_shape, shape=(views, 3, 3))
    check(crop_data["front_bool_idx"], check_shape, shape=(views,))
    check(crop_data["timestamp"], check_shape, shape=(1,))


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_pad3dv():
    # generate fake data
    h, w = 1080, 1920
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    paddings = [(2, 2, 2, 2)] * views
    data["size"] = [(h, w)] * views
    padding_3dv = ANCPad3DV(paddings)
    pad_data = padding_3dv(data)

    check(pad_data["pil_imgs"], check_shape, shape=(h + 4, w + 4))
    check(pad_data["color_imgs"], check_shape, shape=(h + 4, w + 4))
    check(pad_data["gt_seg"], check_shape, shape=(h + 4, w + 4))
    check(pad_data["gt_depth"], check_shape, shape=(h + 4, w + 4))
    check(pad_data["obj_mask"], check_shape, shape=(h + 4, w + 4))
    check(pad_data["front_mask"], check_shape, shape=(h + 4, w + 4))


def test_class_remap():
    # generate fake data
    remap_dict = {"gt_seg": {1: 2}}
    class_remap = ANCClassRemap(remap_dict)

    h, w = 1080, 1920
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    data["image_height"] = h
    data["image_width"] = w

    remap_data = class_remap(data)

    for frame_idx in range(len(remap_data["gt_seg"])):
        for each_seg in remap_data["gt_seg"][frame_idx]:
            assert (each_seg == 1).sum() == 0


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["with_color_imgs"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_to_tensor3dv(with_color_imgs):
    # generate fake data
    to_tensor3dv = ANCToTensor3DV(with_color_imgs)

    h, w = 1080, 1920
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    data["image_height"] = h
    data["image_width"] = w

    tensor_data = to_tensor3dv(data)

    check(tensor_data["imgs"], check_type, instance=torch.Tensor)
    if with_color_imgs:
        check(tensor_data["color_imgs"], check_type, instance=torch.Tensor)
    check(tensor_data["gt_seg"], check_type, instance=torch.Tensor)
    check(tensor_data["gt_depth"], check_type, instance=torch.Tensor)
    check(tensor_data["obj_mask"], check_type, instance=torch.Tensor)
    check(tensor_data["front_mask"], check_type, instance=torch.Tensor)

    check(tensor_data["gt_bev_seg"], check_type, instance=torch.Tensor)
    check(tensor_data["intrinsics"], check_type, instance=torch.Tensor)
    check(tensor_data["distortcoef"], check_type, instance=torch.Tensor)
    check(tensor_data["homography"], check_type, instance=torch.Tensor)
    check(tensor_data["front_bool_idx"], check_type, instance=torch.Tensor)
    check(tensor_data["timestamp"], check_type, instance=torch.Tensor)

    assert "pil_imgs" not in tensor_data

    check(tensor_data["imgs"], check_range, min=0, max=255.0)

    if with_color_imgs:
        assert "color_imgs" in tensor_data
        check(tensor_data["color_imgs"], check_range, min=0, max=1.0)


def test_to_prepare_data_bev():
    # generate fake data
    h, w = 512, 960
    views = 6
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    to_tensor3dv = ANCToTensor3DV(True)

    # all view data will store in key of "img"
    # prepare_bev = ANCPrepareDataBEV("front_side")
    # tensor_data = to_tensor3dv(data)
    # prepare_data = prepare_bev(tensor_data)

    # check(prepare_data["img"], check_shape, shape=(views, 3, h, w))
    # check(prepare_data["gt_seg"], check_shape, shape=(views, 1, h, w))
    # check(prepare_data["gt_bev_seg"], check_shape, shape=(1, h, w))

    # front view data will store in key of "img" and side view data will store
    # in key of "side_img"
    prepare_bev = ANCPrepareDataBEV(
        {
            "front": 1,
            "side": 5,
        }
    )
    data = generate_fake_data(h, w, views, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare_bev(tensor_data)

    check(prepare_data["img"], check_shape, shape=(1, 3, h, w))
    check(prepare_data["side_img"], check_shape, shape=(views - 1, 3, h, w))

    check(prepare_data["gt_bev_seg"], check_shape, shape=(1, h, w))

    # all view data will store in key of "round_img"
    prepare_bev = ANCPrepareDataBEV(
        {
            "round": 6,
        }
    )
    data = generate_fake_data(h, w, views, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare_bev(tensor_data)

    check(prepare_data["round_img"], check_shape, shape=(views, 3, h, w))
    check(prepare_data["gt_bev_seg"], check_shape, shape=(1, h, w))


def test_to_prepare_tempo_data_bev():
    # generate fake data
    h, w = 512, 960
    views = 6
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    to_tensor3dv = ANCToTensor3DV(True)

    # all view data will store in key of "img"
    views_domain2nums = {
        "front": views,
        "side": 0,
        "round": 0,
        "narrow": 0,
    }
    prepare_bev = ANCPrepareTempoDataBEV(views_domain2nums=views_domain2nums)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare_bev(tensor_data)

    check(prepare_data["img"], check_shape, shape=(frames * views, 3, h, w))
    check(prepare_data["gt_bev_seg"], check_shape, shape=(1, h, w))

    # front view data will store in key of "img" and side view data will store
    # in key of "side_img"
    views_domain2nums = {
        "front": 1,
        "side": views - 1,
        "round": 0,
        "narrow": 0,
    }
    prepare_bev = ANCPrepareTempoDataBEV(views_domain2nums=views_domain2nums)
    data = generate_fake_data(h, w, views, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare_bev(tensor_data)

    check(prepare_data["img"], check_shape, shape=(frames, 3, h, w))
    check(
        prepare_data["side_img"],
        check_shape,
        shape=(frames * (views - 1), 3, h, w),
    )

    check(prepare_data["gt_bev_seg"], check_shape, shape=(1, h, w))

    # all view data will store in key of "round_img"
    views_domain2nums = {
        "front": 0,
        "side": 0,
        "round": views,
        "narrow": 0,
    }
    prepare_bev = ANCPrepareTempoDataBEV(views_domain2nums=views_domain2nums)
    data = generate_fake_data(h, w, views, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare_bev(tensor_data)

    check(
        prepare_data["round_img"], check_shape, shape=(frames * views, 3, h, w)
    )
    check(prepare_data["gt_bev_seg"], check_shape, shape=(1, h, w))

    # front view data will store in key of "img"
    # side view data will store in key of "side_img"
    # round view data will store in key of "ruond_img"
    # narrow view data will store in key of "narrow_img"
    views_domain2nums = {
        "front": 1,
        "side": 5,
        "round": 4,
        "narrow": 1,
    }
    prepare_bev = ANCPrepareTempoDataBEV(views_domain2nums=views_domain2nums)
    data = generate_fake_data(h, w, 11, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare_bev(tensor_data)

    check(prepare_data["img"], check_shape, shape=(frames, 3, h, w))
    check(prepare_data["side_img"], check_shape, shape=(frames * 5, 3, h, w))
    check(prepare_data["round_img"], check_shape, shape=(frames * 4, 3, h, w))
    check(prepare_data["narrow_img"], check_shape, shape=(frames, 3, h, w))

    check(prepare_data["gt_bev_seg"], check_shape, shape=(1, h, w))


@pytest.mark.parametrize(
    ["return_relative"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_temporal_homo(return_relative):
    # case1: only return one vcs_range output
    bev_size = (256, 256)
    vcs_range = (-30.0, -51.2, 72.4, 51.2)
    sequence_length = 3

    input = {
        "pose": np.random.randint(0, 100, size=(sequence_length, 4, 4)),
    }
    temporal_homo = ANCTemporalHomo(
        bev_size, vcs_range, return_relative=return_relative
    )

    res = temporal_homo(input)
    assert res["homography_temporal"].shape == (sequence_length - 1, 3, 3)

    # case2: return multi vcs_range output
    bev_size = [(256, 256), (192, 128)]
    vcs_range = [(-30.0, -51.2, 72.4, 51.2), (-12.8, -12.8, 25.6, 12.8)]
    homography_names = ["homography_temporal", "homography_temporal_small"]
    input = {
        "pose": np.random.randint(0, 100, size=(sequence_length, 4, 4)),
    }

    temporal_homo = ANCTemporalHomo(
        bev_size,
        vcs_range,
        return_relative=return_relative,
        homography_names=homography_names,
    )

    res = temporal_homo(input)
    assert res["homography_temporal"].shape == (sequence_length - 1, 3, 3)
    assert res["homography_temporal_small"].shape == (
        sequence_length - 1,
        3,
        3,
    )


@pytest.mark.parametrize(
    ["select_idxs"],
    [
        pytest.param([1, 0]),
        pytest.param([0]),
    ],
)
def test_to_prepare_depth(select_idxs):
    to_tensor3dv = ANCToTensor3DV(True)
    h, w = 512, 960

    select_data = ANCSelectDataByIdx(
        select_idxs=select_idxs,
        input_key="imgs",
        output_key="img",
    )
    views = 5
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = select_data(tensor_data)
    check(prepare_data["img"], check_shape, shape=(3, h, w))
    assert len(prepare_data["img"]) == len(select_idxs)


def test_stack_data():
    to_tensor3dv = ANCToTensor3DV(True)
    h, w = 512, 960

    stack_data = ANCStackData(data_keys=["gt_depth", "imgs"])
    views = 5
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = stack_data(tensor_data)
    check(prepare_data["gt_depth"], check_shape, shape=(views, 1, h, w))
    check(prepare_data["imgs"], check_shape, shape=(views, 3, h, w))


@pytest.mark.parametrize(
    ["input_sequence_length"],
    [
        pytest.param(1),
        pytest.param(2),
    ],
)
def test_to_prepare_depth_pose(input_sequence_length):
    to_tensor3dv = ANCToTensor3DV(True)
    h, w = 512, 960
    # test front_depth_pose_resflow_train task
    prepare25d = ANCPrepareDepthPose(
        with_extra_img=True,
        with_color_img=True,
        input_sequence_length=input_sequence_length,
    )

    views = 1
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare25d(tensor_data)

    check(prepare_data["img"], check_shape, shape=(views, 3, h, w))
    check(prepare_data["extra_img"], check_shape, shape=(views, 3, h, w))
    check(prepare_data["gt_depth"], check_shape, shape=(views, 1, h, w))
    check(prepare_data["obj_mask"], check_shape, shape=(1, h, w))
    assert len(prepare_data["extra_img"]) == 2
    assert len(prepare_data["img"]) == input_sequence_length

    # test front_depth_pose_resflow_val task
    prepare25d = ANCPrepareDepthPose(
        with_color_img=False,
        with_extra_img=False,
        input_sequence_length=input_sequence_length,
    )
    views = 1
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare25d(tensor_data)

    check(prepare_data["img"], check_shape, shape=(views, 3, h, w))
    check(prepare_data["gt_depth"], check_shape, shape=(views, 1, h, w))
    check(prepare_data["obj_mask"], check_shape, shape=(1, h, w))
    assert len(prepare_data["extra_img"]) == 1
    assert len(prepare_data["img"]) == input_sequence_length


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_motion_flow_generator():
    bucket_path = HAT_BUCKET_PATH
    sync_file = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/wenming.meng/FSD_Urban_v4/motion_flow/DG202/val_20210702-105102_276.txt",  # noqa
    )

    root = os.path.join(bucket_path, "unit_test_data/J5FSD/FSD_Urban_v4")
    per_view_shape = {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    }
    collect = ANCCollect3DV(
        load_data_types=["gt_bev_motion_flow"],
        img_idxs=[0, 1, 2, 3],
        gt_bev_motionflow_idxs=[0, -2],
    )
    resize = ANCResize3DV(
        size=[
            (540, 960),
            (600, 960),
            (600, 960),
            (600, 960),
            (600, 960),
            (600, 960),
        ]
    )
    crop = ANCCrop3DV(
        height=[512] * 6,
        width=[960] * 6,
        top=[0, 88, 88, 88, 88, 88],
        left=[0] * 6,
    )

    mf_gen = ANCMotionFlowGenerator(
        bev_size=(512, 512),
        vcs_range=(-30, -51.2, 72.4, 51.2),
        ego_loc=(362, 256),
    )

    trans = torchvision.transforms.Compose([collect, resize, crop, mf_gen])

    dataset = ANCAuto3DV(
        root,
        camera_view_names=list(per_view_shape.keys()),
        per_view_shape=per_view_shape,
        sync_file=sync_file,
        transforms=trans,
    )
    data = dataset[0]
    assert "pil_imgs" in data
    assert "gt_bev_motion_flow" in data


@pytest.mark.parametrize(
    ["nv12_format"],
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_convert_packdate_to_3dv(nv12_format):
    camera_view_names = [
        "camera_front",
        "camera_front_left",
        "camera_front_right",
        "camera_rear_left",
        "camera_rear_right",
        "camera_rear",
    ]
    convert = ANCConvertPackDataTo3DV(calib=True, nv12_format=nv12_format)

    if nv12_format:
        img = np.random.randint(
            0, 255, (int(512 * 960 * 3 / 2)), dtype=np.uint8
        )
    else:
        img = np.random.randint(0, 255, (512, 960, 3), dtype=np.uint8)
    camera_calib = {
        "camera_x": 0,
        "camera_y": 0,
        "camera_z": 1.538,
        "center_u": 1919.9285888671875,
        "center_v": 1085.44140625,
        "distort": [0 for _ in range(8)],
        "focal_u": 2418.281494140625,
        "focal_v": 2418.281494140625,
        "pitch": 0.0009087863419741011,
        "roll": -0.008198394482362016,
        "yaw": 0.005041685210280009,
        "vcs": {
            "rotation": [0 for _ in range(3)],
            "translation": [0 for _ in range(3)],
        },
    }

    data_dict = {
        "img": [img for _ in range(6)],
        "camera_list": camera_view_names,
        "camera_calib": [camera_calib for _ in range(6)],
        "timestamp": np.array(12345678),
    }
    data = convert(data_dict=data_dict)

    if nv12_format:
        img_key = "img"
    else:
        img_key = "pil_imgs"
    assert img_key in data
    assert len(data[img_key]) == 1
    assert len(data[img_key][0]) == 6

    assert "local2cam" in data
    assert "local2vcs" in data


@pytest.mark.parametrize(
    ["enable_vis"],
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_prepare_visualize_ipm(enable_vis):
    bev_size = (512, 512)
    bev_size_small = (192, 128)
    view_idxs = [0, 1, 2, 3, 4, 5]
    img_scale = 1
    block_warp_padding = [(0, 0, 0, 0) for _ in range(6)]
    block_warp_padding_small = [(0, 0, 0, 0) for _ in range(6)]
    return_name = "ipm"
    return_name_small = "ipm_small"
    meta_info = "meta_info"
    meta_info_small = "meta_info_small"
    ipm = ANCVisualizeIpm(
        bev_size=bev_size,
        view_idxs=view_idxs,
        img_scale=img_scale,
        enable_vis=enable_vis,
        meta_name=meta_info,
        block_warp_padding=block_warp_padding,
        return_name=return_name,
    )
    views = 11
    frames = 1
    data_dict = generate_fake_data(bev_size[0], bev_size[1], views, frames)
    data_dict = ipm(data_dict)
    assert "ipm" in data_dict
    check(data_dict["ipm"], check_shape, shape=(bev_size[0], bev_size[1], 3))
    if not enable_vis:
        check(data_dict["ipm"], check_range, min=0, max=0)

    ipm = ANCVisualizeIpm(
        bev_size=[bev_size, bev_size_small],
        view_idxs=view_idxs,
        img_scale=img_scale,
        enable_vis=enable_vis,
        meta_name=[meta_info, meta_info_small],
        block_warp_padding=[
            block_warp_padding,
            block_warp_padding_small,
        ],
        return_name=[return_name, return_name_small],
        vcs_plane_heights=[(0,), (0,)],
    )
    views = 11
    frames = 1
    data_dict = generate_fake_data(bev_size[0], bev_size[1], views, frames)
    data_dict["meta_info_small"] = {}
    data_dict["meta_info_small"]["homography"] = gen_fake_np_data(
        (views, 3, 3), dtype="float32"
    )
    data_dict["meta_info_small"]["homo_offset"] = gen_fake_np_data(
        (views, bev_size_small[0], bev_size_small[1], 2), dtype="float32"
    )
    data_dict = ipm(data_dict)
    assert "ipm" in data_dict
    check(data_dict["ipm"], check_shape, shape=(bev_size[0], bev_size[1], 3))
    assert "ipm_small" in data_dict
    check(
        data_dict["ipm_small"],
        check_shape,
        shape=(bev_size_small[0], bev_size_small[1], 3),
    )
    if not enable_vis:
        check(data_dict["ipm"], check_range, min=0, max=0)
        check(data_dict["ipm_small"], check_range, min=0, max=0)


@pytest.mark.skipif(not hat_sim, reason="hat_sim is required")
def test_nv12_transform():
    w, h = 3840, 2160
    for idx in range(3):
        nv12 = np.random.randint(
            0,
            255,
            size=(int(3 * w * h / 2)),
            dtype=np.uint8,
        )
        data = {"img": [[nv12]], "ratio": [1]}
        transform = ANCNV12Transform3DV(
            ori_size=[(h, w)], pyramid_layer_index=[idx]
        )
        res = transform(data)
        assert res["imgs"][0][0].cpu().numpy().shape == (
            3,
            h / 2 ** (idx + 1),
            w / 2 ** (idx + 1),
        )


@pytest.mark.parametrize(
    ["use_yuv_format"],
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_apply_mask_on_img(use_yuv_format):
    w, h = 3840, 2160
    test_img = np.random.randint(
        0,
        255,
        size=(3, w, h),
        dtype=np.uint8,
    )
    test_img = torch.from_numpy(test_img)
    ignore_mask = np.random.choice(
        [0, 1],
        size=(w, h),
    )
    ignore_mask = ignore_mask.astype(np.bool)
    ignore_mask = torch.from_numpy(ignore_mask)
    data = {
        "imgs": [[test_img]],
        "img_mask": [ignore_mask],
    }
    transform = ANCApplyMaskOnImg(use_yuv_format)
    data = transform(data)
    if use_yuv_format:
        img = cv2.cvtColor(
            data["imgs"][0][0].permute(1, 2, 0).numpy().astype(np.uint8),
            cv2.COLOR_YUV2BGR,
        )
    else:
        img = data["imgs"][0][0].permute(1, 2, 0).numpy().astype(np.uint8)
    assert (img[ignore_mask, :] == 0).all()


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="need SD_Algorithm bucket"
)
def test_BevSegTargetGenerator():

    bucket_path = "/horizon-bucket"

    bev_discrete_obj_lmdb_path = os.path.join(
        bucket_path,
        "SD_Algorithm/07_perception_bev_static/03_pack_package/yunfeng.zhang/driving_11v_gt_pack_merge/det/20221116_det_ut126_ut263_v2/merge_det_UT263_all_bevmap_11v",  # noqa
    )
    bev_static_lmdb_path = os.path.join(
        bucket_path,
        "SD_Algorithm/07_perception_bev_static/03_pack_package/shida.chen/merge_datasets/roadedges_or_solid/v1111/merge_roadedges_or_solid_UT263_all_bevmap",  # noqa
    )
    sync_file_lmdb = os.path.join(
        bucket_path,
        "SD_Algorithm/05_perception_bev_backbone/03_pack_package/yunyi.geng/sync_file_0328/UT263_20220924_selected_list_refactor_by_road_arrow_sync_txt_3",  # noqa
    )
    img_data_path = [
        os.path.join(
            bucket_path,
            "SD_Algorithm/07_perception_bev_static/03_pack_package/shida.chen/om_v3/autolabel_data_20221018_711924_2/rec_lmdb/de_redundancy/merge_UT263_roadedges_or_solid_v5a_v1111/train_FSD_Site_UT263_20220924_front0820_rec_v1228",  # noqa
        ),
        os.path.join(
            bucket_path,
            "SD_Algorithm/07_perception_bev_static/03_pack_package/shida.chen/om_v3/autolabel_data_20221018_711924_2/rec_lmdb/de_redundancy/merge_UT263_roadedges_or_solid_v5a_v1111/train_FSD_Site_UT263_20220924_5v0223_rec_v1228",  # noqa
        ),
        os.path.join(
            bucket_path,
            "SD_Algorithm/07_perception_bev_static/03_pack_package/shida.chen/om_v3/autolabel_data_20221018_711931_2/rec_lmdb/de_redundancy/merge_UT263_roadedges_or_solid_v5a_v1111/train_FSD_Site_UT263_20220924_front0820_rec_v1228",  # noqa
        ),
        os.path.join(
            bucket_path,
            "SD_Algorithm/07_perception_bev_static/03_pack_package/shida.chen/om_v3/autolabel_data_20221018_711931_2/rec_lmdb/de_redundancy/merge_UT263_roadedges_or_solid_v5a_v1111/train_FSD_Site_UT263_20220924_5v0223_rec_v1228",  # noqa
        ),
    ]
    root = os.path.join(
        bucket_path,
        "SD_Algorithm/12_perception_bev_hde/03_hde_data/FSD_Site_UT263_20220924",  # noqa
    )
    HDE_data_path = os.path.join(
        bucket_path,
        "SD_Algorithm/12_perception_bev_hde/03_pack_package/site_data/",
    )

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
            "gt_bev_seg_anno",
            "gt_bev_discrete_obj",
            "timestamp",
            "pose",
            # "wheel_pose",
        ],
        img_idxs=[0],
    )

    roi_vcs_range = (-30.0, -51.2, 72.4, 51.2)
    bev_seg_target_size = (512, 512)

    bev_seg_target_generator = ANCBevSegTargetGenerator(
        vcs_range=roi_vcs_range,
        bev_size=bev_seg_target_size,
    )
    trans = torchvision.transforms.Compose([collect, bev_seg_target_generator])
    # from sync file
    dataset = ANCAuto3DV(
        root,
        camera_view_names=["camera_front"],
        per_view_shape=per_view_shape,
        bev_discrete_obj_lmdb_path=bev_discrete_obj_lmdb_path,
        bev_static_lmdb_path=bev_static_lmdb_path,
        sync_file_lmdb=sync_file_lmdb,
        img_data_path=img_data_path,
        transforms=trans,
        HDE_data_path=HDE_data_path,
    )

    color_map = get_colormap_6c_for_bev()
    begin = 100
    data = dataset[begin]
    seg1 = color_map[np.array(data["gt_bev_seg"])].squeeze()
    pose1 = data["pose"][0]

    data = dataset[begin + 2]
    seg2 = color_map[np.array(data["gt_bev_seg"])].squeeze()
    pose2 = data["pose"][0]

    temporal_homo = ANCTemporalHomo(bev_seg_target_size, roi_vcs_range)
    res = temporal_homo({"pose": [pose2, pose1]})
    tmporal_homo = res["homography_temporal"]
    tmporal_homo = torch.from_numpy(tmporal_homo).float()

    seg1_tensor = img_array2tensor(seg1)

    st = SpatialTransfomer(bev_seg_target_size[0], bev_seg_target_size[1])
    seg1_tensor_warp_tensor = st(seg1_tensor, tmporal_homo)[0]
    seg1_tensor_warp = img_tensor2array(seg1_tensor_warp_tensor)

    res = np.concatenate([seg1, seg2, seg1_tensor_warp], 1)


def test_homo_adaption():
    views, h, w = (6, 128, 128)
    data = {
        "meta_info": {
            "homography": gen_fake_np_data((views, 3, 3), dtype="float32"),
            "homo_offset": gen_fake_np_data((views, h, w, 2), dtype="float32"),
        },
        "meta_info_small": {
            "homography": gen_fake_np_data((views, 3, 3), dtype="float32"),
            "homo_offset": gen_fake_np_data((views, h, w, 2), dtype="float32"),
        },
    }

    homo_adaption = ANCHomoAdaption(
        homo_scale_each_view=[
            1 / 16,
            1 / 8,
            1 / 8,
            1 / 8,
            1 / 8,
            1 / 8,
        ]
    )

    return_data = homo_adaption(data)
    meta_info_names = ["meta_info", "meta_info_small"]
    for meta_info_name in meta_info_names:
        if meta_info_name in return_data:
            cur_homography = return_data[meta_info_name]["homography"]
            cur_homo_offset = return_data[meta_info_name]["homo_offset"]
            assert isinstance(
                cur_homography, torch.Tensor
            ) and cur_homography.shape == (views, 3, 3)
            assert isinstance(
                cur_homo_offset, torch.Tensor
            ) and cur_homo_offset.shape == (views, h, w, 2)


@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS, reason="need HDLTAlgorithm bucket"
)  # noqa
def test_e2e_dynamic_auto3dv():
    bucket_path = HAT_BUCKET_PATH
    unit_test_data = "unit_test_data/J5FSD/users/zongwei.zhou/UT126"
    sync_file = os.path.join(
        bucket_path,
        unit_test_data,  # noqa,
        "e2e_dynamic_sync_file_singlepack.txt",
    )
    homo_file_path = os.path.join(
        bucket_path, unit_test_data, "20221109"  # noqa
    )
    sync_file_lmdb = None
    e2e_dynamic_lmdb_path = os.path.join(
        bucket_path, unit_test_data, "val"  # noqa
    )

    per_view_recs = ["UT126/val_UT126_20221109_D_front0820"]
    img_data_path = [
        os.path.join(
            bucket_path,
            unit_test_data,
            per_view_rec,
        )
        for per_view_rec in per_view_recs
    ]

    per_view_shape = {
        "camera_front": (2160, 3840),
    }
    collect = ANCCollect3DV(
        load_data_types=[
            "e2e_dynamic_anno",
            "timestamp",
            "img_paths",
            "img_name",
            "pack_dir",
        ],
        img_idxs=range(2),
        num_frames_per_iter=2,
        fill_fake_temporal_data=False,
    )
    totensor3dv = ANCToTensor3DV(
        with_color_imgs=False,
    )
    views_domain2nums = {
        "front": 1,
        "side": 0,
        "round": 0,
        "narrow": 0,
    }
    prepare_temporal_data = ANCPrepareTempoDataE2EDynamic(
        views_domain2nums=views_domain2nums,
        length_of_clip=4,
        num_frames_per_iter=2,
    )
    e2e_target_transform = ANCE2EDynamicTargetGenerator(
        vcs_range=(-31.8, -76.8, 102.6, 76.8, -3, 5),
        output_labels_group={"veh": [0], "vru": [2, 1]},
        filter_vcs_range={
            0: (-31.8, -76.8, 102.6, 76.8),
            1: (-31.8, -33.6, 52.2, 33.6),
            2: (-31.8, -33.6, 52.2, 33.6),
        },
        cls_bev_size={0: (224, 256), 1: (280, 224), 2: (280, 224)},
        cls_hm_kernel={0: 11, 1: 9, 2: 9},
        num_frames_per_clip=4,
        num_frames_per_iter=2,
        max_his_odo_len=12,
        trajpred_transform=None,
        category2id_map={
            0: 0,  # "car": 0,
            1: 1,  # "cyclist": 1,
            2: 2,  # "pedestrian": 2,
            3: 0,  # "truck": 3,
            4: 0,  # "tricycle": 4,
            5: 0,  # "bus": 5,
            6: 0,  # "construction": 6,
            7: 0,  # "blur": 7,
            8: -99,  # "other": 8,
            9: 0,  # "van": 9,
            11: 0,  # "bigmot": 10,
            -99: -99,
        },
    )
    transforms = torchvision.transforms.Compose(
        [
            collect,
            totensor3dv,
            e2e_target_transform,
            prepare_temporal_data,
        ]
    )
    camera_view_names = ["camera_front"]
    # from sync file lmdb
    homo_gen = {
        "homo_path": None,
        "calib_path": homo_file_path,
        "spatial_resolution": (0.6, 0.6),
        "H_persp_view_scale": 0.25,
        "vcs_range": (-31.8, -76.8, 102.6, 76.8),
        "camera_view_names": ["camera_front"],
        "task_camera_view_names": ["camera_front"],
        "per_view_shape": per_view_shape,
        "use_distorted_offset": True,
        "homo_transforms": {
            "camera_front": {"Resize": (540, 960), "Crop": (0, 0, 540, 960)}
        },
        "homo_noise": None,
    }
    dataset = ANCAuto3DV(
        root="",
        camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        img_data_path=img_data_path,
        sync_file=sync_file,
        sync_file_lmdb=sync_file_lmdb,
        e2e_dynamic_lmdb_path=e2e_dynamic_lmdb_path,
        transforms=transforms,
        homo_gen=homo_gen,
        num_max_frames=4,
        num_frames_per_iter=2,
        reverse=True,
    )
    data1 = dataset[10]
    data2 = dataset[11]
    assert data1["sample_split_index"] == 0
    assert data2["sample_split_index"] == 1
    assert "motr_targets" in data1
    assert len(data1["motr_targets"]["bev_tracking"]) == 2
    assert "timestamp" in data1
    assert "img_paths" in data1


if __name__ == "__main__":
    pytest.main(["-s", __file__])
