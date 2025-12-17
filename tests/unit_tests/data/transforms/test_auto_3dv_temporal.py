# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import copy

import cv2
import numpy as np
import pytest
import torch
import torchvision  # noqa

from hat.data.transforms.auto_3dv_temporal import (
    ANCPrepareTempoALLFrameDataBEV,
    ANCTemporalApplyMaskOnImg,
    ANCTemporalBev3dTargetGenerator,
    ANCTemporalCrop3DV,
    ANCTemporalMultiViewTargetGenerator,
    ANCTemporalResize3DV,
    ANCToTensorTemporal3DV,
)
from hat.utils.apply_func import _as_list
from hat.utils.package_helper import check_packages_available
from tests.utils import (
    check,
    check_range,
    check_shape,
    check_type,
    gen_fake_np_data,
    gen_fake_pil_data,
)


def generate_fake_data(h=1080, w=1920, views=8, frames=3, multi_frame=True):
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
    if multi_frame:
        data["gt_multi_view"] = []
        for __ in range(frames):
            sync_info = []
            for _ in range(views):

                multi_view_info = {}
                multi_view_info["meta"] = {}
                multi_view_info["meta"]["ignore_mask"] = {}
                multi_view_info["meta"]["ignore_mask"][
                    "ignore_mask_2d"
                ] = gen_fake_pil_data(
                    (h, w), mode="RGB", dtype="uint8", low=0, high=1
                )
                sync_info.append(multi_view_info)
            data["gt_multi_view"].append(sync_info)
    else:
        sync_info = []
        for _ in range(views):
            multi_view_info = {}
            multi_view_info["meta"] = {}
            multi_view_info["meta"]["ignore_mask"] = {}
            multi_view_info["meta"]["ignore_mask"][
                "ignore_mask_2d"
            ] = gen_fake_pil_data(
                (h, w), mode="RGB", dtype="uint8", low=0, high=1
            )
            sync_info.append(multi_view_info)
        data["gt_multi_view"] = sync_info
    return data


def generate_fake_annos(num, clip_len, multi_frame=True):
    if multi_frame:
        clip_annos = []
        clip_multi_view_gt = []
        for _ in range(clip_len):
            annos = []
            multi_view_info = {}
            multi_view_info["occlusion_multi_view"] = {}
            for idx in range(num):
                anno_i = {}
                anno_i["label"] = "Car"
                anno_i["ignore"] = False
                anno_i["location"] = [10, 10, 10]
                anno_i["dimension"] = [3, 3, 3]
                anno_i["yaw"] = 0
                anno_i["visibility"] = True
                anno_i["track_id"] = idx
                annos.append(anno_i)
                multi_view_info["occlusion_multi_view"]["idx"] = 0
            clip_annos.append(annos)
            clip_multi_view_gt.append(multi_view_info)
        data = {
            "gt_bev_dynamic_anno": clip_annos,
            "gt_multi_view": clip_multi_view_gt,
        }
    else:
        annos = []
        multi_view_info = {}
        multi_view_info["occlusion_multi_view"] = {}
        for idx in range(num):
            anno_i = {}
            anno_i["label"] = "Car"
            anno_i["ignore"] = False
            anno_i["location"] = [10, 10, 10]
            anno_i["dimension"] = [3, 3, 3]
            anno_i["yaw"] = 0
            anno_i["visibility"] = True
            anno_i["track_id"] = idx
            annos.append(anno_i)
            multi_view_info["occlusion_multi_view"]["idx"] = 0
        data = {
            "gt_bev_dynamic_anno": annos,
            "gt_multi_view": multi_view_info,
        }
    return data


def generate_fake_multi_view_gt(
    obj_num, view_num, clip_len, h, w, multi_frame=True
):
    if multi_frame:
        clip_multi_view_gt = []
        clip_imgs = []
        for _ in range(clip_len):
            all_view_infos = []
            view_imgs = []
            for __ in range(view_num):
                multi_view_info = {}
                multi_view_info["objects"] = []
                multi_view_info["meta"] = {}
                multi_view_info["meta"]["ignore_mask"] = {}
                multi_view_info["meta"]["ignore_mask"][
                    "ignore_mask_2d"
                ] = torch.from_numpy(
                    gen_fake_np_data((h, w), dtype="uint8", low=0, high=1)
                )
                view_imgs.append(
                    torch.from_numpy(
                        gen_fake_np_data(
                            (h, w), dtype="uint8", low=0, high=255
                        )
                    )
                )
                for idx in range(obj_num):
                    bbox = {}
                    bbox["bbox2d"] = gen_fake_np_data(
                        (4), dtype="float32", low=0, high=100
                    ).tolist()
                    bbox["uid"] = idx
                    bbox["bbox2d_attr"] = {}
                    bbox["bbox2d_attr"]["occlusion"] = "full_visible"
                    multi_view_info["objects"].append(bbox)
                all_view_infos.append(multi_view_info)
            clip_multi_view_gt.append(all_view_infos)
            clip_imgs.append(view_imgs)
        data = {"gt_multi_view": clip_multi_view_gt, "imgs": clip_imgs}
    else:
        all_view_infos = []
        view_imgs = []
        for __ in range(view_num):
            multi_view_info = {}
            multi_view_info["objects"] = []
            multi_view_info["meta"] = {}
            multi_view_info["meta"]["ignore_mask"] = {}
            multi_view_info["meta"]["ignore_mask"][
                "ignore_mask_2d"
            ] = torch.from_numpy(
                gen_fake_np_data((h, w), dtype="uint8", low=0, high=1)
            )
            view_imgs.append(
                torch.from_numpy(
                    gen_fake_np_data((h, w), dtype="uint8", low=0, high=255)
                )
            )
            for idx in range(obj_num):
                bbox = {}
                bbox["bbox2d"] = {}
                bbox["uid"] = idx
                bbox["bbox2d_attr"] = {}
                bbox["bbox2d_attr"]["occlusion"] = "full_visible"
                multi_view_info["objects"].append(bbox)
            all_view_infos.append(multi_view_info)
        data = {
            "gt_multi_view": all_view_infos,
            "imgs": [view_imgs] * clip_len,
        }
    return data


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    [
        "size",
        "interpolation",
        "multi_frame",
    ],
    [
        pytest.param((540, 960), "nearest", False),
        pytest.param((540, 960), "bilinear", True),
    ],
)
def test_resize3dv(size, interpolation, multi_frame):
    # generate fake data
    h, w = 1080, 1920
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames, multi_frame)
    resize_3dv = ANCTemporalResize3DV(
        size=[size] * views, interpolation=interpolation
    )

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
    if not multi_frame:
        for multi_view_info in resize_data["gt_multi_view"]:
            check(
                multi_view_info["meta"]["ignore_mask"]["ignore_mask_2d"],
                check_shape,
                shape=size,
            )
    else:
        for sync_gt_multi_view in resize_data["gt_multi_view"]:
            for multi_view_info in sync_gt_multi_view:
                check(
                    multi_view_info["meta"]["ignore_mask"]["ignore_mask_2d"],
                    check_shape,
                    shape=size,
                )


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["height", "width", "top", "left", "multi_frame"],
    [
        pytest.param(512, 960, None, None, False),
        pytest.param(512, 960, 28, None, True),
        pytest.param(512, 960, 0, None, False),
        pytest.param(512, 960, None, 0, True),
        pytest.param(512, 960, None, 28, False),
        pytest.param(512, 960, None, [28], True),
        pytest.param(512, 960, [0], None, False),
    ],
)
def test_crop3dv(height, width, top, left, multi_frame):
    # generate fake data

    h, w = 1080, 1920
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames, multi_frame)
    data["size"] = [(h, w)] * views
    top = _as_list(top) * views
    left = _as_list(left) * views

    crop_3dv = ANCTemporalCrop3DV([height] * views, [width] * views, top, left)
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

    if not multi_frame:
        for multi_view_info in crop_data["gt_multi_view"]:
            check(
                multi_view_info["meta"]["ignore_mask"]["ignore_mask_2d"],
                check_shape,
                shape=(height, width),
            )
    else:
        for sync_gt_multi_view in crop_data["gt_multi_view"]:
            for multi_view_info in sync_gt_multi_view:
                check(
                    multi_view_info["meta"]["ignore_mask"]["ignore_mask_2d"],
                    check_shape,
                    shape=(height, width),
                )


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["with_color_imgs", "multi_frame"],
    [
        pytest.param(True, False),
        pytest.param(False, True),
    ],
)
def test_to_tensor3dv(with_color_imgs, multi_frame):
    # generate fake data
    to_tensor3dv = ANCToTensorTemporal3DV(with_color_imgs)

    h, w = 1080, 1920
    views = 8
    frames = 3
    data = generate_fake_data(h, w, views, frames, multi_frame)
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

    if not multi_frame:
        for multi_view_info in tensor_data["gt_multi_view"]:
            check(
                multi_view_info["meta"]["ignore_mask"]["ignore_mask_2d"],
                check_type,
                instance=torch.Tensor,
            )
    else:
        for sync_gt_multi_view in tensor_data["gt_multi_view"]:
            for multi_view_info in sync_gt_multi_view:
                check(
                    multi_view_info["meta"]["ignore_mask"]["ignore_mask_2d"],
                    check_type,
                    instance=torch.Tensor,
                )


def test_to_prepare_tempo_data_bev():
    # generate fake data
    h, w = 512, 960
    views = 6
    frames = 3
    data = generate_fake_data(h, w, views, frames)
    to_tensor3dv = ANCToTensorTemporal3DV(True)

    # all view data will store in key of "img"
    views_domain2nums = {
        "front": views,
        "side": 0,
        "round": 0,
        "narrow": 0,
    }
    prepare_bev = ANCPrepareTempoALLFrameDataBEV(
        views_domain2nums=views_domain2nums
    )
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
    prepare_bev = ANCPrepareTempoALLFrameDataBEV(
        views_domain2nums=views_domain2nums
    )
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
    prepare_bev = ANCPrepareTempoALLFrameDataBEV(
        views_domain2nums=views_domain2nums
    )
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
    prepare_bev = ANCPrepareTempoALLFrameDataBEV(
        views_domain2nums=views_domain2nums
    )
    data = generate_fake_data(h, w, 11, frames)
    tensor_data = to_tensor3dv(data)
    prepare_data = prepare_bev(tensor_data)

    check(prepare_data["img"], check_shape, shape=(frames, 3, h, w))
    check(prepare_data["side_img"], check_shape, shape=(frames * 5, 3, h, w))
    check(prepare_data["round_img"], check_shape, shape=(frames * 4, 3, h, w))
    check(prepare_data["narrow_img"], check_shape, shape=(frames, 3, h, w))

    check(prepare_data["gt_bev_seg"], check_shape, shape=(1, h, w))


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
        "imgs": [[test_img], [copy.deepcopy(test_img)]],
        "img_mask": [[ignore_mask], [~ignore_mask]],
    }
    img_mask = copy.deepcopy(data["img_mask"])
    transform = ANCTemporalApplyMaskOnImg(use_yuv_format)
    data = transform(data)
    for idx in range(2):
        if use_yuv_format:
            img = cv2.cvtColor(
                data["imgs"][idx][0].permute(1, 2, 0).numpy().astype(np.uint8),
                cv2.COLOR_YUV2BGR,
            )
        else:
            img = (
                data["imgs"][idx][0].permute(1, 2, 0).numpy().astype(np.uint8)
            )
        assert (img[img_mask[idx][0], :] == 0).all()


def test_temporal_bev3d_target_generator():
    vcs_range = (-35.2, -51.2, 105.6, 51.2)
    category2id_map = {
        "Vehicle": 0,
        "Car": 0,
        "Other": -99,
    }
    cls_hm_kernel = {
        0: 13,
    }
    data = generate_fake_annos(10, 4, True)
    gen = ANCTemporalBev3dTargetGenerator(
        1, (704 // 2, 512 // 2), vcs_range, cls_hm_kernel, category2id_map
    )
    data = gen(copy.deepcopy(data))
    assert isinstance(data["gt_bev_3d"], list)

    data = generate_fake_annos(10, 4, False)
    data = gen(data)
    assert isinstance(data["gt_bev_3d"], dict)


def test_temporal_multiview_target_generator():
    occlusion_attribute_dict = {
        "full_visible": 0,
        "occluded": 1,
        "heavily_occluded": 2,
        "invisible": 3,
    }
    gen = ANCTemporalMultiViewTargetGenerator(
        True,
        occlusion_attribute_dict,
        True,
    )
    data = generate_fake_multi_view_gt(10, 7, 4, 1000, 1000, True)
    data = gen(data)
    for gt_multi_view in data["gt_multi_view"]:
        assert "occlusion_multi_view" in gt_multi_view
    for img_mask in data["img_mask"]:
        assert len(img_mask) == 7

    data = generate_fake_multi_view_gt(10, 7, 4, 1000, 1000, False)
    data = gen(data)
    assert "occlusion_multi_view" in data["gt_multi_view"]
    assert len(data["img_mask"]) == 7


if __name__ == "__main__":
    pytest.main(["-s", __file__])
