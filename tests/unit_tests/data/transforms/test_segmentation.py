# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest

from hat.data.datasets.cityscapes import CITYSCAPES_LABLE_MAPPINGS
from hat.data.transforms.segmentation import (
    FlowRandomAffineScale,
    LabelRemap,
    PolygonToMask,
    ReformatLanePolygon,
    Scale,
    SegOneHot,
    SegRandomAffine,
    SegRandomCrop,
    SegRandomCutOut,
    SegResize,
    SegResizeAffine,
    SegReWeightByArea,
)
from hat.utils.package_helper import check_packages_available
from tests.utils import gen_fake_transforms_data


@pytest.mark.parametrize(
    ["src_h", "src_w", "crop_h", "crop_w", "cat_max_ratio"],
    [
        pytest.param(300, 400, 500, 300, 1.0),
        pytest.param(300, 400, 500, 300, 0.5),
    ],
)
def test_seg_random_crop(src_h, src_w, crop_h, crop_w, cat_max_ratio):
    data = gen_fake_transforms_data(src_w, src_h, layout="hwc")
    seg_random_croper = SegRandomCrop((crop_h, crop_w), cat_max_ratio)
    crop_data = seg_random_croper(data).copy()
    assert crop_data["img_shape"] == crop_data["img"].shape
    assert crop_data["pad_shape"] == crop_data["img"].shape
    assert crop_data["gt_seg"].shape == crop_data["img"].shape[:2]
    assert crop_data["img"].shape[0] <= crop_h
    assert crop_data["img"].shape[1] <= crop_w


def test_seg_reweight_by_area():
    data = gen_fake_transforms_data(40, 40, layout="hwc")
    seg_reweight_by_area = SegReWeightByArea(
        seg_num_classes=2, lower_bound=0.3
    )
    data["gt_seg"][:30] = 0
    data["gt_seg"][30:] = 1
    data = seg_reweight_by_area(data)
    assert "gt_seg_weight" in data
    assert abs(data["gt_seg_weight"].min() - 0.3) < 1e-4
    assert abs(data["gt_seg_weight"][:30].min() - 0.3) < 1e-4
    assert abs(data["gt_seg_weight"][:30].max() - 0.3) < 1e-4
    assert abs(data["gt_seg_weight"][30:].min() - 0.75) < 1e-4
    assert abs(data["gt_seg_weight"][30:].max() - 0.75) < 1e-4


def test_seg_label_remap():
    data = gen_fake_transforms_data(40, 40, layout="chw", return_tensor=True)

    transformed_data = LabelRemap(CITYSCAPES_LABLE_MAPPINGS)(data)

    assert data["gt_seg"].shape == transformed_data["gt_seg"].shape


def test_seg_one_hot():
    data = gen_fake_transforms_data(40, 40, layout="chw", return_tensor=True)
    data["gt_seg"] = data["gt_seg"].unsqueeze(0).unsqueeze(0)

    transformed_data = SegOneHot(10)(data)

    assert transformed_data["gt_seg"].shape == (1, 10, 40, 40)


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_seg_resize():
    data = gen_fake_transforms_data(40, 40, layout="chw", return_tensor=True)
    data["gt_seg"] = data["gt_seg"].unsqueeze(0)

    transformed_data = SegResize((20, 20))(data)

    assert transformed_data["img"].shape == (3, 20, 20)
    assert transformed_data["gt_seg"].shape == (1, 20, 20)


def test_seg_resize_affine():
    data = gen_fake_transforms_data(40, 40, layout="chw", return_tensor=False)
    gt_polygon = data["gt_polygons"][0].copy()

    transformed_data = SegResizeAffine((80, 80))(data)
    assert transformed_data["img"].shape == (3, 80, 80)
    assert transformed_data["gt_seg"].shape == (80, 80)
    assert (
        transformed_data["gt_polygons"][0] == gt_polygon * (80.0 / 40.0)
    ).any()

    transformed_data2 = SegResizeAffine((40, 40))(transformed_data)
    assert transformed_data2["img"].shape == (3, 40, 40)
    assert transformed_data2["gt_seg"].shape == (40, 40)
    assert (transformed_data2["gt_seg"] == data["gt_seg"]).any()
    assert (transformed_data2["gt_polygons"][0] == gt_polygon).any()


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["degrees", "translate", "scale", "translate_p", "scale_p", "c"],
    [
        pytest.param(10, (0.5, 0.5), (0.5, 2), 1.0, 1.0, 3),
        pytest.param(10, (0.5, 0.5), (0.5, 2), 0.5, 0.5, 3),
        pytest.param(10, (0.5, 0.5), (0.5, 2), 1.0, 1.0, 6),
        pytest.param(10, (0.5, 0.5), (0.5, 2), 0.5, 0.5, 6),
    ],
)
def test_seg_random_affine(degrees, translate, scale, translate_p, scale_p, c):
    data = gen_fake_transforms_data(
        40, 40, layout="chw", return_tensor=True, c=c
    )
    data["gt_seg"] = data["gt_seg"].unsqueeze(0)

    transformed_data = SegRandomAffine(
        degrees=degrees,
        translate=translate,
        scale=scale,
        translate_p=translate_p,
        scale_p=scale_p,
    )(data)

    assert (
        transformed_data["img"].shape[-2:]
        == transformed_data["gt_seg"].shape[-2:]
    )

    assert (
        transformed_data["img"].shape[-2:]
        == transformed_data["gt_flow"].shape[-2:]
    )


@pytest.mark.parametrize(
    ["c"],
    [
        pytest.param(3),
        pytest.param(6),
    ],
)
def test_seg_scale(c):
    data = gen_fake_transforms_data(
        40, 40, layout="chw", return_tensor=True, c=c
    )
    data["gt_seg"] = data["gt_seg"].unsqueeze(0).unsqueeze(0)
    data["img"] = data["img"].unsqueeze(0)
    data["gt_flow"] = data["gt_flow"].unsqueeze(0)

    transformed_data = Scale((0.5, 1.0, 2.0))(data)

    assert len(transformed_data["gt_seg"]) == 3
    assert transformed_data["gt_seg"][0].shape == (1, 1, 20, 20)
    assert transformed_data["gt_seg"][1].shape == (1, 1, 40, 40)
    assert transformed_data["gt_seg"][2].shape == (1, 1, 80, 80)
    assert transformed_data["gt_flow"][0].shape == (1, 2, 20, 20)
    assert transformed_data["gt_flow"][1].shape == (1, 2, 40, 40)
    assert transformed_data["gt_flow"][2].shape == (1, 2, 80, 80)
    assert transformed_data["gt_ori_flow"].shape == (1, 2, 40, 40)


@pytest.mark.parametrize(
    ["c"],
    [
        pytest.param(6),
    ],
)
def test_cv_flow_random_affine_scale(c):
    data = gen_fake_transforms_data(
        40, 40, layout="chw", return_tensor=True, c=c
    )
    data["img"] = data["img"]
    data["gt_flow"] = data["gt_flow"]

    transformed_data = FlowRandomAffineScale(0.5, 0.05)(data)
    assert transformed_data["gt_flow"].shape == (2, 40, 40)
    assert transformed_data["img"].shape == (c, 40, 40)


@pytest.mark.parametrize(
    ["c"],
    [
        pytest.param(3),
    ],
)
def test_cv_random_cutout(c):
    data = gen_fake_transforms_data(
        40, 40, layout="hwc", return_tensor=False, c=c
    )
    data["img"] = data["img"]
    before = (data["img"] == 0).sum()
    data["gt_depth"] = data["gt_depth"]
    data["gt_seg"] = data["gt_seg"]

    transformed_data = SegRandomCutOut(
        prob=1,
        n_holes=(2, 6),
        cutout_ratio=[
            (0.05, 0.05),
            (0.02, 0.02),
            (0.07, 0.07),
            (0.1, 0.1),
            (0.2, 0.2),
        ],
        fill_in=(0, 0, 0),
        seg_fill_in=None,
    )(data)
    assert (transformed_data["img"] == 0).sum() > before
    assert transformed_data["img"].shape == (40, 40, c)
    assert transformed_data["gt_depth"].shape == (40, 40)
    assert transformed_data["gt_seg"].shape == (40, 40)


@pytest.mark.skipif(
    not check_packages_available("pycocotools", raise_exception=False),
    reason="need pycocotools",
)
@pytest.mark.parametrize("hw", [5, 40])
@pytest.mark.parametrize("filter_emtpy", [True, False])
@pytest.mark.parametrize("replace_gt_seg", [True, False])
@pytest.mark.parametrize("replace_orig_gt_seg", [True, False])
@pytest.mark.parametrize("add_bbox", [True, False])
def test_polygon_to_mask(
    hw, filter_emtpy, replace_gt_seg, replace_orig_gt_seg, add_bbox
):
    data = gen_fake_transforms_data(hw, hw, layout="hwc", return_tensor=False)
    data.pop("gt_masks", None)
    data.pop("gt_bboxes", None)
    data["gt_polygons"] = [
        np.array(
            [[10, 10], [10, 20], [20, 20], [20, 30], [30, 30], [30, 10]]
        ).astype(np.float32)
    ]
    data["gt_labels"] = [np.array([1])]
    transformed_data = PolygonToMask(
        filter_emtpy=filter_emtpy,
        replace_gt_seg=replace_gt_seg,
        replace_orig_gt_seg=replace_orig_gt_seg and replace_gt_seg,
        add_bbox=add_bbox,
    )(data)

    expected_area = 0 if hw == 5 else 300
    if replace_gt_seg:
        assert transformed_data["gt_seg"].sum() == expected_area
        if replace_orig_gt_seg:
            assert transformed_data["orig_gt_seg"].sum() == expected_area
    else:
        gt_masks = transformed_data["gt_masks"]
        if filter_emtpy and hw == 5:
            assert gt_masks.shape == (0,)
        else:
            assert gt_masks.shape == (1, hw, hw)
            assert gt_masks.sum() == expected_area

    if add_bbox:
        assert "gt_bboxes" in transformed_data
    else:
        assert "gt_bboxes" not in transformed_data


@pytest.mark.parametrize("remove_ignore", [True, False])
@pytest.mark.parametrize("add_attribute_num", [2, 3])
@pytest.mark.parametrize("target_position, target_value", [(-1, 2), (-2, 3)])
def test_reformat_lane_polygon(
    remove_ignore, add_attribute_num, target_position, target_value
):
    data = gen_fake_transforms_data(40, 40, layout="hwc", return_tensor=False)
    data.pop("gt_masks", None)
    data.pop("gt_bboxes", None)
    data["gt_polygons"] = [
        np.array([[10, 10], [10, 20], [20, 20], [30, 10]]).astype(np.float32),
        np.array([[10, 10], [20, 30], [30, 10]]).astype(np.float32),
        np.array([[1, 1], [5, 5], [8, 1]]).astype(np.float32),
    ]
    data["gt_labels"] = [
        np.array([1, 1]),
        np.array([1, 2]),
        np.array([255, 1]),
    ]

    converts = [
        {
            "src_position": 1,
            "src_value": 2,
            "target_position": target_position,
            "target_value": target_value,
        },
    ]
    transformed_data = ReformatLanePolygon(
        add_attribute_num=add_attribute_num,
        class_mapping={1: 0, 2: 1},
        class_position=1,
        converts=converts,
        ignore_index=255,
        remove_ignore=remove_ignore,
    )(data)
    gt_polygons = transformed_data["gt_polygons"]
    if remove_ignore:
        assert len(gt_polygons) == 1
    else:
        assert len(gt_polygons) == 2

    gt_labels = transformed_data["gt_labels"]
    assert len(gt_labels) == len(gt_polygons)
    assert len(gt_labels[0]) == 2 + add_attribute_num
    assert gt_labels[0][target_position] == target_value
