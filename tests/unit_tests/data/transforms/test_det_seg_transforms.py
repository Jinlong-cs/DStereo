import os
import pickle

import numpy as np
import pytest

from hat.core.proj_spec.detection import classname2id
from hat.data.datasets.det_seg_2d_anno_dataset import (
    DetSeg2DAnnoDatasetToDetFormat,
    DetSeg2DAnnoDatasetToROIFormat,
)
from hat.data.transforms.detection import (
    DetAffineAugTransformer,
    IterableDetRoIListTransform,
    IterableDetRoITransform,
    PadDetData,
)
from hat.data.transforms.flank_transform import (
    PadFlankData,
    VehicleFlankAffineAugTransform,
    VehicleFlankRoiTransform,
)
from hat.data.transforms.kps_transform import KPSIterableDetRoITransform
from hat.data.transforms.roi_detection_transform import (
    PadRoIDetData,
    ROIDetectionIterableDetRoITransform,
)
from hat.data.transforms.semantic_seg_transform import (
    SemanticSegAffineAugTransformerEx,
)
from hat.data.transforms.transform_3d import (
    Heatmap3DDetectionLableGenerate,
    Image3DTransform,
    MaskImageEdgeTransform,
)
from tests import HAT_BUCKET_PATH

try:
    import pycocotools
except ImportError:
    pycocotools = None

root = f"{HAT_BUCKET_PATH}/users/zihan.qiu/jenkins_test_transform_data"

input_hw = resize_hw = (640, 1024)
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1


def test_det_transform():
    task = "cyclist_detection"

    classnames = ["cyclist"]
    classname2idxs = list(map(lambda x: classname2id[x], classnames))
    transforms = [
        DetSeg2DAnnoDatasetToDetFormat(
            selected_class_ids=classname2idxs,
            lt_point_id=0,
            rb_point_id=2,
        ),
        IterableDetRoITransform(
            target_wh=input_hw[::-1],
            resize_wh=resize_hw[::-1],
            img_scale_range=(0.7, 1.0 / 0.7),
            roi_scale_range=(0.5, 2.0),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=inter_method,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            pixel_center_aligned=pixel_center_aligned,
            min_valid_area=8,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=2,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            reselect_ratio=-1,
            clip_bbox=False,
            keep_aspect_ratio=True,
        ),
        PadDetData(
            max_gt_boxes_num=300,
            max_ig_regions_num=100,
        ),
    ]

    with open(os.path.join(root, task + ".pkl"), "rb") as handle:
        data = pickle.load(handle)
    for transform in transforms:
        data = transform(data)
    assert isinstance(data, dict)


def test_det_roi_transform():
    classname2idxs = [2]
    task = "rear_plate_detection"

    transforms = [
        DetSeg2DAnnoDatasetToROIFormat(
            selected_class_ids=classname2idxs,
            lt_point_id=10,
            rb_point_id=12,
            parent_lt_point_id=0,
            parent_rb_point_id=2,
            use_parent=True,
            parent_id=1,
        ),
        ROIDetectionIterableDetRoITransform(
            # roi transform
            target_wh=input_hw[::-1],
            resize_wh=resize_hw[::-1],
            img_scale_range=(0.6, 1.0),
            roi_scale_range=(0.7, 1.0 / 0.7),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=pixel_center_aligned,
            inter_method=inter_method,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            min_valid_area=10,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=4,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            clip_bbox=False,
            # person center and head center must in image
            allow_outside_center=True,  # True is better than False
            pixel_center_aligned=pixel_center_aligned,
            keep_aspect_ratio=True,
        ),
        PadRoIDetData(
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
    ]

    with open(os.path.join(root, task + ".pkl"), "rb") as handle:
        data = pickle.load(handle)
    for transform in transforms:
        data = transform(data)
    assert isinstance(data, dict)


def test_det_kps_transform():
    task = "vehicle_wheel_kps"
    transforms = [
        KPSIterableDetRoITransform(
            kps_num=2,
            # roi transform
            target_wh=resize_hw[::-1],
            resize_wh=resize_hw[::-1],
            img_scale_range=(0.5, 2.0),
            roi_scale_range=(0.7, 1.0 / 0.7),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=inter_method,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            min_valid_area=100,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=10,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            clip_bbox=False,
            pixel_center_aligned=pixel_center_aligned,
            min_kps_distance=4,
            keep_aspect_ratio=True,
        ),
        PadDetData(
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
    ]

    with open(os.path.join(root, task + ".pkl"), "rb") as handle:
        data = pickle.load(handle)
    for transform in transforms:
        data = transform(data)
    assert isinstance(data, dict)


def test_seg_transform():
    task = "lane_parsing"
    transforms = [
        SemanticSegAffineAugTransformerEx(
            target_wh=resize_hw[::-1],
            inter_method=10,
            label_scales=[1.0],
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            flip_prob=0.5,
            label_padding_value=-1,
            rand_translation_ratio=0.0,
            center_aligned=False,
            rand_scale_range=(1.0, 1.0),
            resize_wh=resize_hw[::-1],
            adapt_diff_resolution=True,
        ),
    ]

    with open(os.path.join(root, task + ".pkl"), "rb") as handle:
        data = pickle.load(handle)
    for transform in transforms:
        data = transform(data)
    assert isinstance(data, dict)


def test_random_mask_transform():
    task = "vehicle_3d_detection"
    transforms = [
        MaskImageEdgeTransform(
            # left, top, right, bottom
            mask_ranges=((1, 100), (1, 100), (1848, 1920), (1180, 1280)),
            seed=0,
            prob=1.0,
            image_channel_order="hwc",
        )
    ]

    with open(os.path.join(root, task + ".pkl"), "rb") as handle:
        data = pickle.load(handle)
    origin_img = data["img"]
    for transform in transforms:
        data = transform(data)

    assert isinstance(data, dict)
    assert ((origin_img - data["img"]) >= 0).all()


@pytest.mark.skipif(pycocotools is None, reason="need pycocotools")
def test_3d_transform():
    task = "vehicle_3d_detection"
    classnames = ["vehicle"]
    num_classes = len(classnames)
    input_wh = input_hw[::-1]
    down_stride = 4
    keep_res = False  # whether use original image size
    normalize_depth = True
    focal_length_default = 1105.08
    max_depth = 50
    max_objs = 100
    undistort_depth_uv = True

    classid_map = {1: -1, 2: 0, 3: -1, 4: 0, 5: 0, 6: 0, 7: 0, 8: -1}

    transforms = [
        Image3DTransform(
            input_wh=input_wh,
            keep_res=keep_res,
            shift=np.array([0, 0], dtype=np.float32),
            keep_aspect_ratio=False,
        ),
        Heatmap3DDetectionLableGenerate(
            num_classes=num_classes,
            classid_map=classid_map,
            normalize_depth=normalize_depth,
            focal_length_default=focal_length_default,
            alpha_in_degree=False,
            down_stride=down_stride,
            use_bbox2d=True,
            use_project_bbox2d=False,
            enable_ignore_area=True,
            shift=np.array([0, 0], dtype=np.float32),
            filtered_name="__front_0820__",
            min_box_edge=8,  # original image size
            max_depth=max_depth,
            max_objs=max_objs,
            undistort_2dcenter=True,
            undistort_depth_uv=undistort_depth_uv,
            pe_config=None,
        ),
    ]

    with open(os.path.join(root, task + ".pkl"), "rb") as handle:
        data = pickle.load(handle)
    for transform in transforms:
        data = transform(data)
    assert isinstance(data, dict)


@pytest.mark.skipif(pycocotools is None, reason="need pycocotools")
def test_flank_transform():
    task = "vehicle_ground_line"
    transforms = [
        VehicleFlankRoiTransform(
            # roi transform
            target_wh=input_hw[::-1],
            resize_wh=None if resize_hw is None else resize_hw[::-1],
            img_scale_range=(0.5, 2.0),
            roi_scale_range=(0.7, 1.0 / 0.7),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=10,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            min_valid_area=80,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=10,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            pixel_center_aligned=pixel_center_aligned,
            keep_aspect_ratio=True,
            min_flank_width=4,
            min_flank_width_overlap=0.1,
        ),
        PadFlankData(
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
    ]

    with open(os.path.join(root, task + ".pkl"), "rb") as handle:
        data = pickle.load(handle)
    for transform in transforms:
        data = transform(data)
    assert isinstance(data, dict)


def test_detaffineaugtransform():
    transfrom = DetAffineAugTransformer(
        target_wh=(512, 192),
        center_aligned=True,
        flip_prob=0.5,
    )
    with open(os.path.join(root, "detaffineaugtransform.pkl"), "rb") as handle:
        data = pickle.load(handle)
    data = transfrom(data)
    assert isinstance(data, dict)
    assert "img" in data.keys()
    assert "gt_boxes" in data.keys()
    assert "im_hw" in data.keys()
    assert all(data["im_hw"] == data["img"].shape[1:])
    assert len(data["gt_boxes"][0]) == 5


def test_vehicleflankaugtransform():
    transfrom = VehicleFlankAffineAugTransform(
        target_wh=(512, 192),
    )
    with open(os.path.join(root, "vehicle_ground_line.pkl"), "rb") as handle:
        data = pickle.load(handle)
    data = transfrom(data)
    assert isinstance(data, dict)
    assert "img" in data.keys()
    assert "gt_boxes" in data.keys()
    assert "im_hw" in data.keys()
    assert all(data["im_hw"] == data["img"].shape[1:])
    assert len(data["gt_boxes"][0]) == 5
    assert len(data["gt_flanks"][0]) == 9


def test_roi_list_transform():
    task = "cyclist_detection"

    classnames = ["cyclist"]
    classname2idxs = list(map(lambda x: classname2id[x], classnames))
    transforms = [
        DetSeg2DAnnoDatasetToDetFormat(
            selected_class_ids=classname2idxs,
            lt_point_id=0,
            rb_point_id=2,
        ),
        IterableDetRoIListTransform(
            target_wh=input_hw[::-1],
            resize_wh=resize_hw[::-1],
            img_scale_range=(0.7, 1.0 / 0.7),
            roi_scale_range=(0.5, 2.0),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=inter_method,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            pixel_center_aligned=pixel_center_aligned,
            min_valid_area=8,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=2,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            reselect_ratio=-1,
            clip_bbox=False,
            keep_aspect_ratio=True,
            roi_list=np.array(
                [
                    [1536, 431, 1920, 623],
                    [704, 468, 1216, 660],
                    [0, 431, 384, 623],
                    [720, 318, 1232, 510],
                ]
            ),
            append_gt=True,
        ),
        PadDetData(
            max_gt_boxes_num=300,
            max_ig_regions_num=100,
        ),
    ]

    with open(os.path.join(root, task + ".pkl"), "rb") as handle:
        data = pickle.load(handle)
    for transform in transforms:
        data = transform(data)
    assert isinstance(data, dict)


test_vehicleflankaugtransform()
