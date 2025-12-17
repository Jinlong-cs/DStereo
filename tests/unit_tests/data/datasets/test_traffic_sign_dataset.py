# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest

from hat.core.proj_spec.classification import _get_classification_labels
from hat.registry import build_from_registry
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

try:
    import albumentations
except ImportError:
    albumentations = None
try:
    import hatbc
except ImportError:
    hatbc = None

input_hw = (64, 64)
lt_id = 0
rb_id = 2
show_image_info = False
show_origin_class_id = False
show_origin_image = False
show_origin_bbox = False

base_transformer = dict(
    albumentations_type="Compose",
    p=0.5,
    transforms=[
        dict(
            albumentations_type="GaussNoise",
            var_limit=(0.0, 20.0),
            mean=0,
            per_channel=True,
            p=0.3,
        ),
        dict(
            albumentations_type="Affine",
            rotate=(-10, 10),
            shear=(-10, 10),
            cval=(128, 128, 128),
            p=0.3,
        ),
    ],
)

color_transformer = dict(
    albumentations_type="Compose",
    transforms=[
        dict(
            albumentations_type="ColorJitter",
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.0,
            p=0.5,
        ),
        dict(
            albumentations_type="RGBShift",
            r_shift_limit=(-10, 10),
            g_shift_limit=(-10, 10),
            b_shift_limit=(-10, 10),
            p=0.5,
        ),
    ],
)

flip_transformer = dict(
    albumentations_type="Compose",
    transforms=[
        dict(albumentations_type="HorizontalFlip", p=0.5),
    ],
)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HDLTAlgorithm bucket")
@pytest.mark.skipif(
    albumentations is None, reason="albumentations is required"
)
@pytest.mark.skipif(hatbc is None, reason="hatbc is required")
@pytest.mark.parametrize(
    ["task_name", "num_classes"],
    [
        pytest.param("traffic_sign_cn_category_classification", 258),
        pytest.param("traffic_sign_occlusion_classification", 4),
    ],
)
def test_traffic_sign_dataset(task_name, num_classes):
    classname2idxs = list([i for i in range(1, num_classes + 1)])  # noqa
    channel_labels = _get_classification_labels(task_name, str(num_classes))

    rec_path_i = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/mono_traffic_sign_multitask/data.rec"
    )
    anno_path_i = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/mono_traffic_sign_multitask/data.anno.pb_rec",
    )
    dataset_cfg = dict(
        type="TrafficSignDenseBoxImageRecordDataset",
        rec_path=rec_path_i,
        anno_path=anno_path_i,
        to_rgb=True,
        task_type="classification",
        transforms=[
            dict(
                type="DecodeDenseBoxDatasetToDetFormatWithImageInfo",
                selected_class_ids=classname2idxs,
                lt_point_id=lt_id,
                rb_point_id=rb_id,
                show_image_info=False,
                class_id_key="class_id",
            ),
            dict(
                type="TrafficSignROICropResizeTransform",
                crop_method="RandomCropNearBBoxV2",
                max_jitter_ratio=(0.1, 0.1, 0.1, 0.1),
                random_jitter_bilateral=True,
                target_wh=input_hw[::-1],
                filter_valid_roi=True,
                min_valid_area=16,
                min_edge_size=5,
                random_roi_class_id=-1,
                show_origin_image=show_origin_image,
                show_origin_bbox=show_origin_bbox,
                show_debug_info=False,
            ),
            dict(
                type="TrafficSignImageAugmentation",
                base_transformer=base_transformer,
                color_transformer=color_transformer,
                flip_transformer=flip_transformer,
                flip_trans_all=False,
                flip_label_mapping=None,
                color_trans_all=False,
                color_trans_types=None,
                classname2idxs=dict(zip(channel_labels, classname2idxs)),
            ),
            dict(
                type="ClassIdRemap",
                selected_class_ids=classname2idxs,
            ),
            dict(
                type="ClassificationLabelTs",
                to_tensor=True,
                to_yuv=False,
                task_type="classification",
                show_image_info=show_image_info,
                show_origin_class_id=show_origin_class_id,
                show_origin_image=show_origin_image,
                show_origin_bbox=show_origin_bbox,
            ),
        ],
    )

    dataset = build_from_registry(dataset_cfg)

    for index in range(len(dataset)):
        data = dataset[index]
        if len(data) > 0:
            assert "img" in data[0].keys()
        if index > 1:
            break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
