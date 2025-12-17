# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest

from hat.data.transforms.cyckps_transform import CysKpsLabelTransform


def gen_fake_data(img_shape):
    data = {}
    h, w, c = img_shape
    num_rois = 2
    data["img"] = np.random.randint(0, 255, img_shape, dtype=np.uint8)
    data["boxes"] = np.random.uniform(10, 200, (num_rois, 4))
    data["keypoints"] = np.random.uniform(40, 100, (num_rois, 6))
    data["gt_classes"] = np.array([1, 0])
    return data


@pytest.mark.parametrize(
    ["img_shape"],
    [
        pytest.param(
            (256, 256, 3),
        )
    ],
)
def test_cyckps_transform(img_shape):
    data = gen_fake_data(img_shape)
    labeltrans = CysKpsLabelTransform(
        roi_batch_size=4,
        crop_params={
            "border_value": None,
            "crop_method": "densebox",
            "output_wh": [128, 128],
            "inter_method": None,
            "keep_in_image": None,
            "max_crop_scale": 1.2,
            "min_crop_scale": 0.8,
            "norm_len": 106,
            "norm_method": "height",
            "padding_context": None,
        },
    )
    result = labeltrans(data)
    assert result["kps_cls_label"].shape == (4, 2, 8, 8)
    assert result["kps_cls_label_weight"].shape == (4, 2, 8, 8)
    assert result["kps_pos_offset"].shape == (4, 4, 8, 8)
    assert result["kps_pos_offset_weight"].shape == (4, 4, 8, 8)
    assert result["img"].shape == (4, 3, 128, 128)
