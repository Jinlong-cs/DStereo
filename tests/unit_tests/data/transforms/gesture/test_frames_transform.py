# Copyright (c) Horizon Robotics. All rights reserved.
import copy

import numpy as np
import pytest
from init_path import get_data

from hat.data.transforms.gesture import (
    ActionImgClipAlbuTrans,
    ActionImgClipToTensor,
)
from hat.utils.package_helper import check_packages_available

try:
    import albumentations

    assert (
        albumentations.__version__ > "1.0.0"
    ), "albumentations version should be greater than 1.0.0"
except ImportError:
    AlBU_AVAILABLE = False
else:
    AlBU_AVAILABLE = True


@pytest.mark.skipif(not AlBU_AVAILABLE, reason="albumentations is required")
def test_frames_albu(_vis=False):
    (data,) = get_data("albu_input_output.pkl")
    albu_param = {
        "albu_same_in_seq": True,
        "albu_params": [
            {
                "name": "Downscale",  # shift
                "scale_min": 0.4,
                "scale_max": 0.8,
                # 'interpolation': cv2.INTER_LINEAR,
                "p": 0.6,
            },
            {"name": "GaussianBlur", "blur_limit": (3, 7), "p": 0.2},
            {
                "name": "MotionBlur",  # shift
                "blur_limit": (3, 9),
                "p": 0.4,
            },
            {
                "name": "JpegCompression",
                "quality_lower": 50,
                "quality_upper": 90,
                "p": 0.6,
            },
            {"name": "ToGray", "p": 0.3},
            {
                "name": "HueSaturationValue",
                "hue_shift_limit": 20,
                "sat_shift_limit": 30,
                "val_shift_limit": 15,
                "p": 0.6,
            },
            {
                "name": "RandomBrightnessContrast",
                "brightness_limit": 0.30,
                "contrast_limit": 0.20,
                "p": 1.0,
            },
            {"name": "FancyPCA", "alpha": 0.06, "p": 0.4},
            {
                "name": "GaussNoise",
                "var_limit": (10.0, 50.0),
                "mean": 0,
                "p": 0.6,
            },
        ],
    }
    aicat = ActionImgClipAlbuTrans(**albu_param)
    ref_data = copy.deepcopy(data["input"])
    pred_data = aicat(data["input"])
    assert pred_data["frames"][0].shape == (128, 128, 3)

    if _vis:
        import cv2

        for idx, (raw_img, albu_hat_img, albu_gluon_img) in enumerate(
            zip(
                ref_data["frames"],
                pred_data["frames"],
                data["output"]["frames"],
            )
        ):
            img = np.concatenate(
                (raw_img, albu_hat_img, albu_gluon_img), axis=1
            )
            # rgb from rec.
            cv2.imwrite(
                f"albu_aug_{idx}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_totensor():
    (data,) = get_data("albu_input_output.pkl")
    imgclip_to_tensor_params = {
        "to_yuv": True,
        "tensor_layout": "chw",
        "img_layout": "rgb",  # mxnet input layout
        "mean": [128.0, 128.0, 128.0],
        "std": [128.0, 128.0, 128.0],
    }
    aictt = ActionImgClipToTensor(**imgclip_to_tensor_params)
    pred_data = aictt(data["output"])
    assert pred_data["frames"].shape == (8, 3, 128, 128)
    assert pred_data["frame_layout"] == "hwc"
