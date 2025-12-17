import operator
import os

import cv2
import pytest
import torchvision

from hat.data.datasets.pack_dataset import PackDataset
from hat.data.transforms.frame.crop import (
    CropImgPatch,
    DynamicCropImgPatch,
    crop_roi_error_msg,
)
from hat.data.transforms.frame.reader import YUVTurboJPEGDecoder
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

try:
    import pyramid_resizer
except ImportError:
    pyramid_resizer = None


@pytest.mark.skipif(pyramid_resizer is None, reason="need pyramid_resizer")
@pytest.mark.parametrize(
    ["is_buf"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS, reason="HAT_BUCKET is required"
)  # noqa
def test_crop_img_patch(tmpdir, is_buf):

    img_path = os.path.join(
        HAT_BUCKET_PATH,
        "data/orig_data/voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
    )

    # Generate available image can be used in pryamid_resize
    img = cv2.imread(img_path)
    img_w_resize, img_h_resize = 1280, 720
    target_size = (img_w_resize, img_h_resize)
    resized_img = cv2.resize(img, target_size)
    resized_img_path = os.path.join(tmpdir, "resized_img.jpg")
    cv2.imwrite(resized_img_path, resized_img)

    if is_buf:
        decoder = YUVTurboJPEGDecoder(to_string=True)
        with open(resized_img_path, "rb") as f:
            img_buf = f.read()
        data = dict(img_buf=img_buf)
        data = decoder(data)
    else:
        data = dict(
            img=resized_img,
            img_height=img_h_resize,
            img_width=img_w_resize,
            layout="hwc",
        )

    # crop size: (1280, 720) --> (320, 180)
    crop_roi = (0, 0, 320, 180)
    img_h, img_w = (crop_roi[3] - crop_roi[1], crop_roi[2] - crop_roi[0])
    crop = CropImgPatch(
        static_roi=crop_roi, img_color="yuvi420", is_buf=is_buf
    )

    data = crop(data)
    if is_buf:
        img = data["img_buf"]
    else:
        img = data["img"]
    crop_hw = (data["img_height"], data["img_width"])
    assert img is not None
    assert crop_hw == (img_h, img_w)


@pytest.mark.parametrize(
    [
        "pack_path",
        "fp_x",
        "fp_y",
        "w",
        "h",
        "result",
    ],
    [
        pytest.param(
            "users/jianhang.he/hat_unit_test/data/dynamic_crop/ADAS_20230412-154441_286_$Index.pack",  # noqa
            256,
            72,
            512,
            192,
            (720, 318, 1232, 510),
        ),
    ],
)
@pytest.mark.skipif(
    (not HAT_BUCKET_EXISTS) or crop_roi_error_msg,
    reason="HAT_BUCKET and crop_roi is required",
)  # noqa
def test_dynamic_crop_img_patch(pack_path, fp_x, fp_y, w, h, result):

    pack_path = os.path.join(
        HAT_BUCKET_PATH,
        pack_path,
    )
    roi_input = (
        fp_x,
        fp_y,
        w,
        h,
    )
    val_transforms = torchvision.transforms.Compose(
        [YUVTurboJPEGDecoder(to_string=True), DynamicCropImgPatch(roi_input)]
    )
    dataset = PackDataset(
        pack_path=pack_path,
        pix_format="rgb",
        camera_view_names=["camera_rear"],
        camera_calib=True,
        transforms=val_transforms,
        to_buf=True,
        camera_calib_key="calib_all",
        crop_roi=True,
        with_cam=False,
    )

    for data in dataset:
        assert operator.eq(data["transform_meta"][-1]["eval_roi"], result)
        break
