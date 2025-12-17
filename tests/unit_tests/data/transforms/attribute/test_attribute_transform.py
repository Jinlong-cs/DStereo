import os

import cv2
import pytest

from hat.data.transforms.attribute.attribute_transform import AttrCV2CropInput
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS, reason="HAT_BUCKET is required"
)  # noqa
def test_attr_crop_input_resize(_vis=False):

    img_path = os.path.join(
        HAT_BUCKET_PATH,
        "data/orig_data/voc/VOCdevkit/VOC2007/JPEGImages/000341.jpg",
    )

    img = cv2.imread(img_path)
    img_w_resize, img_h_resize = 128, 128
    target_size = (img_h_resize, img_w_resize)

    crop_resizer = AttrCV2CropInput(
        model_input_hw=target_size,
        detection_task_name="vehicle",
        center_crop_prob=0.5,
    )
    img_meta = dict(
        layout="hwc",
        img=img,
        img_shape=img.shape,
        img_anno={
            "vehicle": [
                {"data": [313.0, 113.0, 393.0, 200.0]},
                {"data": [193.0, 137.0, 312.0, 222.0]},
            ]
        },
    )

    transformed_img_meta = crop_resizer(img_meta)

    if _vis:
        for idx, roi_img in enumerate(
            transformed_img_meta["img"].permute(0, 2, 3, 1)
        ):
            cv2.imwrite(f"transformed_img_{idx}.jpg", roi_img.numpy())

    assert transformed_img_meta["img"].shape[-2:] == target_size
