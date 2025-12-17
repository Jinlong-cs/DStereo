import os
import pickle

import cv2
import horizon_plugin_pytorch as horizon
import pytest
from horizon_plugin_pytorch.march import March

from hat.data.transforms.frame.reader import YUVTurboJPEGDecoder
from hat.data.transforms.frame.resize import (
    BPUPyramidResizer,
    CV2AdptiveResolutionInput,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

try:
    import pyramid_resizer  # noqa: F401

    _PYRAMID_RESIZER = True
except ImportError:
    _PYRAMID_RESIZER = False

try:
    import hat_sim
except ImportError:
    hat_sim = None


@pytest.mark.parametrize(
    [
        "pyramid_type",
        "layout",
        "march",
    ],
    [
        pytest.param("ips", "chw", "j3"),
        pytest.param("ips", "hwc", "j3"),
        pytest.param("ips", "chw", "j5"),
        pytest.param("ipu", "chw", "j5"),
        pytest.param("ipu", "hwc", "j3"),
    ],
)
@pytest.mark.skipif(
    not _PYRAMID_RESIZER, reason="pyramid_resizer is required"
)  # noqa
@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS, reason="HAT_BUCKET is required"
)  # noqa
def test_bpu_pyramid_resize(tmpdir, pyramid_type, layout, march):
    if horizon.get_march() == March.BAYES and hat_sim is None:
        return

    if march == "j5":
        horizon.set_march(March.BAYES)
    elif march == "j3":
        horizon.set_march(March.BERNOULLI2)
    else:
        horizon.set_march(March.BERNOULLI)

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

    decoder = YUVTurboJPEGDecoder(to_string=True)

    with open(resized_img_path, "rb") as f:
        img_buf = f.read()

    data = dict(
        img_buf=img_buf,
        layout=layout,
    )
    data = decoder(data)
    yuvi420_buf = data["img_buf"]
    img_h, img_w = data["img_height"], data["img_width"]

    # test size: (1280, 720) --> (320, 180)
    resizer = BPUPyramidResizer(
        scale_wh=(0.25, 0.25),
        pyramid_type=pyramid_type,
        pyramid_idx=1,
    )

    data = resizer(data)
    yuvi420_buf_new = data["img_buf"]
    resize_hw = (data["img_height"], data["img_width"])
    assert yuvi420_buf_new
    assert resize_hw == (img_h // 4, img_w // 4)

    resizer_data = pickle.dumps(resizer)
    assert resizer_data

    resizer_new = pickle.loads(resizer_data)
    data_new = dict(img_buf=yuvi420_buf, img_height=img_h, img_width=img_w)
    data_new = resizer_new(data_new)
    yuvi420_buf_new2 = data_new["img_buf"]
    resize_hw2 = (data_new["img_height"], data_new["img_width"])
    assert yuvi420_buf_new2
    assert resize_hw2 == (img_h // 4, img_w // 4)


@pytest.mark.parametrize(
    ["scale_type"],
    [
        pytest.param("MIN"),
    ],
)
@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS, reason="HAT_BUCKET is required"
)  # noqa
def test_adaptive_resolution_input_resize(scale_type):

    img_path = os.path.join(
        HAT_BUCKET_PATH,
        "data/orig_data/voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
    )

    img = cv2.imread(img_path)
    img_w_resize, img_h_resize = 320, 180
    target_size = (img_h_resize, img_w_resize)

    resizer = CV2AdptiveResolutionInput(
        model_input_hw=target_size, scale_type=scale_type
    )
    img_meta = dict(
        layout="hwc",
        img=img,
        img_shape=img.shape,
        img_height=img.shape[0],
        img_width=img.shape[1],
    )

    # test size: (xxx, xxx) --> (320, 180)
    resized_img_meta = resizer(img_meta)

    assert resized_img_meta["img"].shape[:2] == target_size


if __name__ == "__main__":
    pytest.main(["-s", __file__])
