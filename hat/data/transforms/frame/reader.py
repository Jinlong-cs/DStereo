# Copyright (c) Horizon Robotics. All rights reserved.


import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from pyramid_resizer import pyramid_resizer
except ImportError:
    pyramid_resizer = None


__all__ = ["ImgBufToYUV444", "YUVTurboJPEGDecoder", "ImgBufDecoder"]


@OBJECT_REGISTRY.register
class YUVTurboJPEGDecoder(object):
    """YUV JPEG Decoder.

    Args:
        to_string: Whether to decode jpeg into yuvi420_buf.
        If to_string is `False`, will return (Y, U, V) in np.ndarray type.
    """

    @require_packages("pyramid_resizer")
    def __init__(
        self,
        to_string: bool = True,
    ):
        self.to_string = to_string

    def __call__(self, data):
        img_bytes = data["img_buf"]
        if self.to_string:
            (
                yuv42x,
                img_w,
                img_h,
                sample_type,
            ) = pyramid_resizer.imdecode_yuv42x(img_bytes, decode=False)
            if sample_type == "yuvi420":
                yuvi420_buf = yuv42x
            elif sample_type == "yuv422p":
                yuvi420_buf = pyramid_resizer.yuv422p_str2yuvi420_str(
                    yuv42x, img_w, img_h
                )
            else:
                raise NotImplementedError(
                    "Sample type only support `yuvi420` and `yuv422p`."
                )

            data["img_buf"] = yuvi420_buf
            data["img_height"], data["img_width"] = img_h, img_w
            data["color_space"] = "yuv"
            return data
        else:
            y_img, u_img, v_img = pyramid_resizer.imdecode_yuv42x(
                img_bytes, decode=True
            )
            return y_img, u_img, v_img


@OBJECT_REGISTRY.register
class ImgBufToYUV444(object):
    def __call__(self, data):
        data["img"] = pyramid_resizer.yuvi420_str2yuv444_np(
            data["img_buf"], data["img_width"], data["img_height"]
        )
        data["img_shape"] = data["img"].shape

        return data


@OBJECT_REGISTRY.register
class ImgBufDecoder(object):
    """Decode JPEG from img_buf.

    Args:
        to_rgb: Whether to convert image from bgr to rgb.
    """

    def __init__(self, to_rgb: bool = False):
        self.to_rgb = to_rgb

    def __call__(self, data):
        image_np = np.frombuffer(data["img_buf"], dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        color_space = "bgr"
        if self.to_rgb:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            color_space = "rgb"

        data["img"] = image_np
        data["img_shape"] = image_np.shape
        data["color_space"] = color_space
        data["img_height"] = data.get("img_height", image_np.shape[0])
        data["img_width"] = data.get("img_width", image_np.shape[1])

        return data
