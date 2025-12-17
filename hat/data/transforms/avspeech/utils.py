import os
import subprocess
from typing import Sequence

import cv2
import numpy as np

__all__ = ["to_tuple"]

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple.

    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be
                (offset - value, offset + value).
            If value is tuple, return value would be
                value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError(
            "Argument param must be either scalar (int, float) or tuple"
        )

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


def _brightness_contrast_adjust_non_uint(
    img, alpha=1, beta=0, beta_by_max=False
):
    dtype = img.dtype
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)

    return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)


def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = np.dtype("uint8")

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = np.arange(0, max_value + 1).astype("float32")

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += beta * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img


def is_grayscale_image(image):
    return (len(image.shape) == 2) or (
        len(image.shape) == 3 and image.shape[-1] == 1
    )


def is_rgb_image(image):
    return len(image.shape) == 3 and image.shape[-1] == 3


def call_subproc_cmd(cmd, env=None):
    if env is None:
        env = os.environ.copy()
    assert type(cmd) == str or type(cmd) == list
    if isinstance(cmd, str):
        subprocess.check_call(cmd, shell=True, env=env)
    elif isinstance(cmd, list):
        subprocess.check_call(cmd, env=env)
    else:
        raise NotImplementedError
