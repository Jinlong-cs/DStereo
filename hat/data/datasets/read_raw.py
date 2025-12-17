# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Optional

import cv2
import numba as nb
import numpy as np

__all__ = ["read_raw", "vis_raw"]


@nb.jit()
def _transfer_12bit_mipi_data(raw_data, out_raw_data, width, height):
    pixel_length = 12
    size_src = int(width * height * pixel_length // 8)

    for i in range(int(size_src / 3)):
        src_p0 = raw_data[i * 3]
        src_p1 = raw_data[i * 3 + 1]
        src_px_lsb = raw_data[i * 3 + 2]

        tar_p0_msb = src_p0
        tar_p1_msb = src_p1
        tar_p0_lsb = ((src_px_lsb >> 0) & 0x0F) << 4
        tar_p1_lsb = ((src_px_lsb >> 4) & 0x0F) << 4

        out_raw_data[i * 2] = (tar_p0_msb << 4) + (tar_p0_lsb >> 4)
        out_raw_data[i * 2 + 1] = (tar_p1_msb << 4) + (tar_p1_lsb >> 4)

    return out_raw_data


def _transfer_mipi_data(raw_data, out_raw_data, width, height, bit_nums):
    if bit_nums == 12:
        out_raw_data = _transfer_12bit_mipi_data(
            raw_data, out_raw_data, width, height
        )
    else:
        raise NotImplementedError(
            "Don't support mipi mode when bit_nums is not equal to "
            "12, current bit_nums is %d" % bit_nums
        )

    return out_raw_data


def read_raw(
    raw_bytes: np.ndarray,
    height: int,
    width: int,
    bit_nums_upper: int,
    mipi: Optional[int] = 1,
    channels: Optional[int] = 1,
    pre_offset: Optional[int] = 0,
) -> np.ndarray:
    """Output raw data array given the raw path and data information.

    Args:
        raw_bytes: raw file of binary bytes.
        image_info: dict including following keys: height, width,
            bit_nums_upper, mipi, channels, pre_offset.

    Raises:
        ValueError: when bit_nums is bigger than 32.

    Returns:
        output raw data.
    """

    if pre_offset is None:
        pre_offset = 0

    out_raw_data = np.zeros(height * width * channels)

    if bit_nums_upper <= 8:
        np_type = np.uint8
        pre_offset = pre_offset
    elif bit_nums_upper <= 16:
        np_type = np.uint16
        pre_offset = pre_offset // 2
    elif bit_nums_upper <= 32:
        np_type = np.uint32
        pre_offset = pre_offset // 4
    else:
        raise ValueError(
            "Don't support bit_nums_upper which is bigger than 32, "
            "current bit_nums_upper is %d" % bit_nums_upper
        )

    out_raw_data = out_raw_data.astype(np_type)
    if not mipi:
        raw_data = np.frombuffer(raw_bytes, dtype=np_type)
        out_raw_data = raw_data[pre_offset:]
    else:
        raw_data = np.frombuffer(raw_bytes, dtype=np.uint8)
        out_raw_data = _transfer_mipi_data(
            raw_data, out_raw_data, width, height, bit_nums_upper
        )

    out_raw_data = np.reshape(out_raw_data, (height, width, channels))

    return out_raw_data


def vis_raw(image_info: Dict, write_path: str):
    """Visualize raw data and save it to write_path.

    Args:
        image_info: dict including following keys: raw_path, height,
            width, bit_nums_upper, mipi, channels, pre_offset.
        write_path: path to save 8 bit visualized data with
            jpg/png/bmp/etc suffix.
    """
    out_raw_data = read_raw(image_info["raw_path"], image_info)
    cv2.imwrite(
        write_path,
        out_raw_data[:, :, ::-1] >> (image_info["bit_nums_upper"] - 8),
    ).astype(np.uint8)
