# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.

import json
from typing import Dict, Tuple

import cv2
import numpy as np

from hat.utils.pack_type.mxrecord import MXRecord, unpack
from hat.utils.pack_type.recordio_pb2 import RecordUnit


def encode_img(
    img: np.ndarray, quality: int = 95, img_fmt: str = ".jpg"
) -> bytes:
    """encode_img.

    对图片进行编码返回编码后的字符串


    Args:
        img : 图片的矩阵, 形状是 HxWxC 的
        quality : 压缩图像的质量，默认为95%
        img_fmt : 压缩图片的格式，默认为 ".jpg"

    Returns:
        str 图片压缩后的buf str
    """
    jpg_formats = [".JPG", ".JPEG"]
    png_formats = [".PNG"]
    encode_params = None
    if img_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
    ret, buf = cv2.imencode(img_fmt, img, encode_params)
    assert ret, "failed to encode image"
    return buf


def decode_img(s: bytes, iscolor: int = -1) -> np.ndarray:
    """decode_img.

    对图片压缩后的buff str进行解码, 得到解压后的图片数组

    Args:
        s: 图片压缩后的bytes字符串
        iscolor: 是否是彩色图片. 默认是 -1.

    Returns:
        np.ndarray 解码后的图片数组
    """
    img = np.frombuffer(s, dtype=np.uint8)
    img = cv2.imdecode(img, iscolor)
    return img


def img_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert image color to rgb.

    Args:
        img: image in color bgr or gray.
    """
    assert isinstance(img, np.ndarray)

    assert img.ndim in [2, 3]

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            return img[:, :, ::-1]
        else:
            raise ValueError("Invalid image shape %s" % img.shape)


class Real3DRecReader:
    def __init__(self, rec_file, idx_file=None, decode_image=True):
        """Real3d Rec Reader.

        rec_file: the path of rec
        decode_image: whether to return the decoded image.
        """
        assert (
            RecordUnit is not None
        ), "gluon_horizon and auto_matrix is required"
        assert rec_file.endswith(".rec")
        if idx_file is None:
            idx_file = rec_file + ".idx"
        self.rec = MXRecord(rec_file, idx_path=idx_file, writable=False)
        self._len = len(self.rec.record.keys)
        self.decode_image = decode_image
        self.rec_file = rec_file

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        item = self.rec.read(idx)
        _, s = unpack(item)
        rec_data = RecordUnit()
        rec_data.ParseFromString(s)
        rec_data = rec_data.body
        assert len(rec_data.data) == 2
        assert len(rec_data.extra) == 0
        image_buf = rec_data.data[0].value
        label_buf = rec_data.data[1].value
        label = json.loads(label_buf)
        if self.decode_image:
            image = decode_img(image_buf, iscolor=1)
            return image, label
        return image_buf, label
