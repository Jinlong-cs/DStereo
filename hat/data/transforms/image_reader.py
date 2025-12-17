# Copyright (c) Horizon Robotics. All rights reserved.
import os

import cv2
import msgpack
import numpy as np

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.lmdb import Lmdb


@OBJECT_REGISTRY.register
class MVImgJsonReader(object):
    def __init__(
        self,
        img_dir,
        to_rgb=True,
    ):
        self.img_dir = img_dir
        self.to_rgb = to_rgb

    def __call__(self, data):

        label = data["meta"]

        imgs_list = []
        image_keys = []
        for cam in label["img_orders"]:
            img_file_name = os.path.join(
                self.img_dir,
                label["view_anno"][cam]["meta"]["image_key"] + ".jpg",
            )

            img = cv2.imread(img_file_name)
            if self.to_rgb:
                img = img[:, :, ::-1]

            imgs_list.append(img)
            image_keys.append(label["view_anno"][cam]["meta"]["image_key"])

        data["imgs"] = imgs_list
        data["image_keys"] = image_keys

        return data


@OBJECT_REGISTRY.register
class MVImgLmdbReader(object):
    def __init__(
        self,
        img_path,
        decode_img=True,
        to_rgb=True,
    ):
        lmdb_common_config = {
            "writable": False,
            "map_size": 10485760,
            "meminit": True,
            "map_async": False,
            "sync": True,
        }
        self._img_lmdb = Lmdb(uri=img_path, **lmdb_common_config)
        self.decode_img = decode_img
        self.to_rgb = to_rgb

    def __call__(self, data):

        label = data["meta"]
        key = label["lmdb_key"]
        raw_img = self._img_lmdb.get(key)
        if self.decode_img:
            img_bytes_list = msgpack.unpackb(
                raw_img,
                raw=True,
            )
            imgs = [
                cv2.imdecode(
                    np.frombuffer(img_byte, dtype=np.uint8),
                    flags=cv2.IMREAD_COLOR,
                )
                for img_byte in img_bytes_list
            ]

            if self.to_rgb:
                imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        else:
            imgs = raw_img

        data["imgs"] = imgs

        return data
