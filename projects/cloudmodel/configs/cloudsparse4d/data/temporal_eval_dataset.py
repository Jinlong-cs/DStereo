# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os
from typing import Callable, List, Optional

import numpy as np

from hat.data.datasets.temporal_dataset import TemporalLabelDataset
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TemporalSDJsonDataset(TemporalLabelDataset):
    """Temporal dataset with Json annotations for SD format.

    Args:
        json_file: the path of json anno file.
        transforms: Config dict of transformations.
        max_interval: max_interval between 2 clip. if abs(cur_ts - pre_ts) >
            max_interval, cur_ts will be regarded as another clip.
        max_len_in_clip: max length of one clip. if length of a clip >
            max_len_in_clip, it will be splited to different clips.
            if max_len_in_clip <= 0, length of a clip can be any
            positive integer.
        event_key: name of clip_id, it can be `event_id`,
            `scene_token` or None.
        eval_class: name of the evaluate class.
        category_name2id: name dict mapping names to indexes.
        sub_category_name2id: sub-category name dict mapping names to indexes.
    """

    def __init__(
        self,
        json_file: str,
        transforms: Optional[List[Callable]] = None,
        max_interval: int = 600,
        max_len_in_clip: int = -1,
        event_key: Optional[str] = None,  # "event_id", "scene_token"
        eval_class: Optional[str] = "vehicle",
        category_name2id: dict = None,
        sub_category_name2id: dict = None,
    ):
        ts_list = []
        assert os.path.exists(json_file), "json file does not exist!"
        with open(json_file, "r") as f:
            self.json_lines = list(f.readlines())
            for line in self.json_lines:
                label = json.loads(line)
                ts_item = {
                    "timestamp": label["timestamp"],
                }
                if event_key and event_key in label:
                    ts_item[ts_item] = label[event_key]
                ts_list.append(ts_item)

        super(TemporalSDJsonDataset, self).__init__(
            sorted_ts_list=ts_list,
            transforms=transforms,
            max_interval=max_interval,
            max_len_in_clip=max_len_in_clip,
            event_key=event_key,
        )
        self.category_name2id = category_name2id
        self.sub_category2id = {}
        for k, v in sub_category_name2id.items():
            self.sub_category2id[k.lower()] = v
        self.eval_class = eval_class
        logger.info(f"Loading {json_file}.")

    def parse_label(self, index):
        label = json.loads(self.json_lines[index])
        # reformat
        label = self.reformat_label(label)
        return label

    def corver_cloudbev2mvt4d(self, obj):
        obj["in_vcs"]["dim"] = obj["in_vcs"]["dim"][::-1]
        return obj

    def reformat_label(self, label):
        plate, timestamp = label["timestamp"].split("_")
        img_orders = list(label["imgs_meta"].keys())
        objects = []
        for obj in label[self.eval_class]["objects"]:
            obj = self.corver_cloudbev2mvt4d(obj)
            objects.append(
                {
                    "uid": obj["uid"],
                    "category_id": self.category_name2id[self.eval_class],
                    "in_lidar": obj["in_lidar"],
                    "in_vcs": obj["in_vcs"],
                    "sub_category_id": self.sub_category2id[
                        obj["attrs"]["sub_type"].lower()
                    ],
                    "velocity": [0, 0, 0],
                }
            )
        view_anno = {}
        for cam in img_orders:
            view_anno[cam] = {}
            meta = label["imgs_meta"][cam]
            distort = meta["calib"]["distort"]
            meta["calib"]["distort"] = {"param": distort}
            meta["calib"]["img_wh"] = [
                meta["calib"]["image_width"],
                meta["calib"]["image_height"],
            ]
            meta["timestamp"] = timestamp
            view_anno[cam] = {"meta": meta}
            order_obj = label[self.eval_class]["view_anno"][cam]["objects"]
            view_anno[cam]["objects"] = {}
            for uuid, obj in order_obj.items():
                view_anno[cam]["objects"][uuid] = {
                    "category_id": self.category_name2id[self.eval_class],
                    "ignore": obj["ignore"],
                    "sub_category_id": self.sub_category2id[
                        obj["attrs"]["sub_type"].lower()
                    ],
                    "bbox": obj["bbox"],
                    "bbox_2d": obj["bbox_2d"],
                }
        return {
            "timestamp": timestamp,
            "objects": objects,
            "img_orders": img_orders,
            "view_anno": view_anno,
            "plate": plate,
        }


@OBJECT_REGISTRY.register
class TemporalMonoJsonDataset(TemporalLabelDataset):
    """Temporal dataset with Json annotations for Mono format.

    Args:
        json_file: the path of json anno file.
        transforms: Config dict of transformations.
        max_interval: max_interval between 2 clip. if abs(cur_ts - pre_ts) >
            max_interval, cur_ts will be regarded as another clip.
        max_len_in_clip: max length of one clip. if length of a clip >
            max_len_in_clip, it will be splited to different clips.
            if max_len_in_clip <= 0, length of a clip can be any
            positive integer.
        event_key: name of clip_id, it can be `event_id`,
            `scene_token` or None.
        eval_class: name of the evaluate class.
        category_name2id: name dict mapping names to indexes.
        sub_category_name2id: sub-category name dict mapping names to indexes.
    """

    def __init__(
        self,
        json_file: str,
        transforms: Optional[List[Callable]] = None,
        max_interval: int = 600,
        max_len_in_clip: int = -1,
        event_key: Optional[str] = None,  # "event_id", "scene_token"
        eval_class: Optional[str] = "vehicle",
        category_name2id: dict = None,
        sub_category_name2id: dict = None,
    ):
        ts_list = []
        with open(json_file, "r") as f:
            self.json_lines = list(f.readlines())
            for line in self.json_lines:
                label = json.loads(line)
                ts_item = {
                    "timestamp": label["image_key"].split("__")[0],
                }
                if event_key and event_key in label:
                    ts_item[ts_item] = label[event_key]
                ts_list.append(ts_item)

        super(TemporalMonoJsonDataset, self).__init__(
            sorted_ts_list=ts_list,
            transforms=transforms,
            max_interval=max_interval,
            max_len_in_clip=max_len_in_clip,
            event_key=event_key,
        )
        self.category_name2id = category_name2id
        self.sub_category2id = {}
        for k, v in sub_category_name2id.items():
            self.sub_category2id[k.lower()] = v
        self.eval_class = eval_class
        logger.info(f"Loading {json_file}.")

    def parse_label(self, index):
        label = json.loads(self.json_lines[index])
        # reformat
        label = self.reformat_label(label)
        return label

    def reformat_label(self, label):
        timestamp = label["image_key"].split("_")[0]
        objects = []

        for uid, obj in enumerate(label[self.eval_class]):
            in_vcs = {
                "location": obj["location"],
                "dim": obj["dimensions"][::-1],
                "yaw": obj["rotation_y"],
            }
            objects.append(
                {
                    "uid": uid,
                    "category_id": self.category_name2id[self.eval_class],
                    "in_lidar": {},
                    "in_vcs": in_vcs,
                }
            )
        img_orders = {"front"}
        view_anno = {}
        img_shape = (
            [label["height"], label["width"]]
            if "height" in label
            else [label["hight"], label["width"]]
        )
        K = np.array(label["calib"]).reshape(3, -1)[:, :3]
        d = np.array(label["distCoeffs"]).reshape(-1)
        if label["lidar_to_camera"] is None:
            from hat.core.virtual_camera.utils import parse_extrinsicParam

            poseMat_vcs2cam = parse_extrinsicParam(label["calib_all"], False)
            T_vcs2cam = poseMat_vcs2cam
        else:
            T_vcs2cam = np.array(
                label["lidar_to_camera"]["Tr_vel2cam"]
            ).reshape(4, 4)
        view_anno["front"] = {
            "meta": {
                "image_key": label["image_key"].replace(".jpg", ""),
                "image_source": label["image_source"],
                "Tr_vel2cam": np.eye(4),
                "shape": img_shape,
                "ignore_mask": label["ignore_mask"],
                "calib": label["calib"],
            },
            "objects": {},
        }
        calib_params = {
            "camera_front": {
                "K": K,
                "d": d,
                "T_vcs2cam": T_vcs2cam,
            },
        }
        # hard-code for mono
        for cam in [
            "camera_front_left",
            "camera_front_right",
            "camera_front_30fov",
            "camera_rear",
            "camera_rear_left",
            "camera_rear_right",
        ]:
            calib_params[cam] = {
                "K": np.eye(3),
                "d": np.zeros(8),
                "T_vcs2cam": np.eye(4),
            }
        return {
            "timestamp": timestamp,
            "objects": objects,
            "img_orders": img_orders,
            "view_anno": view_anno,
            "calib_params": calib_params,
        }

    def __getitem__(self, index):
        label = self.parse_label(index)
        imgs_meta = {
            "dataset_index": index,
            "meta": label,
            "calib_params": label["calib_params"],
        }

        if self.transforms is not None:
            for transform in self.transforms:
                imgs_meta = transform(imgs_meta)
        return imgs_meta
