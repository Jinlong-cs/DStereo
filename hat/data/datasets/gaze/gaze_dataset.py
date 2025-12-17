# Copyright (c) Horizon Robotics. All rights reserved.
import collections
import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch.utils.data as data
from torch.utils.data import ConcatDataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.mxrecord import unpack
from hat.utils.pack_type.recordio_pb2 import RecordUnit
from .utils import (
    R2xyz,
    calculate_rotation_matrix,
    eye_ldmk_transform,
    parse_gaze_mtl_label,
)

__all__ = ["GazeRecDataset", "GazeDataset"]


@OBJECT_REGISTRY.register
class GazeRecDataset(data.Dataset):
    """Gaze dataset wrapping over RecordIOPB file containing.

    A json structure as data, and a json structure as label.

    Args:
        filename: Rec path
        input_size: input (w, h) for gaze_eyeldmk network
        transforms: List of transforms
        rotate_3d_augm: Whether to do rotate augm online
        norm: Normalization image
            1. The x-axis of the human head coordinate system
               is aligned with the x-axis of the virtual camera
            2. The z-axis of the virtual camera is aligned with
               the target point in the image
            3. Move the virtual camera to a fixed position, the
               distance from the image target point is s
        embedding_gaze_sup: Pairwise image, rotate their gaze embedding
            gaze_gt2 * inv(gaze_gt1) * gaze_embedding, add rotation gaze
            embedding supervise
    """

    def __init__(
        self,
        filename: str,
        input_size: Tuple,
        transforms: Optional[List] = None,
        rotate_3d_augm: bool = False,
        norm: bool = False,
        embedding_gaze_sup: bool = False,
        angle_form: str = "degree",
        gazemap_settings: Dict = None,
    ):
        self.idx_file = filename + ".idx"
        self.rec_file = filename
        self.resize_shape = input_size
        self.transforms = transforms
        self.rotate_3d_augm = rotate_3d_augm
        self.norm = norm
        self.embedding_gaze_sup = embedding_gaze_sup

        self.angle_form = angle_form
        self.gazemap_settings = gazemap_settings

        self.pack_type = PackTypeMapper["mxrecord"]
        self.pack_file = self.pack_type(
            uri=self.rec_file, idx_path=self.idx_file, writable=False
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.pack_file.read(self.samples[idx])
        _, s = unpack(record)
        rec_data = RecordUnit()
        rec_data.ParseFromString(s)
        rec_data = rec_data.body

        parsed_label = parse_gaze_mtl_label(
            json.loads(bytes.decode(rec_data.data[-1].value)),
            rotate_3d_augm=self.rotate_3d_augm,
            angle_form=self.angle_form,
        )

        data = collections.defaultdict(dict)
        # online augm
        if self.rotate_3d_augm:
            assert len(rec_data.data) == 2
            # Unpacked data will be [image, label],
            # where label is dict
            image = cv2.imdecode(
                np.frombuffer(rec_data.data[0].value, dtype=np.uint8), flags=-1
            )
            data["img"] = image
        # offline augm
        else:
            assert len(rec_data.data) in [4, 6, 8]
            assert len(rec_data.extra) == 0

            # Unpacked data will be [image, horizon_pos, vertical_pos,
            # mirror_horizon_pos, mirror_vertical_pos, label],
            # where label is dict

            if len(rec_data.data) in [6, 8]:  # for train and val
                image = cv2.imdecode(
                    np.frombuffer(rec_data.data[0].value, dtype=np.uint8),
                    flags=-1,
                )
                horizon_img = cv2.imdecode(
                    np.frombuffer(rec_data.data[1].value, dtype=np.uint8),
                    flags=-1,
                )
                vertical_img = cv2.imdecode(
                    np.frombuffer(rec_data.data[2].value, dtype=np.uint8),
                    flags=-1,
                )
                mirror_horizon_img = cv2.imdecode(
                    np.frombuffer(rec_data.data[3].value, dtype=np.uint8),
                    flags=-1,
                )
                mirror_vertical_img = cv2.imdecode(
                    np.frombuffer(rec_data.data[4].value, dtype=np.uint8),
                    flags=-1,
                )
            elif len(rec_data.data) == 4:  # for test
                image = cv2.imdecode(
                    np.frombuffer(rec_data.data[0].value, dtype=np.uint8),
                    flags=-1,
                )
                horizon_img = cv2.imdecode(
                    np.frombuffer(rec_data.data[1].value, dtype=np.uint8),
                    flags=-1,
                )
                vertical_img = cv2.imdecode(
                    np.frombuffer(rec_data.data[2].value, dtype=np.uint8),
                    flags=-1,
                )
                mirror_horizon_img = horizon_img
                mirror_vertical_img = vertical_img
            else:
                raise ValueError

            if self.gazemap_settings and self.gazemap_settings["active"]:
                gazemap = np.frombuffer(
                    rec_data.data[-3].value, dtype=np.uint8
                )
                gazemap_weight = np.frombuffer(
                    rec_data.data[-2].value, dtype=np.uint8
                )
                gazemap = np.reshape(
                    gazemap, (image.shape[0], image.shape[1], 2)
                )
                gazemap_weight = np.reshape(gazemap_weight, (1, 1, 2))
                gazemap = gazemap.astype(dtype=np.float32)
                gazemap_weight = gazemap_weight.astype(dtype=np.float32)

                if self.gazemap_settings["ignore_sample_with_eyeball_only"]:
                    # ignore eyeball if iris is ignored
                    gazemap_weight[:, :, 0] *= gazemap_weight[:, :, 1]
                gazemap_weight[:, :, 0] *= self.gazemap_settings[
                    "eyeball_weight"
                ]
                gazemap_weight[:, :, 1] *= self.gazemap_settings["iris_weight"]
            else:
                # fake gazemap and gazemap_weight
                gazemap = np.zeros((image.shape[0], image.shape[1], 2))
                gazemap_weight = np.zeros((1, 1, 2))

            # Gazemap data
            parsed_label["gt_gazemap"] = gazemap
            parsed_label["gt_gazemap_weight"] = gazemap_weight

            parsed_label["gt_normed_eye_ldmk"] = eye_ldmk_transform(
                eye_ldmk=parsed_label["gt_eye_ldmk"],
                eye_bbox=parsed_label["gt_eye_bbox"],
                resize_shape=self.resize_shape,
            )
            data["img"] = image
            data["horizon_img"] = horizon_img
            data["vertical_img"] = vertical_img
            data["mirror_horizon_img"] = mirror_horizon_img
            data["mirror_vertical_img"] = mirror_vertical_img

        data["gaze_label"] = parsed_label
        data["layout"] = "hwc"

        # data transoform
        if self.transforms:
            data = self.transforms(data)

        # remove unused key
        if "horizon_img" in data.keys():
            data.pop("horizon_img")
        if "vertical_img" in data.keys():
            data.pop("vertical_img")
        if "mirror_horizon_img" in data.keys():
            data.pop("mirror_horizon_img")
        if "mirror_vertical_img" in data.keys():
            data.pop("mirror_vertical_img")
        if "gt_face_ldmks" in data["gaze_label"].keys():
            data["gaze_label"].pop("gt_face_ldmks")
        if "gt_face_bbox" in data["gaze_label"].keys():
            data["gaze_label"].pop("gt_face_bbox")
        if "intrinsics_K" in data["gaze_label"].keys():
            data["gaze_label"].pop("intrinsics_K")
        if "origin_image_shape" in data["gaze_label"].keys():
            data["gaze_label"].pop("origin_image_shape")
        if "gt_eye_ldmk" in data["gaze_label"].keys():
            data["gaze_label"]["gt_normed_eye_ldmk"] = data["gaze_label"][
                "gt_eye_ldmk"
            ]
            data["gaze_label"].pop("gt_eye_ldmk")

        if self.embedding_gaze_sup:
            # get gaze dir rotation matrix
            # Check unvalid gaze label if change to valid
            data["gt_R_gaze_left"] = calculate_rotation_matrix(
                data["gaze_label"]["gt_gaze"][:2]
            )
            data["gt_R_gaze_right"] = calculate_rotation_matrix(
                data["gaze_label"]["gt_gaze"][2:]
            )

        # do norm, not need position map
        if self.norm:
            data["img"] = data["img"][0, None, :, :]  # [1, 192, 320]
            head_pose = data["gaze_label"]["gt_head_pose"]
            R = cv2.Rodrigues(head_pose)[0]
            x, y, z = R2xyz(R)
            head_pose = (
                np.array([x, y, z]) * 180 / 3.14 * 8 / 75
            )  # (-75, 75) -> (-8, 8)
            data["gt_head_pose"] = head_pose.astype(np.float32)
            # data['label']['gt_head_pose'] = head_pose
        return data


@OBJECT_REGISTRY.register
class GazeDataset(ConcatDataset):
    def __init__(
        self,
        rec_list: List,
        input_size: Tuple,
        transforms: Optional[List] = None,
        rotate_3d_augm: bool = False,
        norm: bool = False,
        embedding_gaze_sup: bool = False,
        angle_form: str = "degree",
        gazemap_settings: Dict = None,
    ):
        self.transforms = transforms
        self.rec_list = rec_list
        self.input_size = input_size
        self.rotate_3d_augm = rotate_3d_augm
        self.norm = norm
        self.embedding_gaze_sup = embedding_gaze_sup
        self.angle_form = angle_form
        self.gazemap_settings = gazemap_settings

        self._init_pack()

        super(GazeDataset, self).__init__(self.dataset_list)

    def _init_pack(self):
        self.dataset_list = []
        for filename in self.rec_list:
            self.dataset_list.append(
                GazeRecDataset(
                    filename=filename,
                    input_size=self.input_size,
                    transforms=self.transforms,
                    rotate_3d_augm=self.rotate_3d_augm,
                    norm=self.norm,
                    embedding_gaze_sup=self.embedding_gaze_sup,
                    angle_form=self.angle_form,
                    gazemap_settings=self.gazemap_settings,
                )
            )
