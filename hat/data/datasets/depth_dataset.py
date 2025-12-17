# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch.utils.data as data

from hat.data.datasets.utils import decode_img
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXIndexedRecordIO, unpack
from hat.utils.pack_type.recordio_pb2 import RecordUnit

__all__ = ["DepthDatasetRec", "DepthRecReader"]

logger = logging.getLogger(__name__)


class DepthRecReader:
    def __init__(
        self,
        rec_file: str,
        idx_file: str = None,
        decode_image: bool = True,
        with_parsing: bool = False,
    ):
        """Depth Rec Reader.

        Args:
            rec_file: The path of rec
            idx_file: Index file related to data_path.
            decode_image: Whether to return the decoded image.
            with_parsing: Whether parsing label in rec file.
        """

        assert rec_file.endswith(".rec")
        if idx_file is None:
            idx_file = rec_file + ".idx"

        self.rec = MXIndexedRecordIO(idx_file, rec_file, "r")
        self._len = len(open(idx_file, "r").readlines())
        self.decode_image = decode_image
        self.with_parsing = with_parsing

        logging.info(f"dataset length: {self._len}")

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Tuple["np.ndarray", Dict]:
        item = self.rec.read_idx(idx)
        _, s = unpack(item)
        rec_data = RecordUnit()
        rec_data.ParseFromString(s)
        rec_data = rec_data.body
        assert len(rec_data.data) in [3, 4]
        assert len(rec_data.extra) == 0
        image_buf = rec_data.data[0].value
        depth_label_buf = rec_data.data[1].value

        depth_label = cv2.imdecode(
            np.frombuffer(depth_label_buf, dtype=np.uint8), flags=-1
        )
        depth_label = depth_label.astype("float32")
        if self.with_parsing:
            parsing_label_buf = rec_data.data[2].value
            parsing_label = cv2.imdecode(
                np.frombuffer(parsing_label_buf, dtype=np.uint8), flags=-1
            )
        info_meta_buffer = (
            rec_data.data[3].value
            if self.with_parsing
            else rec_data.data[2].value
        )
        info_meta = json.loads(info_meta_buffer)
        if self.decode_image:
            image = decode_img(image_buf)
        else:
            image = image_buf
        ret = {"img": image, "gt_depth": depth_label, "info_meta": info_meta}
        if self.with_parsing:
            ret["gt_seg"] = parsing_label
        return ret

    def __del__(self):
        self.rec.close()


@OBJECT_REGISTRY.register
class DepthDatasetRec(data.Dataset):
    """Depth dataset class in rec fashion.

    Args:
        paths: Paths for rec path.
        transforms: Transforms that applies to the data.
        depth_scale_facotr: Scale factor for scale the depth.
        with_parsing: Whether to use parsing label as auxilary task.
        to_rgb: whether convter to rgb.
        select_sample: Whether reselect index when invalid data.
        mode: Model state.
    """

    def __init__(
        self,
        paths: List[str],
        transforms: Optional[Callable] = None,
        depth_scale_factor: float = 256.0,
        with_parsing: Optional[bool] = False,
        virtual_cam_params: Optional[np.array] = None,
        to_rgb: Optional[bool] = False,
        select_sample: Optional[bool] = False,
        mode: Optional[str] = "train",
    ):
        super(DepthDatasetRec, self).__init__()

        assert mode in ["train", "val", "test"]

        self.paths = paths
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.with_parsing = with_parsing
        self.select_sample = select_sample
        self.depth_scale_factor = depth_scale_factor
        self.mode = mode

        self._load_recs(self.paths)
        self.num_samples = self.acc_lengths[-1]
        self._index = -1
        self.image_rec_idxes = self._image_assign_rec_idx(self.acc_lengths)

        if virtual_cam_params is not None:
            assert (
                virtual_cam_params.size == 9
            ), "Size of camera \
                instrinsic value must equal to 9."
        self.virtual_cam_params = virtual_cam_params
        # camera extrinsic parameter encoding
        self.extrinsic_camera = ExtrinsicCameraParamEncoding(
            encode_method="vector",
            param_type={
                "rpy": ["roll", "pitch", "yaw"],
                "local2cam": ["camera_z"],
                "vcs2cam_rot": [0, 1, 2],
                "vcs2cam_trans": [0, 1, 2],
            },
        )

        logging.info(f"dataset total length: {self.num_samples}")

    @staticmethod
    def _image_assign_rec_idx(acc_lengths):
        rec_assign_idx = []
        start_idx = 0
        for i, part in enumerate(acc_lengths):
            rec_assign_idx.extend(list([i] * part)[start_idx:])
            start_idx = part
        return rec_assign_idx

    def _load_recs(self, paths):
        assert isinstance(paths, list)
        self.recs = []
        for rec_path in paths:
            logger.info(f"loading {rec_path}")
            rec_reader = DepthRecReader(
                rec_path, with_parsing=self.with_parsing
            )
            self.recs.append(rec_reader)
        lengths = [len(rec) for rec in self.recs]
        self.acc_lengths = np.cumsum(lengths)

    def __len__(self):
        return self.num_samples

    # TODO (yunfeng.zhang): parse camera parameters
    def _parse_camera_param(self, img_info):
        camera_params_info = {}
        if "camera_params" in img_info:
            camera_params = img_info["camera_params"]
            if "virtual_cam_params" in camera_params:
                virtual_cam_params = np.array(
                    camera_params["virtual_cam_params"], dtype=np.float32
                ).reshape(3, 3)
            else:
                virtual_cam_params = self.virtual_cam_params

            camera_params_info.update(self.extrinsic_camera(camera_params))

            # camera intrinsic parameters
            intrinsic_cam_params = np.array(
                [
                    camera_params["focal_u"],
                    camera_params["focal_v"],
                    camera_params["center_u"],
                    camera_params["center_v"],
                    camera_params["fov"],
                ],
                dtype=np.float32,
            )
            camera_params_info["intrinsic_cam_params"] = intrinsic_cam_params
        else:
            if self.virtual_cam_params is not None:
                virtual_cam_params = self.virtual_cam_params.reshape(3, 3)
            else:
                virtual_cam_params = self.virtual_cam_params

        camera_params_info["virtual_cam_params"] = virtual_cam_params
        return camera_params_info

    def _get_rec_from_idx(self, idx):
        """Get recio from index."""
        assert idx < len(self)
        for length_idx in range(len(self.acc_lengths)):
            length_i = self.acc_lengths[length_idx]
            if idx > length_i:
                continue
            rec = self.recs[length_idx]
            if length_idx == 0:
                previous_idx = 0
            else:
                previous_idx = self.acc_lengths[length_idx - 1]
            idx_in_rec = idx - previous_idx
            assert idx_in_rec >= 0
            if idx_in_rec >= len(rec):
                idx_in_rec = len(rec) - 1
            return rec, idx_in_rec

    def _prepare_data(self, index):
        rec, idx = self._get_rec_from_idx(index)
        data_info = rec[idx]
        img = data_info["img"]
        img_info = data_info["info_meta"]
        _, encoded_image = cv2.imencode(".jpg", img)

        if "img_name" in img_info:
            image_name = img_info["img_name"]
        elif "img_url" in img_info:
            img_url = img_info["img_url"]
            image_name = os.path.basename(img_url)

        depth_label = data_info["gt_depth"] / self.depth_scale_factor

        color_space = "bgr"
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.     # noqa
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color_space = "rgb"

        data = {
            "image_name": image_name,
            "image_height": img.shape[0],
            "image_width": img.shape[1],
            "img": img,
            "gt_depth": depth_label,
            "color_space": color_space,
            "layout": "hwc",
            "img_url": img_info["img_url"],
            "img_buf": encoded_image.tobytes(),
            "img_shape": img.shape,
        }

        camera_params = self._parse_camera_param(img_info)
        data.update(camera_params)

        if self.with_parsing:
            parsing_label = data_info["gt_seg"]
            parsing_label = merge_parsing_labels(parsing_label)
            data["gt_seg"] = parsing_label
        return data

    def _rand_another(self):
        """Get another random index."""
        return np.random.choice(len(self))

    def __getitem__(self, index):
        while True:
            data_dict = self._prepare_data(index)

            if self.transforms:
                data_dict = self.transforms(data_dict)
                if self.mode == "train":
                    # TODO (yunfeng.zhang): data verification
                    if not data_dict.get("valid", True) and self.select_sample:
                        index = self._rand_another()
                        continue
            if data_dict.get("valid", None) is not None:
                data_dict.pop("valid")
            return data_dict


class ExtrinsicCameraParamEncoding(object):
    def __init__(self, encode_method="normal", param_type=None):
        assert encode_method in ["normal", "vector", "map"]
        self.method = encode_method
        if param_type is None:
            self.param_type = {
                "rpy": ["roll", "pitch", "yaw"],
                "local2cam": ["camera_x", "camera_y", "camera_z"],
                "vcs2cam_rot": [0, 1, 2],
                "vcs2cam_trans": [0, 1, 2],
            }
        else:
            self.param_type = param_type
        param_type_keys = list(self.param_type.keys())
        assert set(param_type_keys).issubset(
            ["rpy", "local2cam", "vcs2cam_rot", "vcs2cam_trans"]
        )

    def _get_rpy(self, camera_params):
        if "rpy" not in self.param_type:
            rpy = np.array([], dtype=np.float32)
        else:
            param_lst = []
            for param_name in self.param_type["rpy"]:
                param_lst.append(camera_params[param_name])
            rpy = np.array(param_lst, dtype=np.float32)
        return rpy

    def _get_local2cam(self, camera_params):
        if "local2cam" not in self.param_type:
            local2cam = np.array([], dtype=np.float32)
        else:
            param_lst = []
            for param_name in self.param_type["local2cam"]:
                param_lst.append(camera_params[param_name])
            local2cam = np.array(param_lst, dtype=np.float32)
        return local2cam

    def _get_vcs2cam_rot(self, camera_params):
        if "vcs2cam_rot" in self.param_type:
            vcs2cam_rot = np.array(
                camera_params["vcs"]["rotation"], dtype=np.float32
            )
            param_lst = []
            for idx in self.param_type["vcs2cam_rot"]:
                param_lst.append(vcs2cam_rot[idx])
            vcs2cam_rot = np.array(param_lst, dtype=np.float32)
        else:
            vcs2cam_rot = np.array([], dtype=np.float32)
        return vcs2cam_rot

    def _get_vcs2cam_trans(self, camera_params):
        if "vcs2cam_trans" in self.param_type:
            vcs2cam_trans = np.array(
                camera_params["vcs"]["translation"], dtype=np.float32
            )
            param_lst = []
            for idx in self.param_type["vcs2cam_trans"]:
                param_lst.append(vcs2cam_trans[idx])
            vcs2cam_trans = np.array(param_lst, dtype=np.float32)
        else:
            vcs2cam_trans = np.array([], dtype=np.float32)
        return vcs2cam_trans

    def _encode_normal(self, camera_params):
        extrinsic_cam_params = {}

        extrinsic_cam_params["cam_rpy"] = self._get_rpy(camera_params)
        extrinsic_cam_params["local2cam"] = self._get_local2cam(camera_params)
        vcs2cam_rot = self._get_vcs2cam_rot(camera_params)
        extrinsic_cam_params["vcs2cam_rot"] = vcs2cam_rot
        vcs2cam_trans = self._get_vcs2cam_trans(camera_params)
        extrinsic_cam_params["vcs2cam_trans"] = vcs2cam_trans
        return extrinsic_cam_params

    def _encode_map(self, camera_params):
        pass

    def _encode_vector(self, camera_params):
        extrinsic_cam_params = {}
        rpy = self._get_rpy(camera_params)
        local2cam = self._get_local2cam(camera_params)
        vcs2cam_rot = self._get_vcs2cam_rot(camera_params)
        vcs2cam_trans = self._get_vcs2cam_trans(camera_params)
        all_extrinsic_params = np.concatenate(
            [rpy, local2cam, vcs2cam_rot, vcs2cam_trans], axis=0
        )
        extrinsic_cam_params["extrinsic_cam_params"] = all_extrinsic_params[
            :, None, None
        ]
        return extrinsic_cam_params

    def __call__(self, camera_params):
        if self.method == "normal":
            extrinsic_cam_params = self._encode_normal(camera_params)
        elif self.method == "vector":
            extrinsic_cam_params = self._encode_vector(camera_params)
        elif self.method == "map":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return extrinsic_cam_params


def merge_parsing_labels(label):
    merge_dict = {
        0: [0, 1, 4, 20, 21],  # road
        1: [9, 10, 11, 14],  # vehicle
        2: [12, 13],  # person
        3: [5, 7, 8, 15, 16, 17, 18, 19],  # fence&bollard&pole..
        4: [3, 6],  # tree
        5: [22],  # column
        6: [2],  # sky
        7: [23],  # background
    }
    new_label = np.full_like(label, fill_value=6)
    for idx, old_labels in merge_dict.items():
        for old_idx in old_labels:
            new_label[label == old_idx] = idx
    return new_label
