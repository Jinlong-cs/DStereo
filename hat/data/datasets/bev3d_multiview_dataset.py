from __future__ import absolute_import, print_function
import json
import logging
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import cv2
import numpy as np
import torch
from numpy import ndarray
from scipy.spatial.transform import Rotation
from torchvision.transforms import Compose

from hat.core.virtual_camera.camera_base import CameraParam
from hat.data.datasets.bev import HomoGenerator
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXIndexedRecordIO, unpack
from hat.utils.pack_type.recordio_pb2 import RecordUnit

__all__ = [
    "MultiViewRecDataset",
    "MultiViewImgDataset",
]

logger = logging.getLogger(__name__)


def get_mat_from_trans_rpy(lidar_params, seq="xyz"):
    r = Rotation.from_euler(seq, lidar_params["rpy"])
    rot = r.as_matrix()
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = lidar_params["xyz"]
    mat[3, 3] = 1.0
    return mat


def get_random_noise(homo_noise: Union[List, ndarray]):
    """Get random noise according to the upper bound.

    Args:
        homo_noise: The noise range of rpy and xyz.
    """
    d_rpy_limit = homo_noise[:3] / 180 * math.pi
    d_xyz_limit = homo_noise[3:6]

    d_rpy = [
        random.random() * rad_limit * 2 - rad_limit
        for rad_limit in d_rpy_limit
    ]
    d_xyz = [
        random.random() * trans_limit * 2 - trans_limit
        for trans_limit in d_xyz_limit
    ]

    return d_rpy, d_xyz


def get_noise_matrix(
    homo_noise: Union[List, ndarray],
    T_vcs2cam: ndarray,
    noise_type: str = "random",
):
    """Generate noise extrinsic parameter matrix for T_vcs2cam.

    Args:
        homo_noise: The noise range of rpy and xyz,
            the units are degrees and meters.
        T_vcs2cam: Pose Matrix for world/vcs coordinate to
            cammera coordinate. Default: np.identity(4).
        noise_type: There are two types of noise, 'random' and 'specific'.
    """
    if noise_type == "random":
        noise_rpy, noise_xyz = get_random_noise(homo_noise)
        R_cam2noisecam = Rotation.from_euler(
            "xyz", noise_rpy, degrees=False
        ).as_matrix()

        T_cam2noisecam = np.zeros((4, 4))
        T_cam2noisecam[:3, :3] = R_cam2noisecam
        T_cam2noisecam[:3, 3] = noise_xyz
        T_cam2noisecam[3, 3] = 1
        T_vcs2noisecam = T_cam2noisecam @ T_vcs2cam

        return T_vcs2noisecam
    elif noise_type == "specific":
        T_cam2vcs = np.linalg.inv(T_vcs2cam)
        T_camnew2cam = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )
        T_camnew2vcs = T_cam2vcs @ T_camnew2cam

        R_camnew2vcs = T_camnew2vcs[:3, :3]
        xyz = T_camnew2vcs[:3, 3]

        # add noise
        T_noisecamnew2vcs = np.zeros((4, 4))
        noise_rpy = homo_noise[:3] / 180 * math.pi
        R_noise = Rotation.from_euler(
            "xyz", noise_rpy, degrees=False
        ).as_matrix()
        R_noisecam2vcs = R_noise @ R_camnew2vcs

        T_noisecamnew2vcs[:3, :3] = R_noisecam2vcs
        T_noisecamnew2vcs[:3, 3] = xyz + homo_noise[3:6]
        T_noisecamnew2vcs[3, 3] = 1

        T_cam2vcs = T_noisecamnew2vcs @ np.linalg.inv(T_camnew2cam)
        return np.linalg.inv(T_cam2vcs)


def _get_calib_params_from_anno(
    anno: Dict[str, Any],
    camera_view_names: Optional[Sequence[str]],
    view_shapes: Optional[Dict[str, Sequence[int]]],
    standardized_cam_calibs=None,
    return_extra: bool = False,
    homo_noise: Union[List, ndarray] = None,
    noise_type: str = None,
    noise_view: str = None,
):
    """Get calib params recorded in annotation.

    Args:
        anno: Meta info from data.
        camera_view_names: Each view name.
        view_shapes: The shape of each view.
        standardized_cam_calibs: Standardized calibration parameters.
        return_extra: Whether to return extra info of transformation matrix.
        homo_noise: The noise range of rpy and xyz,
            the units are degrees and meters.
        noise_type: There are two types of noise, 'random' and 'specific'.
        noise_view: The camera view names with noise.
    """

    # parse all calibration parameters
    calib_dict = {}
    mat_lidar2vcs = None
    for cam_name in camera_view_names:
        cam_k = (
            f"camera_{cam_name}"
            if f"camera_{cam_name}" in view_shapes
            else cam_name
        )
        if (
            cam_name in anno["view_anno"]
            and "calib" in anno["view_anno"][cam_name]["meta"]
        ):
            calib_params = deepcopy(
                anno["view_anno"][cam_name]["meta"]["calib"]
            )
            if (
                "image_width" not in calib_params
                or "image_height" not in calib_params
            ):
                calib_params["image_width"] = view_shapes[cam_k][1]
                calib_params["image_height"] = view_shapes[cam_k][0]

            if (
                standardized_cam_calibs is not None
                and cam_k in standardized_cam_calibs
            ):
                calib_params.update(standardized_cam_calibs[cam_k])

            if return_extra:
                (
                    calib_getter,
                    local2vcs,
                    local2cam,
                ) = CameraParam.init_cam_param_by_dict(
                    calib_params, is_virtual=False, return_extra=return_extra
                )
            else:
                calib_getter = CameraParam.init_cam_param_by_dict(
                    calib_params, is_virtual=False
                )

            if mat_lidar2vcs is None and calib_params.get("lidar", None):
                mat_lidar2vcs = get_mat_from_trans_rpy(
                    calib_params["lidar"], "xyz"
                )

        else:
            # when a view is miss, use identity matrix as its
            # calibration matrix
            logger.debug(
                f"view {cam_name} misses, using identity matrix "
                "as calib mat."
            )
            calib_getter = CameraParam(distcoeffs=np.zeros((8,)).astype(float))
            if return_extra:
                local2cam, local2vcs = np.eye(4), np.eye(4)

        T_vcs2cam = calib_getter.poseMat_vcs2cam
        if homo_noise is not None:
            if noise_view == "all" or cam_name == noise_view:
                T_vcs2cam = get_noise_matrix(homo_noise, T_vcs2cam, noise_type)

        calib_dict[cam_k] = {
            "K": calib_getter.camera_matrix,
            "d": calib_getter.distcoeffs,
            "T_vcs2cam": T_vcs2cam,
            "T_lidar2vcs": calib_getter.poseMat_lidar2vcs,
        }
        if return_extra:
            if homo_noise is not None:
                if noise_view == "all" or cam_name == noise_view:
                    local2cam = get_noise_matrix(
                        homo_noise, local2cam, noise_type
                    )
            calib_dict[cam_k].update(
                T_local2cam=local2cam, T_local2vcs=local2vcs
            )
    if mat_lidar2vcs is None:
        mat_lidar2vcs = np.identity(4)
    calib_dict["T_lidar2vcs"] = mat_lidar2vcs

    return calib_dict


def _generate_homography_from_anno(
    anno,
    camera_view_names,
    view_shapes,
    homo_cfg,
    standardized_cam_calibs=None,
):
    """Generate homography from calib params recorded in annotation."""
    # parse all calibration parameters
    calib_dict = _get_calib_params_from_anno(
        anno,
        camera_view_names,
        view_shapes,
        standardized_cam_calibs=standardized_cam_calibs,
    )
    homo_cfg["calib_para"] = calib_dict
    homo_gen = HomoGenerator(**homo_cfg)
    homography = homo_gen.get_homography()
    homo_offset = homo_gen.get_homo_offset()
    return homography, homo_offset


@OBJECT_REGISTRY.register
class MultiViewRecDataset(torch.utils.data.Dataset):
    """A rec dataset of multiview data that a item contains several images.

    Args:
        rec_path: Path of the rec file
        rec_idx_file: Path of the rec idx file
        camera_view_names: Name of each camera view
        view_shapes: Shape (image_height, image_width) of each camera view
        homo_cfg: Config dict of homography grid computation
        key_type: Type of key
        decode_img: Whether to decode image
        to_rgb: Whether to convert to rgb
        transforms: Config dict of transformations
        cv_format: Opencv image decode flags, e.g., cv2.IMREAD_COLOR
    """

    def __init__(
        self,
        rec_path: str,
        rec_idx_file: str,
        camera_view_names: Optional[Sequence[str]] = None,
        view_shapes: Optional[Dict[str, Sequence[int]]] = None,
        homo_cfg: Optional[Dict[str, Any]] = None,
        key_type: Optional[Type] = int,
        decode_img: Optional[bool] = True,
        to_rgb: Optional[bool] = True,
        transforms: Optional[List[Callable]] = None,
        cv_format: Optional[int] = cv2.IMREAD_COLOR,
        standardized_cam_calibs=None,
        flag_path: Optional[str] = None,
    ):
        self.rec_path = rec_path
        self.rec_idx_file = rec_idx_file
        self.camera_view_names = camera_view_names
        self.view_shapes = view_shapes
        self.homo_cfg = homo_cfg
        self.key_type = key_type
        self.decode_img = decode_img
        self.to_rgb = to_rgb
        self.cv_format = cv_format
        self.transforms = transforms
        self.standardized_cam_calibs = standardized_cam_calibs

        logger.info(f"Loading {rec_path}.")

        self._open()

        if flag_path and os.path.exists(flag_path):
            logger.info(f"Loading {flag_path} as sequence flag.")
            self.flag = np.load(flag_path)
        else:
            self._set_default_group_flag()

    def _set_default_group_flag(self):
        self.flag = np.arange(len(self._rec_io.keys), dtype=np.int64)

    def _open(self):
        self._rec_io = MXIndexedRecordIO(
            idx_path=self.rec_idx_file,
            uri=self.rec_path,
            flag="r",
            key_type=self.key_type,
        )

    def _decode_record(self, record):
        _, s = unpack(record)
        rec_data = RecordUnit()
        rec_data.ParseFromString(s)
        rec_data = rec_data.body
        assert len(rec_data.extra) == 0
        label_buf = rec_data.data[0].value
        data_list = []
        for i in range(1, len(rec_data.data)):
            data = rec_data.data[i].value
            if self.decode_img:
                data = np.frombuffer(data, dtype=np.uint8)
                data = cv2.imdecode(data, self.cv_format)
                if self.to_rgb:
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                data = data.copy()
            data_list.append(data)

        label = json.loads(label_buf)
        return data_list, label

    def __getstate__(self):
        # The dataset should be picklable when launching
        # multiprocess workers
        state = self.__dict__.copy()
        state["_rec_io"] = None
        if self._rec_io is not None:
            state["_rec_io"] = pickle.dumps(self._rec_io)
        return state

    def __setstate__(self, state):
        # after unpickled, you should invoke `_fork` operation.
        rec_io_stat = state.pop("_rec_io")
        state["_rec_io"] = None
        self.__dict__ = state.copy()
        if rec_io_stat is not None:
            self._rec_io = pickle.loads(rec_io_stat)

    def _fork(self):
        # Reset _rec_io for rec file in hdfs will cause dead lock.
        assert (
            sys.version_info[0] >= 3
        ), "Using multiprocessing in python2 will cause bugs! "
        "Please use python3."
        if self._rec_io is not None:
            if hasattr(self._rec_io, "reset"):
                self._rec_io.reset()
        else:
            self._open()

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(
                "maximum length is %d, indexed by %d" % (len(self), idx)
            )

        try:
            rec_data = self._rec_io.read_idx(idx)
            imgs, label = self._decode_record(rec_data)
            if self.camera_view_names is not None:
                for cam_name in self.camera_view_names:
                    cam_k = f"camera_{cam_name}"
                    if (
                        cam_name in label["view_anno"]
                        and "calib" in label["view_anno"][cam_name]["meta"]
                    ):
                        calib_params = label["view_anno"][cam_name]["meta"][
                            "calib"
                        ]
                        if (
                            "image_width" not in calib_params
                            or "image_height" not in calib_params
                        ):
                            calib_params["image_width"] = self.view_shapes[
                                cam_k
                            ][1]
                            calib_params["image_height"] = self.view_shapes[
                                cam_k
                            ][0]
        except Exception:
            next_idx = np.random.randint(len(self))
            logger.debug(
                f"{idx}th-frame decode record error, try random {next_idx}th-frame"  # noqa
            )
            return self.__getitem__(next_idx)

        imgs_meta = {"imgs": imgs, "meta": label}

        # generate homography and offset
        if self.homo_cfg is not None:
            homography, homo_offset = _generate_homography_from_anno(
                label,
                self.camera_view_names,
                self.view_shapes,
                self.homo_cfg,
                self.standardized_cam_calibs,
            )
            imgs_meta["homography"] = homography
            imgs_meta["homo_offset"] = homo_offset

        if self.standardized_cam_calibs is not None:
            imgs_meta["standardized_calib_all"] = self.standardized_cam_calibs

        if self.transforms is not None:
            imgs_meta = self.transforms(imgs_meta)
        return imgs_meta

    def __len__(self):
        return len(self._rec_io.keys)


@OBJECT_REGISTRY.register
class MultiViewImgCollect:
    """Collect images from different views.

    Args:
        target_keys: Image keys that model has.
        collect_keys: View keys that model has.
    """

    def __init__(
        self,
        target_keys: Optional[List] = None,
        collect_keys: Optional[List] = None,
    ):
        assert len(target_keys) == len(collect_keys)
        self.target_keys = target_keys
        self.collect_keys = collect_keys

    def __call__(
        self, camera_view_imgs: Dict, suffix: Optional[str] = None
    ) -> Dict:
        ret = {}
        for i, key in enumerate(self.target_keys):
            collect_imgs = []
            for c_k in self.collect_keys[i]:
                collect_imgs.append(camera_view_imgs[c_k])
            t_k = key if suffix is None else key + suffix
            ret[t_k] = torch.stack(collect_imgs)
        return ret


@OBJECT_REGISTRY.register
class MultiViewImgDataset(torch.utils.data.Dataset):
    """Multiview dataset that reads images.

    If annotation file is provided, which will be also read.

    Notes:
        1. It reads image with name in {ts}__{cam_name}.jpg or
            {ts}__{cam_name}__*.jpg format. Please be aware that
            there two ``_``.

        2. In reading raw images mode, i.e., ``anno_json_file`` is set to
            None, then ``calib_path`` must be set.

        3. In reading images and annotations mode, i.e., ``anno_json_file``
            is set, then the ``calib`` should be contained in annotations.

    Args:
        img_dir: Directory of images, which name must be formatted
            like {time_stamp}__{camera_name}.jpg
        camera_list: Camera names
        view_shapes: Shape (image_height, image_width) of each camera view
        anno_json_file: Annotation json file path
        calib_path: Calibration parameters path, there are two use cases:
            1. In reading raw image mode, it must be set.
            2. In reading images and annotations mode, it can be optionally
            set, to provid parameters when a view is miss
        homo_cfg: Config dict of homography grid computation
        to_rgb: Whether to convert to rgb
        nums_to_read: Nums of images to read
        transforms: Config dict of transformations
    """

    def __init__(
        self,
        img_dir: str,
        camera_list: List[str],
        view_shapes: Optional[Dict[str, Sequence[int]]] = None,
        img_pad_value: Optional[float] = 0,
        anno_json_file: Optional[str] = None,
        calib_path: Optional[str] = None,
        homo_cfg: Optional[Dict[str, Any]] = None,
        to_rgb: Optional[bool] = True,
        nums_to_read: Optional[int] = -1,
        transforms: Optional[
            Union[List[Callable], Dict[str, List[Callable]]]
        ] = None,
        standardized_cam_calibs=None,
        anno_view_key_map=None,
        multi_view_collect=None,
    ):
        if anno_json_file is None:
            assert (
                calib_path is not None
            ), "``calib_path`` must be set in reading raw image mode."
        self.img_dir = img_dir
        self.camera_list = camera_list
        self.view_shapes = view_shapes
        self.img_pad_value = img_pad_value
        self.homo_cfg = homo_cfg
        self.to_rgb = to_rgb
        if isinstance(transforms, (list, tuple)):
            transforms = Compose(transforms)
        elif isinstance(transforms, dict):
            _transforms = {}
            for k, v in transforms.items():
                if isinstance(v, (list, tuple)):
                    _transforms[k] = Compose(v)
                else:
                    _transforms[k] = v
            transforms = _transforms
        self.transforms = transforms

        if calib_path is not None:
            self.calib_dict = {}
            for cam in camera_list:
                calib_file = os.path.join(calib_path, f"{cam}.json")
                assert os.path.exists(calib_file), f"{calib_file} not exists."
                with open(calib_file, "r") as f:
                    self.calib_dict[cam] = json.load(f)
        else:
            self.calib_dict = None

        img_paths = defaultdict(dict)
        # reads image with name in
        # 1. {ts}__{cam_name}.jpg
        # 2. {plate}__{cam_name}__{ts}.jpg format
        for file_name in os.listdir(img_dir):
            if not file_name.endswith(".jpg"):
                continue
            name = file_name.rstrip(".jpg")
            name = name.split("__")
            if len(name) == 2:
                # {ts}__{cam_name}.jpg
                ts, cam_name = name
            elif len(name) == 3:
                # {plate}__{cam_name}__{ts}.jpg
                _, cam_name, ts = name
            else:
                raise ValueError
            img_paths[ts][cam_name] = file_name
        ts_list = list(img_paths.keys())
        ts_list = list(set(ts_list))
        if anno_json_file is not None:
            with open(anno_json_file, "r") as f:
                lines = f.readlines()
            annos = [json.loads(line) for line in lines]
            ts_anno = {}
            for anno in annos:
                ts = anno["timestamp"].split("_")
                if len(ts) == 1:
                    ts = ts[0]
                elif len(ts) == 2:
                    ts = ts[1]
                else:
                    raise ValueError
                assert ts in ts_list, "Img timestamp not in anno file!"
                ts_anno[anno["timestamp"]] = anno
            if self.calib_dict is None:
                self.calib_dict = {}
                for _, meta in ts_anno.items():
                    for view in self.camera_list:
                        if view not in meta["imgs_meta"]:
                            continue
                        self.calib_dict[view] = meta["imgs_meta"][view][
                            "calib"
                        ]
                    if len(self.calib_dict) == len(self.camera_list):
                        break
        else:
            ts_anno = None
        if nums_to_read > 0:
            ts_list = ts_list[:nums_to_read]
            if ts_anno is not None:
                ts_anno = {ts: ts_anno[ts] for ts in ts_list}

        self.ts_anno = ts_anno
        self.ts_list = ts_list
        self.img_paths = img_paths
        self.standardized_cam_calibs = standardized_cam_calibs
        self.anno_view_key_map = anno_view_key_map
        self.multi_view_collect = multi_view_collect

    def __len__(self):
        return len(self.ts_anno)

    def _get_view_data(
        self, img, cam, img_path=None, color_space=None, ts=None
    ):
        uv_map = None
        _cam = self._map_anno_view_key(cam)
        if self.transforms is not None:
            if img_path is None:
                img_buf = cv2.imencode(".jpg", img)[1].tostring()
            else:
                img_buf = open(img_path, "rb").read()

            img_frame = {"img_buf": img_buf, "layout": "hwc"}
            if color_space is not None:
                img_frame["color_space"] = color_space

            if self.ts_anno is not None and ts is not None:
                img_frame["calib_all"] = deepcopy(
                    self.ts_anno[ts]["imgs_meta"][_cam]["calib"]
                )
            elif self.calib_dict is not None and cam in self.calib_dict:
                img_frame["calib_all"] = deepcopy(self.calib_dict[cam])

            if (
                self.standardized_cam_calibs is not None
                and cam in self.standardized_cam_calibs
            ):
                img_frame["standardized_calib_all"] = deepcopy(
                    img_frame.get("calib_all", {})
                )
                img_frame["standardized_calib_all"].update(
                    self.standardized_cam_calibs[cam]
                )
            if isinstance(self.transforms, dict):
                transforms = self.transforms[cam]
            else:
                transforms = self.transforms
            transform_output = transforms(img_frame)
            img = transform_output["img"]
            uv_map = transform_output.get("uv_map")
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy())
        if isinstance(uv_map, np.ndarray):
            uv_map = torch.from_numpy(uv_map)
        return img, uv_map

    def _map_anno_view_key(self, key):
        if self.anno_view_key_map is not None:
            _key = self.anno_view_key_map[key]
        else:
            _key = key
        return _key

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(
                "maximum length is %d, indexed by %d" % (len(self), idx)
            )
        ts = list(self.ts_anno.keys())[idx]
        view_imgs = {}
        view_imgs_ori = {}
        view_uv_maps = {}
        meta = {"timestamp": ts, "view_anno": {}}
        for cam in self.camera_list:
            _cam = self._map_anno_view_key(cam)
            if self.calib_dict is not None and cam in self.calib_dict:
                calib_params = self.calib_dict[cam]
                if (
                    "image_width" not in calib_params
                    or "image_height" not in calib_params
                ):
                    calib_params["image_width"] = self.view_shapes[cam][1]
                    calib_params["image_height"] = self.view_shapes[cam][0]

            if (
                ts not in self.ts_anno
                or _cam not in self.ts_anno[ts]["imgs_meta"]
            ):
                img = np.full(
                    (self.view_shapes[cam][0], self.view_shapes[cam][1], 3),
                    self.img_pad_value,
                    np.uint8,
                )
                view_imgs_ori[cam] = torch.from_numpy(img.copy())
                img, uv_map = self._get_view_data(img, cam)
                view_imgs[cam] = img
                view_uv_maps[cam] = uv_map
                img_shape = img.shape[1:]
                img_meta = {
                    "image_key": f"{ts}__{cam}__PAD",
                    "shape": img_shape,
                }
                if self.calib_dict is not None and cam in self.calib_dict:
                    img_meta.update(
                        {
                            "calib": deepcopy(self.calib_dict[cam]),
                        }
                    )
            else:
                img_path = os.path.join(
                    self.img_dir,
                    self.ts_anno[ts]["imgs_meta"][_cam]["image_key"] + ".jpg",
                )
                img = cv2.imread(img_path)
                if self.to_rgb:
                    img = img[:, :, ::-1]
                view_imgs_ori[cam] = torch.from_numpy(img.copy())
                img, uv_map = self._get_view_data(
                    img, cam, img_path, "rgb", ts
                )
                view_imgs[cam] = img
                view_uv_maps[cam] = uv_map
                # img_orders cam
                img_shape = img.shape[1:]
                if self.ts_anno is not None:
                    img_meta = deepcopy(self.ts_anno[ts]["imgs_meta"][_cam])
                else:
                    img_meta = {
                        "image_key": f"{ts}__{cam}",
                        "shape": img_shape,
                        "calib": deepcopy(self.calib_dict[cam]),
                    }
            meta["view_anno"][_cam] = {"meta": img_meta}

        meta["img_orders"] = self.camera_list
        if self.multi_view_collect is not None:
            imgs_meta = self.multi_view_collect(view_imgs)
            imgs_meta.update(
                self.multi_view_collect(view_imgs_ori, suffix="_ori")
            )
            imgs_meta["meta"] = json.dumps(meta)
        else:
            imgs_meta = {
                "img": torch.stack(list(view_imgs.values())),
                "img_ori": torch.stack(list(view_imgs_ori.values())),
                "meta": json.dumps(meta),
            }

        # generate homography and offset
        if self.homo_cfg is not None:
            # parse all calibration parameters
            homography, homo_offset = _generate_homography_from_anno(
                meta,
                self.camera_list,
                self.view_shapes,
                self.homo_cfg,
                self.standardized_cam_calibs,
            )
            imgs_meta["meta_info"] = dict(  # noqa
                homography=torch.from_numpy(
                    np.concatenate(list(homography.values()), axis=0).astype(
                        np.float32
                    )
                ),
                homo_offset=torch.from_numpy(
                    np.concatenate(list(homo_offset.values()), axis=0).astype(
                        np.float32
                    )
                ),
            )
        return imgs_meta
