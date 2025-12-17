# Copyright (c) Horizon Robotics. All rights reserved.
import base64
import copy
import json
import logging
import os
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch.utils.data as data
import yaml

from hat.core.box3d_utils import (
    cal_rot_y_from_3dbox_corners_in_camera,
    get_3dbox_corners,
)
from hat.core.virtual_camera.cameras import FisheyeCamera, PinholeCamera
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.lmdb import Lmdb, LmdbReadList
from .image_auto2d import Auto2dFromImage
from .pack_dataset import PackDataset
from .utils import Real3DRecReader

__all__ = [
    "Real3DDataset",
    "Auto3dFromImage",
    "Real3DDatasetRec",
    "PbRec2DDataset",
    "Real3DPackDataset",
    "Real3DDatasetLMDBFromBEV",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Real3DDataset(data.Dataset):
    """Real3D dataset class.

    Args:
        paths: Paths for image roots and annotations.
        num_classes: Number of classes.
        transforms: Transforms that applies to the data.
        num_dist: Length of dist coeffs.
        view: Camera view.
    """

    def __init__(
        self,
        paths: List[str],
        num_classes: int,
        num_dist: int,
        transforms: Optional[Callable] = None,
        view: Optional[str] = None,
    ):
        super(Real3DDataset, self).__init__()
        self.paths = paths
        self.num_classes = num_classes
        self.transforms = transforms
        self.num_dist = num_dist
        self.view = view

        assert (
            self.num_classes == 1
            or self.num_classes == 3
            or self.num_classes == 6
        ), "currently the number of classes must be 1 or 3 or 6"

        self._load_annotations(self.paths)
        self.num_samples = len(self._images)

    @staticmethod
    def get_category_id_dict(num_classes):
        if num_classes == 1:
            category_id_dict = {
                1: -99,  # Pedestrian -> Pedestrian
                2: 0,  # Car        -> Car
                3: -99,  # Cyclist    -> Cyclist
                4: 0,  # Bus        -> Car
                5: 0,  # Truck      -> Car
                6: 0,  # SpecialCar -> Car
                7: 0,  # Blur       -> ignore
                8: -99,  # Other    -> ignore
            }
        elif num_classes == 2:
            category_id_dict = {
                1: 0,  # Pedestrian -> Pedestrian
                2: -99,  # Car        -> Car
                3: 1,  # Cyclist    -> Cyclist
                4: -99,  # Bus        -> Car
                5: -99,  # Truck      -> Car
                6: -99,  # SpecialCar -> Car
                7: -99,  # Blur       -> ignore
                8: -99,  # Other    -> ignore
            }  # 'Dontcare' -> Ignore
        elif num_classes == 3:
            category_id_dict = {
                1: 0,  # Pedestrian -> Pedestrian
                2: 1,  # Car        -> Car
                3: 2,  # Cyclist    -> Cyclist
                4: 1,  # Bus        -> Car
                5: 1,  # Truck      -> Car
                6: 1,  # SpecialCar -> Car
                7: 1,  # Blur       -> ignore
                8: -99,  # Other    -> ignore
            }  # 'Dontcare' -> Ignore
        elif num_classes == 6:
            category_id_dict = {
                1: 0,  # Pedestrian -> Pedestrian
                2: 1,  # Car        -> Car
                3: 2,  # Cyclist    -> Cyclist
                4: 3,  # Bus        -> Bus
                5: 4,  # Truck      -> Truck
                6: 5,  # SpecialCar -> SpecialCar
                7: -99,  # Blur     -> ignore
                8: -99,  # Other    -> ignore
            }  # 'Dontcare' -> Ignore
        return category_id_dict

    def _load_annotations(self, paths):
        img_dir, anno_path = paths["img_dir"], paths["anno_path"]
        anno_list = []
        if isinstance(img_dir, (list, tuple)):
            assert len(img_dir) == len(
                anno_path
            ), "img_dir and anno_path are not matched, {} vs. {}".format(
                len(img_dir), len(anno_path)
            )

            for _img_dir, _anno_path in zip(img_dir, anno_path):
                anno = json.load(open(_anno_path))
                for im in anno["images"]:
                    im["img_path"] = os.path.join(_img_dir, im["file_name"])
                anno_list += [anno]
        elif isinstance(img_dir, str):
            for _anno_path in anno_path:
                logger.info(f"loading {_anno_path}")
                anno = json.load(open(_anno_path))
                for im in anno["images"]:
                    im["img_path"] = os.path.join(img_dir, im["file_name"])
                anno_list += [anno]
        else:
            raise NotImplementedError
        anno = anno_list[0]
        for _anno in anno_list[1:]:
            anno["images"] += _anno["images"]
            anno["annotations"] += _anno["annotations"]

        self._images = anno["images"]
        annos_by_image = {im["id"]: [] for im in self._images}
        for ann in anno["annotations"]:
            img_id = ann["image_id"]
            annos_by_image[img_id] += [ann]
        self._annos_by_image = annos_by_image

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_info = self._images[index]
        img_id = img_info["id"]
        img_path = img_info["img_path"]
        img = cv2.imread(img_path)
        calib = np.array(img_info["calib"], dtype=np.float32)
        dist_coeffs = img_info.get("distCoeffs", [0] * self.num_dist)
        dist_coeffs = dist_coeffs[: self.num_dist]
        anns = self._annos_by_image[img_id]

        data_dict = {
            "image_name": img_info["file_name"],
            "image_height": img.shape[0],
            "image_width": img.shape[1],
            "img": img,
            "imgs": [img],
            "color_space": "bgr",
            "annotations": anns,
            "calibration": calib,
            "dist_coeffs": dist_coeffs,
            "image_id": str(img_id),
            "ignore_mask": img_info["ignore_mask"],
            "view": self.view,
            "timestamp": img_info["timestamp"],
        }

        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict


@OBJECT_REGISTRY.register
class Auto3dFromImage(Auto2dFromImage):  # noqa: D205,D400
    """Auto3d from image.

    Args:
        calibration: With shape (3,4), for example, \
            calibration = ndarray([[1548, 0,    963, 0], \
                                   [0,    1548, 577, 0], \
                                   [0,    0,    1,   0]])
        dist_coeffs: For example, the length is 4, \
            dist_coeffs = [-0.3477686047554016,   0.13457843661308289, \
                           0.0004990333109162748, 8.008156873984262e-05]
    """

    def __init__(
        self,
        data_path: str,
        calibration: np.ndarray,
        dist_coeffs: List[int],
        transforms: Optional[Callable] = None,
        to_rgb: Optional[bool] = False,
        return_src_img: Optional[bool] = False,
        view: Optional[str] = None,
    ):
        super(Auto3dFromImage, self).__init__(
            data_path, transforms, to_rgb, return_src_img
        )
        self.calibration = calibration
        self.dist_coeffs = dist_coeffs
        self.view = view

    def __getitem__(self, item):
        data = super(Auto3dFromImage, self).__getitem__(item)
        # add calibration and dist_coeffs to data
        data["calibration"] = self.calibration
        data["dist_coeffs"] = self.dist_coeffs
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        return repr_str


@OBJECT_REGISTRY.register
class Real3DDatasetRec(data.Dataset):
    """Real3D dataset class in rec fashion.

    Args:
        paths: Paths for rec path.
        num_classes: Number of classes.
        transforms: Transforms that applies to the data.
        num_dist: Length of dist coeffs.
        view: Camera view.
        track_params: [Mode, pre_index, cur_index, range].
            if you select Mode is Front_back, you can set the range of the
            data fragments, and which index you want. for example, my data have
            3 continuous frame and I want to use first and third, you can set
            track_params=[2, 0, 2, 3]
        select_sample: Whether reselect index when invalid data.
        to_rgb: whether convter to rgb.
    """

    def __init__(
        self,
        paths: List[str],
        num_classes: int,
        num_dist: int,
        transforms: Optional[Callable] = None,
        view: Optional[str] = None,
        track_params: List[int] = None,
        select_sample: Optional[bool] = False,
        crop_kwargs: Optional[Dict] = None,
        to_rgb: Optional[bool] = False,
    ):
        super().__init__()
        self.paths = paths
        self.num_classes = num_classes
        self.transforms = transforms
        self.num_dist = num_dist
        self.view = view
        if track_params is None:
            self.track_params = [0, 0, 0, 1]
        else:
            self.track_params = track_params
        assert (
            self.num_classes == 1
            or self.num_classes == 2
            or self.num_classes == 3
            or self.num_classes == 6
        ), "currently the number of classes must be 1 or 2 or 3 or 6"

        self.crop_kwargs = crop_kwargs
        self.offline_calibs = None
        if (
            self.crop_kwargs is not None
            and self.crop_kwargs["dynamic_roi"]
            and self.crop_kwargs["calib_yaml_path"] is not None
        ):
            self.offline_calibs = yaml.load(
                open(self.crop_kwargs["calib_yaml_path"], "r"),
                Loader=yaml.FullLoader,
            )

        self.select_sample = select_sample
        self.to_rgb = to_rgb

    def __getstate__(self):
        state = self.__dict__
        state["recs"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.recs = []
        self.mat_vcsgnd2img = []
        for rec_path in self.paths:
            logger.info(f"loading {rec_path}")
            rec_reader = Real3DRecReader(rec_path)
            self.recs.append(rec_reader)
            if (
                self.crop_kwargs is not None
                and self.crop_kwargs["dynamic_roi"]
                and self.offline_calibs is not None
            ):  # noqa
                rec_flag = "/".join(rec_path.split("/")[-4:])
                self.mat_vcsgnd2img.append(
                    self.offline_calibs[rec_flag]["mat_vcsgnd2img"]
                )
            elif (
                self.crop_kwargs is not None
                and self.crop_kwargs["dynamic_roi"]
            ):
                # TODO, generate calib from rec
                raise NotImplementedError
        lengths = [len(rec) for rec in self.recs]
        self.acc_lengths = np.cumsum(lengths)
        self.num_samples = self.acc_lengths[-1]
        self.indices = list(range(self.num_samples))
        logging.info(f"real3d rec dataset length: {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def _get_data_from_idx(self, idx):
        """Get recio from index."""
        for length_idx in range(len(self.acc_lengths)):
            length_i = self.acc_lengths[length_idx]
            if idx > length_i:
                continue
            rec = self.recs[length_idx]
            mat_vcsgnd2img = None
            if (
                self.crop_kwargs is not None
                and self.crop_kwargs["dynamic_roi"]
            ):
                mat_vcsgnd2img = self.mat_vcsgnd2img[length_idx]
            if length_idx == 0:
                previous_idx = 0
            else:
                previous_idx = self.acc_lengths[length_idx - 1]
            idx_in_rec = idx - previous_idx
            assert idx_in_rec >= 0
            if idx_in_rec >= len(rec):
                idx_in_rec = len(rec) - 1
            img, label = rec[idx_in_rec]
            return img, label, mat_vcsgnd2img

    def _prepare_data(self, index):
        img, label, mat_vcsgnd2img = self._get_data_from_idx(index)
        img_info = label["meta"]
        img_id = img_info.get("id", img_info.get("image_key"))
        calib = np.array(img_info["calib"], dtype=np.float32)
        dist_coeffs = np.array(
            img_info.get("distCoeffs", [0] * self.num_dist), dtype=np.float32
        )
        Tr_vel2cam = np.array(
            img_info.get("Tr_vel2cam", np.eye(4)), dtype=np.float32
        )
        if "file_name" in img_info:
            image_name = img_info["file_name"]
        elif "image_key" in img_info:
            image_name = img_info["image_key"] + ".jpg"

        if mat_vcsgnd2img is None:
            mat_vcsgnd2img = np.array(
                [
                    [img.shape[1] / 2, 0, 0],
                    [img.shape[0] / 2, 0, 0],
                    [1, 0, 0],
                ],
                dtype=np.float32,
            )

        color_space = "bgr"
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.  # noqa
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color_space = "rgb"

        anns = label["objects"]
        data_dict = {
            "image_name": image_name,
            "image_height": img.shape[0],
            "image_width": img.shape[1],
            "img": img,
            "imgs": [img],
            "color_space": color_space,
            "layout": "hwc",
            "annotations": anns,
            "calibration": calib[:, :3],
            "dist_coeffs": dist_coeffs,
            "Tr_vel2cam": Tr_vel2cam,
            "image_id": str(img_id),
            "ignore_mask": img_info["ignore_mask"],
            "view": self.view,
            "index": index,
            "mat_vcsgnd2img": mat_vcsgnd2img,
            "timestamp": img_info["timestamp"],
        }
        if "Tr_vcs2cam" in img_info:
            data_dict.update(
                {
                    "Tr_vcs2cam": np.array(
                        img_info["Tr_vcs2cam"], dtype=np.float32
                    ),
                }
            )
        else:
            data_dict.update({"Tr_vcs2cam": np.eye(4, dtype=np.float32)})

        if "camera_model" in img_info:
            data_dict.update({"camera_model": img_info["camera_model"]})

        if "parsing" in img_info:
            gt_seg = img_info["parsing"]
            gt_seg = bytes(gt_seg, "utf-8")
            gt_seg = base64.b64decode(gt_seg)
            gt_seg = np.fromstring(gt_seg, np.uint8).reshape(
                img.shape[0], img.shape[1]
            )
            data_dict.update({"gt_seg": gt_seg})

        if "point_cloud" in img_info:
            gt_pcl = img_info["point_cloud"]
            gt_pcl = bytes(gt_pcl, "utf-8")
            gt_pcl = base64.b64decode(gt_pcl)
            gt_pcl = np.fromstring(gt_pcl, np.float32).reshape(-1, 8)
            data_dict.update({"gt_pcl": gt_pcl})

        return data_dict

    def _cell(self, index):
        if self.track_params[0]:
            index -= index % self.track_params[3]
            index_pre = index + self.track_params[1]
            index += self.track_params[2]

        data_dict = self._prepare_data(index)
        if self.track_params[0]:
            img_pre, label_pre, _ = self._get_data_from_idx(index_pre)
            data_dict.update(
                {
                    "imgs": [img_pre, data_dict["img"]],
                    "track_mode": self.track_params[0],
                }
            )

        if self.track_params[0] == 2:
            anns = [label_pre["objects"], data_dict["annotations"]]
            data_dict.update({"annotations": anns})

        data_dict.update({"valid": True})
        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __getitem__(self, index):
        if self.select_sample:
            flag = False
            while not flag:
                data_dict = self._cell(index)
                # check valid param
                flag = data_dict.get("valid", True)
                if not flag:
                    index = self.indices[np.random.randint(self.num_samples)]
        else:
            data_dict = self._cell(index)

        return data_dict


@OBJECT_REGISTRY.register
class Real3DDatasetLMDBFromBEV(Real3DDatasetRec):
    def __init__(
        self,
        paths,
        label_data_names,
        syncf_data_names,
        camera_names,
        **kwargs,
    ):
        super().__init__(paths, **kwargs)
        if self.offline_calibs is not None:
            raise NotImplementedError(
                "offline_calibs not supported in Real3DDatasetLMDBFromBEV"
            )
        self.label_data_names = label_data_names
        self.syncf_data_names = syncf_data_names
        self.camera_names = camera_names
        self.category_map = {
            "Pedestrian": 1,
            "Car": 2,
            "Cyclist": 3,
            "Bus": 4,
            "Truck": 5,
            "SpecialCar": 6,
            "Tricycle": 7,
            "Other": 8,
            "Construction": 6,
            "Blur": 6,
        }

    def __getstate__(self):
        state = self.__dict__
        state["label_lmdb"] = None
        for cam_name in self.camera_names:
            state[f"{cam_name}_lmdb"] = None
        state["sync_info"] = None
        state["num_samples"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.label_lmdb is None:
            self.label_lmdb = LmdbReadList(
                [
                    os.path.join(p, n)
                    for p, n in zip(self.paths, self.label_data_names)
                ],
            )
        if self.sync_info is None:
            self.sync_info = [
                Lmdb(os.path.join(p, n), writable=False)
                for p, n in zip(self.paths, self.syncf_data_names)
            ]
            self.num_samples_sep = []
            for f in self.sync_info:
                self.num_samples_sep.append(len(f))
            self.num_samples_sep = [
                len(f) * len(self.camera_names) for f in self.sync_info
            ]
        for cam_name in self.camera_names:
            if getattr(self, f"{cam_name}_lmdb") is None:
                setattr(
                    self,
                    f"{cam_name}_lmdb",
                    LmdbReadList(
                        [os.path.join(p, cam_name) for p in self.paths],
                    ),
                )
        self.num_samples = sum(self.num_samples_sep)
        logging.info(f"real3d lmdb dataset path: {self.paths[0]}")
        logging.info(f"real3d lmdb dataset length: {self.num_samples}")

    def _get_sf_part_id(self, idx):
        """Get the index of the sync file and the offset in the sync file."""
        idx, offset = 0, idx // len(self.camera_names)
        for i, ds_len in enumerate(self.num_samples_sep):
            ds_len //= len(self.camera_names)
            if offset >= ds_len:
                offset -= ds_len
            else:
                idx = i
                break
        return idx, offset

    def _get_data_from_idx(self, idx):
        """Get recio from index."""
        si, offset = self._get_sf_part_id(idx)
        multi_frame_sync_info = json.loads(
            self.sync_info[si].read(offset).decode()
        )
        while True:
            selected_cam = self.camera_names[idx % len(self.camera_names)]
            try:
                timestamp = multi_frame_sync_info[selected_cam]
                if isinstance(timestamp, list):
                    timestamp = timestamp[0]
                label_key = os.path.join(
                    multi_frame_sync_info["pack_dir"],
                    timestamp,
                )
                label_data = self.label_lmdb.read(label_key)
                label_data = json.loads(label_data.decode())
                label = label_data[selected_cam]
                break
            except KeyError:
                logger.warn(
                    f"no label data read for [{selected_cam}], "
                    "please check pack or self.camera_names"
                )
        image_key = os.path.join(
            os.path.dirname(label_key),
            selected_cam,
            os.path.basename(label_key) + ".jpg",
        )
        image_raw = getattr(self, f"{selected_cam}_lmdb").read(image_key)
        image_data = cv2.imdecode(
            np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR
        )

        param_dict = copy.deepcopy(label["meta"]["calib"])
        param_dict["image_width"] = image_data.shape[1]
        param_dict["image_height"] = image_data.shape[0]
        Camera = FisheyeCamera if "fisheye" in selected_cam else PinholeCamera
        camera = Camera.init_cam_param_by_dict(
            param_dict=param_dict,
            is_virtual=False,
        )

        cam_info = label["meta"]
        calib = cam_info["calib"]
        cam_info["Tr_vcs2cam"] = camera.poseMat_vcs2cam
        cam_info["Tr_vel2cam"] = camera.poseMat_lidar2cam
        cam_info["distCoeffs"] = calib["distort"]
        cam_info["calib"] = camera.camera_matrix

        cam_info["ignore_mask"] = cam_info["ignore_mask"]["ignore_mask"]

        label["meta"] = cam_info
        objs = label["objects"]
        filtered_objs = []
        for obj in objs:
            bbox2d = copy.deepcopy(obj["bbox2d"])
            if bbox2d is None:
                continue
            # TODO@fangquan.hu: update bbox2d to x1y1x2y2
            bbox2d = [  # from x1x2y1y2 to x1y1wh
                bbox2d[0],
                bbox2d[2],
                bbox2d[1] - bbox2d[0],
                bbox2d[3] - bbox2d[2],
            ]
            bbox3d = obj["bbox3d"]
            corner_pts_bbox3d = get_3dbox_corners(
                loc=[
                    bbox3d["location"][0],
                    bbox3d["location"][1],
                    bbox3d["location"][2] - bbox3d["dim"][0] / 2,
                ],
                dim=[
                    bbox3d["dim"][2],
                    bbox3d["dim"][1],
                    bbox3d["dim"][0],
                ],
                heading_angle=bbox3d["yaw"],
                coord_system="vcs",
            )
            loc_cam = camera.project_vcs2cam([bbox3d["location"]])[0]
            loc_cam[1] += bbox3d["dim"][0] / 2
            corner_pts_bbox3d_cam = camera.project_vcs2cam(corner_pts_bbox3d)
            rot_y_cam = cal_rot_y_from_3dbox_corners_in_camera(
                corner_pts_bbox3d_cam
            )
            bbox3d_cam = {
                "location": loc_cam.astype(np.float32),
                "dim": bbox3d["dim"],
                "rotation_y": rot_y_cam,
            }
            new_obj = {
                "bbox": bbox2d,
                "bbox_2d": bbox2d,
                "in_camera": bbox3d_cam,
                "ignore": obj["ignore"]["single_view"],
                "category_id": self.category_map[obj["bbox3d_attr"]["label"]],
                "occlusion": obj["bbox2d_attr"]["occlusion"],
                "image_id": label["meta"]["image_key"],
            }
            filtered_objs.append(new_obj)
        label["objects"] = filtered_objs
        return image_data, label, None


@OBJECT_REGISTRY.register
class Real3DPackDataset(PackDataset):
    def __init__(self, vc_transforms, **kwargs):
        super().__init__(**kwargs)
        assert len(self.views) == 1, "only support single view for real3d."
        self.vc_transforms = vc_transforms

    def __getitem__(self, idx):
        data_dict = super().__getitem__(idx)
        view = self.views[0]
        src_camera = data_dict["meta_info"]["src_cam"][view]
        data_dict = {
            "img": data_dict["img"],
            "org_image": data_dict["img"],
            "image_name": "{}_{}".format(data_dict["timestamp"], view),
            "img_id": data_dict["timestamp"],
            "layout": "hwc",
            "img_shape": data_dict["img"].shape,
            "image_width": data_dict["img"].shape[1],
            "image_height": data_dict["img"].shape[0],
            "pad_shape": data_dict["img"].shape,
            "calibration": src_camera.camera_matrix,
            "dist_coeffs": src_camera.distcoeffs,
            "Tr_vel2cam": src_camera.poseMat_lidar2cam,
            "Tr_vcs2cam": src_camera.poseMat_vcs2cam,
        }
        if self.vc_transforms is not None:
            for tfm in self.vc_transforms:
                if "ToTensor" in str(tfm):
                    data_dict["dst_image"] = data_dict["img"]
                data_dict = tfm(data_dict)
        return data_dict


@OBJECT_REGISTRY.register
class PbRec2DDataset(data.Dataset):
    """2D Pb rec dataset class in rec fashion.

    Args:
        paths: Paths for rec path.
        num_classes: Number of classes.
        transforms: Transforms that applies to the data.
        to_rgb: whether convter to rgb.
    """

    def __init__(
        self,
        paths: List[str],
        transforms: Optional[Callable] = None,
        to_rgb: Optional[bool] = False,
    ):
        super(PbRec2DDataset, self).__init__()
        self.paths = paths
        self.transforms = transforms
        self._load_recs(self.paths)
        self.num_samples = self.acc_lengths[-1]
        self.indices = list(range(self.num_samples))
        self.to_rgb = to_rgb

    def _load_recs(self, paths):
        assert isinstance(paths, list)
        self.recs = []
        for rec_path in paths:
            logger.info(f"loading {rec_path}")
            rec_reader = Real3DRecReader(rec_path)
            self.recs.append(rec_reader)
        lengths = [len(rec) for rec in self.recs]
        self.acc_lengths = np.cumsum(lengths)

    def __len__(self):
        return self.num_samples

    def _get_rec_from_idx(self, idx):
        """Get recio from index."""
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
        img, label = rec[idx]
        color_space = label["color_space"]
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.  # noqa
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        gt_bboxes = np.array(label["gt_bboxes"], dtype=np.float32)
        gt_classes = np.array(label["gt_classes"], dtype=np.int64)
        data_dict = {
            "img": img,
            "img_shape": img.shape,
            "color_space": color_space,
            "layout": label["layout"],
            "img_name": label["img_name"],
            "img_height": label["img_height"],
            "img_width": label["img_width"],
            "gt_bboxes": gt_bboxes,
            "gt_classes": gt_classes,
        }

        return data_dict

    def __getitem__(self, index):
        data_dict = self._prepare_data(index)
        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict
