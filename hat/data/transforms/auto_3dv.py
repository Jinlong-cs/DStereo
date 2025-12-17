# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import math
import random
from collections import OrderedDict
from logging import warning
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from hat.core.affine import get_vcs2bev_img_mat
from hat.core.bev_elevation_utils import compute_vismask, get_roi_resize_data
from hat.core.traj_pred_utils import SeqCenter, TdtCoordHelper
from hat.core.virtual_camera.utils import (
    parse_extrinsicParam,
    transform_euler2rotMat,
)
from hat.data.datasets.bev import ANCConvertSoftwareOffset, HomoGenerator
from hat.data.datasets.bev3d_multiview_dataset import (
    _get_calib_params_from_anno,
)
from hat.data.transforms.online_mapping_utils import (
    get_gt_online_mapping,
    get_roi_vcs_range_box,
    multi_line_segment_intersection,
)
from hat.data.transforms.real3d import (
    draw_heatmap,
    get_gaussian2D,
    get_reg_map,
)
from hat.metrics.bev.bev_discobj_eval import is_in_range
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, img_array2tensor
from hat.utils.filesystem import join_path
from hat.utils.package_helper import require_packages
from hat.visualize.generate_video import draw_ground
from hat.visualize.online_mapping import visualize_ipm

try:
    from hat_sim import IPSPyramid, nv122yuv444
except ImportError:
    IPSPyramid, nv122yuv444 = None, None

try:
    import torchvision
    import torchvision.transforms.functional as F
except ImportError:
    torchvision = None
    F = None

try:
    import torchvision
    import torchvision.transforms.functional as F
except ImportError:
    torchvision = None
    F = None


__all__ = [
    "ANCConvertPackDataTo3DV",
    "ANCResize3DV",
    "ANCCrop3DV",
    "ANCToTensor3DV",
    "ANCNormalize3DV",
    "ANCPrepareDepthPose",
    "ANCSelectDataByIdx",
    "ANCStackData",
    "ANCPrepareDataBEV",
    "ANCBev3dTargetGenerator",
    "ANCConvertReal3dTo3DV",
    "ANCBevSegTargetGenerator",
    "ANCBevSegAnnoGenerator",
    "ANCE2EDynamicTargetGenerator",
    "ANCPrepareTempoDataE2EDynamic",
    "ANCSetTemporalClearFlag",
    "ANCVisualizeIpm",
    "ANCOnlineMappingTargetGenerator",
    "ANCCrossPointTargetGenerator",
]

PIL_INTERP_CODES = {
    "nearest": F.InterpolationMode.NEAREST,
    "bilinear": F.InterpolationMode.BILINEAR,
}


def get_ctoff_map(wh: int, center: float) -> np.ndarray:
    """Calculate the regression map of bev center offset.

    Args:
        wh: the inserted kernel size.(order (w,h))
        center: object center in bev.(order u, v)

    Returns:
        bev center offset regression map
    """

    w, h = wh
    radius = (w // 2, h // 2)
    n, m = radius
    center_int = (int(center[0]), int(center[1]))
    center_offset = np.array(center) - np.array(center_int)
    x, y = center_int

    y_grid = np.arange(y - m, y + m + 1)
    x_grid = np.arange(x - n, x + n + 1)

    y_reg = center[1] - y_grid
    x_reg = center[0] - x_grid

    y_reg[m] = center_offset[1]
    x_reg[n] = center_offset[0]
    xv, yv = np.meshgrid(x_reg, y_reg)
    ct_off_reg_map = np.concatenate(
        [xv[:, :, np.newaxis], yv[:, :, np.newaxis]], axis=-1
    )
    return ct_off_reg_map


@OBJECT_REGISTRY.register
class ANCBevDiscObjLidarFilter(object):
    """If the nearest point cloud distance < threshold, rate_visible equal 0.

    Args:
        lidar_filter_cfg: Filted by lidar config,
            .. code-block:: json

                {
                    "name2annolabel": dict, A mapping from multimodal lmdb data
                        name to annotation label.

                    "cls2disthred": dict, a mapping from name to distance
                        threshold of lidar to cloud point.

                    "cls2exposure_pts": dict, a mapping from name to number of
                        exposure points.

                    "lidar_blind_vcs_range": tuple, lidar blind zone vcs range.
                        ordered by (bottom, right, top, left).

                    "judge_front_pts": bool, default False.
                        If False,indicating that at least N points are exposed.
                        If True,indicating that the first N points are exposed.

                }

    """

    def __init__(
        self,
        lidar_filter_cfg: Optional[dict] = None,
    ):
        self.lidar_filter_cfg = lidar_filter_cfg
        self.judge_front_pts = lidar_filter_cfg.get("judge_front_pts", False)
        if lidar_filter_cfg:
            self.lidar_blind_vcs_range = lidar_filter_cfg[
                "lidar_blind_vcs_range"
            ]

    def __call__(self, obj, auxiliary_info, name):
        assert "name2annolabel" in self.lidar_filter_cfg
        anno_label = self.lidar_filter_cfg["name2annolabel"][name]
        lidar_info = auxiliary_info[anno_label]
        assert "cls2disthred" in self.lidar_filter_cfg
        distance_thred = self.lidar_filter_cfg["cls2disthred"][name]
        assert "cls2exposure_pts" in self.lidar_filter_cfg
        exposure_pts = self.lidar_filter_cfg["cls2exposure_pts"][name]
        for info in lidar_info:
            data_vcs = info["data_vcs"]
            if obj["pts"][0][0] == data_vcs[0][0]:
                vcs_points = np.array(data_vcs)
                if vcs_points.shape != (4, 2):
                    in_blind = False
                else:
                    in_blind = (
                        is_in_range(vcs_points[0], self.lidar_blind_vcs_range)
                        or is_in_range(
                            vcs_points[1], self.lidar_blind_vcs_range
                        )
                        or is_in_range(
                            vcs_points[2], self.lidar_blind_vcs_range
                        )
                        or is_in_range(
                            vcs_points[3], self.lidar_blind_vcs_range
                        )
                    )
                if in_blind:
                    obj["rate_visible"] = 2
                    break
                if vcs_points.shape[1] == 2:
                    vcs_points = np.concatenate(
                        (vcs_points, np.zeros((vcs_points.shape[0], 1))),
                        axis=1,
                    )
                no_lidar_pack = info.get("no_lidar_pack", False)
                if no_lidar_pack:
                    obj["rate_visible"] = 0
                else:
                    nearest_point_distance_m = np.array(
                        info["nearest_point_distance_m"]
                    )
                    nearest_flag = nearest_point_distance_m < distance_thred
                    if self.judge_front_pts:
                        if nearest_flag[:exposure_pts].sum() == exposure_pts:
                            obj["rate_visible"] = 2
                        else:
                            obj["rate_visible"] = 0
                    else:
                        if nearest_flag.sum() > exposure_pts:
                            obj["rate_visible"] = 2
                        else:
                            obj["rate_visible"] = 0
                break
        return obj


@OBJECT_REGISTRY.register
class ANCBevDiscObjClassMatcher(object):
    """Generate group pair for joint optimization in discrete object detection.

    Args:
        ct_dist_type: Center distance type used to match object pair.

            .. code-block:: none

                l2_ct_dist: l2 center distance.

                height_ct_dist: projection of the center distance in
                    the direction of the long side.

                width_ct_dist: projection of the center distance in
                    the direction of the short side.

        class_match_cfg: Parameter config for pair match of
            different classes.
    """

    def __init__(
        self,
        ct_dist_type: str,
        class_match_cfg: Mapping,
    ):
        assert ct_dist_type in [
            "l2_ct_dist",
            "height_ct_dist",
            "width_ct_dist",
        ]
        self.ct_dist_type = ct_dist_type
        self.class_match_cfg = class_match_cfg

    @staticmethod
    def _get_yaw_diff(yaw_s, yaw_t):
        yaw_diff = yaw_s[:, None] - yaw_t[None, :]
        return np.abs(yaw_diff)

    @staticmethod
    def _get_ct_dist(ct_s, ct_t, mode="L1"):
        if mode == "L1":
            ct_dist = ct_s[:, None, :] - ct_t[None, :, :]
        elif mode == "L2":
            ct_dist = np.linalg.norm(
                ct_s[:, None, :] - ct_t[None, :, :], axis=-1, keepdims=True
            )
        else:
            raise NotImplementedError
        return ct_dist

    def _get_pair_by_ctrot_dist(
        self,
        info_source,
        info_target,
        ct_dist_type,
        source_class_id=0,
        target_class_id=1,
        yaw_match_threshold=0.1,
        ct_dist_thresh=6,
    ):
        """Get matched pair by center and rotation distance.

        Args:
            info_source: Source class infomation used to match.
            info_target: Target class information used to match.
            ct_dist_type: Used to determine the method of ct dist computation.
            yaw_match_threshold: Threshold for yaw difference.
            ct_dist_thresh: Threshold for center distance difference.

        Returns: matched pair index list.

        """
        assert (
            "yaw" in info_source
            and "yaw" in info_target
            and "vcs_loc" in info_source
            and "vcs_loc" in info_target
        )
        s_yaw = info_source["yaw"]
        t_yaw = info_target["yaw"]
        s_loc = info_source["vcs_loc"]
        t_loc = info_target["vcs_loc"]
        yaw_diff = self._get_yaw_diff(s_yaw, t_yaw)
        ct_dist = self._get_ct_dist(s_loc, t_loc, mode="L1")
        s_idx, t_idx = np.where(yaw_diff < yaw_match_threshold)
        matched_src = np.zeros(shape=len(s_yaw), dtype=np.int)
        matched_pair = []
        for m, n in zip(s_idx, t_idx):
            # if one object has matched, skip
            if matched_src[m] == 1:
                continue
            if source_class_id == target_class_id and matched_src[n] == 1:
                continue
            ct_d = ct_dist[m][n]
            points = info_source["points"][m]
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            length = np.linalg.norm(points[0] - points[1])
            width = np.linalg.norm(points[1] - points[2])
            if length > width:
                length_vector = points[1] - points[0]
            else:
                length_vector = points[2] - points[1]
            # compute the angle between center distance vector
            # and the length vector of two objects.
            cos_theta = np.dot(ct_d, length_vector) / (
                np.linalg.norm(ct_d) * np.linalg.norm(length_vector)
            )
            theta = np.arccos(cos_theta)
            if cos_theta < 0:
                theta = np.pi - theta

            if ct_dist_type == "l2_ct_dist":
                # l2 center distance
                dist = np.linalg.norm(ct_d)
            elif ct_dist_type == "width_ct_dist":
                # compute the projection distance in short side
                sin_theta = np.sin(theta)
                dist = np.linalg.norm(ct_d) * sin_theta
            elif ct_dist_type == "height_ct_dist":
                # compute the projection distance in long side
                dist = np.linalg.norm(ct_d) * cos_theta
            else:
                raise NotImplementedError
            if dist <= ct_dist_thresh:
                matched_src[m] = 1
                if source_class_id == target_class_id:
                    matched_src[n] = 1
                    if m == n:
                        continue
                matched_pair.append([m, n])
        return matched_pair

    def __call__(self, info_source, info_target):
        matched_pair = self._get_pair_by_ctrot_dist(
            info_source, info_target, self.ct_dist_type, **self.class_match_cfg
        )
        return matched_pair


@OBJECT_REGISTRY.register
class ANCConvertPackDataTo3DV(object):
    """Convert the data read out by pack to BEV format.

    BEV task need homography and homo offset, support homogene with
    calibration paras from packs in this class.
    And transform origin imgs to PIL imgs, the result of this
    transformation can consistent with auto_3dv output results.
    Additional ground line drawing on original images.

    Args:
        homo_gen: HomoGenerator dict for compute homography.
        homo_gen_small: HomoGenerator dict for compute
             homography of small range.
        homo_gen_high_sp: HomoGenerator dict for compute
             homography of high spatial resolution.
        calib: Whether get calibration.
        ground_level: Whether draw ground level on imgs.
        nv12_format: format of input image is NV12 or Not.
        num_frames_per_iter: hope to split the temporal data of each sample,
        and each iter only outputs a specific number of frames
        temporal_bev: Whether suit temporal fusion.

    """

    def __init__(
        self,
        homo_gen: dict = None,
        homo_gen_small: dict = None,
        homo_gen_high_sp: dict = None,
        convert_software_offset: dict = None,
        convert_software_offset_small: dict = None,
        calib: bool = False,
        ground_level: bool = False,
        nv12_format: bool = True,
        num_frames_per_iter: int = 1,
        temporal_bev: bool = False,
    ):
        self.homo_gen = homo_gen
        self.homo_gen_small = homo_gen_small
        self.homo_gen_high_sp = homo_gen_high_sp
        self.convert_software_offset = convert_software_offset
        self.convert_software_offset_small = convert_software_offset_small
        self.calib = calib
        self.ground_level = ground_level
        self.nv12_format = nv12_format
        self.num_frames_per_iter = num_frames_per_iter
        self.temporal_bev = temporal_bev
        self._meta_info, self._meta_info_small, self._local_calib = (
            None,
            None,
            None,
        )
        self._meta_info_high_sp = None

    def reformat_calibration(self, view_calib):

        K = [
            [view_calib["focal_u"], 0, view_calib["center_u"]],
            [0, view_calib["focal_v"], view_calib["center_v"]],
            [0, 0, 1],
        ]
        K = np.array(K)
        d_coef = np.array(view_calib["distort"])
        T_vcs2cam = parse_extrinsicParam(view_calib)
        cam2local_eular = np.array(
            [view_calib["roll"], view_calib["pitch"], view_calib["yaw"]]
        )
        cam2local_trans = np.array(
            [
                view_calib["camera_x"],
                view_calib["camera_y"],
                view_calib["camera_z"],
            ]
        )
        T_local2vcs = np.eye(4, dtype=np.float)
        rot_local2vcs = transform_euler2rotMat(
            np.array(view_calib["vcs"]["rotation"])
        )
        T_local2vcs[:3, :3] = rot_local2vcs
        T_local2vcs[0:3, 3] = np.array(view_calib["vcs"]["translation"]).T

        return {
            "K": K,
            "d_coef": d_coef,
            "T_vcs2cam": T_vcs2cam,
            "cam2local_eular": cam2local_eular,
            "cam2local_trans": cam2local_trans,
            "T_local2vcs": T_local2vcs,
        }

    def get_local_calib(self, calibrations):
        intrinsics = []
        distortcoef = []
        local2vcs = []
        local2cam = []
        calibrations = _as_list(calibrations)
        for i in range(len(calibrations)):
            para = self.reformat_calibration(calibrations[i])
            # intrinsics
            intrinsics.append(para["K"])
            # distortcoef
            distortcoef.append(para["d_coef"])
            # local camera to vcs coord
            local2vcs_ = np.eye(4, dtype=np.float)
            R_local2vcs = transform_euler2rotMat(
                np.array(calibrations[i]["vcs"]["rotation"])
            )
            local2vcs_[:3, :3] = R_local2vcs
            local2vcs_[0:3, 3] = np.array(
                calibrations[i]["vcs"]["translation"]
            ).T
            local2vcs.append(local2vcs_)
            # local camera to Horizon camera
            Hcam2local = np.eye(4, dtype=np.float)
            cam_x = calibrations[i]["camera_x"]
            cam_y = calibrations[i]["camera_y"]
            cam_z = calibrations[i]["camera_z"]
            roll = calibrations[i]["roll"]
            pitch = calibrations[i]["pitch"]
            yaw = calibrations[i]["yaw"]
            R_Hcam2local = transform_euler2rotMat([roll, pitch, yaw])
            Hcam2local[:3, :3] = R_Hcam2local
            Hcam2local[0:3, 3] = np.array([cam_x, cam_y, cam_z]).T

            # camera coord (opencv axis) to Horizon camera
            opencvCam2Hcam = np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
            )
            cam2local = Hcam2local @ opencvCam2Hcam
            local2cam_ = np.linalg.inv(cam2local)
            local2cam.append(local2cam_)

        intrinsics = np.stack(intrinsics, axis=0)
        distortcoef = np.stack(distortcoef, axis=0)
        local2vcs = np.stack(local2vcs, axis=0)
        local2cam = np.stack(local2cam, axis=0)

        local_calib = (intrinsics, distortcoef, local2vcs, local2cam)
        return local_calib

    def get_meta_info(self, data_dict, homo_gen, convert_software_offset):
        calib_para = {}
        for view, calib in zip(
            data_dict["camera_list"], data_dict["camera_calib"]
        ):
            calib_para[view] = self.reformat_calibration(calib)
        homo_generator = HomoGenerator(
            calib_path=None,
            homo_path=None,
            calib_para=calib_para,
            **homo_gen,
        )
        meta_info = homo_generator.get_meta_info()
        if convert_software_offset is not None:
            converter = ANCConvertSoftwareOffset(**convert_software_offset)
            meta_info["homo_offset"] = converter()
        return meta_info

    def __call__(self, data_dict: Mapping):
        assert "img" in data_dict

        # homography & homo offset
        if self.homo_gen is not None:
            if self._meta_info is None:
                self._meta_info = self.get_meta_info(
                    data_dict, self.homo_gen, self.convert_software_offset
                )
            data_dict["meta_info"] = self._meta_info
            for mate_name, meta_value in data_dict["meta_info"].items():
                data_dict["meta_info"][
                    mate_name
                ] = ANCToTensor3DV._meta_to_tensor(meta_value)
        if self.homo_gen_small is not None:
            if self._meta_info_small is None:
                self._meta_info_small = self.get_meta_info(
                    data_dict,
                    self.homo_gen_small,
                    self.convert_software_offset_small,
                )
            data_dict["meta_info_small"] = self._meta_info_small
            for mate_name, meta_value in data_dict["meta_info_small"].items():
                data_dict["meta_info_small"][
                    mate_name
                ] = ANCToTensor3DV._meta_to_tensor(meta_value)
        if self.homo_gen_high_sp is not None:
            if self._meta_info_high_sp is None:
                self._meta_info_high_sp = self.get_meta_info(
                    data_dict,
                    self.homo_gen_high_sp,
                    self.convert_software_offset_small,
                )
            data_dict["meta_info_high_sp"] = self._meta_info_high_sp
            for mate_name, meta_value in data_dict[
                "meta_info_high_sp"
            ].items():
                data_dict["meta_info_high_sp"][
                    mate_name
                ] = ANCToTensor3DV._meta_to_tensor(meta_value)

        if self.homo_gen or self.homo_gen_small:
            data_dict["img_vis"] = []
            per_view_shape = (
                self.homo_gen["per_view_shape"]
                if self.homo_gen
                else self.homo_gen_small["per_view_shape"]
            )

            if self.nv12_format:
                data_dict["ratio"] = []

            for view, img in zip(data_dict["camera_list"], data_dict["img"]):
                height, width = per_view_shape[view]
                if self.nv12_format:
                    ratio = ((height * width * 3 // 2) / img.shape[0]) ** 0.5
                    assert (
                        ratio % 1 == 0
                    ), "please recheck img's shape from pack."
                    ratio = int(ratio)
                    height, width = height // ratio, width // ratio
                    bgr_img = cv2.cvtColor(
                        img.reshape((height * 3 // 2, width)),
                        cv2.COLOR_YUV2BGR_NV12,
                    )
                    data_dict["ratio"].append(ratio)
                    if ratio != 1:
                        bgr_img = cv2.resize(bgr_img, None, fx=ratio, fy=ratio)
                else:
                    assert (
                        height % img.shape[0] == 0
                        and width % img.shape[1] == 0
                    ), "please recheck img's shape from pack."
                    ratio_h = height // img.shape[0]
                    ratio_w = width // img.shape[1]
                    bgr_img = cv2.cvtColor(
                        img,
                        cv2.COLOR_RGB2BGR,
                    )
                    if ratio_h != 1 or ratio_w != 1:
                        bgr_img = cv2.resize(
                            bgr_img, None, fx=ratio_w, fy=ratio_h
                        )

                data_dict["img_vis"].append(bgr_img)

        if self.calib:
            assert "camera_calib" in data_dict
            if self._local_calib is None:
                self._local_calib = self.get_local_calib(
                    data_dict["camera_calib"]
                )
            (intrinsics, distortcoef, local2vcs, local2cam) = self._local_calib
            data_dict["K"] = torch.from_numpy(intrinsics)
            data_dict["dist"] = torch.from_numpy(distortcoef)
            data_dict["local2vcs"] = torch.from_numpy(local2vcs)
            data_dict["local2cam"] = torch.from_numpy(local2cam)

        if self.ground_level:
            assert "img_vis" in data_dict
            assert "camera_calib" in data_dict
            assert len(data_dict["camera_calib"]) == len(data_dict["img_vis"])
            data_dict["img_vis"] = [
                draw_ground(img, cali)
                for img, cali in zip(
                    data_dict["img_vis"], data_dict["camera_calib"]
                )
            ]

        if self.nv12_format:
            if "img_vis" in data_dict:
                # 如果是nv12格式的图像，那么pil_imgs中不会被使用到，只是为了兼容考虑
                data_dict["pil_imgs"] = [
                    [
                        torchvision.transforms.PILToTensor()(
                            Image.fromarray(img)
                        )
                        for img in data_dict["img_vis"]
                    ]
                ]
            data_dict["img"] = [data_dict["img"]]

        else:
            # 表示使用RGB格式的图像输入,会额外的对图像做jpg的编解码
            # 以便和训练时候的图像输入处理流程对齐
            data_dict["pil_imgs"] = []

            for image in data_dict["img"]:
                _, image_encode = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                image = cv2.imdecode(image_encode, cv2.IMREAD_COLOR)
                data_dict["pil_imgs"].append(Image.fromarray(image))
            data_dict["pil_imgs"] = [data_dict["pil_imgs"]]

            data_dict.pop("img")

        if "img_vis" in data_dict:
            data_dict["img_vis"] = [
                [
                    torch.from_numpy(img.transpose(2, 0, 1))
                    for img in data_dict["img_vis"]
                ]
            ]

        if "camera_calib" in data_dict:
            data_dict.pop("camera_calib")
        data_dict["timestamp"] = torch.tensor(
            data_dict["timestamp"] / 1000.0
        ).unsqueeze(0)
        data_dict["temporal_info"] = {
            "num_frames_per_iter": self.num_frames_per_iter
        }
        data_dict.pop("camera_list")

        if self.temporal_bev:
            # 添加和temporal_fusion相关的变量
            assert "pack_start_flag" in data_dict
            data_dict["temporal_clr_flag"] = data_dict[
                "pack_start_flag"
            ] or data_dict.get("odo_invaild_flag", False)
            data_dict["return_latest_flag"] = True
            data_dict.pop("pack_start_flag")
        return data_dict


@OBJECT_REGISTRY.register
class ANCCollect3DV(object):
    """Selecting data what we really need from all frames.

    Args:
        load_data_types: which type data to load. Depth training
            only need gt_depths; Pose training need intrinsics, obj_mask.
        img_idxs: frame indexs of input images.
        gt_depth_idxs: frame indexs of gt depth to load.
        gt_bev_seg_idxs: frame indexs of gt bev seg to load.
        gt_bev_elevation_idxs: frame indexs of gt bev elevation
            to load.
        gt_bev_freespace_idxs: frame indexs of gt bev freespace
            to load.
        gt_bev_elevation_vismask_idxs: frame indexs of gt bev
            elevation vismask to load.
        gt_bev_3d_idxs: frame indexs of gt bev 3d to load.
        gt_multi_view_idx: frame indexs of gt multi view to load.
        gt_bev_motionflow_idxs: frame indexs bev
            motion flow to load.
        gt_bev_parking_idx: frame index of gt bev psd object
        gt_bev_discobj_idx: frame index of gt bev discrete object
            (e.g., crosswalk, road arrow, etc.) to load
        obj_mask_idx: frame index of object mask to load.
        timestamp_idx: timestamp index to load
        pose_idxs: timestamp index to load pose
        fill_fake_temporal_data: whether fill fake temporal data or not.
            When using single-frame data for temporal training, setting to true
            to extend single-frame data to multiple-frame length.
        num_frames_per_iter: number of frames in a iteration for e2e task,
            default 1.
    """

    def __init__(
        self,
        load_data_types: List[str],
        img_idxs: List[int],
        gt_depth_idx: int = 1,
        gt_bev_seg_idxs: Sequence[int] = (0,),
        gt_bev_elevation_idxs: Sequence[int] = (0,),
        gt_bev_freespace_idxs: Sequence[int] = (0,),
        gt_bev_elevation_vismask_idxs: Sequence[int] = (0,),
        gt_bev_3d_idx: int = 0,
        gt_multi_view_idx: int = 0,
        gt_bev_motionflow_idxs: Sequence[int] = (0,),
        gt_bev_parking_idx: int = 0,
        gt_bev_discobj_idx: int = 0,
        gt_om_idx: int = 0,
        gt_bev_crosspoint_idx: int = 0,
        img_paths_idx: int = 0,
        obj_mask_idx: int = 1,
        timestamp_idx: int = 0,
        pose_idxs: Sequence[int] = (0,),
        occlusion_idxs: Sequence[int] = (0,),
        fill_fake_temporal_data: bool = False,
        num_frames_per_iter: int = 1,
        lidar_idxs: Sequence[int] = (0,),
        lidar_pose_idxs: Sequence[int] = (0,),
        load_tag_types: Sequence[str] = (
            "event_id",
            "city",
            "light",
            "scene",
            "time",
            "weather",
        ),
        pose_type_for_eval: str = "",
        load_object_tag_types: Sequence[str] = ("cutin",),
    ):
        self.load_data_types = load_data_types
        self.img_idxs = img_idxs
        self.img_paths_idx = img_paths_idx
        self.gt_bev_freespace_idxs = gt_bev_freespace_idxs
        self.gt_depth_idx = gt_depth_idx
        self.gt_bev_seg_idxs = gt_bev_seg_idxs
        self.gt_bev_elevation_idxs = gt_bev_elevation_idxs
        self.gt_bev_freespace_idxs = gt_bev_freespace_idxs
        self.gt_bev_elevation_vismask_idxs = gt_bev_elevation_vismask_idxs
        self.gt_bev_3d_idx = gt_bev_3d_idx
        self.gt_multi_view_idx = gt_multi_view_idx
        self.gt_bev_motionflow_idxs = gt_bev_motionflow_idxs
        self.gt_om_idx = gt_om_idx
        self.gt_bev_crosspoint_idx = gt_bev_crosspoint_idx
        self.gt_bev_parking_idx = gt_bev_parking_idx
        self.gt_bev_discobj_idx = gt_bev_discobj_idx
        self.obj_mask_idx = obj_mask_idx
        self.timestamp_idx = timestamp_idx
        self.pose_idxs = pose_idxs
        self.occlusion_idxs = occlusion_idxs
        self.fill_fake_temporal_data = fill_fake_temporal_data
        self.num_frames_per_iter = num_frames_per_iter
        self.lidar_idxs = lidar_idxs
        self.lidar_pose_idxs = lidar_pose_idxs
        self.load_tag_types = load_tag_types
        self.pose_type_for_eval = pose_type_for_eval
        self.load_object_tag_types = load_object_tag_types

    def _squeeze_list(
        self,
        list_data,
    ):
        if isinstance(list_data, Sequence) and len(list_data) == 1:
            return self._squeeze_list(list_data[0])
        else:
            return list_data

    def _load_img(self, frames):
        pil_imgs = []
        for idx in self.img_idxs:
            # Only frame 0 is collected for non-sequential data
            if self.fill_fake_temporal_data:
                pil_imgs.append(frames[0].img())
            else:
                # Judgment here is to support long-short clips mixed temporal
                # training. For example, When length of long-short clips are
                # 5 and 3, it is necessary to extend the length of short clip
                # to 5 so as to unify the logic of temporal training.
                if idx < len(frames):
                    pil_imgs.append(frames[idx].img())
                else:
                    pil_imgs.append(frames[len(frames) - 1].img())
        return pil_imgs

    def _load_pose(self, frames):
        pose = []
        for idx in self.pose_idxs:
            if self.fill_fake_temporal_data:
                pose.append(frames[0].eye_pose)
            else:
                if idx < len(frames):
                    pose.append(frames[idx].pose())
                else:
                    pose.append(frames[len(frames) - 1].pose())
        return pose

    def _load_wheel_pose_base(
        self, frames, task_name=None, enable_eye_pose=True
    ):
        wheel_pose = []
        for idx in self.pose_idxs:
            if self.fill_fake_temporal_data and enable_eye_pose:
                wheel_pose.append(frames[0].eye_pose)
            else:
                if idx < len(frames):
                    frame_idx = idx
                else:
                    frame_idx = len(frames) - 1
                if not task_name:
                    wheel_pose.append(frames[frame_idx].wheel_pose())
                elif task_name == "vehicle":
                    wheel_pose.append(frames[frame_idx].bev3d_vehicle_pose())
                elif task_name == "vrumerge":
                    wheel_pose.append(frames[frame_idx].bev3d_vrumerge_pose())
                else:
                    raise AssertionError(
                        "task_name only support vehicle and vrumerge"
                    )
        return wheel_pose

    def _load_point_cloud(self, frames, valid_lidar_idxs: List[int]):
        point_clouds = []
        for idx in self.lidar_idxs:
            if self.fill_fake_temporal_data:
                point_clouds.append(frames[valid_lidar_idxs[0]].point_cloud())
            else:
                idx_idx = idx if idx < len(valid_lidar_idxs) else -1
                point_clouds.append(
                    frames[valid_lidar_idxs[idx_idx]].point_cloud()
                )
        return point_clouds

    def _load_point_cloud_wheel_pose(
        self,
        frames,
        valid_lidar_idxs: List[int],
        valid_lidar_timestamp: List[float],
    ):
        point_cloud_wheel_pose = []
        for idx in self.lidar_pose_idxs:
            if self.fill_fake_temporal_data:
                point_cloud_wheel_pose.append(frames[0].eye_pose)
            else:
                if idx < len(valid_lidar_idxs):
                    point_cloud_wheel_pose.append(
                        frames[valid_lidar_idxs[idx]].wheel_pose(
                            valid_lidar_timestamp[idx]
                        )
                    )
                else:
                    point_cloud_wheel_pose.append(
                        frames[valid_lidar_idxs[-1]].wheel_pose(
                            valid_lidar_timestamp[-1]
                        )
                    )
        return point_cloud_wheel_pose

    def _load_wheel_pose(self, frames):
        return self._load_wheel_pose_base(frames)

    def _load_bev3d_pose(self, frame, task_name, enable_eye_pose=True):
        return self._load_wheel_pose_base(frame, task_name, enable_eye_pose)

    def __call__(self, data_dict: Mapping):
        frames = data_dict.pop("frames")

        data_dict["pil_imgs"] = self._load_img(frames)

        if "origin_imgs" in self.load_data_types:
            data_dict["origin_imgs"] = copy.deepcopy(data_dict["pil_imgs"])

        if "gt_depth" in self.load_data_types:
            data_dict["gt_depth"] = frames[self.gt_depth_idx].depth()

        if "gt_bev_seg" in self.load_data_types:
            data_dict["gt_bev_seg"] = self._squeeze_list(
                [frames[idx].bev_seg() for idx in self.gt_bev_seg_idxs]
            )
        if "occlusion" in self.load_data_types:
            data_dict["occlusion"] = self._squeeze_list(
                [frames[idx].bev_occlusion() for idx in self.occlusion_idxs]
            )
        if "gt_bev_seg_anno" in self.load_data_types:
            data_dict["gt_bev_static_anno"] = self._squeeze_list(
                [
                    frames[idx].read_static_anno()
                    for idx in self.gt_bev_seg_idxs
                ]
            )

        if "gt_bev_freespace" in self.load_data_types:
            data_dict["gt_bev_freespace_raw"] = self._squeeze_list(
                [
                    frames[idx].bev_freespace()
                    for idx in self.gt_bev_freespace_idxs
                ]
            )
            if "bev_freespace_mask" in self.load_data_types:
                data_dict["bev_freespace_mask"] = self._squeeze_list(
                    [
                        frames[idx].bev_freespace_mask()
                        for idx in self.gt_bev_freespace_idxs
                    ]
                )

        if "gt_bev_elevation" in self.load_data_types:
            data_dict["gt_bev_elevation_raw"] = self._squeeze_list(
                [
                    frames[idx].bev_elevation()
                    for idx in self.gt_bev_elevation_idxs
                ]
            )

        if "gt_bev_elevation_vismask" in self.load_data_types:
            data_dict["gt_bev_elevation_vismask_raw"] = self._squeeze_list(
                [
                    frames[idx].bev_elevation_vismask()
                    for idx in self.gt_bev_elevation_vismask_idxs
                ]
            )

        if "gt_bev_3d" in self.load_data_types:
            if self.num_frames_per_iter > 1:
                data_dict["gt_bev_dynamic_anno"] = self._squeeze_list(
                    [
                        frames[self.gt_bev_3d_idx + idx].bev_3d()
                        for idx in range(self.num_frames_per_iter)
                    ]
                )
            else:
                data_dict["gt_bev_dynamic_anno"] = frames[
                    self.gt_bev_3d_idx
                ].bev_3d()

        if "gt_multi_view" in self.load_data_types:
            if self.num_frames_per_iter > 1:
                data_dict["gt_multi_view"] = self._squeeze_list(
                    [
                        frames[self.gt_bev_3d_idx + idx].multi_view()
                        for idx in range(self.num_frames_per_iter)
                    ]
                )
            else:
                data_dict["gt_multi_view"] = frames[
                    self.gt_multi_view_idx
                ].multi_view()

        if "gt_bev_motion_flow" in self.load_data_types:
            data_dict["gt_bev_motion_flow"] = self._squeeze_list(
                [
                    frames[idx].bev_motion_flow()
                    for idx in self.gt_bev_motionflow_idxs
                ]
            )

        if "gt_bev_parking_obj" in self.load_data_types:
            data_dict["gt_bev_parking_obj"] = frames[
                self.gt_bev_parking_idx
            ].bev_parking_obj()

        if "gt_bev_discrete_obj" in self.load_data_types:
            data_dict["gt_bev_discrete_raw"] = frames[
                self.gt_bev_discobj_idx
            ].bev_discrete_obj()

        if "bev_occlusion_mask" in self.load_data_types:
            data_dict["bev_occlusion_mask"] = frames[
                self.gt_bev_discobj_idx
            ].bev_occlusion_mask()

        if "gt_online_mapping" in self.load_data_types:
            data_dict["gt_online_mapping"] = frames[self.gt_om_idx].om()

        if "gt_bev_crosspoint" in self.load_data_types:
            data_dict["gt_bev_crosspoint"] = frames[
                self.gt_bev_crosspoint_idx
            ].bev_crosspoint()

        # load object mask of t-1 in front view, only for 2.5d task
        if "obj_mask" in self.load_data_types:
            data_dict["obj_mask"] = frames[self.obj_mask_idx].front_seg()

        # only load timestamp of t
        if "timestamp" in self.load_data_types:
            if self.num_frames_per_iter > 1:
                # obtain the timestamp for every frame in the iter.
                data_dict["timestamp"] = np.array(
                    [
                        frames[self.timestamp_idx + idx].timestamp()
                        for idx in range(self.num_frames_per_iter)
                    ]
                ).reshape(-1)
            else:
                data_dict["timestamp"] = frames[self.timestamp_idx].timestamp()

        if "pose" in self.load_data_types:
            data_dict["pose"] = self._load_pose(frames)

        if "wheel_pose" in self.load_data_types:
            data_dict["pose"] = self._load_wheel_pose(frames)

        if "bev3d_vehicle_pose" in self.load_data_types:
            data_dict["pose"] = self._load_bev3d_pose(frames, "vehicle")

        if "bev3d_vrumerge_pose" in self.load_data_types:
            data_dict["pose"] = self._load_bev3d_pose(frames, "vrumerge")

        if self.pose_type_for_eval:
            pose = []
            if "bev3d_vehicle_pose" == self.pose_type_for_eval:
                pose = self._load_bev3d_pose(frames, "vehicle", False)
            if "bev3d_vrumerge_pose" == self.pose_type_for_eval:
                pose = self._load_bev3d_pose(frames, "vrumerge", False)
            if pose:
                if isinstance(self.gt_bev_3d_idx, Sequence):
                    pose = self._squeeze_list(
                        [pose[idx].tolist() for idx in self.gt_bev_3d_idx]
                    )
                else:
                    pose = pose[self.gt_bev_3d_idx].tolist()
            data_dict["ego_pose"] = pose

        # only return pack_dir of frame 0
        if "pack_dir" in self.load_data_types:
            data_dict["pack_dir"] = frames[0].pack_dir

        if "clip_key" in self.load_data_types:
            data_dict["clip_key"] = frames[0].clip_key

        # return img_paths of every frame.
        if "img_paths" in self.load_data_types:
            data_dict["img_paths"] = frames[0].img_paths

        # NOTE: e2e_dynamic_anno save whole clip's annotations through a
        # single str, here only get the frames[0]'s anno.
        if "e2e_dynamic_anno" in self.load_data_types:
            data_dict["e2e_dynamic_anno"] = frames[0].e2e_dynamic_anno()

        if "point_cloud" in self.load_data_types:
            valid_lidar_idx = [
                sweep_idx
                for sweep_idx, frame in enumerate(frames)
                if frame.frame_sync_info["lidar"]
            ]
            valid_lidar_timestamp = [
                frames[valid_lidar_idx[i]]
                .frame_sync_info["lidar"]
                .split(".")[0]
                for i in range(len(valid_lidar_idx))
            ]

            data_dict["point_clouds"] = self._load_point_cloud(
                frames, valid_lidar_idx
            )

            data_dict["have_lidar_input"] = (
                frames[0].frame_sync_info["lidar"] is not None
            )

            if "lidar_wheel_pose" in self.load_data_types:
                data_dict["lidar_pose"] = self._load_point_cloud_wheel_pose(
                    frames, valid_lidar_idx, valid_lidar_timestamp
                )

            if "lidar_detection_gt" in self.load_data_types:
                data_dict["lidar_gt"] = frames[0].lidar_detection()
                data_dict["object_token"] = frames[0].frame_sync_info[
                    "gt_name"
                ]

        # scene/image tag and event info
        if "tag_info" in self.load_data_types:
            if self.num_frames_per_iter > 1:
                data_dict["tag_info"] = []
                for idx in range(
                    self.gt_bev_3d_idx,
                    self.gt_bev_3d_idx + self.num_frames_per_iter,
                ):
                    tag_dict = {}
                    tags = frames[idx].tag_info
                    for tag in self.load_tag_types:
                        tag_dict[tag] = tags.get(tag, "")
                    data_dict["tag_info"].append(tag_dict)
                data_dict["tag_info"] = self._squeeze_list(
                    data_dict["tag_info"]
                )
            else:
                tags = frames[self.gt_bev_3d_idx].tag_info
                data_dict["tag_info"] = {}
                for tag in self.load_tag_types:
                    data_dict["tag_info"][tag] = tags.get(tag, "")

        # object tag
        if "object_tag_info" in self.load_data_types:
            if self.num_frames_per_iter > 1:
                data_dict["object_tag_info"] = []
                for idx in range(
                    self.gt_bev_3d_idx,
                    self.gt_bev_3d_idx + self.num_frames_per_iter,
                ):
                    tag_dict = {}
                    tags = frames[idx].object_tag_info
                    for tag in self.load_object_tag_types:
                        tag_dict[tag] = tags.get(tag, {})
                    data_dict["object_tag_info"].append(tag_dict)
                data_dict["object_tag_info"] = self._squeeze_list(
                    data_dict["object_tag_info"]
                )
            else:
                tags = frames[self.gt_bev_3d_idx].object_tag_info
                data_dict["object_tag_info"] = {}
                for tag in self.load_object_tag_types:
                    data_dict["object_tag_info"][tag] = tags.get(tag, {})

        return data_dict


@OBJECT_REGISTRY.register
class ANCResize3DV(object):
    """Resize PIL Images to the given size and modify intrinsics.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
    interpolation: Desired interpolation. Default is 'nearest'.
    resize_depth: whether resize gt depth.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        size: Union[Sequence, Sequence[Sequence]],
        interpolation: str = "nearest",
        resize_depth: bool = True,
    ):
        self.size = size if isinstance(size[0], Sequence) else [size]
        assert interpolation in PIL_INTERP_CODES
        self.interpolation = interpolation
        self.resize_depth = resize_depth

    def _resize(self, data: Union[Image.Image, Sequence], size, interpolation):
        if isinstance(data, Sequence):
            assert len(data) == len(size)
            return [
                self._resize(data_i, size, interpolation)
                for data_i, size in zip(data, size)
            ]
        else:
            return F.resize(data, size, interpolation)

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'
        if len(data["pil_imgs"][0]) != len(self.size):
            duplicate = len(data["pil_imgs"][0]) / len(self.size)
            self.size = self.size * int(duplicate)

        data["pil_imgs"] = [
            self._resize(
                pil_img, self.size, PIL_INTERP_CODES[self.interpolation]
            )
            for pil_img in data["pil_imgs"]
        ]
        # in-place modify on data['size'] will affect self.size,
        # use copy.deepcopy fix it
        data["size"] = copy.deepcopy(self.size)

        if "gt_seg" in data:
            data["gt_seg"] = [
                self._resize(
                    gt_seg, self.size, PIL_INTERP_CODES[self.interpolation]
                )
                for gt_seg in data["gt_seg"]
            ]

        if "gt_depth" in data and self.resize_depth:
            data["gt_depth"] = self._resize(
                data["gt_depth"], self.size, PIL_INTERP_CODES["nearest"]
            )

        if "color_imgs" in data:
            data["color_imgs"] = [
                self._resize(
                    color_img, self.size, PIL_INTERP_CODES[self.interpolation]
                )
                for color_img in data["color_imgs"]
            ]

        if "front_mask" in data:
            data["front_mask"] = self._resize(
                data["front_mask"], self.size[0], PIL_INTERP_CODES["nearest"]
            )

        if "obj_mask" in data:
            data["obj_mask"] = self._resize(
                data["obj_mask"], self.size[0], PIL_INTERP_CODES["nearest"]
            )

        if "intrinsics" in data:
            # scale intrinsics matrix
            data["intrinsics"][0, :] *= self.size[0][1]
            data["intrinsics"][1, :] *= self.size[0][0]
        if "gt_multi_view" in data and data["gt_multi_view"]:
            for idx, multi_view_info in enumerate(data["gt_multi_view"]):
                if multi_view_info:
                    ignore_mask_2d = multi_view_info["meta"]["ignore_mask"][
                        "ignore_mask_2d"
                    ]
                    ignore_mask_2d = self._resize(
                        ignore_mask_2d,
                        self.size[idx],
                        PIL_INTERP_CODES["nearest"],
                    )
                    multi_view_info["meta"]["ignore_mask"][
                        "ignore_mask_2d"
                    ] = ignore_mask_2d

        return data

    def __repr__(self):
        return "Resize3DV"


@OBJECT_REGISTRY.register
class ANCBevSegTargetGenerator(object):
    """Generate gt_bev_seg from gt_bev_static_anno to calculate bevseg loss\
    or calculate MIOU metric.

    Args:
        vcs_range: vcs range.(order is (bottom,right,top,left))
        line_width: Thickness of bevseg midline elements.
                    such as roadedges,solid_lanes
        bev_size:bev_size
    """

    def __init__(
        self,
        vcs_range: Sequence[float],
        line_width: int = 2,
        bev_size: Sequence[int] = (512, 512),
        gt_name: str = "gt_bev_seg",
    ):
        self.bev_size = bev_size
        self.labels = {
            "roadedges": (1, 1, 1),
            "roadarrows": (2, 2, 2),
            "solid_lanes": (3, 3, 3),
            "stoplines": (4, 4, 4),
            "crosswalks": (5, 5, 5),
            "ignores": (255, 255, 255),
            "outrange_ignores": (255, 255, 255),
        }
        self.vcs2bev = get_vcs2bev_img_mat(vcs_range, bev_size)
        self.line_width = line_width
        self.gt_name = gt_name

    def __call__(self, data):
        bev_seg_dot = data["gt_bev_static_anno"]
        if "gt_bev_discrete_raw" in data:
            det_info = data["gt_bev_discrete_raw"]
            if "det" in det_info:
                if "stoplines" in det_info["det"]:
                    bev_seg_dot["stoplines"] = det_info["det"]["stoplines"]
                if "arrows" in det_info["det"]:
                    bev_seg_dot["roadarrows"] = det_info["det"]["arrows"]
                if "crosswalks" in det_info["det"]:
                    bev_seg_dot["crosswalks"] = det_info["det"]["crosswalks"]
                if "ignores" in det_info["det"]:
                    bev_seg_dot["ignores"] += det_info["det"]["ignores"]
        bev_height, bev_width = self.bev_size
        bev_map = np.zeros((bev_height, bev_width, 3), dtype="uint8")

        # draw bevseg img
        for (category, edges) in bev_seg_dot.items():
            if edges is not None:
                if category in [
                    "crosswalks",
                    "roadarrows",
                    "ignores",
                    "outrange_ignores",
                ]:
                    # draw plane element
                    for edge in edges:
                        if edge:
                            if "pts" not in edge:
                                continue
                            edge = edge["pts"]
                            output_list = []
                            for _current_edge in edge:
                                [x1, y1, x2, y2] = _current_edge
                                pt1 = np.squeeze(
                                    self.vcs2bev
                                    @ np.array(
                                        [x1, y1, 1.0], dtype=np.float
                                    ).reshape((3, 1))
                                )
                                pt2 = np.squeeze(
                                    self.vcs2bev
                                    @ np.array(
                                        [x2, y2, 1.0], dtype=np.float
                                    ).reshape((3, 1))
                                )
                                output_list.append(
                                    [int(pt1[0] + 0.5), int(pt1[1] + 0.5)]
                                )
                                output_list.append(
                                    [int(pt2[0] + 0.5), int(pt2[1] + 0.5)]
                                )
                            edge = np.array([output_list])
                            cv2.fillConvexPoly(
                                bev_map, edge, self.labels[category]
                            )
                elif category in ["roadedges", "solid_lanes", "stoplines"]:
                    # draw line element
                    for edge in edges:
                        if "pts" not in edge:
                            continue
                        edge = edge["pts"]
                        for _current_edge in edge:
                            [x1, y1, x2, y2] = _current_edge
                            pt1 = np.squeeze(
                                self.vcs2bev
                                @ np.array(
                                    [x1, y1, 1.0], dtype=np.float
                                ).reshape((3, 1))
                            )
                            pt2 = np.squeeze(
                                self.vcs2bev
                                @ np.array(
                                    [x2, y2, 1.0], dtype=np.float
                                ).reshape((3, 1))
                            )
                            bev_map = cv2.line(
                                bev_map,
                                (int(pt1[0] + 0.5), int(pt1[1] + 0.5)),
                                (int(pt2[0] + 0.5), int(pt2[1] + 0.5)),
                                self.labels[category],
                                self.line_width,
                            )
        data[self.gt_name] = Image.fromarray(bev_map[:, :, 0]).convert("I")
        return data

    def __repr__(self):
        return "BevSegTargetGenerator"


@OBJECT_REGISTRY.register
class ANCBevSegAnnoGenerator(object):
    """Generate gt_bev_seg_anno from gt_bev_static_anno to\
    calculate BevSegInstanceEval metric.

    Args:
        vcs_range: vcs range.(order is (bottom,right,top,left))
        max_anno_num: Maximum number of each bevseg annotation
        bev_size: bev_size.(order is (h,w))
        gt_name: return gt name.

    """

    def __init__(
        self,
        vcs_range: Sequence[float],  # (bottom, right, top, left)
        bev_size: Sequence[float],  # (Height, Width)
        max_anno_num: int = 20000,
        gt_name: str = "gt_bev_seg_anno",
    ):
        self.vcs_range = vcs_range
        self.max_anno_num = max_anno_num
        self.gt_name = gt_name

        self.x_perpixel = abs(vcs_range[2] - vcs_range[0]) / bev_size[0]
        self.x_perpixel_half = self.x_perpixel / 2

    def get_arrange(self, x1, y1, x2, y2):
        # generate lines at 0.01m intervals from endpoints
        output_list = []
        if abs(x2 - x1) < 0.01:
            output_list.append([x1, y1])
            output_list.append([x2, y2])
            return output_list
        else:
            if x1 > x2:
                x_list = np.arange(x2, x1, 0.01).tolist()
                x_list.append(x1)
            else:
                x_list = np.arange(x1, x2, 0.01).tolist()
                x_list.append(x2)
            x_list = [float(format(x, ".2f")) for x in x_list]

            x_list = [
                _item
                for _item in x_list
                if (
                    abs(_item % self.x_perpixel - self.x_perpixel_half) < 0.001
                )
            ]
            k = (y1 - y2) / (x1 - x2)
            b = y1 - x1 * k
            y_list = [k * x + b for x in x_list]
            output_list = [[x, y] for (x, y) in zip(x_list, y_list)]
            return output_list

    def convert_points(self, input_arrays):
        """Convert arrays to the set of points needed for evaluation.

        If some category less than 20,000 points in a category,
            fill it to 20,000 by (1000.0,1000.0)
        """
        out_set = set()
        for input_array in list(input_arrays):
            if "pts" not in input_array:
                continue
            input_array = input_array["pts"]
            for [x1, y1, x2, y2] in list(input_array):
                current_list = self.get_arrange(x1, y1, x2, y2)
                for current_dots in current_list:
                    current_dots[0] = float(format(current_dots[0], ".2f"))
                    current_dots[1] = float(format(current_dots[1], ".2f"))
                    if (
                        current_dots[0] < self.vcs_range[2]
                        and current_dots[0] > self.vcs_range[0]
                        and current_dots[1] < self.vcs_range[3]
                        and current_dots[1] > self.vcs_range[1]
                    ):
                        out_set.add(
                            str(current_dots[0]) + "," + str(current_dots[1])
                        )
        out = [one.split(",") for one in list(out_set)]
        out = [[float(one[0]), float(one[1])] for one in out]
        assert len(out) < self.max_anno_num
        out.extend([[1000.0, 1000.0]] * (self.max_anno_num - len(out)))
        return out

    def get_validation_anno_target(self, input_dict):
        output_dict = {}

        for category in ["roadedges", "solid_lanes"]:
            edges = input_dict[category]
            if edges is not None:
                edges_new = self.convert_points(edges)
                output_dict[category] = edges_new
        return output_dict

    def __call__(self, data):

        data[self.gt_name] = self.get_validation_anno_target(
            data["gt_bev_static_anno"]
        )
        return data

    def __repr__(self):
        return "BevSegAnnoGenerator"


@OBJECT_REGISTRY.register
class ANCCrop3DV(object):  # noqa: D205,D400
    """Crop the given list of image at specified location/output size
     and modify intrinsics/homography matrix. \

     The image can be a PIL Image or a Tensor,
     in which case it is expected to have [..., H, W] shape,
     where ... means an arbitrary number of leading dimensions.

    Args:
    heights: Height of the crop boxs in each view.
    widths: Width of the crop boxs in each view.
    top: Vertical component of the top left corner \
        of the crop box in each view. Setting None means random crop.
    left: Horizontal component of the top left corner \
        of the crop box in each view. Setting None means random crop.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        height: Union[Sequence, int],
        width: Union[Sequence, int],
        top: Union[Sequence, int],
        left: Union[Sequence, int],
    ):

        self.top = top if isinstance(top, Sequence) else [top]
        self.left = left if isinstance(left, Sequence) else [left]
        self.height = height if isinstance(height, Sequence) else [height]
        self.width = width if isinstance(width, Sequence) else [width]

    def _crop(
        self, data: Union[Image.Image, Sequence], top, left, height, width
    ):
        if isinstance(data, Sequence):
            assert (
                len(data) == len(top) == len(left) == len(height) == len(width)
            ), "please check length of data and length of argument, must be same"  # noqa
            return [
                self._crop(_, t, l, h, w)
                for _, t, l, h, w in zip(data, top, left, height, width)
            ]
        else:
            return F.crop(data, top, left, height, width)

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'

        nums = len(data["size"])
        assert (
            len(data["pil_imgs"][0])
            == len(self.top)
            == len(self.left)
            == len(self.height)
            == len(self.width)
        )

        for i in range(nums):
            assert data["size"][i][0] >= self.height[i]
            assert data["size"][i][1] >= self.width[i]

        # generate a random integer if self.top is None
        top = [
            np.random.randint(data["size"][i][0] - self.height[i] + 1)
            if self.top[i] is None
            else self.top[i]
            for i in range(nums)
        ]
        left = [
            np.random.randint(data["size"][i][1] - self.width[i] + 1)
            if self.left[i] is None
            else self.left[i]
            for i in range(nums)
        ]

        data["pil_imgs"] = [
            self._crop(pil_img, top, left, self.height, self.width)
            for pil_img in data["pil_imgs"]
        ]

        if "imgs" in data:
            data["imgs"] = [
                self._crop(pil_img, top, left, self.height, self.width)
                for pil_img in data["imgs"]
            ]

        if "gt_seg" in data:
            data["gt_seg"] = [
                self._crop(gt_seg, top, left, self.height, self.width)
                for gt_seg in data["gt_seg"]
            ]

        if "obj_mask" in data:
            data["obj_mask"] = self._crop(
                data["obj_mask"],
                top[0],
                left[0],
                self.height[0],
                self.width[0],
            )

        if "gt_depth" in data:
            for i in range(len(data["gt_depth"])):
                depth_w, depth_h = data["gt_depth"][i].size
                scale_h, scale_w = (
                    depth_h // data["size"][i][0],
                    depth_w // data["size"][i][1],
                )
                data["gt_depth"][i] = self._crop(
                    data["gt_depth"][i],
                    top[i] * scale_h,
                    left[i] * scale_w,
                    self.height[i] * scale_h,
                    self.width[i] * scale_w,
                )

        if "color_imgs" in data:
            data["color_imgs"] = [
                self._crop(color_img, top, left, self.height, self.width)
                for color_img in data["color_imgs"]
            ]

        if "front_mask" in data:
            data["front_mask"] = self._crop(
                data["front_mask"],
                top[0],
                left[0],
                self.height[0],
                self.width[0],
            )

        if "intrinsics" in data:
            # modify intrinsics matrix
            data["intrinsics"][0, 2] -= left[0]
            data["intrinsics"][1, 2] -= top[0]
        if "gt_multi_view" in data and data["gt_multi_view"]:
            for idx, multi_view_info in enumerate(data["gt_multi_view"]):
                if multi_view_info:
                    ignore_mask_2d = multi_view_info["meta"]["ignore_mask"][
                        "ignore_mask_2d"
                    ]
                    ignore_mask_2d = self._crop(
                        ignore_mask_2d,
                        top[idx],
                        left[idx],
                        self.height[idx],
                        self.width[idx],
                    )
                    multi_view_info["meta"]["ignore_mask"][
                        "ignore_mask_2d"
                    ] = ignore_mask_2d

        return data

    def __repr__(self):
        return "Crop3DV"


@OBJECT_REGISTRY.register
class ANCPad3DV(object):  # noqa: D205,D400
    """Pad the given list of image and modify intrinsics/homography matrix.

     The image can be a PIL Image or a Tensor,
     in which case it is expected to have [..., H, W] shape,
     where ... means an arbitrary number of leading dimensions.

    Args:
        paddings: Padding on each border of multi view data.
            NOTE: usage of padding is same as F.pad().
            NOTE: make sure the length of paddings must be equal to length of imgs.  # noqa
        img_fill: Pixel fill value for image.
        gt_seg_fill : Pixel fill value for gt seg.
        gt_depth_fill : Pixel fill value for gt depth.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        paddings: Union[int, Sequence],
        img_fill: int = 0,
        gt_seg_fill: int = -1,
        gt_depth_fill: int = 0,
    ):
        self.paddings = paddings
        self.img_fill = img_fill
        self.gt_seg_fill = gt_seg_fill
        self.gt_depth_fill = gt_depth_fill

    def _pad(
        self,
        data: Union[Image.Image, Sequence],
        padding,
        fill,
    ):
        if isinstance(data, Sequence):
            assert len(data) == len(padding)
            return [self._pad(_, p, fill) for _, p in zip(data, padding)]
        else:
            return F.pad(data, padding, fill)

    def __call__(self, data: Mapping):

        assert len(data["size"]) == len(self.paddings)

        assert "pil_imgs" in data, 'input data must has "pil_imgs"'

        data["pil_imgs"] = [
            self._pad(pil_img, self.paddings, fill=self.img_fill)
            for pil_img in data["pil_imgs"]
        ]

        if "gt_seg" in data:
            data["gt_seg"] = [
                self._pad(gt_seg, self.paddings, fill=self.img_fill)
                for gt_seg in data["gt_seg"]
            ]

        if "obj_mask" in data:
            data["obj_mask"] = self._pad(
                data["obj_mask"], self.paddings[0], fill=0
            )  # noqa

        if "gt_depth" in data:
            data["gt_depth"] = self._pad(
                data["gt_depth"], self.paddings, fill=self.gt_depth_fill
            )

        if "color_imgs" in data:
            data["color_imgs"] = [
                self._pad(color_img, self.paddings, self.img_fill)
                for color_img in data["color_imgs"]
            ]

        if "front_mask" in data:
            data["front_mask"] = self._pad(
                data["front_mask"], self.paddings[0], fill=0
            )  # noqa

        if "intrinsics" in data:
            # modify intrinsics matrix
            front_padding = self.paddings[0]
            left = (
                front_padding[0]
                if isinstance(front_padding, Sequence)
                else front_padding
            )
            top = (
                front_padding[1]
                if isinstance(front_padding, Sequence)
                else front_padding
            )

            data["intrinsics"][0, 2] += top
            data["intrinsics"][1, 2] += left

        return data

    def __repr__(self):
        return "Pad3DV"


@OBJECT_REGISTRY.register
class ANCClassRemap(object):
    """Remap segmentation gt with specific remap dict.

    Args:
        remap_dict: remap dict.

    """

    def __init__(self, remap_dict: Optional[Mapping] = None):
        self.remap_dict = remap_dict if remap_dict else {}

    @staticmethod
    def remap(data, remap_dict):
        if isinstance(data, Sequence):
            return [ANCClassRemap.remap(_, remap_dict) for _ in data]
        else:
            return np.vectorize(remap_dict.get)(np.array(data))

    def __call__(self, data):

        for key_name, each_map in self.remap_dict.items():
            assert key_name in data
            data[key_name] = ANCClassRemap.remap(data[key_name], each_map)

        return data

    def __repr__(self):
        return "ClassRemap"


@OBJECT_REGISTRY.register
class ANCToTensor3DV(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to torch.tensor.

    Args:
    with_color_imgs: whether return original color imags.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        with_color_imgs: bool = True,
    ):
        self.with_color_imgs = with_color_imgs
        self.pil2tensor = torchvision.transforms.PILToTensor()

    def _array_to_tensor(self, data: np.ndarray, normalize=True):
        """Convert a ``numpy.ndarray`` to torch.tensor."""
        default_float_dtype = torch.get_default_dtype()
        if data.ndim == 2:
            data = data[:, :, None]
        data = torch.from_numpy(data.transpose((2, 0, 1))).contiguous()
        if normalize:
            assert isinstance(data, torch.ByteTensor)
            data.to(dtype=default_float_dtype).div(255)
        return data

    def _pil_to_tensor(self, data: Image.Image, normalize=True):
        """Convert a ``PIL Image`` to torch.tensor."""
        default_float_dtype = torch.get_default_dtype()

        img = self.pil2tensor(data)
        if normalize:
            assert isinstance(img, torch.ByteTensor)
            return img.to(dtype=default_float_dtype).div(255)
        else:
            return img

    def _map_to_tensor(
        self, data: Union[Image.Image, Sequence], normalize=True
    ):
        # convert map to tensor
        # step1. data = data.transpose(2, 0, 1)
        # step2. if normalize, normalize data to [0,1]
        if isinstance(data, (list, tuple)):
            return [self._map_to_tensor(_, normalize) for _ in data]
        elif isinstance(data, dict):
            return {
                key: self._map_to_tensor(data[key], normalize)
                for key in data.keys()
            }
        elif isinstance(data, np.ndarray):
            return self._array_to_tensor(data, normalize)
        elif isinstance(data, Image.Image):
            return self._pil_to_tensor(data, normalize)
        else:
            raise TypeError

    def _undistort_coord(self, inputsize, intrinsic, distor_coeff):
        # scale intrinsics matrix
        intrinsic = intrinsic.copy()
        # intrinsic[0, :] *= inputsize[1]     #width
        # intrinsic[1, :] *= inputsize[0]     # high

        mapx, mapy = cv2.initUndistortRectifyMap(
            intrinsic,
            distor_coeff,
            None,
            intrinsic,
            (inputsize[1], inputsize[0]),
            cv2.CV_32FC1,
        )

        mapx = (mapx - inputsize[1] * 0.5) / (inputsize[1] * 0.5)
        mapy = (mapy - inputsize[0] * 0.5) / (inputsize[0] * 0.5)

        img_ud_coord = np.concatenate(
            [mapx[:, :, np.newaxis], mapy[:, :, np.newaxis]], axis=2
        )
        return img_ud_coord

    @staticmethod
    def _meta_to_tensor(data: Union[str, np.ndarray, Sequence]):
        if isinstance(data, (list, tuple)):
            return [ANCToTensor3DV._meta_to_tensor(_) for _ in data]
        elif isinstance(data, dict):
            return {
                key: ANCToTensor3DV._meta_to_tensor(data[key])
                for key in data.keys()
            }
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, (str, int, float, torch.Tensor)):
            return data
        else:
            raise TypeError

    def _del_keys(self, data: dict, del_keys: Union[str, Sequence[str]]):
        del_keys = _as_list(del_keys)
        for key in del_keys:
            data.pop(key, None)

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'
        if "color_imgs" in data:
            data["color_imgs"] = self._map_to_tensor(data["color_imgs"])
            # generate undistorted image to calcu project loss that are normalized to [0,1] # noqa
            data["ud_coord"] = self._undistort_coord(
                data["color_imgs"][0][0].shape[1:3],
                data["intrinsics"],
                data["distortcoef"],
            )
            data["ud_coord"] = torch.from_numpy(data["ud_coord"])
            # generate undistorted coordinates for depth and resflow
        else:
            data["color_imgs"] = self._map_to_tensor(data["pil_imgs"])

        data["imgs"] = self._map_to_tensor(data["pil_imgs"], normalize=False)
        if "gt_seg" in data:
            data["gt_seg"] = self._map_to_tensor(
                data["gt_seg"], normalize=False
            )

        if "obj_mask" in data:
            data["obj_mask"] = self._map_to_tensor(
                data["obj_mask"], normalize=False
            )

        if "gt_depth" in data:
            data["gt_depth"] = self._map_to_tensor(
                data["gt_depth"], normalize=False
            )

        if "front_mask" in data:
            data["front_mask"] = self._map_to_tensor(
                data["front_mask"], normalize=False
            )

        if "intrinsics" in data:
            data["intrinsics"] = torch.from_numpy(data["intrinsics"])

        if "distortcoef" in data:
            data["distortcoef"] = torch.from_numpy(data["distortcoef"])

        if "local2vcs" in data:
            data["local2vcs"] = torch.from_numpy(data["local2vcs"])

        if "local2cam" in data:
            data["local2cam"] = torch.from_numpy(data["local2cam"])

        if "K" in data:
            data["K"] = torch.from_numpy(data["K"])

        if "dist" in data:
            data["dist"] = torch.from_numpy(data["dist"])

        bevseg_gt_names = ["gt_bev_seg", "gt_bev_seg_small"]
        bevseg_del_keys = ["gt_bev_static_anno", "gt_bev_discrete_raw"]
        for bevseg_gt_name in bevseg_gt_names:
            if bevseg_gt_name in data:

                self._del_keys(data, bevseg_del_keys)
                data[bevseg_gt_name] = self._map_to_tensor(
                    data[bevseg_gt_name], normalize=False
                )

        homography_temporal_names = [
            "homography_temporal",
            "homography_temporal_small",
        ]
        for homography_temporal_name in homography_temporal_names:
            if homography_temporal_name in data:
                data[homography_temporal_name] = torch.from_numpy(
                    data[homography_temporal_name]
                )

        if "occlusion" in data:
            data["occlusion"] = self._map_to_tensor(
                data["occlusion"], normalize=False
            )

        elevation_gt_names = ["gt_bev_elevation", "gt_bev_elevation_small"]
        for elevation_gt_name in elevation_gt_names:
            if elevation_gt_name in data:
                self._del_keys(data, "gt_bev_elevation_raw")
                data[elevation_gt_name] = self._meta_to_tensor(
                    data[elevation_gt_name]
                )

        ele_vismask_gt_names = [
            "gt_bev_elevation_vismask",
            "gt_bev_elevation_vismask_small",
        ]
        for ele_vismask_gt_name in ele_vismask_gt_names:
            if ele_vismask_gt_name in data:
                self._del_keys(data, "gt_bev_elevation_vismask_raw")
                data[ele_vismask_gt_name] = self._meta_to_tensor(
                    data[ele_vismask_gt_name]
                )

        disc_res_names = [
            "bev_discrete_obj",
            "bev_discrete_obj_small",
            "bev_arrow",
            "bev_arrow_small",
            "bev_roadmarking",
            "bev_roadmarking_small",
            "bev_parkinglock",
            "bev_parkinglock_small",
            "bev_cementcolumn",
            "bev_cementcolumn_small",
        ]
        disc_del_keys = ["bev_occlusion_mask", "gt_bev_discrete_raw"]
        for disc_res_name in disc_res_names:
            if f"gt_{disc_res_name}" in data:
                self._del_keys(data, disc_del_keys)
                data[f"gt_{disc_res_name}"] = self._map_to_tensor(
                    data[f"gt_{disc_res_name}"], normalize=False
                )
                assert f"annos_{disc_res_name}" in data
                data[f"annos_{disc_res_name}"] = self._meta_to_tensor(
                    data[f"annos_{disc_res_name}"]
                )

        parking_res_keys = [
            "bev_psd_obj",
            "bev_psd_obj_small",
            "bev_parkingrod_obj",
            "bev_parkingrod_obj_small",
        ]
        for parking_res_key in parking_res_keys:
            if f"gt_{parking_res_key}" in data:
                data[f"gt_{parking_res_key}"] = self._map_to_tensor(
                    data[f"gt_{parking_res_key}"], normalize=False
                )
                assert f"annos_{parking_res_key}" in data
                data[f"annos_{parking_res_key}"] = self._meta_to_tensor(
                    data[f"annos_{parking_res_key}"]
                )
                if "gt_bev_parking_obj" in data:
                    data.pop("gt_bev_parking_obj")

        if "gt_bev_seg_anno" in data:
            _gt_bev_seg_anno = {}
            for k in data["gt_bev_seg_anno"].keys():
                if data["gt_bev_seg_anno"][k] is not None:
                    _gt_bev_seg_anno[k] = torch.from_numpy(
                        np.array(data["gt_bev_seg_anno"][k])
                    )
            if "occlusion" in data:
                # occlusion need to be B*H*W in bevseg instance eval
                _gt_bev_seg_anno["occlusion"] = copy.deepcopy(
                    torch.squeeze(data["occlusion"])
                )
            data["gt_bev_seg_anno"] = _gt_bev_seg_anno

        freespace_gt_names = [
            "gt_bev_freespace",
            "gt_bev_freespace_small",
            "gt_bev_ele_freespace",
            "gt_bev_ele_freespace_small",
        ]
        for freespace_gt_name in freespace_gt_names:
            if freespace_gt_name in data:
                self._del_keys(data, "gt_bev_freespace_raw")
                data[freespace_gt_name] = self._meta_to_tensor(
                    data[freespace_gt_name]
                )

        om_gt_names = ["om_target", "om_target_small"]
        om_del_keys = ["gt_online_mapping", "pack_dir", "img_paths"]
        for om_gt_name in om_gt_names:
            if om_gt_name in data:
                self._del_keys(data, om_del_keys)
                for group, group_gt in data[om_gt_name].items():
                    for k, v in group_gt.items():
                        data[om_gt_name][group][k] = torch.from_numpy(v)

        if "e2e_dynamic_det_targets" in data:
            for k in data["e2e_dynamic_det_targets"].keys():
                data["e2e_dynamic_det_targets"][k] = self._to_tensor(
                    data["e2e_dynamic_det_targets"][k], normalize=False
                )
        crosspoint_target_names = [
            "crosspoint_target",
            "crosspoint_target_small",
        ]
        crosspoint_del_keys = [
            "bev_occlusion_mask",
            "gt_bev_crosspoint",
            "pack_dir",
            "img_paths",
        ]
        for crosspoint_target_name in crosspoint_target_names:
            if crosspoint_target_name in data:

                self._del_keys(data, crosspoint_del_keys)
                for group, group_gt in data[crosspoint_target_name].items():
                    for k, v in group_gt.items():
                        data[crosspoint_target_name][group][
                            k
                        ] = torch.from_numpy(v)

        if "timestamp" in data:
            data["timestamp"] = ANCToTensor3DV._meta_to_tensor(
                data["timestamp"]
            )

        if "front_bool_idx" in data:
            data["front_bool_idx"] = torch.from_numpy(data["front_bool_idx"])

        if "axisangle" in data:
            data["axisangle"] = torch.from_numpy(data["axisangle"])
            data["translation"] = torch.from_numpy(data["translation"])

        if "meta_info" in data:
            for mate_name, meta_value in data["meta_info"].items():
                data["meta_info"][mate_name] = ANCToTensor3DV._meta_to_tensor(
                    meta_value
                )
        if "meta_info_small" in data:
            for mate_name, meta_value in data["meta_info_small"].items():
                data["meta_info_small"][
                    mate_name
                ] = ANCToTensor3DV._meta_to_tensor(meta_value)
        if "meta_info_high_sp" in data:
            for mate_name, meta_value in data["meta_info_high_sp"].items():
                data["meta_info_high_sp"][
                    mate_name
                ] = ANCToTensor3DV._meta_to_tensor(meta_value)
        if "gt_multi_view" in data and data["gt_multi_view"]:
            for multi_view_info in data["gt_multi_view"]:
                if multi_view_info:
                    ignore_mask_2d = multi_view_info["meta"]["ignore_mask"][
                        "ignore_mask_2d"
                    ]
                    ignore_mask_2d = self._map_to_tensor(ignore_mask_2d)
                    ignore_mask_2d = ignore_mask_2d.squeeze(0)
                    ignore_mask_2d = ignore_mask_2d > 0
                    multi_view_info["meta"]["ignore_mask"][
                        "ignore_mask_2d"
                    ] = ignore_mask_2d

        if not self.with_color_imgs:
            data.pop("color_imgs")
        data.pop("pil_imgs")
        return data

    def __repr__(self):
        return "ToTensor3DV"


@OBJECT_REGISTRY.register
class ANCNormalize3DV(object):  # noqa: D205,D400
    """Normalize a tensor image with mean and standard deviation,and
       scale gt depth with a specified coefficient if need.

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of std for each channel.
        depth_scale: coefficient to scale depth.
    """

    def __init__(
        self,
        mean: Union[float, Sequence[float]],
        std: Union[float, Sequence[float]],
        depth_scale: float = 1.0,
    ):
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale
        self.norm_img_names = ["img", "side_img", "round_img", "narrow_img"]

    def _normalize(self, data: Union[Image.Image, Sequence], mean, std):
        if isinstance(data, Sequence):
            return [self._normalize(_, mean, std) for _ in data]
        else:
            return F.normalize(data, mean, std)

    def __call__(self, data: Mapping):
        for key in self.norm_img_names:
            if key in data:
                data[key] = self._normalize(data[key], self.mean, self.std)
        if "gt_depth" in data:
            data["gt_depth"] = [
                depth * self.depth_scale for depth in data["gt_depth"]
            ]
        return data

    def __repr__(self):
        return "Normalize3DV"


@OBJECT_REGISTRY.register
class ANCMotionFlowGenerator(object):
    """Generate gt bev motion flow map from 3d bbox annotation of two frames.

    Args:
        bev_size : bev size, order is (h,w).
        vcs_range : vcs range.
            order is (bottom,right,top,left).
        ego_loc : location of ego car in bev pixel
            coordinate. order is (h,w).
    """

    def __init__(
        self,
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        ego_loc: Sequence[int],
    ):
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x
        self.ego_loc = ego_loc

    def pad_to_square(self, array, pad_value=0):
        """Pad the array to square array.

        Args:
        kernel (ndarray): the input numpy array
        pad_value (int): value used for pad
        """
        h, w = array.shape[0:2]
        pad_dim = max(h, w) + w // 2
        before_1 = after_1 = int((pad_dim - h) // 2)
        before_2 = after_2 = int((pad_dim - w) // 2)
        padded_array = np.pad(
            array,
            ((before_1, after_1), (before_2, after_2), (0, 0)),  # noqa
            "constant",
            constant_values=pad_value,
        )
        return padded_array

    def rotate(
        self,
        image: np.ndarray,
        angle: float,
        center: Optional[tuple] = None,
        scale: float = 1.0,
    ):
        """Rotate the input img or array.

        Args:
        image: input iamge
        angle: rotate angle, degree
        center: center to rotate
        scale: transform scale

        """
        (h, w) = image.shape[:2]
        max_dim = max((h, w))
        output_dim = (max_dim, max_dim)

        if center is None:
            center = tuple((np.array(image.shape[:2]) // 2).astype("float"))

        # Perform the rotation
        # NOTE: The angle is degree
        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(image, M, output_dim, flags=cv2.INTER_NEAREST)

        return rotated

    def __call__(self, data: dict):
        """Generate gt bev motion flow map.

        Args:
            data : The dict contains at leaset annotations

        """
        assert "gt_bev_motion_flow" in data
        annotations = data.pop("gt_bev_motion_flow")

        bev_motion_flow = np.zeros((*self.bev_size, 3), dtype=np.float32)

        cur_frame, pre_frame = annotations
        cur_ego_location = cur_frame.pop("ego_location")
        cur_ego_yaw = cur_frame.pop("ego_yaw")

        cur_seq_center = SeqCenter(
            pos_x=cur_ego_location[0],
            pos_y=cur_ego_location[1],
            yaw=cur_ego_yaw,
        )

        obs_d_yaw = []
        pre_obs_location = []
        obs_w_bev, obs_h_bev, obs_location, obs_yaw_vcs = [], [], [], []
        for track_id, obs_data in cur_frame.items():
            if track_id not in pre_frame:
                continue
            obs_d_yaw.append(
                obs_data["obs_yaw"] - pre_frame[track_id]["obs_yaw"]
            )

            obs_w_bev.append(
                obs_data["obs_dimension"][0] / self.m_perpixel[1]
            )  # pixel
            obs_h_bev.append(
                obs_data["obs_dimension"][1] / self.m_perpixel[0]
            )  # pixel

            obs_yaw_vcs.append(obs_data["obs_yaw"] - cur_ego_yaw)
            obs_location.append(np.array(obs_data["obs_location"]))
            pre_obs_location.append(
                np.array(pre_frame[track_id]["obs_location"])
            )

        obs_num = len(obs_location)
        obs_vcs_coord = TdtCoordHelper.global_phy_to_local_phy(
            np.stack(obs_location + pre_obs_location),
            cur_seq_center,
        )

        obs_vcs_ct = obs_vcs_coord[:obs_num]
        pre_obs_vcs_ct = obs_vcs_coord[obs_num:]

        obs_dxy_vcs = obs_vcs_ct - pre_obs_vcs_ct
        obs_dx_vcs, obs_dy_vcs = obs_dxy_vcs[:, 0], obs_dxy_vcs[:, 1]

        # convert vcs to bev
        obs_bev_ct = (
            -obs_vcs_ct / np.array(self.m_perpixel) + np.array(self.ego_loc)
        )[:, [1, 0]]

        for bev_ct, w, h, yaw, d_x, d_y, d_yaw, in zip(
            obs_bev_ct,
            obs_w_bev,
            obs_h_bev,
            obs_yaw_vcs,
            obs_dx_vcs,
            obs_dy_vcs,
            obs_d_yaw,
        ):
            if (
                0 <= bev_ct[0] < self.bev_size[1]
                and 0 <= bev_ct[1] < self.bev_size[0]
            ):
                reg_map = get_reg_map((int(w), int(h)), (d_x, d_y, d_yaw))
                reg_map = self.pad_to_square(reg_map, 0)
                # rotate the insert hm
                reg_map = self.rotate(reg_map, np.rad2deg(yaw))
                draw_heatmap(bev_motion_flow, reg_map, bev_ct, op="overwrite")
        data["gt_bev_motion_flow"] = bev_motion_flow
        return data

    def __repr__(self):
        return "MotionFlowGenerator"


@OBJECT_REGISTRY.register
class ANCBev3dTargetGenerator(object):
    """Generate gound truth labels for bev_3d.

    Args:
        num_classes: Number of classes
        bev_size : bev size.(order is (h,w))
        vcs_range : vcs range.(order is
            (bottom,right,top,left))
        cls_hm_kernel : the gaussian kernel size of bev
            heatmap for each category.(e.g. {cls1: 3})
        category2id_map : A mapping from raw category (str)
            to training category_id which starts from 0
        cls_dimension : the average dimension of each category,
            each column stands for a class.
        max_objs : Maximum number of objects used in the training and
            inference. This number should be large enough
        enable_ignore : Whether to use ignore_mask. If true, the
            ignored object will be drew in the bev3d_ignore_mask heatmap
            and not participate in the loss calculation, else the object
            will be regarded as the background.
        filter_vcs_range : filter vcs range.(order is
            (bottom,right,top,left)), if setting, the gt annotations will
            only contains in filter range.
        ego_ignore_range : filter gt that overlap with
            ego car, order is (bottom ,right, top, left).
        use_category_decouple : If True, foreground/background
            classification(1-channel) with category classification(c-channel)
            is used to distinguish the object category.

        roi_label_seq : the list of roi_labels
            which can replace label.
        enable_roi_label_seq : the list of labels
            which choose roi_label to replace.
        use_occlusion_attribute : Whether to generate occlusion
            attributes. If true, the target will generate an occlusion
            attribute heatmap.
        occlusion_attribute_seq : the list of occlsuion labels.
            Sort according to the degree of occlusion from small to large.
        occlusion_ignore_id : Default occlusion attributes ignore id.
        category_class_weight : Inter-class weight mapping.
        length_limitation : Mapping from cls_id to interval, which
            limits the box3d's length. For one bbxo3d, its length(L) should
            statisfy: limitation[0] < L < limitation[1].
        bigobj_hm_kernel_cfg : Big object kernel size setting,
            have tow keys, respectively, "length_thresh" and "kernel_size".
            If one box3d's length exceed ${length_thresh}, the size of
            gaussian2D kernel will be set as ${kernel_size}.
        use_psc_rot: whether use psc to encode the heading angle.
        N_steps_PSC_rot: nums of steps to decode the heading angle by PSC.
            Refer to https://horizonrobotics.feishu.cn/docx/EwYEdysIBodB69xn6Mdc7HkSnif. # noqa
        use_dual_freq: wether to use dual frequency for heading angle, True not implemented yet.
        background_reweight_cfg: Config of reweight strategy for background
            3D bbox, for exmaple, cyclist bbox in vehicle task or tricycle
            bbox in vru task.
        gt_name: return gt name of data, egs: data[gt_name] = gt_bev_3d
        anno_name: return anno name of data. egs: data[anno_name] = anno_bev_3d
        roi_background_weight_cfg: Config of roi background weight finetune,
            e.g. {"roi_range": [bottom, right, top, left], "weight": 1.5},
            `roi_range` is sub-region in vcs coordinate,
            `weight` is background weight factor in roi_range'.
        bigobj_length_thresh_cfg : Big object length threshold setting,
            contains length thresholds (m) for each category to be defined as big vehicle.
            Currently used in rot loss reweight,
            if one box3d's length exceed ${length_thresh},
            its rot weight will be multipled by its aspect ratio,
            which is defined as (length/width + width/length)/2.
    """

    def __init__(
        self,
        num_classes: int,
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        cls_hm_kernel: Mapping,
        category2id_map: Mapping,
        cls_dimension: np.ndarray = None,
        max_objs: int = 100,
        enable_ignore: bool = True,
        filter_vcs_range: Sequence[float] = None,
        ego_ignore_range: Sequence[float] = None,
        use_category_decouple: bool = False,
        ignore_obj_cls: bool = False,
        roi_label_seq: Sequence[str] = (),
        enable_roi_label_seq: Sequence[str] = (),
        use_occlusion_attribute: bool = False,
        occlusion_attribute_seq: Sequence[str] = (),
        occlusion_ignore_id: float = -99.0,
        category_class_weight: Optional[Mapping] = None,
        length_limitation: Optional[Mapping] = None,
        bigobj_hm_kernel_cfg: Optional[Mapping] = None,
        use_psc_rot: bool = False,
        N_steps_PSC_rot: int = None,
        use_dual_freq: bool = False,
        background_reweight_cfg: Optional[Mapping] = None,
        gt_name: str = "gt_bev_3d",
        anno_name: str = "annos_bev_3d",
        roi_background_weight_cfg: Optional[Mapping] = None,
        roi_weight_cfg: Optional[Mapping] = None,
        bigobj_length_thresh_cfg: Optional[Mapping] = None,
    ):
        self.num_classes = num_classes
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.max_objs = max_objs
        self.cls_dimension = cls_dimension
        self.category2id_map = category2id_map
        self.enable_ignore = enable_ignore
        self.filter_vcs_range = filter_vcs_range
        self.ego_ignore_range = ego_ignore_range
        self.use_category_decouple = use_category_decouple
        self.roi_label_seq = roi_label_seq
        self.ignore_obj_cls = ignore_obj_cls
        self.enable_roi_label_seq = enable_roi_label_seq
        self.use_occlusion_attribute = use_occlusion_attribute
        self.occlusion_attribute_seq = occlusion_attribute_seq
        self.occlusion_ignore_id = occlusion_ignore_id
        self.category_class_weight = category_class_weight
        self.length_limitation = length_limitation
        self.bigobj_hm_kernel_cfg = bigobj_hm_kernel_cfg
        self.gt_name = gt_name
        self.anno_name = anno_name
        self.bigobj_length_thresh_cfg = bigobj_length_thresh_cfg

        self.m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x

        self.cls2kernel = {}
        for category, kernel_size in cls_hm_kernel.items():
            self.cls2kernel[category] = np.array(
                [kernel_size, kernel_size], dtype=np.float32
            )
        if self.category_class_weight is None:
            self.category_class_weight = {}

        if self.length_limitation is None:
            self.length_limitation = {}

        self.use_psc_rot = use_psc_rot
        self.N_steps_PSC_rot = N_steps_PSC_rot
        self.use_dual_freq = use_dual_freq
        if self.use_psc_rot and self.use_dual_freq:
            raise NotImplementedError
        elif self.use_psc_rot:
            assert self.N_steps_PSC_rot >= 3
            self.rot_encode_size = self.N_steps_PSC_rot
        else:
            self.rot_encode_size = 2  # cos, sin

        if background_reweight_cfg is None:
            background_reweight_cfg = {}
        self.background_reweight_cfg = background_reweight_cfg

        if roi_background_weight_cfg is None:
            roi_background_weight_cfg = {}
        self.roi_background_weight_cfg = roi_background_weight_cfg
        self.roi_weight_cfg = roi_weight_cfg

    def get_roi_xyxy(self, roi_range):
        roi_lt_x = int(
            ((self.vcs_range[3] - roi_range[3]) / self.m_perpixel[1])
        )
        roi_lt_y = int(
            ((self.vcs_range[2] - roi_range[2]) / self.m_perpixel[0])
        )
        roi_rb_x = int(
            ((self.vcs_range[3] - roi_range[1]) / self.m_perpixel[1])
        )
        roi_rb_y = int(
            ((self.vcs_range[2] - roi_range[0]) / self.m_perpixel[0])
        )
        roi_lt_x = max(roi_lt_x, 0)
        roi_lt_y = max(roi_lt_y, 0)

        return roi_lt_x, roi_lt_y, roi_rb_x, roi_rb_y

    def finetune_roi_background_weight(
        self, roi_weight_hm: np.ndarray, bev3d_weight_hm: np.ndarray
    ):
        """Finetune the background weight in roi range.

        Args:
            roi_weight_hm: Result of background weight finetuning.
            bev3d_weight_hm: Weight map where value == 0 is background.
        """
        roi_range = self.roi_background_weight_cfg["roi_range"]
        weight = self.roi_background_weight_cfg["weight"]

        roi_lt_x, roi_lt_y, roi_rb_x, roi_rb_y = self.get_roi_xyxy(roi_range)

        bev3d_weight_hm_roi = bev3d_weight_hm[
            roi_lt_y:roi_rb_y, roi_lt_x:roi_rb_x
        ]
        if bev3d_weight_hm_roi.shape == bev3d_weight_hm.shape:
            warning("roi_range is same as vcs_range....")
        roi_weight_tmp = np.ones_like(bev3d_weight_hm_roi)
        roi_weight_tmp[bev3d_weight_hm_roi == 0] = weight
        roi_weight_hm[roi_lt_y:roi_rb_y, roi_lt_x:roi_rb_x] = roi_weight_tmp
        return roi_weight_hm

    def finetune_roi_weight(
        self,
        roi_weight_hm: np.ndarray,
        bev3d_length_map: np.ndarray,
    ):
        """Finetune the weight in roi range.

        Args:
            roi_weight_hm: Result of roi weight finetuning.
        """
        roi_range = self.roi_weight_cfg["roi_range"]
        weight = self.roi_weight_cfg["weight"]

        roi_lt_x, roi_lt_y, roi_rb_x, roi_rb_y = self.get_roi_xyxy(roi_range)

        roi_weight_hm[roi_lt_y:roi_rb_y, roi_lt_x:roi_rb_x] = weight

        if self.bigobj_hm_kernel_cfg is not None:
            big_object_length_thresh = self.bigobj_hm_kernel_cfg[
                "length_thresh"
            ]
            roi_bigobj = np.zeros(bev3d_length_map.shape, dtype=np.bool)
            roi_bigobj[roi_lt_y:roi_rb_y, roi_lt_x:roi_rb_x] = True
            bw = self.bigobj_hm_kernel_cfg["weight"]
            bigobj_mask = np.logical_and(
                bev3d_length_map > big_object_length_thresh, roi_bigobj
            )
            roi_weight_hm[bigobj_mask] *= bw
        return roi_weight_hm

    def reweight_background(
        self,
        need_background_reweight: bool,
        label: str,
        cls_id: int,
        bev_ct_int: tuple,
        background_gaussian_hm: np.ndarray,
        background_reweight_hm: np.ndarray,
    ):
        """Generate weight map for certain background 3D bbox.

        Args:
            need_background_reweight: Expected background
                3D bbox will be reweighted when calculate loss,
                e.g. tricycle in vru task will be reweighted,
                foreground 3D bbox will be not.
            label: Category name of one 3D bbox.
            cls_id: Category id in cls task.
            bev_ct_int: Bbox3D center in bev(grid index).
            background_gaussian_hm: gaussian heatmap.
            background_reweight_hm: reweight heatmap.
        """

        if need_background_reweight:
            reweight_kernel_size = self.background_reweight_cfg.get(label)[
                "kernel"
            ]
            reweight_value = self.background_reweight_cfg.get(label)["weight"]
            if isinstance(reweight_kernel_size, (int, float)):
                reweight_kernel_size = [reweight_kernel_size] * 2
            reweight_kernel_size = np.array(
                reweight_kernel_size,
                dtype=np.float32,
            )
        else:
            reweight_kernel_size = self.cls2kernel[cls_id]
            reweight_value = 1.0

        background_reweight_insert_hm = get_gaussian2D(
            reweight_kernel_size, alpha=1
        )
        background_reweight_insert_wh = background_reweight_insert_hm.shape[
            :2
        ][::-1]
        draw_heatmap(
            background_gaussian_hm,
            background_reweight_insert_hm,
            bev_ct_int,
            [background_reweight_hm],
            [get_reg_map(background_reweight_insert_wh, reweight_value)],
        )
        return background_gaussian_hm, background_reweight_hm

    def __call__(self, data: dict):
        """Generate bev_3d labels.

        Args:
            data : The dict contains at leaset annotations

        """
        assert "gt_bev_dynamic_anno" in data
        annotations = data["gt_bev_dynamic_anno"]

        # build bev3d gt heatmaps
        # Now(20230303), bev3d_hm have three roles,
        # 1. When enable_vehicle_cls is False, the model don't
        # output vehicle category info, bev_hm(1-c) takes the
        # role on distinguishing the fore/background.
        # When enable_vehicle_cls is True, three are two schemes
        # to judge vehicle category(classification task), so:
        # 2. use_category_decouple is False, bev3d_hm(n-c, n is
        # the num of vehicle category) distinguish the
        # back/foreground and judge the vehicle category. By the way,
        # when all scores of n category are lower than threshshold,
        # it is background.
        # 3. use_category_decouple is True, the model output a new
        # tensor named ben3d_cls_hm(n-c), bev3d_hm(1-c) only
        # distinguish the fore/background, and bev3d_cls_hm judge
        # the vehicle category. we name this scheme as <category_decouple>.
        if self.use_category_decouple:
            bev3d_hm = np.zeros((*self.bev_size, 1), dtype=np.float32)
            bev3d_cls_hm = np.zeros(
                (*self.bev_size, self.num_classes), dtype=np.float32
            )
        else:
            bev3d_hm = np.zeros(
                (*self.bev_size, self.num_classes), dtype=np.float32
            )
        if self.use_occlusion_attribute:
            if data["gt_multi_view"]:
                multi_view_infos_exist_flag = True
                occlusion_multi_view = data["gt_multi_view"][
                    "occlusion_multi_view"
                ]
                occlusion_id_map = (
                    np.ones(self.bev_size, dtype=np.float32)
                    * self.occlusion_ignore_id
                )
            else:
                multi_view_infos_exist_flag = False
            occlusion_num_classes = len(self.occlusion_attribute_seq)
            occlusion_cls_hm = np.zeros(
                (*self.bev_size, occlusion_num_classes), dtype=np.float32
            )
        class_id_map = np.ones(self.bev_size, dtype=np.float32) * -99
        valid_rot_reweight_mask = np.zeros(self.bev_size, dtype=np.bool)
        category_class_weight_hm = np.ones(self.bev_size, dtype=np.float32)
        roi_weight_hm = np.ones(self.bev_size, dtype=np.float32)
        bev3d_roi_weight_hm = np.ones(self.bev_size, dtype=np.float32)
        bev3d_dim = np.zeros((*self.bev_size, 3), dtype=np.float32)  # h, w, l
        bev3d_rot = np.zeros(
            (*self.bev_size, self.rot_encode_size), dtype=np.float32
        )

        bev3d_ct_offset = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # bev center offest
        bev3d_loc_z = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_weight_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_point_pos_mask = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_ignore_mask = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_ignore_obj_cls = np.array([self.ignore_obj_cls], dtype=np.bool)
        background_reweight_hm = np.ones(self.bev_size, dtype=np.float32)
        background_gaussian_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_rot_reweight_mask = np.ones((self.bev_size), dtype=np.float32)
        bev3d_length_map = np.ones((self.bev_size), dtype=np.float32)
        bev3d_aspect_ratio_map = np.ones((self.bev_size), dtype=np.float32)

        # used for eval
        vcs_loc_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        vcs_dim_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        vcs_rot_z_ = np.zeros((self.max_objs), dtype=np.float32)
        vcs_cls_ = np.zeros((self.max_objs), dtype=np.float32) - 99
        vcs_ignore_ = np.zeros((self.max_objs), dtype=np.bool_)
        vcs_visible_ = np.zeros((self.max_objs), dtype=np.float32)
        if self.use_occlusion_attribute:
            vcs_occlusion_ = np.zeros((self.max_objs), dtype=np.float32) - 99
        object_tags = {}
        if "object_tag_info" in data:
            for object_tag in data["object_tag_info"]:
                object_tags[object_tag] = np.zeros(
                    (self.max_objs), dtype=np.bool_
                )

        # In our train/val data, tow bbox3ds having different category
        # maybe overlap heavily. This will affect the training process,
        # so we filter the overlapped bbox3d.
        # {overlap_thresh} is a hyper-param, for:
        # bbox3d1(x1, y1, z1), bbox3d2(x2, y2, z2),
        # when [abs(x1 - x2) + abs(y1 - y2)] <  {overlap_thresh},
        # they will be set as ignore, not involved in training.
        overlap_thresh = 0.1
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                loci = np.array(annotations[i]["location"][:2])
                locj = np.array(annotations[j]["location"][:2])
                loc_diff = np.abs(loci - locj).sum()
                if loc_diff <= overlap_thresh:
                    annotations[i]["ignore"] = True
                    annotations[j]["ignore"] = True

        count_id = 0
        for ann_idx, anno in enumerate(annotations):  # noqa [B007]
            if count_id > (self.max_objs):
                break
            label = anno["label"]
            # 2D detailed classification label in data from
            # 4D GT link production
            need_background_reweight = False
            if (
                label.lower() in self.background_reweight_cfg
                and not anno["ignore"]
            ):
                need_background_reweight = True

            if (
                self.enable_roi_label_seq
                and label in self.enable_roi_label_seq
                and self.category2id_map.get(label, -99) > -99
            ):
                roi_label = anno.get("roi_2d_type_label", None)
                if roi_label is not None:
                    if roi_label in self.roi_label_seq:
                        label = roi_label
                    else:
                        anno["ignore"] = True
            # The classification of occlusion attributes are obtained
            # from multi view lmdb, while annos are recorded in bev3d lmdb.
            # So use uid to match the two.
            if self.use_occlusion_attribute and multi_view_infos_exist_flag:
                uid = anno["track_id"]
                if uid in occlusion_multi_view:
                    occlusion_id = occlusion_multi_view[uid]
                else:
                    occlusion_id = occlusion_num_classes - 1
                    anno["ignore"] = True

            cls_id = int(self.category2id_map.get(label, -99))
            if (cls_id <= -99 and not need_background_reweight) or (
                not self.enable_ignore and anno.get("ignore", False)
            ):
                continue
            vcs_loc = anno["location"]
            vcs_length = anno["dimension"][-1]
            vcs_width = anno["dimension"][1]
            if self.filter_vcs_range:
                if not (
                    self.filter_vcs_range[0]
                    < vcs_loc[0]
                    < self.filter_vcs_range[2]
                    and self.filter_vcs_range[1]
                    < vcs_loc[1]
                    < self.filter_vcs_range[3]
                ):
                    continue

            if self.ego_ignore_range:
                if (
                    self.ego_ignore_range[0]
                    < vcs_loc[0]
                    < self.ego_ignore_range[2]
                    and self.ego_ignore_range[1]
                    < vcs_loc[1]
                    < self.ego_ignore_range[3]
                ):
                    continue

            bev_ct = (
                ((self.vcs_range[3] - vcs_loc[1]) / self.m_perpixel[1]),
                ((self.vcs_range[2] - vcs_loc[0]) / self.m_perpixel[0]),
            )  # (u, v)
            bev_ct_int = (int(bev_ct[0]), int(bev_ct[1]))
            if (
                0 <= bev_ct_int[0] < self.bev_size[1]
                and 0 <= bev_ct_int[1] < self.bev_size[0]
            ):
                if self.background_reweight_cfg:
                    (
                        background_gaussian_hm,
                        background_reweight_hm,
                    ) = self.reweight_background(
                        need_background_reweight,
                        label.lower(),
                        cls_id,
                        bev_ct_int,
                        background_gaussian_hm,
                        background_reweight_hm,
                    )
                # For one 3d bbox of background category,
                # it don't need to generate foreground target.
                if need_background_reweight:
                    continue

                length_limitation = self.length_limitation.get(
                    cls_id, (0.0, np.inf)
                )
                if not (
                    length_limitation[0]
                    < anno["dimension"][-1]
                    < length_limitation[1]
                ):
                    anno["ignore"] = True
                vcs_loc_[count_id] = vcs_loc
                vcs_cls_[count_id] = cls_id
                vcs_dim_[count_id] = anno["dimension"]
                # transfer the yaw to value [-pi,pi]
                vcs_yaw = np.arctan2(np.sin(anno["yaw"]), np.cos(anno["yaw"]))
                vcs_rot_z_[count_id] = vcs_yaw
                if "object_tag_info" in data:
                    for object_tag in data["object_tag_info"]:
                        if (
                            "track_id" in anno
                            and anno["track_id"]
                            in data["object_tag_info"][object_tag]
                        ):
                            object_tags[object_tag][count_id] = 1
                if anno.get("ignore", False):
                    vcs_ignore_[count_id] = 1
                # Visibility reflects the occlusion degree of the target
                if anno.get("visibility", False):
                    vcs_visible_[count_id] = anno["visibility"]
                if (
                    self.use_occlusion_attribute
                    and multi_view_infos_exist_flag
                ):
                    vcs_occlusion_[count_id] = occlusion_id

                # draw bev3d heatmap
                kernel_size = self.cls2kernel[cls_id]
                # ugly code, optimize below code seg in the future
                # TODO @zihao.lu
                if self.bigobj_hm_kernel_cfg is not None:
                    big_object_length_thresh = self.bigobj_hm_kernel_cfg[
                        "length_thresh"
                    ]
                    big_object_kernel_size = self.bigobj_hm_kernel_cfg[
                        "kernel_size"
                    ]
                    if anno["dimension"][-1] > big_object_length_thresh:
                        kernel_size = np.array(
                            [
                                big_object_kernel_size,
                                big_object_kernel_size,
                            ],
                            dtype=np.float32,
                        )

                insert_bev_hm = get_gaussian2D(kernel_size, alpha=1)
                insert_bev_hm_wh = insert_bev_hm.shape[:2][::-1]
                if anno.get("ignore", False):
                    insert_ignore_mask = np.ones_like(insert_bev_hm)
                else:
                    insert_ignore_mask = np.zeros_like(insert_bev_hm)
                if self.cls_dimension is not None:
                    ori_dim = np.array(anno["dimension"])
                    avg_dim = self.cls_dimension[cls_id]
                    residual_dim = np.log(abs(ori_dim) / avg_dim)
                    ann_dim = list(residual_dim)
                else:
                    ann_dim = anno["dimension"]

                if self.use_psc_rot:
                    phase_shift_targets = tuple(
                        np.cos(vcs_yaw + 2 * np.pi * x / self.N_steps_PSC_rot)
                        for x in range(self.N_steps_PSC_rot)
                    )
                    if self.use_dual_freq:
                        raise NotImplementedError
                else:
                    phase_shift_targets = (np.cos(vcs_yaw), np.sin(vcs_yaw))

                insert_bev_reg_map_list = [
                    get_reg_map(insert_bev_hm_wh, ann_dim),
                    get_reg_map(
                        insert_bev_hm_wh,
                        phase_shift_targets,
                    ),
                    get_reg_map(insert_bev_hm_wh, vcs_loc[-1]),
                    get_ctoff_map(insert_bev_hm_wh, bev_ct),
                ]
                bev_reg_map_list = [
                    bev3d_dim,
                    bev3d_rot,
                    bev3d_loc_z,
                    bev3d_ct_offset,
                ]
                insert_bev_reg_map_list.append(
                    get_reg_map(insert_bev_hm_wh, cls_id)
                )
                bev_reg_map_list.append(class_id_map)

                # Variant of length/width ratio to highlight bigObj
                aspect_ratio = (
                    vcs_length / vcs_width + vcs_width / vcs_length
                ) / 2
                if not math.isnan(aspect_ratio):
                    insert_bev_reg_map_list.append(
                        get_reg_map(insert_bev_hm_wh, vcs_length)
                    )
                    bev_reg_map_list.append(bev3d_length_map)
                    insert_bev_reg_map_list.append(
                        get_reg_map(insert_bev_hm_wh, aspect_ratio)
                    )
                    bev_reg_map_list.append(bev3d_aspect_ratio_map)

                insert_bev_reg_map_list.append(
                    get_reg_map(
                        insert_bev_hm_wh,
                        self.category_class_weight.get(cls_id, 1.0),
                    )
                )
                bev_reg_map_list.append(category_class_weight_hm)

                if self.use_category_decouple:
                    draw_heatmap(bev3d_hm[:, :, 0], insert_bev_hm, bev_ct_int)
                else:
                    draw_heatmap(
                        bev3d_hm[:, :, cls_id], insert_bev_hm, bev_ct_int
                    )
                if (
                    self.use_occlusion_attribute
                    and multi_view_infos_exist_flag
                ):
                    insert_bev_reg_map_list.append(
                        get_reg_map(insert_bev_hm_wh, occlusion_id)
                    )
                    bev_reg_map_list.append(occlusion_id_map)

                draw_heatmap(
                    bev3d_weight_hm,
                    insert_bev_hm,
                    bev_ct_int,
                    bev_reg_map_list,
                    insert_bev_reg_map_list,
                )
                draw_heatmap(bev3d_ignore_mask, insert_ignore_mask, bev_ct_int)

                bev3d_point_pos_mask[bev_ct_int[1], bev_ct_int[0]] = 1
                count_id += 1

        if self.roi_background_weight_cfg:
            roi_weight_hm = self.finetune_roi_background_weight(
                roi_weight_hm, bev3d_weight_hm
            )
        if self.roi_weight_cfg:
            bev3d_roi_weight_hm = self.finetune_roi_weight(
                bev3d_roi_weight_hm, bev3d_length_map
            )

        bev3d_background_weight = np.maximum(
            roi_weight_hm, background_reweight_hm
        )
        one_hot_pool = np.eye(self.num_classes)
        valid_class_mask = class_id_map >= 0
        valid_class_id = class_id_map[valid_class_mask].astype(np.int32)
        if self.use_category_decouple:
            bev3d_cls_hm[valid_class_mask] = one_hot_pool[valid_class_id]
        else:
            bev3d_hm[valid_class_mask] *= one_hot_pool[valid_class_id]
        if self.use_occlusion_attribute and multi_view_infos_exist_flag:
            occlusion_one_hot_pool = np.eye(occlusion_num_classes)
            valid_occlusion_cls_mask = occlusion_id_map >= 0
            valid_occlusion_id = occlusion_id_map[
                valid_occlusion_cls_mask
            ].astype(np.int32)
            occlusion_cls_hm[
                valid_occlusion_cls_mask
            ] = occlusion_one_hot_pool[valid_occlusion_id]

        if self.bigobj_length_thresh_cfg:
            for (
                class_id,
                length_thresh,
            ) in self.bigobj_length_thresh_cfg.items():
                # get bigObjs based on length_thresh for rot reweight
                valid_rot_reweight_mask = np.logical_or(
                    valid_rot_reweight_mask,
                    np.logical_and(
                        class_id_map == class_id,
                        bev3d_length_map >= length_thresh,
                    ),
                )
                if (
                    self.bigobj_hm_kernel_cfg is not None
                    and self.bigobj_hm_kernel_cfg.get("rot_reweight", False)
                ):
                    valid_rot_reweight_mask = np.logical_or(
                        valid_rot_reweight_mask,
                        bev3d_length_map
                        >= self.bigobj_hm_kernel_cfg["length_thresh"],
                    )
            bev3d_rot_reweight_mask[
                valid_rot_reweight_mask
            ] = bev3d_aspect_ratio_map[valid_rot_reweight_mask]

        gt_bev_3d = {
            "bev3d_hm": bev3d_hm,
            "bev3d_dim": bev3d_dim,
            "bev3d_rot": bev3d_rot,
            "bev3d_ct_offset": bev3d_ct_offset,
            "bev3d_loc_z": bev3d_loc_z[:, :, np.newaxis],
            "bev3d_weight_hm": bev3d_weight_hm[:, :, np.newaxis],
            "bev3d_point_pos_mask": bev3d_point_pos_mask[:, :, np.newaxis],
            "bev3d_ignore_mask": bev3d_ignore_mask[:, :, np.newaxis],
            "bev3d_ignore_obj_cls": bev3d_ignore_obj_cls,
            "bev3d_category_class_weight": category_class_weight_hm[
                :, :, np.newaxis
            ],
            "bev3d_background_weight": bev3d_background_weight[
                :, :, np.newaxis
            ],
            "bev3d_roi_weight": bev3d_roi_weight_hm[:, :, np.newaxis],
            "bev3d_rot_reweight_mask": bev3d_rot_reweight_mask[
                :, :, np.newaxis
            ],
        }
        if self.use_category_decouple:
            gt_bev_3d["bev3d_cls_hm"] = bev3d_cls_hm
        annos_bev_3d = {
            "vcs_loc_": vcs_loc_,
            "vcs_cls_": vcs_cls_,
            "vcs_rot_z_": vcs_rot_z_,
            "vcs_dim_": vcs_dim_,
            "vcs_ignore_": vcs_ignore_,
            "vcs_visible_": vcs_visible_,
        }
        if self.use_occlusion_attribute:
            gt_bev_3d["bev3d_occlusion_hm"] = occlusion_cls_hm
            gt_bev_3d["bev3d_ignore_occlusion"] = np.array(
                [not multi_view_infos_exist_flag], dtype=np.bool
            )
            annos_bev_3d["vcs_occlusion_"] = vcs_occlusion_
        for object_tag in object_tags:
            annos_bev_3d["tag_" + object_tag] = object_tags[object_tag]
        data[self.gt_name] = gt_bev_3d
        data[self.anno_name] = annos_bev_3d
        return data

    def __repr__(self):
        return "Bev3dTargetGenerator"


@OBJECT_REGISTRY.register
class ANCE2EDynamicTargetGenerator(object):
    """Generate gound truth labels for e2e dynamic detection.

    Args:
        vcs_range: vcs range. (order is (bottom,right,top,left))
        output_labels_group (dict): dict contains target cls and
            their relations, (e.g. {"veh": [0], "vru": [1, 2]}),
            the heatmap of cls in the same group are concatenated
            on axis 0 and predicted by the same decoder or head.
            The name for every group is f"{group_name}_gt" and
            contains gt in the form of bev-3d.
        filter_vcs_range (dict): filter vcs range for specific cls.
        cls_bev_size: bev_size for sepecific cls, used to generate heatmap.
            (e.g. {0: (224, 256), 1: (280, 224), 2: (280, 224)})
        cls_hm_kernel (dict): the gaussian kernel size of bev
            heatmap for each category.(e.g. {cls1: 3})
        num_frames_per_clip: number of frames in a clip.
        num_frames_per_iter: number of frames in a iteration (sub-clip).
        max_his_odo_len: maximum of history odometry length. default:12.
        use_psc_rot: whether use psc to encode the heading angle.
        N_steps_PSC_rot: nums of steps to decode the heading angle by PSC.
            Refer to https://horizonrobotics.feishu.cn/docx/EwYEdysIBodB69xn6Mdc7HkSnif. # noqa
        trajpred_transform: data transformer for trajectory prediction.
        category2id_map: A mapping from raw category to training category_id.
    """

    # following the format in data packing.
    ego_columns = [
        "ego_x",
        "ego_y",
        "ego_z",
        "ego_height",
        "ego_width",
        "ego_length",
        "ego_yaw",
        "ego_vx",
        "ego_vy",
        "ego_wy",
    ]
    obs_columns = [
        "obs_x",
        "obs_y",
        "obs_z",
        "obs_height",
        "obs_width",
        "obs_length",
        "obs_yaw",
        "track_id",
        "classification",
        "obs_vx",
        "obs_vy",
        "obs_vz",
        "obs_wy",
        # "maneuver_lc",
    ]

    def __init__(
        self,
        vcs_range: Sequence[float],
        output_labels_group: dict,
        filter_vcs_range: dict,
        cls_bev_size: dict,
        cls_hm_kernel: Mapping,
        num_frames_per_clip: int = 1,
        num_frames_per_iter: int = 1,
        max_his_odo_len: int = 12,
        use_psc_rot: bool = False,
        N_steps_PSC_rot: int = 3,
        trajpred_transform=None,
        category2id_map: dict = None,
    ):

        self.cls_bev_size = cls_bev_size
        self.vcs_range = vcs_range
        self.num_frames_per_clip = num_frames_per_clip
        self.num_frames_per_iter = num_frames_per_iter
        self.max_his_odo_len = max_his_odo_len
        self.output_labels_group = output_labels_group
        self.filter_vcs_range = filter_vcs_range
        self.trajpred_transform = trajpred_transform
        self.cur_ego_index = self.max_his_odo_len - 1
        self.cls_labels = []
        self.category2id_map = category2id_map

        for _, cls_list in self.output_labels_group.items():
            self.cls_labels.extend(cls_list)

        for cls_id in self.cls_labels:
            assert cls_id in cls_bev_size
            assert cls_id in filter_vcs_range
            assert cls_id in cls_hm_kernel

        self.cls2kernel = {}
        for category, kernel_size in cls_hm_kernel.items():
            self.cls2kernel[category] = np.array(
                [kernel_size, kernel_size], dtype=np.float32
            )
        # 由于ct_offset是同一个group中所有类别共享一个，
        # 所以需要得到一个cls_id到group_id的映射。
        self.cls_id2group_id = {}
        for group_id, group_name in enumerate(self.output_labels_group.keys()):
            for cls_id in self.output_labels_group[group_name]:
                self.cls_id2group_id[cls_id] = group_id

        self.use_psc_rot = use_psc_rot
        self.N_steps_PSC_rot = N_steps_PSC_rot

        if self.use_psc_rot:
            assert self.N_steps_PSC_rot >= 3
            self.rot_encode_size = self.N_steps_PSC_rot
        else:
            self.rot_encode_size = 2  # cos, sin

    def __call__(self, data):
        assert "e2e_dynamic_anno" in data, KeyError(
            "e2e_dynamic_anno should be in data..."
        )
        # 将[t,t-1,t-2,...] 转成 [..., t-2, t-2,t]
        data["timestamp"] = torch.flip(data["timestamp"], [0])
        # 由于在sync_info将同一个sub_clip里将frame改成了倒序，
        # 所以这里需要将origin_imgs的顺序再改回来
        if "origin_imgs" in data:
            data["origin_imgs"] = data["origin_imgs"][::-1]
        # e2e targets
        e2e_dynamic_det_targets = []

        # collect heatmap, heatmap_weight and ignore mask for every cls group
        clip_hm = [[] for i in range(len(self.output_labels_group))]
        clip_ignore_mask = [[] for i in range(len(self.output_labels_group))]
        clip_hm_weight = [[] for i in range(len(self.output_labels_group))]
        clip_ct_offset = [[] for i in range(len(self.output_labels_group))]

        (
            annos,
            annos_num_list,
            anno_timestamps_list,
            ego_rawArrs,
            target_rawArrs,
        ) = data.pop("e2e_dynamic_anno")
        # NOTE: annos contains all clip(multi-frames)'s annotation, \
        #  here should split accroding annos_num_list \
        # to get each frames's annotations.
        annos_list_ori = np.split(
            annos, annos_num_list.cumsum().tolist(), axis=0
        )
        targets_raws_list_ori = np.split(
            target_rawArrs, annos_num_list.cumsum().tolist(), axis=0
        )
        annos_list, targets_raws_list = [], []
        cls_id_idx = self.obs_columns.index("classification")
        use_hde_data = True if ego_rawArrs.shape[-1] < 20 else False
        for idx in range(len(annos_list_ori)):
            # anno_ori is read-only, to write subclass, change writable
            # to true by copy.
            anno_i = annos_list_ori[idx].copy()
            # anno[:,  0] -> obj_idxes
            # anno[:,  1] -> labels
            # anno[:,  2] -> vcs_y
            # anno[:,  3] -> vcs_x
            # anno[:,  4] -> width
            # anno[:,  5] -> length
            # anno[:,  6] -> vcs_rot[0]: cosin
            # anno[:,  7] -> vcs_rot[1]: sine
            # anno[:,  8] -> bev_loc_z
            # anno[:,  9] -> height
            # anno[:,  10] -> confidences of gt
            # anno[:,  11] -> ignore
            # anno[:,  12] -> vx
            # anno[:,  13] -> vy
            # anno[:,  14] -> vz
            # in order to be compatible with both hde and 4dgt format, split \
            # anno_i, 4dgt add extra accelerated speed (ax, ay, az, ayaw) \
            # than hde.
            anno_i = anno_i[:, :15]
            anno_i[:, 1] = np.array(
                [self.category2id_map[int(v)] for v in anno_i[:, 1]]
            )
            annos_list.append(anno_i)
            targets_raws_i = targets_raws_list_ori[idx].copy()
            # targets_raws[:, 0] -> world_x
            # targets_raws[:, 1] -> world_y
            # targets_raws[:, 2] -> world_z
            # targets_raws[:, 3] -> height
            # targets_raws[:, 4] -> width
            # targets_raws[:, 5] -> length
            # targets_raws[:, 6] -> world_yaw
            # targets_raws[:, 7] -> obj_idxes
            # targets_raws[:, 8] -> labels
            # targets_raws[:, 9] -> vx
            # targets_raws[:, 10] -> vy
            # targets_raws[:, 11] -> vz
            # targets_raws[:, 12] -> wy
            # targets_raws[:, 13] -> maneuver_lc
            # in order to be compatible with both hde and 4dgt format, split \
            # targets_raws_i, 4dgt add extra accelerated speed (ax, ay, az) \
            # than hde.
            targets_raws_i = targets_raws_i[:, :13]
            targets_raws_i[:, cls_id_idx] = np.array(
                [
                    self.category2id_map[int(v)]
                    for v in targets_raws_i[:, cls_id_idx]
                ]
            )
            targets_raws_list.append(targets_raws_i)

        # the split annos_list's last's element is empty, pop it
        annos_list.pop(-1)
        # annos is the current sub_clip.
        # sample_split_index is the sub_clip index of current clips.
        anno_idxes = range(
            data["sample_split_index"] * self.num_frames_per_iter,
            (data["sample_split_index"] + 1) * self.num_frames_per_iter,
        )

        for anno_idx in anno_idxes:
            anno = annos_list[anno_idx]
            anno = torch.tensor(anno, dtype=torch.float)

            # ego information containes (self.max_his_odo_len) frames history
            # ego information, the index for the ego of anno_idx frames
            # should consider this.
            ego_current_index = self.max_his_odo_len - 1 + anno_idx

            # NOTE: first process ignore attribute
            if anno.size(-1) > 11:
                ignore = anno[:, 11]
                anno = anno[ignore == 0]

            # filter anno according to filter_vcs_range and cls
            mask = torch.zeros(anno.shape[0]).bool()
            for cls in self.cls_labels:
                filter_vcs_range_cls = self.filter_vcs_range[cls]
                vcs_x_filter = (filter_vcs_range_cls[0] <= anno[:, 3]) & (
                    anno[:, 3] <= filter_vcs_range_cls[2]
                )
                vcs_y_filter = (filter_vcs_range_cls[1] <= anno[:, 2]) & (
                    anno[:, 2] <= filter_vcs_range_cls[3]
                )
                cls_filter = anno[:, 1] == cls
                mask = mask | (vcs_x_filter & vcs_y_filter & cls_filter)
            anno = anno[mask]

            # save bbox in vcs coordinate for generate heatmap
            boxes_np = anno[:, 2:6].numpy().copy()

            # NOTE: generate normalized bbox(without rot) in bev img coordinate
            # anno[2] vcs_y -> norm_bev_img_x
            # anno[3] vcs_x -> norm_bev_img_y
            # anno[4] width(m) -> norm_bbox_width
            # anno[5] length(m) -> norm_bbox_length
            # anno[:, 2:6] -> bbox (cx,cy,w,h) in bev_imgs coor
            # anno[8] bev_loc_z(m) -> norm vcs bev_loc_z
            # anno[9] height(m) -> norm bbox_height
            anno[:, 2] = (self.vcs_range[3] - anno[:, 2]) / abs(
                self.vcs_range[3] - self.vcs_range[1]
            )
            anno[:, 3] = (self.vcs_range[2] - anno[:, 3]) / abs(
                self.vcs_range[2] - self.vcs_range[0]
            )
            anno[:, 4] = anno[:, 4] / abs(
                self.vcs_range[3] - self.vcs_range[1]
            )
            anno[:, 5] = anno[:, 5] / abs(
                self.vcs_range[2] - self.vcs_range[0]
            )
            # NOTE: 这里正确逻辑应该是 anno[:,8] - vcs_range[4] 之后进行归一化，
            # 但因为目前跟踪和检测在当前配置下训练的权重提供给了下游任务训练，因此
            # 暂时不做修改。
            # 在当前配置下，随着L2667-L2668行的截断会导致中心点在自车中心点下方的
            # 目标其预测的目标值大于等于0，所以对这类目标的z值预测会不准。
            # TODO: 下次发版(20230807)前需要fix之后重训一版结果。
            if use_hde_data:
                anno[:, 8] = anno[:, 8] - ego_rawArrs[ego_current_index, 2]
            anno[:, 8] = anno[:, 8] / abs(
                self.vcs_range[5] - self.vcs_range[4]
            )
            anno[:, 9] = anno[:, 9] / abs(
                self.vcs_range[5] - self.vcs_range[4]
            )
            anno[:, 8:10][anno[:, 8:10] > 1] = 1.0
            anno[:, 8:10][anno[:, 8:10] < 0] = 0.0

            if self.use_psc_rot:
                # convert to vcs_yaw from cos/sin
                vcs_yaw = torch.atan2(anno[:, 7:8], anno[:, 6:7])
                phase_shift_targets = tuple(
                    np.cos(vcs_yaw + 2 * np.pi * x / self.N_steps_PSC_rot)
                    for x in range(self.N_steps_PSC_rot)
                )
                yaws = torch.cat(phase_shift_targets, dim=-1)
            else:
                yaws = anno[:, 6:8]

            det_target = {
                "obj_idxes": anno[:, 0].long(),
                "labels": anno[:, 1].long(),
                "boxes": anno[:, 2:6],
                "yaws": yaws,
                "bev_loc_z": anno[:, 8],
                "heights": anno[:, 9],
                "scores": anno[:, 10],
            }

            # anno[12: 15] vcs vx, vy, vz
            if anno.size(1) > 12:
                det_target.update(
                    {
                        "velocities": anno[:, 12:15],
                    }
                )

            # generate heatmap, hm weight and ignore mask for every cls
            cls_hm = [
                np.zeros((self.cls_bev_size[cls_id]), dtype=np.float32)
                for cls_id in self.cls_labels
            ]
            cls_hm_weight = [
                np.zeros((self.cls_bev_size[cls_id]), dtype=np.float32)
                for cls_id in self.cls_labels
            ]
            cls_ignore_mask = [
                np.zeros((self.cls_bev_size[cls_id]), dtype=np.float32)
                for cls_id in self.cls_labels
            ]
            # 初始化ct_offset, 每个group一个
            ct_offset = [
                np.zeros(
                    (*self.cls_bev_size[cls_list[0]], 2), dtype=np.float32
                )
                for groud_id, cls_list in self.output_labels_group.items()
            ]
            for target_idx, cls_id in enumerate(det_target["labels"]):
                # 针对每个类别生成heatmap，heatmap weight和ignore mask，针对每
                # 一个group生成对应的ct_offset。
                cls_id_int = int(cls_id.item())
                insert_bev_hm = get_gaussian2D(
                    self.cls2kernel[cls_id_int], alpha=1
                )

                filter_vcs_range_cls = self.filter_vcs_range[cls_id_int]
                bev_size_cls = self.cls_bev_size[cls_id_int]
                bev_ct_x = (
                    (filter_vcs_range_cls[3] - boxes_np[target_idx, 0])
                    / (filter_vcs_range_cls[3] - filter_vcs_range_cls[1])
                    * bev_size_cls[1]
                )
                bev_ct_y = (
                    (filter_vcs_range_cls[2] - boxes_np[target_idx, 1])
                    / (filter_vcs_range_cls[2] - filter_vcs_range_cls[0])
                    * bev_size_cls[0]
                )

                bev_ct = (bev_ct_x, bev_ct_y)
                bev_ct_int = (int(bev_ct_x), int(bev_ct_y))
                insert_ignore_mask = np.zeros_like(insert_bev_hm)
                draw_heatmap(
                    cls_hm[self.cls_labels.index(cls_id_int)],
                    insert_bev_hm,
                    bev_ct_int,
                )
                draw_heatmap(
                    cls_ignore_mask[self.cls_labels.index(cls_id_int)],
                    insert_ignore_mask,
                    bev_ct_int,
                )
                draw_heatmap(
                    cls_hm_weight[self.cls_labels.index(cls_id_int)],
                    insert_bev_hm,
                    bev_ct_int,
                )
                # 生成ct_offset，由于每个group中所有类别共享一个ct_offset，所以
                # 首先根据cls_id2group_id得到当前类别所在的group_id，
                # 然后根据group_id得到对应的ct_offset。
                insert_bev_hm_wh = insert_bev_hm.shape[:2][::-1]
                group_id = self.cls_id2group_id[cls_id_int]
                bev_reg_map_list = [ct_offset[group_id]]
                insert_bev_reg_map_list = [
                    get_ctoff_map(insert_bev_hm_wh, bev_ct)
                ]
                draw_heatmap(
                    np.zeros_like(
                        cls_hm_weight[self.cls_labels.index(cls_id_int)]
                    ),
                    insert_bev_hm,
                    bev_ct_int,
                    bev_reg_map_list,
                    insert_bev_reg_map_list,
                )

            # 将各个group中的heatmap, heatmap weight, ignore mask, ct_offset合并起来。
            # 1）对于heatmap, heatmap weight, ignore mask而言，每个类别都有一个独立变量，
            # 将每个group中的类别的这些变量heatmap在axis 0上进行拼接。
            # 2）对于ct_offset而言，每个group中所有类别共享一个ct_offset，所以不需要进行拼接。
            # 每个group的名字为f"{group_name}_gt"，并且包含与bev-3d相同格式的gt。
            # 由于每个group中所有类别共享一个ct_offset，所以ct_offset不需要进行拼接。
            for group_id, group in enumerate(self.output_labels_group.keys()):

                group_cls_list = self.output_labels_group[group]
                group_hm = [
                    cls_hm[self.cls_labels.index(i)] for i in group_cls_list
                ]
                group_hm_weight = [
                    cls_hm_weight[self.cls_labels.index(i)]
                    for i in group_cls_list
                ]
                group_ignore_mask = [
                    cls_ignore_mask[self.cls_labels.index(i)]
                    for i in group_cls_list
                ]

                group_hm = np.stack(group_hm, axis=0)
                group_hm_weight = np.stack(group_hm_weight, axis=0)
                group_ignore_mask = np.stack(group_ignore_mask, axis=0)

                clip_hm[group_id].append(torch.from_numpy(group_hm))
                clip_hm_weight[group_id].append(
                    torch.from_numpy(group_hm_weight)
                )
                clip_ignore_mask[group_id].append(
                    torch.from_numpy(group_ignore_mask)
                )
                clip_ct_offset[group_id].append(
                    torch.from_numpy(ct_offset[group_id])
                )

            e2e_dynamic_det_targets.append(det_target)

        for group_index, group_name in enumerate(
            self.output_labels_group.keys()
        ):
            heat_gt = {
                "bev3d_hm": torch.stack(clip_hm[group_index], dim=0),
                "bev3d_ignore_mask": torch.stack(
                    clip_ignore_mask[group_index], dim=0
                ),
                "bev3d_weight_hm": torch.stack(
                    clip_hm_weight[group_index], dim=0
                ),
                "bev3d_background_weight": torch.ones_like(
                    torch.stack(clip_hm_weight[group_index], dim=0)
                ),
                "bev3d_ct_offset": torch.stack(
                    clip_ct_offset[group_index], dim=0
                ).permute(
                    (0, 3, 1, 2)
                ),  # shape: frame, 2, h, w
            }
            data.update({f"{group_name}_gt": {"gt_bev_3d": heat_gt}})

        data["motr_targets"] = {}
        data["motr_targets"]["bev_tracking"] = e2e_dynamic_det_targets

        ego_raws, targets_raws = [], []

        odo_end_idx = self.num_frames_per_clip + (self.max_his_odo_len - 1)
        ego_his_arrs = ego_rawArrs[:odo_end_idx].copy()
        # ego_rawArrs[:, 0] -> world_ego_x
        # ego_rawArrs[:, 1] -> world_ego_y
        # ego_rawArrs[:, 2] -> world_ego_z
        # ego_rawArrs[:, 3] -> ego_height
        # ego_rawArrs[:, 4] -> ego_width
        # ego_rawArrs[:, 5] -> ego_length
        # ego_rawArrs[:, 6] -> world_ego_yaw
        # ego_rawArrs[:, 7] -> ego_vx
        # ego_rawArrs[:, 8] -> ego_vy
        # ego_rawArrs[:, 9] -> ego_wy
        # in order to be compatible with both hde and 4dgt format, split \
        # ego_rawArrs, 4dgt add extra liosam odometry information than \
        # hde.
        ego_rawArrs = ego_rawArrs[self.cur_ego_index :][:, :10]

        # add columns of cur-frame-id
        ego_raws = np.concatenate(
            [
                ego_rawArrs.copy(),
                np.arange(ego_rawArrs.shape[0])[:, np.newaxis],
            ],
            axis=1,
        ).astype(np.float32)

        for idx in range(ego_rawArrs.shape[0]):
            targets_raws_idx = targets_raws_list[idx]
            targets_raws_with_index = np.concatenate(
                [
                    np.ones_like(targets_raws_list[idx][:, :1]) * idx,
                    targets_raws_idx,
                ],
                axis=-1,
            ).astype(np.float32)
            targets_raws.append(targets_raws_with_index)
        targets_raws = np.concatenate(targets_raws, axis=0)

        egoXCol = self.ego_columns.index("ego_x")
        egoYCol = self.ego_columns.index("ego_y")
        egoWCal = self.ego_columns.index("ego_yaw")
        ego_his_arrs = ego_his_arrs[:, [egoXCol, egoYCol, egoWCal]]

        trajectory_pred = {
            "ego_raw": pd.DataFrame(
                data=ego_raws, columns=self.ego_columns + ["frame_id"]
            ),  # do not contains history ego information
            # "ego_rotmat": ego_rotmat,   # ----------- old mr --------
            "targets_raw": pd.DataFrame(
                data=targets_raws, columns=["frame_id"] + self.obs_columns
            ),  # do not contains history information
        }

        # ego information for every frames containes (self.max_his_odo_len)
        # frames history ego information, the index of current ego info is
        # (self.max_his_odo_len - 1)
        ego_info_clip = []
        for anno_idx in anno_idxes:
            ego_iter = ego_his_arrs[anno_idx : anno_idx + self.max_his_odo_len]
            ego_info_clip.append(torch.from_numpy(ego_iter))
        ego_info_clip = torch.stack(ego_info_clip, dim=0)

        data["odo_info"] = ego_info_clip

        data["pack_names"] = "_".join(data["pack_dir"].split("/")[-2:])

        data["motr_targets"]["trajectory_pred"] = trajectory_pred
        if self.trajpred_transform is not None:
            for trans in self.trajpred_transform:
                data = trans(data)

        return data


@OBJECT_REGISTRY.register
class ANCPrepareDepthPose(object):
    """Build input data for 2.5D task.

    e.g stack mulit view datas on batch axis. convert tensor type and so on.

    Args:
        with_extra_img: whther return extra img.
            set True when depth/pose training and False for validation.
        with_color_img: whther return color img.
            Used for depth/pose training.
        input_sequence_length: img number input to backbone, 2 means
            two img input to backbone(order is (t-1,t)), 1 means single img
            input to backbone.
    """

    def __init__(
        self,
        with_extra_img: bool = False,
        with_color_img: bool = False,
        input_sequence_length: int = 1,
    ) -> None:

        self.with_extra_img = with_extra_img
        self.with_color_img = with_color_img
        assert input_sequence_length in (1, 2)
        self.input_sequence_length = input_sequence_length

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'
        data["view"] = "front"
        cat_data = [torch.stack(_) for _ in data["imgs"]]

        if self.with_extra_img:
            # durning train process, need three frames,
            # suppose the order in data['imgs'] is [t,t-1,t-2]
            # and t-1 frame is current frame(
            # it means we will cal depth loss for t-1 frame)
            data["extra_img"] = [cat_data[0], cat_data[2]]  # t,t-2
            if self.input_sequence_length == 1:
                data["img"] = [cat_data[1]]  # t-1
            else:
                data["img"] = [cat_data[2], cat_data[1]]  # t-2,t-1
        else:
            # durning val process,only need two frames,
            # suppose the order in data['imgs'] is [t-1,t-2]
            data["extra_img"] = [cat_data[1]]  # t-2
            if self.input_sequence_length == 1:
                data["img"] = [cat_data[0]]  # t-1
            else:
                data["img"] = [cat_data[1], cat_data[0]]  # t-2,t-1

        if self.with_color_img:
            color_cat_data = []
            for color_img_i in data["color_imgs"]:
                color_cat_data.append(torch.stack(color_img_i))
            data["color_imgs"] = color_cat_data

        data.pop("imgs")

        if "gt_depth" in data:
            data["gt_depth"] = torch.stack(data["gt_depth"])

        if "front_mask" in data:
            data["front_mask"] = data["front_mask"].float()

        if "obj_mask" in data:
            data["obj_mask"] = (
                (data["obj_mask"] <= 17) * (data["obj_mask"] >= 9)
            ).float()
        # class idx between 9 and 17 means dynamic class
        # (auto 33class parsing model). e.g. bus, car, person and so on.

        if "size" in data:
            data.pop("size")

        return data


@OBJECT_REGISTRY.register
class ANCSelectDataByIdx(object):
    """Select the specified data from the input data by idx.

    Args:
        select_idxs: the index list to select.
        input_key: data key in input dict to select.
        output_key: data key to store data selected.
            NOTE: output_key and input_key is same means in-place modification.

    """

    def __init__(
        self,
        select_idxs: Sequence[int],
        input_key: str = "imgs",
        output_key: str = "img",
    ):
        self.select_idxs = select_idxs
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, data: Mapping):
        assert self.input_key in data
        assert isinstance(data[self.input_key], Sequence)
        data[self.output_key] = [
            data[self.input_key][idx] for idx in self.select_idxs
        ]
        return data

    def __repr__(self):
        return "SelectDataByIdx"


@OBJECT_REGISTRY.register
class ANCStackData(object):
    """Stack specific data in input dict.

    Args:
        data_keys: a key list to stack.

    """

    def __init__(self, data_keys: Union[str, Sequence[str]]):
        self.data_keys = _as_list(data_keys)

    def __call__(self, data: Mapping):
        for key in self.data_keys:
            if key not in data:
                continue
            assert isinstance(data[key], Sequence)
            if isinstance(data[key][0], Sequence):
                data[key] = [torch.stack(_) for _ in data[key]]
            else:
                data[key] = torch.stack(data[key])
        return data

    def __repr__(self):
        return "StackData"


@OBJECT_REGISTRY.register
class ANCPrepareDataBEV(object):
    """Build input data for BEV task.

    e.g stack mulit view datas on batch axis. convert tensor type and so on.

    Args:
        views_domain2nums: a dict to map view domains to
            corresponding view nums, specifying how to organize data.
            view domains contain ["front", "side", "round", "narrow"]

            .. code-block:: none

                # the `views_domain2nums` of 7v-wide model is
                {
                    "front": 1,
                    "side": 5,
                    "round": 0,
                    "narrow": 1,
                }

        single_frame: Whether it is single frame input, default to True.
    """

    def __init__(
        self,
        views_domain2nums: dict,
        single_frame: bool = True,
    ) -> None:

        self.views_domain2nums = views_domain2nums
        self.single_frame = single_frame
        views_domain2img_keys = OrderedDict()
        for view_domain in self.views_domain2nums.keys():
            img_key = view_domain + "_img" if view_domain != "front" else "img"
            views_domain2img_keys[view_domain] = img_key
        self.views_domain2img_keys = views_domain2img_keys

    def _stack_data_by_view(
        self, data: Sequence, view_idxs: Sequence = None
    ) -> torch.Tensor:
        """Stack data according to views.

        Args:
            data: data need to stack, e.g. imgs, seg. shape=
            [[frame0],[frame1],[frame2],[frame3]], each frame's contains:
            frame0: [view1, view2, view3, view4,...]
            view_idxs: views_idxs to stack the data,
            support 3 types:
            [type:1]: 1 -> [1];
            [type:2]: [1,2,3];
            [type:3]: None (means stack all); Defaults to None.

        Returns:
            list of stacked data by frames.
        """
        if view_idxs is not None:
            view_idxs = _as_list(view_idxs)
        else:
            view_idxs = list(range(len(data[0])))
        stack_data = [
            torch.stack([_data[idx] for idx in view_idxs]) for _data in data
        ]
        return stack_data

    def _prepare_common_data(self, data: Mapping):
        data.pop("imgs")
        data.pop("object_tag_info", None)

        bevseg_gt_names = ["gt_bev_seg", "gt_bev_seg_small"]
        for bevseg_gt_name in bevseg_gt_names:
            if bevseg_gt_name in data:
                data[bevseg_gt_name] = data[bevseg_gt_name].long()

        if "occlusion" in data:
            data["occlusion"] = data["occlusion"].long()

        bev3d_rm_names = [
            "gt_bev_dynamic_anno",
            "gt_multi_view",
            "image_id",
            "image_name",
        ]
        bev3d_gt_group_name = [
            ["gt_bev_3d", "annos_bev_3d"],
            ["gt_bev_3d_small", "annos_bev_3d_small"],
        ]
        for one_group_name in bev3d_gt_group_name:
            gt_name, anno_name = one_group_name
            if gt_name in data:
                for bev3d_rm_name in bev3d_rm_names:
                    if bev3d_rm_name in data:
                        if bev3d_rm_name == "gt_multi_view":
                            data[bev3d_rm_name].pop(
                                "occlusion_multi_view", None
                            )
                        else:
                            data.pop(bev3d_rm_name, None)
                for k in data[gt_name].keys():
                    if (
                        k == "bev3d_ignore_obj_cls"
                        or k == "bev3d_ignore_occlusion"
                    ):
                        data[gt_name][k] = torch.from_numpy(data[gt_name][k])
                    else:
                        data[gt_name][k] = torch.from_numpy(
                            data[gt_name][k].transpose(2, 0, 1)
                        )
                assert anno_name in data
                for k in data[anno_name].keys():
                    data[anno_name][k] = torch.from_numpy(data[anno_name][k])

        if "size" in data:
            data.pop("size")

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'
        data["view"] = self.views_domain2nums
        views = list(self.views_domain2nums.values())
        start_view_idx_list = np.cumsum([0] + views[:-1])
        end_view_idx_list = np.cumsum(views)
        for view_domain, start_view_idx, end_view_idx in zip(
            self.views_domain2nums.keys(),
            start_view_idx_list,
            end_view_idx_list,
        ):
            if self.views_domain2nums[view_domain] == 0:
                continue
            view_idxs = list(range(start_view_idx, end_view_idx))
            view_cat_data = self._stack_data_by_view(data["imgs"], view_idxs)
            img_key = self.views_domain2img_keys[view_domain]
            data[img_key] = [view_cat_data[0]]  # t
            if not self.single_frame:
                data[img_key].insert(0, view_cat_data[1])  # t-1,t

        self._prepare_common_data(data)
        return data


@OBJECT_REGISTRY.register
class ANCTemporalHomo(object):
    """Calculate the temporal homography matrix with giving vcs pose.

    if return_relative is False, calculate the absolute temporal homography,Suppose input 3 poses, namely pose_t0, pose_t1, pose_t2,
    and then return 2 homography matrices, representing time t0 to t1, and t0 to t2.  # noqa

    if return_relative is True, calculate the relative temporal homography,Suppose input 3 poses, namely pose_t0, pose_t1, pose_t2,
    and then return 2 homography matrices, representing time t0 to t1, and t1 to t2.  # noqa

    if add_eye_mat is True, append a eye metric to the end of the homography.  # noqa

    (pose_t0 represents the pose from the vehicle to the world coordinate system at time t0)  # noqa

    Args:
        bev_size: bev size.(order is (h,w))
        vcs_range: The range of vcs, in meters.(order is
            (bottom,right,top,left))
        return_relative: if True, return relative pose, otherwise return absolute pose.
        homography_names: return homography names of data.
        add_eye_mat: whether append an eye metric to the end of the homography
            matrix, designed for the cache feat in the temporal fusion module.  # noqa
    """

    def __init__(
        self,
        bev_size: Union[Sequence[int], Sequence[Sequence[int]]],
        vcs_range: Union[Sequence[float], Sequence[Sequence[float]]],
        return_relative: bool = False,
        homography_names: Union[str, Sequence[str]] = "homography_temporal",
        add_eye_mat: bool = False,
    ) -> None:

        if not isinstance(bev_size[0], Sequence):
            bev_size = [bev_size]
        if not isinstance(vcs_range[0], Sequence):
            vcs_range = [vcs_range]
        homography_names = _as_list(homography_names)
        assert (
            len(bev_size) == len(vcs_range) == len(homography_names)
        ), "The length of bev_size, vcs_range and homography_names must be the same"  # noqa
        self.homography_names = homography_names
        self.vcs2bev_img = []
        for one_vcs_range, one_bev_size in zip(vcs_range, bev_size):
            self.vcs2bev_img.append(
                get_vcs2bev_img_mat(one_vcs_range, one_bev_size)
            )
        self.return_relative = return_relative
        self.add_eye_mat = add_eye_mat

    def __call__(self, data):
        # generate temporal bev homograpy matrix
        if "pose" in data:
            homography_temporal = [[] for _ in range(len(self.vcs2bev_img))]
            transforms = data.pop("pose")
            transfrom_cur, transforms_pre_all = transforms[0], transforms[1:]
            for frame_idx, transform_pre in enumerate(transforms_pre_all):
                if self.return_relative:
                    transfrom_cur, transform_pre = (
                        transforms[frame_idx],
                        transforms[frame_idx + 1],
                    )
                transfrom_cur2pre = (
                    np.linalg.inv(transform_pre) @ transfrom_cur
                )

                rotation = transfrom_cur2pre[:3, :3]
                euler = R.from_matrix(rotation).as_euler(
                    "xyz", degrees=True
                )  # order is (roll,pitch,yaw)
                # set the pitch and roll angle to zero, and only consider the yaw angle according vcs coordinate  # noqa
                euler[0] = 0
                euler[1] = 0
                rotation_2d = R.from_euler(
                    "xyz", euler, degrees=True
                ).as_matrix()[:2, :2]
                trans_2d = transfrom_cur2pre[:2, 3]
                vcs_transform = np.eye(3, dtype="float32")
                vcs_transform[:2, :2] = rotation_2d
                vcs_transform[:2, 2] = trans_2d

                for i, vcs2bev_img in enumerate(self.vcs2bev_img):
                    h_cur2pre = (
                        vcs2bev_img
                        @ vcs_transform
                        @ np.linalg.inv(vcs2bev_img)
                    )
                    homography_temporal[i].append(h_cur2pre)
            if self.add_eye_mat:
                for i in range(len(self.vcs2bev_img)):
                    homography_temporal[i].append(np.eye(3, dtype="float32"))
            for i, homography_name in enumerate(self.homography_names):
                data[homography_name] = np.stack(
                    homography_temporal[i]
                ).astype("float32")
        return data


@OBJECT_REGISTRY.register
class ANCConvertReal3dTo3DV(object):
    """Convert real3d annotations to gt_bev_3d.

    Args:
        category_id_dict: The category mapping dict for label exchange.
        append_eye_pose: whether append eye pose or not, setting to True
            when using real3d data for temporal training.
    """

    def __init__(self, category_id_dict: dict, append_eye_pose: bool = False):
        super().__init__()
        self.category_id_dict = category_id_dict
        self.append_eye_pose = append_eye_pose

    def __call__(self, data_dict: Mapping):
        anno_multi_view = data_dict.pop("annotations")
        data_dict.pop("num_classes")

        gt_bev_3d = []

        for anno_each_view in anno_multi_view:
            for anno in anno_each_view:
                if "in_camera" in anno:
                    anno.update(anno.pop("in_camera"))
                gt_object = {}
                gt_object["dimension"] = anno["dim"]
                gt_object["location"] = anno["location"]
                gt_object["score"] = 1
                gt_object["category_id"] = anno["category_id"]
                gt_object["label"] = self.category_id_dict[anno["category_id"]]
                gt_object["ignore"] = anno["ignore"]
                gt_object["rotation_y"] = anno["rotation_y"]
                gt_object["image_id"] = anno["image_id"]
                gt_bev_3d.append(gt_object)

        data_dict["gt_bev_dynamic_anno"] = gt_bev_3d
        # collect pack_dir for eval result
        data_dict["pack_dir"] = data_dict["image_name"][0].split("__")[0]
        if self.append_eye_pose:
            eye_pose = np.eye(4)
            data_dict["pose"] = [eye_pose] * len(data_dict["pil_imgs"])

        return data_dict

    def __repr__(self):
        return "ConvertReal3dTo3DV"


# used for fsd pack inference
@OBJECT_REGISTRY.register
class ANCHomoAdaption(object):  # noqa: D205,D400
    """Modify homography matrix of each view and convert to tensor.
        only use for pack inference.

    Args:
        homo_scale_each_view: homography matrix scale factor.
    """

    def __init__(self, homo_scale_each_view: Sequence):
        self.homo_scale_each_view = homo_scale_each_view

    def __call__(self, data):
        meta_info_names = ["meta_info", "meta_info_small"]
        for meta_info_name in meta_info_names:
            if meta_info_name in data:
                if "homography" in data[meta_info_name]:
                    # modify homography matrix
                    assert len(data[meta_info_name]["homography"]) == len(
                        self.homo_scale_each_view
                    )
                    for i, (h, scale) in enumerate(
                        zip(
                            data[meta_info_name]["homography"],
                            self.homo_scale_each_view,
                        )
                    ):
                        scale_mat = np.eye(3, dtype="float32")
                        scale_mat[0, 0] = scale
                        scale_mat[1, 1] = scale
                        data[meta_info_name]["homography"][i] = scale_mat @ h
                    data[meta_info_name]["homography"] = np.stack(
                        data[meta_info_name]["homography"], axis=0
                    )
                    data[meta_info_name]["homography"] = torch.from_numpy(
                        data[meta_info_name]["homography"]
                    )
            if "homo_offset" in data[meta_info_name]:
                data[meta_info_name]["homo_offset"] = np.stack(
                    data[meta_info_name]["homo_offset"], axis=0
                )
                data[meta_info_name]["homo_offset"] = torch.from_numpy(
                    data[meta_info_name]["homo_offset"]
                )
        return data


@OBJECT_REGISTRY.register
class ANCPrepareTempoDataBEV(ANCPrepareDataBEV):
    """Build input temporal data for BEV task.

    NOTE: PrepareTempoDataBEV used for BEV temporal task,
    e.g. bev_motion, bev_multitask contains (bev_motion
    bev_seg and bev_3d. etc.).

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'
        data["view"] = self.views_domain2nums
        views = list(self.views_domain2nums.values())
        start_view_idx_list = np.cumsum([0] + views[:-1])
        end_view_idx_list = np.cumsum(views)
        for view_domain, start_view_idx, end_view_idx in zip(
            self.views_domain2nums.keys(),
            start_view_idx_list,
            end_view_idx_list,
        ):
            if self.views_domain2nums[view_domain] == 0:
                continue
            view_idxs = list(range(start_view_idx, end_view_idx))
            view_cat_data = self._stack_data_by_view(data["imgs"], view_idxs)

            frame_num = len(view_cat_data)
            img_key = self.views_domain2img_keys[view_domain]
            data[img_key] = [torch.cat(view_cat_data, dim=0)]

        meta_info_names = ["meta_info", "meta_info_small"]
        for meta_info_name in meta_info_names:
            if meta_info_name in data:
                if "homography" in data[meta_info_name]:
                    data[meta_info_name]["homography"] = data[meta_info_name][
                        "homography"
                    ].repeat([frame_num, 1, 1, 1])
                if "homo_offset" in data[meta_info_name]:
                    data[meta_info_name]["homo_offset"] = data[meta_info_name][
                        "homo_offset"
                    ].repeat([frame_num, 1, 1, 1])

        self._prepare_common_data(data)
        return data


@OBJECT_REGISTRY.register
class ANCPrepareTempoDataE2EDynamic(ANCPrepareTempoDataBEV):
    """Build input temporal data for E2E dynamic task.

    NOTE: PrepareTempoDataE2EDynamic used for E2E temporal task.

    Args:
        length_of_clip: numbers of frames in a clip.
        num_frames_per_iter: numbers of frames in every iter.
        reverse_imgs_order: whether reverse the order of imgs. The original
            order is [t, t-1, t-2, t-3] for the temporal fusion module.
            However, for specific task like det that do not apply temporal
            fusion, the order should be reversed.
    """

    def __init__(
        self,
        length_of_clip: int,
        num_frames_per_iter: int,
        reverse_imgs_order: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.length_of_clip = length_of_clip
        self.num_frames_per_iter = num_frames_per_iter
        self.reverse_imgs_order = reverse_imgs_order
        self.split_interval_frames = length_of_clip // num_frames_per_iter

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'
        data["view"] = self.views_domain2nums

        # reverse the order of imgs for temporal fusion.
        if self.reverse_imgs_order:
            data["imgs"] = data["imgs"][::-1]

        views = list(self.views_domain2nums.values())
        start_view_idx_list = np.cumsum([0] + views[:-1])
        end_view_idx_list = np.cumsum(views)
        for view_domain, start_view_idx, end_view_idx in zip(
            self.views_domain2nums.keys(),
            start_view_idx_list,
            end_view_idx_list,
        ):
            if self.views_domain2nums[view_domain] == 0:
                continue
            view_idxs = list(range(start_view_idx, end_view_idx))
            stack_img = self._stack_data_by_view(
                data["imgs"],
                view_idxs=view_idxs,
            )
            img_key = self.views_domain2img_keys[view_domain]
            data[img_key] = [torch.cat(stack_img, dim=0)]

        meta_info_names = ["meta_info", "meta_info_small"]
        for meta_info_name in meta_info_names:
            if meta_info_name in data:
                if "homography" in data[meta_info_name]:
                    data[meta_info_name]["homography"] = data[meta_info_name][
                        "homography"
                    ].repeat(self.num_frames_per_iter, 1, 1, 1)
                if "homo_offset" in data[meta_info_name]:
                    data[meta_info_name]["homo_offset"] = data[meta_info_name][
                        "homo_offset"
                    ].repeat(self.num_frames_per_iter, 1, 1, 1)
        self._prepare_common_data(data)
        return data


@OBJECT_REGISTRY.register
class ANCVisualizeIpm(object):
    """Visualize ipm fusion image under bev.

    Visualize ipm result for 6v image.

    Args:
        bev_size: bev size.(order is (h,w))
        view_idxs: Specify which image to use form bird eye view.
            Optional 4v (order is front, rear, left, right) or 6v(order
            is front, front_left, front_right, rear_left, rear_right, rear)
        img_scale: Scale of per view.
        block_warp_padding: block_warp_padding parameter,
            order is (left,right,up,bottom).
        vis_single_frame: only vis frame 0.
        enable_vis: Whether to fuse image intt bev eye bird view.
            If enable_vis = False return a image which all value is 0.
            If enable_vis = True return a IPM image.
        vcs_plane_heights: the heights of multi-height vcs planes.
            Default to (0, ), representing the single ground plane.
            more info, please refer to `HomoGenerator` in \
                `hat_internal/data/datasets/bev.py`
        return_name: name of function return. Optional ipm, ipm_small
        meta_name: name of meta info. Optional meta_info, meta_info_small
    """

    def __init__(
        self,
        bev_size: Union[Sequence[int], Sequence[Sequence[int]]],
        view_idxs: Sequence[int],
        img_scale: float,
        block_warp_padding: Sequence,
        vis_single_frame: bool = True,
        enable_vis: bool = False,
        vcs_plane_heights: Union[
            Sequence[float], Sequence[Sequence[float]]
        ] = (0,),
        return_name: Union[str, Sequence[str]] = "ipm",
        meta_name: Union[str, Sequence[str]] = "meta_info",
    ) -> None:
        if not isinstance(bev_size[0], Sequence):
            bev_size = [bev_size]
        if not isinstance(block_warp_padding[0][0], Sequence):
            block_warp_padding = [block_warp_padding]
        if not isinstance(vcs_plane_heights[0], Sequence):
            vcs_plane_heights = [vcs_plane_heights]
        meta_name = _as_list(meta_name)
        return_name = _as_list(return_name)
        self.bev_size = bev_size
        self.view_idxs = view_idxs
        self.img_scale = img_scale
        self.block_warp_padding = block_warp_padding
        self.vis_single_frame = vis_single_frame
        self.enable_vis = enable_vis
        self.vcs_plane_heights = vcs_plane_heights
        self.meta_name = meta_name
        self.return_name = return_name
        assert (
            len(self.bev_size)
            == len(self.block_warp_padding)
            == len(self.meta_name)
            == len(self.vcs_plane_heights)
            == len(self.return_name)  # noqa
        ), "The length of bev_size_list, block_warp_padding_list, meta_name_list and return_name must be the same"  # noqa

    def __call__(self, data):
        for i, meta_name in enumerate(self.meta_name):
            height = self.bev_size[i][0]
            width = self.bev_size[i][1]
            ipm_size = (height, width, 3)
            bird_eye_view = np.zeros(ipm_size, np.uint8)

            frame_num = 1 if self.vis_single_frame else len(data["pil_imgs"])
            data[self.return_name[i]] = []
            for frame_idx in range(frame_num):
                if self.enable_vis and meta_name in data:
                    imgs_list = []
                    homo_offset_list = []
                    block_warp_list = []
                    for view_idx in self.view_idxs:
                        homo_offset_list.append(
                            torch.from_numpy(
                                np.expand_dims(
                                    data[meta_name]["homo_offset"][
                                        view_idx
                                        * len(self.vcs_plane_heights[i])
                                    ],
                                    axis=0,
                                )
                            ).float()
                        )
                        if isinstance(
                            data["pil_imgs"][frame_idx][0], torch.Tensor
                        ):
                            resize_h = int(
                                data["pil_imgs"][frame_idx][view_idx].shape[1]
                                * self.img_scale  # noqa
                            )
                            resize_w = int(
                                data["pil_imgs"][frame_idx][view_idx].shape[2]
                                * self.img_scale  # noqa
                            )
                            img_torch = (
                                F.resize(
                                    data["pil_imgs"][frame_idx][view_idx],
                                    (resize_h, resize_w),  # noqa
                                )
                                .unsqueeze(0)
                                .float()
                            )
                        else:
                            resize_w = int(
                                data["pil_imgs"][frame_idx][view_idx].size[0]
                                * self.img_scale  # noqa
                            )
                            resize_h = int(
                                data["pil_imgs"][frame_idx][view_idx].size[1]
                                * self.img_scale  # noqa
                            )
                            cur_img = data["pil_imgs"][frame_idx][
                                view_idx
                            ].resize(
                                (resize_w, resize_h)
                            )  # (w, h)
                            cur_img = np.array(cur_img)
                            img_torch = img_array2tensor(cur_img)
                        imgs_list.append(img_torch)
                        block_warp_list.append(
                            self.block_warp_padding[i][view_idx]
                        )
                    bird_eye_view = visualize_ipm(
                        imgs_list, homo_offset_list, block_warp_list
                    )
                    if Image.fromarray(bird_eye_view).mode == "RGB":
                        bird_eye_view = cv2.cvtColor(
                            bird_eye_view, cv2.COLOR_RGB2BGR
                        )
                data[self.return_name[i]].append(
                    torch.from_numpy(bird_eye_view)
                )
            if len(data[self.return_name[i]]) == 1:
                data[self.return_name[i]] = data[self.return_name[i]][0]

        return data

    def __repr__(self):
        return "VisualizeIpm"


@OBJECT_REGISTRY.register
class ANCNV12Transform3DV(object):
    """Pyramid resize nv12 Images to the given layer and convert yuv444.

    The transformation of RGB format data is as follows:
        Auto3DV -> Collect3DV -> Reseze -> Crop -> ToTensor(to yuv)
        -> Normalize

    transformation for nv12 format data should as follows:
        PackDataset -> ConvertPackDataTo3DV -> NV12Transform -> Crop
        -> Normalize

    This transformer contain nv12 pyramid resize and convert nv12 to yuv444.

    Args:
        ori_size: Shape of nv12 img when convert to bgr like (h, w).
        layer_index: Img pyramid's index, range is 0 to 5,
            while 0 for 1/2, 5 for 1/64.
    """

    def __init__(
        self,
        ori_size: Union[tuple, Sequence[tuple]],
        pyramid_layer_index: Union[int, Sequence[int]],
        save_nv12: bool = False,
    ):
        if nv122yuv444 is None:
            raise ImportError(
                "Unable to import nv122yuv444 from hat-sim,"
                "please make sure the version hat-sim>=1.0.1"
            )
        self.ori_size = ori_size
        self.pyramid_layer_index = pyramid_layer_index
        self.save_nv12 = save_nv12

    def _pyramid_resize(
        self,
        data: Union[np.array, Sequence],
        ori_size: Union[tuple, Sequence],
        layer_index: Union[int, Sequence],
        ratio: Union[int, Sequence],
    ):
        if isinstance(data, Sequence):
            assert len(data) == len(layer_index)
            return [
                self._pyramid_resize(
                    data_i, ori_size_i, layer_index_i, ratio_i
                )
                for data_i, ori_size_i, layer_index_i, ratio_i in zip(
                    data, ori_size, layer_index, ratio
                )
            ]
        else:
            ori_size_h, ori_size_w = ori_size[0] // ratio, ori_size[1] // ratio
            assert ori_size_h * ori_size_w * 3 // 2 == data.size
            target_h, target_w = ori_size[0] // ratio, ori_size[1] // ratio
            if 0 <= layer_index <= 4:
                for _ in range(layer_index + 1):
                    target_w = (target_w >> 1) & (~0x1)
                    target_h = (target_h >> 1) & (~0x1)
                output_rois = [0, 0, target_w, target_h]
                sim_pyr = IPSPyramid(
                    data, ori_size_w, ori_size_h, 0, 0, output_rois
                )
                sim_pyr.build_pyramid()
                target = np.random.randint(
                    0,
                    255,
                    size=(int(3 * target_w * target_h / 2),),
                    dtype=np.uint8,
                )
                sim_pyr.crop_resize(target, layer_index)
            elif layer_index == -1:
                target = data
            else:
                raise ValueError("layer_index not in [-1, 0, 1, 2, 3, 4]")

            return target, target_h, target_w

    def _nv12_to_yuv444(self, data: np.array, height: int, width: int):
        yuv444 = np.zeros(shape=(3, height, width), dtype=np.uint8)
        nv122yuv444(data, yuv444, width, height)
        return yuv444

    def __call__(self, data: Mapping):
        assert "img" in data, 'input data must has "img"'
        assert (
            len(data["img"][0])
            == len(self.ori_size)
            == len(self.pyramid_layer_index)
        )

        ratio = data.pop("ratio")
        delta_layer = [math.log(i, 2) for i in ratio]
        pyramid_layer_index = list(
            map(
                lambda x: int(x[0] - x[1]),
                zip(self.pyramid_layer_index, delta_layer),
            )
        )
        data["nv12"] = [
            self._pyramid_resize(
                img, self.ori_size, pyramid_layer_index, ratio
            )
            for img in data["img"]
        ]
        data["imgs"] = [
            self._nv12_to_yuv444(img, target_h, target_w)
            for img, target_h, target_w in data["nv12"][0]
        ]
        data["imgs"] = [
            [torch.from_numpy(data_i).float() for data_i in data["imgs"]]
        ]
        data.pop("img")

        if self.save_nv12:
            data["nv12"] = [
                [torch.from_numpy(i[0]).float() for i in data["nv12"][0]]
            ]
        else:
            data.pop("nv12")

        return data

    def __repr__(self):
        return "NV12Transform3DV"


@OBJECT_REGISTRY.register
class ANCSetRPYAugParam(object):
    """Assignment rpy perspective augmentation param.

    Args:
        prob: the prob to perturb roll, pitch, yaw.
        aug_degree(seg): the perturbed degree added
            to the origin rpy, if set to `1`,
            means random-> [-1,1].
    """

    def __init__(
        self,
        prob: float = 0.0,
        aug_degree: float = 1,  # deg
    ) -> None:
        super(ANCSetRPYAugParam, self).__init__()
        self.prob = prob
        self.aug_degree = aug_degree

    def __call__(self, data: Mapping):
        data.setdefault("aug_transforms", {})
        data["aug_transforms"]["rpy_augmentation"] = {
            "prob": self.prob,
            "aug_degree": self.aug_degree,  # deg
        }
        return data

    def __repr__(self):
        return "SetRPYAugParam"


@OBJECT_REGISTRY.register
class ANCMultiViewTargetGenerator(object):
    """Multi view data processer.

    Currently, it is only used to process data for occlusion attributes

    Args:
        occlusion_attribute: Whether use occlusion attributes.
        occlusion_attribute_dict: occlusion category to id dict.
            Specifically, if the category id is smaller, it indicates
            a lower degree of occlusion.
        use_ignore_mask_img: Whether use ignore mask.
    """

    def __init__(
        self,
        occlusion_attribute: bool = False,
        occlusion_attribute_dict: Mapping[str, int] = None,
        use_ignore_mask_img: bool = False,
    ):
        self.occlusion_attribute = occlusion_attribute
        if self.occlusion_attribute:
            self.occlusion_attribute_dict = occlusion_attribute_dict
        self.use_ignore_mask_img = use_ignore_mask_img

    def get_occlusion_attribute(self, bbox, occlusion_multi_view_dict):
        bbox_occlusion = bbox["bbox2d_attr"]["occlusion"]
        if bbox_occlusion == "unknown":
            bbox_occlusion = "invisible"
        bbox_uid = bbox["uid"]
        if bbox_uid not in occlusion_multi_view_dict:
            occlusion_attribute = self.occlusion_attribute_dict[bbox_occlusion]
        elif (
            occlusion_multi_view_dict[bbox_uid]
            > self.occlusion_attribute_dict[bbox_occlusion]
        ):
            occlusion_attribute = self.occlusion_attribute_dict[bbox_occlusion]
        else:
            occlusion_attribute = occlusion_multi_view_dict[bbox_uid]
        return occlusion_attribute

    def __call__(self, data: Mapping):
        assert "gt_multi_view" in data, 'input data must has "gt_multi_view"'
        if not data["gt_multi_view"]:
            data["gt_multi_view"] = {}
            return data
        else:
            if self.occlusion_attribute:
                occlusion_multi_view_dict = {}
            if self.use_ignore_mask_img:
                ignore_mask = [
                    torch.zeros(img.size()[1:], dtype=torch.bool)
                    for img in data["imgs"][0]
                ]
            assert len(data["gt_multi_view"]) == len(data["imgs"][0])
            for idx, multi_view_info in enumerate(data["gt_multi_view"]):
                if multi_view_info:
                    for bbox in multi_view_info["objects"]:
                        if self.occlusion_attribute and bbox["bbox2d"]:
                            occlusion_attribute = self.get_occlusion_attribute(
                                bbox, occlusion_multi_view_dict
                            )
                            occlusion_multi_view_dict.update(
                                {bbox["uid"]: occlusion_attribute}
                            )
                    if self.use_ignore_mask_img:
                        ignore_mask[idx] = multi_view_info["meta"][
                            "ignore_mask"
                        ]["ignore_mask_2d"]

            data["gt_multi_view"] = {}
            if self.use_ignore_mask_img:
                if "img_mask" in data:
                    for index, mask in data["img_mask"]:
                        assert mask.size() == ignore_mask[index].size()
                        data["img_mask"][index] = torch.logical_or(
                            mask, ignore_mask[index]
                        )
                else:
                    data["img_mask"] = ignore_mask
            if self.occlusion_attribute:
                data["gt_multi_view"][
                    "occlusion_multi_view"
                ] = occlusion_multi_view_dict
            return data

    def __repr__(self):
        return "MultiViewTargetGenerator"


@OBJECT_REGISTRY.register
class ANCApplyMaskOnImg(object):
    """Apply Mask on original image.

    Mask is a torch.Tensor bool format data, the size of mask equal to
    the img w,h size.
    If one index of mask is 'True', means the index of img will be blackened.
    else, the index of img will be retained.

    Args:
        use_yuv_format: Whether the format of the image is yuv.
    """

    def __init__(
        self,
        use_yuv_format: bool = False,
    ):
        self.use_yuv_format = use_yuv_format

    def _apply_mask_on_img(self, img, img_mask):
        if self.use_yuv_format:
            img[0, img_mask] = 0
            img[1, img_mask] = 128
            img[2, img_mask] = 128
        else:
            img[:, img_mask] = 0

    def __call__(self, data: Mapping):
        if "img_mask" in data:
            img_masks = data.pop("img_mask")
            assert len(img_masks) == len(data["imgs"][0])
            for img_mask, img in zip(img_masks, data["imgs"][0]):
                self._apply_mask_on_img(img, img_mask)
        return data

    def __repr__(self):
        return "ApplyMaskOnImg"


@OBJECT_REGISTRY.register
class ANCSetTemporalClearFlag(object):
    """
    Set temporal clear flag according to the sub_clip_index and pack_name.

    Args:
        clr_mode: the mode of clear temporal flag.
            Only "clip", "pack" and None supported now.
            None: always clear temporal model.
            "clip": clear temporal model if a new clip start.
            "pack": clear temporal model if a new pack start.
    """

    def __init__(
        self,
        clr_mode: Optional[str] = None,
    ):
        assert clr_mode in [None, "clip", "pack"]
        self.clr_mode = clr_mode
        self.his_pack_name = None

    def __call__(self, clip_data: Dict):

        if self.clr_mode is None:
            clip_data["temporal_clr_flag"] = True
            return clip_data
        elif self.clr_mode == "clip":
            # clear temporal model if a new clip start.
            assert "sample_split_index" in clip_data
            if clip_data["sample_split_index"] == 0:
                clip_data["temporal_clr_flag"] = True
            else:
                clip_data["temporal_clr_flag"] = False
        else:
            if "pack_names" in clip_data:
                _key = "pack_names"
            elif "pack_path" in clip_data:
                _key = "pack_path"
            else:
                raise ValueError(
                    f"clip data not obtain key pack_names or pack_path, keys: {list(clip_data.keys())}"  # noqa
                )
            # clear temporal model if a new pack start.
            if (
                self.his_pack_name is None
                or clip_data[_key] != self.his_pack_name
            ):
                if _key == "pack_path":
                    clip_data["temporal_clr_flag"] = True
                elif (_key == "pack_names") and (
                    clip_data["sample_split_index"] == 0
                ):
                    clip_data["temporal_clr_flag"] = True
                else:
                    clip_data["temporal_clr_flag"] = False
            else:
                clip_data["temporal_clr_flag"] = False

            self.his_pack_name = clip_data[_key]

        if "sample_split_index" in clip_data:
            clip_data.pop("sample_split_index")

        return clip_data


class BgrToYuv444V2GPU(torch.nn.Module):
    """
    GPU Processer of BgrToYuv444V2.

    BgrToYuv444V2 implements by calling rgb2centered_yuv functions which
    has been verified to get the basically same YUV output on J5.

    Args:
        rgb_input : The input is rgb input or bgr.
        swing: "studio" for YUV studio swing (Y: -112~107,
                U, V: -112~112).
                "full" for YUV full swing (Y, U, V: -128~127).
                default is "full"
    """

    def __init__(self, rgb_input: bool = False, swing: str = "full"):
        super(BgrToYuv444V2GPU, self).__init__()
        self.rgb_input = rgb_input
        assert swing in ["studio", "full"]
        self.swing = swing

        if self.swing == "studio":
            weight = [[66, 129, 25], [-38, -74, 112], [112, -94, -18]]
            offset = [16, 128, 128]
        else:
            weight = [[77, 150, 29], [-43, -84, 127], [127, -106, -21]]
            offset = [0, 128, 128]
        if not self.rgb_input:
            weight = [w[::-1] for w in weight]
        self.weight = (
            torch.tensor(weight).to(torch.float32).unsqueeze(2).unsqueeze(3)
        )
        self.offset = (
            torch.tensor(offset).to(torch.float32).reshape(1, 3, 1, 1)
        )
        self.bias = torch.ones(3) * 128

    def _convert_color(self, input):
        input = input.to(torch.float32)
        if self.weight.device != input.device:
            self.weight = self.weight.to(input.device)
            self.bias = self.bias.to(input.device)
            self.offset = self.offset.to(input.device)
        res = torch.nn.functional.conv2d(input, self.weight, self.bias) / 256
        res += self.offset
        return res.to(torch.int32).float()

    def _to_yuv(self, img):
        centered_yuv = self._convert_color(img)
        return centered_yuv

    def __call__(self, data):
        image = data["img"] if isinstance(data, dict) else data
        ndim = image.ndim
        if ndim == 3:
            image = torch.unsqueeze(image, 0)
        if image.dtype is not torch.uint8:
            image = image.to(dtype=torch.uint8)
        if image.shape[1] == 6:
            image1 = self._convert_color(image[:, :3])
            image2 = self._convert_color(image[:, 3:])
            image = torch.cat((image1, image2), dim=1)
        else:
            image = self._convert_color(image)
        if ndim == 3:
            image = image[0]
        if isinstance(data, dict):
            data["img"] = image
            return data
        else:
            return image


@OBJECT_REGISTRY.register
class ANCConvertToYuv(torch.nn.Module):
    """Convert a 'rgb format tensor' to 'yuv format tensor'.

    Args:
    rgb_input: whether the input image is in rgb format.

    """

    def __init__(
        self,
        rgb_input: bool = True,
    ):
        super(ANCConvertToYuv, self).__init__()
        self.convert_img_names = ["img", "side_img", "round_img", "narrow_img"]
        self.yuv_converter = BgrToYuv444V2GPU(rgb_input=rgb_input)

    def _convert_to_yuv(self, data: Sequence):
        if isinstance(data, Sequence):
            return [self._convert_to_yuv(item) for item in data]
        else:
            return self.yuv_converter(data)

    def __call__(self, data: Mapping):

        for key in self.convert_img_names:
            if key in data:
                data[key] = self._convert_to_yuv(data[key])
        return data


@OBJECT_REGISTRY.register
class ANCBevDiscreteTargetGenerator(object):
    """Generate ground truth labels for bev discrete objects.

    Discrete objects includes arrow, crosswalk, junction, etc.

    .. attention:: (TODO ben.hu, xiangyu.li) Integrate
        BevDiscreteTargetGenerator into Bev3dTargetGenerator.

    Args:
        num_classes: Number of classes.
        bev_size: bev size.(order is (h,w)).
        vcs_range: vcs range,
            order is bottom,right,top,left.
        category2id_map: A mapping from raw category (str)
            to training category_id which starts from 0.
        name2label: A mapping from multimodal lmdb data name to label.
        name2group: A mapping from multimodal lmdb data name to group,
            some names belong to one group. for example, crosswalk and arrow
            belong to det, stopline belong to om.
        smallobj_dilate_cfg: small object dilate config.

            .. code-block:: none

                {
                    "kernel": tuple, dilate kernel.

                    "dilate_cls_id": list, the category id that needs to be dilated.  # noqa

                    "iterations": int, number of iternation.

                }

        vis_mask_vcs_range_cfg: Raw offline vismask image size
            as key, and corresponding vcs range as value.
        use_vis_mask: Whether use vismask in model training.
        max_objs: Maximum number of objects used in the training and
            inference. This number should be large enough
        vis_mask_expand_scale: Expand target box scale used to
            calculate visible rate. Default 1.0.
        vcs_bbox_area_thresh: filter the bbox area less than
            this number. default is set to 0.
        visible_threshold: The threshold of gt's visibility.
        ignore_miss_cls: Ignore the class with missing annotation in gt.
        ignore_miss_clsid: Ignored cls_id with missing annotation in gt.
        ego_ignore_range: Ego range
            (bottom, right, top, left) to be ignored, (-0.6, -0.5, 2.0, 0.5)
            recommended based on the minimum tire diameter,
            wheelbase and track.
        class_group_matcher: Class object for pair match of
            different classes.
        valid_vcs_range_percls: VCS range for every class. Different
            categories of vcs range in the same task may be different.
        lidar_filter: The nearby point cloud information packed
            in the gt Other_info used to filter the shelted points and
            filter for different classes.

            .. code-block:: none

                Other_info: {

                    "CementColumn": [[

                        "data_vcs": list, vcs coordinates of the
                            four points of the rotate box.

                        "nearest_point_cloud_vcs": list, the nearest point
                            cloud vcs coordinates of each of the four points.

                        "nearest_point_distance_m": list, the distance from
                            each of the four points to the nearest point cloud.

                    ], ....],

                }

        N_steps_PSC_rot: nums of steps to decode the heading
            angle by PSC. Refer to
            https://horizonrobotics.feishu.cn/docx/EwYEdysIBodB69xn6Mdc7HkSnif.
        psc_phase_factor: Factor number to align the rotation period
            of the taraget object to 2pi.
            Refer to https://arxiv.org/abs/2211.06368.
            rotation period of arrow is 2*pi, so psc_phase_factor is 1.
            rotation period of crosswalk is pi, so psc_phase_factor is 2.
        return_ignore_obj: Whether return ignore instances or not.
        res_key: result key of output.
    """

    def __init__(
        self,
        num_classes: int,
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        category2id_map: Mapping,
        name2label: Mapping,
        name2group: Mapping,
        smallobj_dilate_cfg: Optional[dict] = None,
        vis_mask_vcs_range_cfg: Optional[Sequence[float]] = None,
        use_vis_mask: bool = False,
        max_objs: int = 100,
        vis_mask_expand_scale: float = 1.0,
        vcs_bbox_area_thresh: float = 0.0,
        visible_threshold: float = 0.0,
        ignore_miss_cls: bool = False,
        ignore_miss_clsid: int = None,
        ego_ignore_range: Optional[Sequence[float]] = None,
        class_group_matcher: Optional[object] = None,
        valid_vcs_range_percls: Optional[dict] = None,
        lidar_filter: Optional[object] = None,
        psc_phase_factor: int = 1,
        N_steps_PSC_rot: Optional[int] = None,
        return_ignore_obj: bool = True,
        res_key: str = "bev_discrete_obj",
    ):
        self.num_classes = num_classes
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.category2id_map = category2id_map
        self.max_objs = max_objs
        self.vcs_bbox_area_thresh = vcs_bbox_area_thresh
        self.name2label = name2label
        self.name2group = name2group
        self.smallobj_dilate_cfg = smallobj_dilate_cfg
        self.visible_threshold = visible_threshold

        self.ignore_miss_cls = ignore_miss_cls
        self.ignore_miss_clsid = ignore_miss_clsid

        self.ego_ignore_range = ego_ignore_range
        self.vis_mask_expand_scale = vis_mask_expand_scale

        self.m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x
        if use_vis_mask:
            assert (
                vis_mask_vcs_range_cfg is not None
            ), "pls define vismask vcsrange in cconfig first."
            self.vis_mask_vcs_range_cfg = vis_mask_vcs_range_cfg
        self.use_vis_mask = use_vis_mask

        self.class_group_matcher = class_group_matcher
        self.lidar_filter = lidar_filter
        self.return_ignore_obj = return_ignore_obj
        self.res_key = res_key

        # get valid bev size for every categroy
        self.valid_bev_range = {}
        for cls_id in range(self.num_classes):
            self.valid_bev_range[cls_id] = (
                0,
                0,
                self.bev_size[0],
                self.bev_size[1],
            )
        self.valid_vcs_range_percls = valid_vcs_range_percls
        if valid_vcs_range_percls:
            for cate, valid_range in valid_vcs_range_percls.items():
                cls_id = category2id_map[cate]
                # top, left, bottom, right
                self.valid_bev_range[cls_id] = get_roi_vcs_range_box(
                    bev_size, vcs_range, valid_range
                )

        # psc setting
        self.psc_phase_factor = psc_phase_factor
        self.N_steps_PSC_rot = N_steps_PSC_rot
        if self.N_steps_PSC_rot is not None:
            assert N_steps_PSC_rot >= 3
            self.rot_encode_size = self.N_steps_PSC_rot
        else:
            self.rot_encode_size = 2  # cos, sin

    @staticmethod
    def _vcs2bev_coord(vcs_pt, vcs_range, spatial_resolution):
        u = int((vcs_range[3] - vcs_pt[1]) / spatial_resolution[1] + 0.5)
        v = int((vcs_range[2] - vcs_pt[0]) / spatial_resolution[0] + 0.5)
        return [u, v]

    def get_vertices_from_bev_box(self, wh, ct, yaw):
        """Get vertices of bounding box with format (w,h,cx,cy, yaw) in vcs.

        Args:
            wh: [w, h] in vcs along yaw direction and vertical to yaw.
            ct: [cx, cy], coordinate of center of bounding box in vcs.
            yaw: yaw of bounding box in vcs.

        Returns: coordinate (x, y) in vcs, 4 vertices of a bbox.

        """
        w, h = wh
        ctx, cty = ct

        p0 = [0.5 * w, 0.5 * h, 1]
        p1 = [0.5 * w, -0.5 * h, 1]
        p2 = [-0.5 * w, -0.5 * h, 1]
        p3 = [-0.5 * w, 0.5 * h, 1]
        points = np.array([p0, p1, p2, p3])
        R = np.array(
            [
                [np.cos(yaw), np.sin(yaw), ctx],
                [-np.sin(yaw), np.cos(yaw), cty],
                [0, 0, 1],
            ]
        )
        points = R.dot(points.T).T
        return points[:, :2]

    def vcs2bev_coord(self, pt):
        """Convert point pt(x, y) in vcs to bev u-v coordinate.

        Args:
            pt: [x, y] coordinate of point in vcs

        Returns: (u, v) in bev u-v coordinate

        """
        return self._vcs2bev_coord(pt, self.vcs_range, self.m_perpixel)

    def get_rotated_gaussian2D(self, wh, yaw, alpha=0.54, sigma=None):
        """Create Gaussian heatmap with rotation.

        Args:
            wh: [w, h] in vcs along yaw direction and vertical to yaw.
            yaw: yaw of bounding box in vcs.
            alpha: heatmap scaling.
            sigma: Gaussian kernel standard deviation.

        Returns: heatmap of bbox.

        """
        radius = (np.array([*wh]) / 2 * alpha).astype("int32")
        if sigma is None:
            sigma = np.array(
                [[(radius[0] * 2 + 1) / 6, 0], [0, (radius[1] * 2 + 1) / 6]]
            )
        else:
            sigma = np.array(sigma)
        var = sigma * sigma

        R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        var_rotated = np.matmul(R, np.matmul(var, R.transpose()))

        points = self.get_vertices_from_bev_box(wh, (0, 0), yaw)
        w_rotated = points[:, 0].max() - points[:, 0].min()
        h_rotated = points[:, 1].max() - points[:, 1].min()

        rw, rh = int(w_rotated / 2), int(h_rotated / 2)
        x, y = np.ogrid[-rw : rw + 1, -rh : rh + 1]
        xy_grid = np.array(np.meshgrid(x, y), dtype=np.float32).transpose(
            1, 2, 0
        )

        heatmap = np.exp(
            -np.sum(
                np.matmul(xy_grid, np.linalg.inv(var_rotated)) * xy_grid,
                axis=-1,
            )
            * 0.5
        )
        heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0
        assert heatmap.max() == 1, "heatmap value max must be 1!"
        return heatmap

    def get_annotations(self, lines, label, timestamp, name, rate_visible):
        """Create annotation from lmdb data.

        Args:
            lines: multiple sides of the target box.
            label: label of the target.
            timestamp: the timestamp when the target was generated.
            name: the label name of the target.
            rate_visible: visibility of the target box.
                When target is fully visible, valuse is 1.
                To distinguish from the visibility of 1,
                default value is 2.
        Return:
            anno: the annotations of the target.

        """
        points = np.array(lines).reshape((-1, 2)).astype(np.float32)
        (cx, cy), _, _ = cv2.minAreaRect(points)
        if name == "arrows":
            # arrow has orientation property, the range of yaw is [-180~180],
            # the orientation is first point to second point. the yaw of others
            # is the angle down from the positive x-axis.
            length = np.sqrt(np.sum(np.square(points[0] - points[1])))
            width = np.sqrt(np.sum(np.square(points[2] - points[3])))
            tmp = points[1] - points[0]
            yaw = np.arctan2(tmp[1], tmp[0])
        else:
            if len(points) == 4 and name == "stoplines":
                length = np.sqrt(np.sum(np.square(points[1] - points[0])))
                width = 0.5
                tmp = points[1] - points[0]
            else:
                rect = cv2.minAreaRect(points)
                points = cv2.boxPoints(rect)
                length = np.sqrt(np.sum(np.square(points[0] - points[1])))
                width = np.sqrt(np.sum(np.square(points[1] - points[2])))
                if length > width:
                    tmp = points[1] - points[0]
                else:
                    length, width = width, length
                    tmp = points[2] - points[1]
            yaw = np.arctan2(tmp[1], tmp[0])
            if yaw > np.pi / 2:
                yaw = -(np.pi - yaw)
            elif yaw < -np.pi / 2:
                yaw = np.pi + yaw

        anno = {
            "dimension": [length, width],
            "yaw": yaw,
            "location": [cx, cy],
            "area": length * width,
            "cls_id": label,
            "ignore": False,
            "rate_visible": rate_visible,
            "timestamp": timestamp,
            "lines": [lines],
            "cls_name": name,
        }
        return anno

    def gt_filter(self, annotations, vis_mask, expand_scale):
        """Filter annotation by computing visible rate.

        Args:
            annotations: the annotations of the target.
            vis_mask: visible area mask.
            expand_scale: Expand target box scale used to
            calculate iou. Default 1.0.

        """
        for anno in annotations:
            vcs_loc = anno["location"]
            vcs_dim = anno["dimension"]
            yaw = anno["yaw"]
            bev_loc = self.vcs2bev_coord(vcs_loc)
            bev_dim = [
                (vcs_dim[0] / self.m_perpixel[0]) * expand_scale,
                (vcs_dim[1] / self.m_perpixel[1]) * expand_scale,
            ]
            box = self.get_vertices_from_bev_box(
                bev_dim, bev_loc, yaw + np.pi / 2
            )
            box = np.int0(box)
            bbox_mask = np.zeros(self.bev_size, np.uint8)
            cv2.drawContours(bbox_mask, [box], -1, (1), thickness=-1)
            Area_ori = np.sum(bbox_mask > 0)
            bbox_visible = bbox_mask[vis_mask > 0]
            Area_visible = np.sum(bbox_visible)
            rate_visible = Area_visible / (Area_ori + 1e-6)
            anno["rate_visible"] = rate_visible
            # As for arrow, if rate_visible < 0.7 and head region is invisible,
            # the subtype of the arrow can not be recognized.
            # Refer to https://horizonrobotics.feishu.cn/wiki/KjzgwHlclihBOMkdhQNc6MT7n9f # noqa
            if anno["cls_name"] == "arrows" and rate_visible < 0.7:
                lines = anno["lines"][0]
                vcs_front_pts = [lines[1][:2], lines[1][2:]]
                bev_front_pts = [
                    self.vcs2bev_coord(pt) for pt in vcs_front_pts
                ]
                for pt in bev_front_pts:
                    if not is_in_range(
                        pt, (0, 0, self.bev_size[1] - 1, self.bev_size[0] - 1)
                    ):
                        anno["ignore"] = True
                        break
                    if vis_mask[pt[1], pt[0]] == 0:
                        anno["ignore"] = True
                        break

    def draw_ignore_mask(self, ignore_mask, lines):
        """Draw ignore mask according to ignore area.

        Args:
            ignore_mask: ignore area mask.
            lines: multiple sides of the target box.

        """
        for line in lines:
            if len(line) == 0:
                continue
            if "pts" in line:
                line = line["pts"]
            pt_csv = np.array(line).reshape((-1, 2)).astype(np.float32)
            rect = cv2.minAreaRect(pt_csv)
            box = cv2.boxPoints(rect)

            box_bev = []
            for i in range(len(box)):
                box_bev.append(self.vcs2bev_coord(box[i]))
            box_bev = np.array(box_bev)
            cv2.drawContours(ignore_mask, [box_bev], -1, (1), thickness=-1)

    def get_class_ignore_mask(
        self, objs: dict, ignore_miss_clsid: int
    ) -> np.ndarray:
        """Generate class ignore mask for missing label.

        Annotate the missing label with ignore, due to annotations from
        different periods for bev roadmarking task. For example, stopline
        is annotated in om, but om not annotate other labels like
        crosswalk, diamond, ... Currently, crosswalk and stopline in the
        same classification head, could cause incorrect negative samples.
        Current data annotation contains the following formats: 1) om;
        2) det + om; 3) det; 4) direct with label keys such as crosswalks,
        stoplines.

        Args:
            objs: includes the annotation of objects, include
                keys such as om, det to group different types
                of annotation.
            ignore_miss_clsid: class id to be ignored.

        Returns:
            (1, 1, num_classes)
        """
        bev_discobj_cls_ignore = np.zeros(
            (self.num_classes,), dtype=np.float32
        )
        if "om" in objs:
            if "det" not in objs:
                # data only annotated with stoplines, ignore other classes
                bev_discobj_cls_ignore += 1
                bev_discobj_cls_ignore[ignore_miss_clsid] = 0
            if "freespace" in objs:
                # data only annotated with crosswalk, ignore junction
                bev_discobj_cls_ignore[ignore_miss_clsid] = 1
        elif "det" in objs:
            # data annotated without stoplines, ignore stopline class
            bev_discobj_cls_ignore[ignore_miss_clsid] = 1
        else:
            # previous data only annotated with crosswalk and stopline
            bev_discobj_cls_ignore += 1
            if (
                "crosswalks" in self.name2label
                and "stoplines" in self.name2label
            ):
                cross_label_name = self.name2label["crosswalks"]
                cross_label_id = self.category2id_map[cross_label_name]
                bev_discobj_cls_ignore[cross_label_id] = 0

                stopline_label_name = self.name2label["stoplines"]
                stopline_label_id = self.category2id_map[stopline_label_name]
                bev_discobj_cls_ignore[stopline_label_id] = 0

        return bev_discobj_cls_ignore.reshape((1, 1, -1))

    @staticmethod
    def _get_selected_anno(annotations, select_idxs, select_cls_id):
        """Get annotations by specified index.

        Args:
            annotations: annotations of objects.
            select_idxs: specified index of annotations.
            select_cls_id: class id of selected annotaions.
        """
        num_select_idx = len(select_idxs)
        select_anno_info = {
            "yaw": np.zeros((num_select_idx,), dtype=np.float32),
            "vcs_dim": np.zeros((num_select_idx, 2), dtype=np.float32),
            "vcs_loc": np.zeros((num_select_idx, 2), dtype=np.float32),
            "cls_id": select_cls_id,
            "points": [],
        }
        for idx in range(num_select_idx):
            anno = annotations[select_idxs[idx]]
            select_anno_info["yaw"][idx] = anno["yaw"]
            select_anno_info["vcs_dim"][idx] = np.array(anno["dimension"])
            select_anno_info["vcs_loc"][idx] = np.array(anno["location"])
            lines = anno["lines"][0]
            points = np.array(lines).reshape((-1, 2)).astype(np.float32)
            select_anno_info["points"].append(points)
        return select_anno_info

    def _get_class_group_anno(
        self,
        annotations,
        source_anno_idx,
        target_anno_idx,
        source_cls_idx,
        target_cls_idx,
    ):

        source_anno_info = self._get_selected_anno(
            annotations, source_anno_idx, source_cls_idx
        )
        target_anno_info = self._get_selected_anno(
            annotations, target_anno_idx, target_cls_idx
        )

        return source_anno_info, target_anno_info

    def _draw_bev_discobj_instmap(self, vcs_dim, yaw, instance_id):
        """Get instance heatmap which different object has different id.

        Args:
            vcs_dim: wh in vcs coordinate.
            vcs_loc: vcs center location.
            yaw: rotation angle.
            instance_id: instance id of selected object.
        """

        bev_dim = [
            vcs_dim[0] / self.m_perpixel[0],
            vcs_dim[1] / self.m_perpixel[1],
        ]
        # get object gaussian heatmap, size is (h, w)
        insert_bev_hm = self.get_rotated_gaussian2D(
            bev_dim, yaw + np.pi / 2, alpha=1.0
        )
        # create instance mask with value of defined instace id
        # by generated gaussian map
        insert_bev_inst_mask = (insert_bev_hm > 0).astype(
            np.float32
        ) * instance_id
        return insert_bev_inst_mask

    def _group_class_map(
        self,
        annotations,
        source_anno_idx,
        target_anno_idx,
        source_cls_idx,
        target_cls_idx,
    ):
        """Represent mathced objects in the form of instance map.

        The size of instance map is (h, w, num_classes). On the channel of
        specified category, different intergers represent different instance,
        that is instance id. The matching relationship between specified
        categories is determined by instance id, the instance id difference
        between matched objects is 1. For example, number 1 in instance map
        represents the first instance of the source category, then number 2
        in instance map is the instance of the target category that matches
        it.

        Args:
            annotations: annotations of objects.
            source_anno_idx: index of the source category annotations.
            target_anno_idx: index of the target category annotations.
            source_cls_idx: class id of the source category.
            target_cls_idx: class id of the target category.
        """
        bev_discobj_instances = np.zeros(
            (*self.bev_size, self.num_classes), dtype=np.float32
        )
        if not all(
            (len(annotations), len(source_anno_idx), len(target_anno_idx))
        ):
            return bev_discobj_instances
        # get annotations of classes needed to match
        info_source, info_target = self._get_class_group_anno(
            annotations,
            source_anno_idx,
            target_anno_idx,
            source_cls_idx,
            target_cls_idx,
        )
        matched_pair = self.class_group_matcher(info_source, info_target)
        # if no match exists, return
        if len(matched_pair) == 0:
            return bev_discobj_instances
        inst_cnt = 1
        for pair_idx in matched_pair:
            m, n = pair_idx
            # get the matched object instance map mask, size is (h, w)
            insert_bev_inst_mask = self._draw_bev_discobj_instmap(
                info_target["vcs_dim"][n], info_target["yaw"][n], inst_cnt
            )
            bev_ct_int = self.vcs2bev_coord(info_target["vcs_loc"][n])
            # get the matched object instance target, size
            # is (h, w, num_classes)
            draw_heatmap(
                bev_discobj_instances[:, :, info_target["cls_id"]],
                insert_bev_inst_mask,
                bev_ct_int,
            )
            # get the matched object instance map mask, size is (h, w)
            insert_bev_inst_mask = self._draw_bev_discobj_instmap(
                info_source["vcs_dim"][m], info_source["yaw"][m], inst_cnt + 1
            )
            bev_ct_int = self.vcs2bev_coord(info_source["vcs_loc"][m])
            # get the matched object instance target, size
            # is (h, w, num_classes)
            draw_heatmap(
                bev_discobj_instances[:, :, info_source["cls_id"]],
                insert_bev_inst_mask,
                bev_ct_int,
            )
            # the instance id of the next match
            inst_cnt += 2
        return bev_discobj_instances

    def __call__(self, data: dict) -> dict:
        """Generate bev discrete object labels, e.g., arrow, crosswalk, etc.

        Args:
            data: The dict contains at least annotations

        """
        assert "gt_bev_discrete_raw" in data
        objs = data.get("gt_bev_discrete_raw")
        auxiliary_info = objs.get("Other_info", None)
        if self.ignore_miss_cls:
            assert 0 <= self.ignore_miss_clsid < self.num_classes
            bev_discobj_cls_ignore = self.get_class_ignore_mask(
                objs, self.ignore_miss_clsid
            )  # (1, 1, num_classes)

        ignore_mask = np.zeros(self.bev_size, np.uint8)
        timestamp = str(int(data["timestamp"][0] * 1000))
        annotations = []
        for name, label in self.name2label.items():
            if name not in objs:
                proto_objs = objs.get(self.name2group[name], {name: []})
            else:
                proto_objs = objs
            for obj in proto_objs.get(name, []):
                # filted by lidar
                if self.lidar_filter and auxiliary_info is not None:
                    obj = self.lidar_filter(obj, auxiliary_info, name)
                _obj = obj.copy()
                # The value is 1 when the vismask is used and
                #   the target is fully visible.
                # The value is 2 when no vismask is used or
                #   lidar is used to mask but the target box is visible
                rate_visible = (
                    _obj.get("rate_visible", 2.0)
                    if isinstance(_obj, dict)
                    else 2
                )
                if "type" in obj:
                    label = obj["type"]
                if "pts" in obj:
                    obj = obj["pts"]
                if len(obj) == 0:
                    continue
                # filter label not in target category
                if label not in self.category2id_map:
                    continue
                cls_id = int(self.category2id_map.get(label, 0))
                anno = self.get_annotations(
                    obj, cls_id, timestamp, name, rate_visible
                )
                if name == "sod3d":
                    anno["dimension"].append(_obj["height"])
                annotations.append(anno)

            ignore_objs = proto_objs.get("ignores", None)
            if ignore_objs:
                self.draw_ignore_mask(ignore_mask, ignore_objs)
                # ignore objs for validation.
                # Set cls_id as -2 for labeled ignore instances.
                if self.return_ignore_obj:
                    for obj in ignore_objs:
                        _obj = obj.copy()
                        if "pts" in obj:
                            obj = obj["pts"]
                        if len(obj) == 0:
                            continue
                        anno = self.get_annotations(
                            obj, -2, timestamp, name, 2
                        )
                        anno["ignore"] = True
                        if name == "sod3d":
                            anno["dimension"].append(_obj["height"])
                        annotations.append(anno)

        if self.use_vis_mask:
            assert "bev_occlusion_mask" in data
            vis_mask_ = data.get("bev_occlusion_mask")
            if vis_mask_:
                vis_mask = compute_vismask(
                    vis_mask_,
                    visible_flag=[1],
                    occlusion_flag=[0, 1, 2, 3],
                )
                vismask_size = vis_mask.shape[:2]
                assert vismask_size in self.vis_mask_vcs_range_cfg
                vis_mask_vcs_range = self.vis_mask_vcs_range_cfg[vismask_size]
                # If it exceeds the vismask range, it will be
                # treated as visible
                vis_mask = get_roi_resize_data(
                    vis_mask,
                    vis_mask_vcs_range,
                    self.vcs_range,
                    self.bev_size,
                    pad_index=1,
                )
                self.gt_filter(
                    annotations, vis_mask, self.vis_mask_expand_scale
                )
                ignore_mask[vis_mask == 0] = 0

        bev_discobj_hm = np.zeros(
            (*self.bev_size, self.num_classes), dtype=np.float32
        )
        dim_num = 3 if "sod3d" in self.name2label else 2
        bev_discobj_wh = np.zeros(
            (*self.bev_size, dim_num), dtype=np.float32
        )  # (height, width[, length])

        bev_discobj_rot = np.zeros(
            (*self.bev_size, self.rot_encode_size), dtype=np.float32
        )  # cos, sin
        bev_discobj_ct_offset = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # (u, v)
        bev_discobj_weight_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev_discobj_ignore = np.array(ignore_mask, dtype=np.float32)

        # used for eval
        vcs_discobj_loc = np.zeros((self.max_objs, 2), dtype=np.float32)
        vcs_discobj_dim = np.zeros((self.max_objs, dim_num), dtype=np.float32)
        vcs_discobj_cls = np.zeros((self.max_objs), dtype=np.float32) - 1
        vcs_discobj_ignore = np.zeros((self.max_objs,), dtype=np.float32)
        vcs_discobj_rate_visible = (
            np.zeros((self.max_objs,), dtype=np.float32) - 1
        )
        vcs_discobj_yaw = np.zeros((self.max_objs,), dtype=np.float32)

        if self.class_group_matcher:
            assert hasattr(self.class_group_matcher, "class_match_cfg")
            class_match_cfg = self.class_group_matcher.class_match_cfg
            assert (
                "source_class_id" in class_match_cfg
                and "target_class_id" in class_match_cfg
            )
            source_cls_id = class_match_cfg["source_class_id"]
            target_cls_id = class_match_cfg["target_class_id"]
            assert (
                source_cls_id < self.num_classes
                and target_cls_id < self.num_classes
            )
            source_anno_idx = []
            target_anno_idx = []

        count_id = 0
        for anno_idx, anno in enumerate(annotations):  # noqa [B007]
            if anno["ignore"]:
                continue
            if count_id >= self.max_objs:
                break
            if (
                anno["area"] < self.vcs_bbox_area_thresh
                or anno.get("rate_visible", 2) < self.visible_threshold
            ):
                anno["ignore"] = True
                continue

            cls_id = anno["cls_id"]
            vcs_yaw = anno["yaw"]  # in rad
            vcs_dim = anno["dimension"]  # (w, h), w along yaw direction
            vcs_loc = anno["location"]  # (x, y) in vcs

            if self.ego_ignore_range is not None:
                if (
                    self.ego_ignore_range[0]
                    <= vcs_loc[0]
                    <= self.ego_ignore_range[2]
                    and self.ego_ignore_range[1]
                    <= vcs_loc[1]
                    <= self.ego_ignore_range[-1]
                ):
                    anno["ignore"] = True
                    continue

            bev_ct_int = self.vcs2bev_coord(vcs_loc)
            bev_ct = (
                (self.vcs_range[3] - vcs_loc[1]) / self.m_perpixel[1],
                (self.vcs_range[2] - vcs_loc[0]) / self.m_perpixel[0],
            )  # (u, v)
            bev_dim = [
                vcs_dim[0] / self.m_perpixel[0],
                vcs_dim[1] / self.m_perpixel[1],
            ]
            # get class valid bev range
            cls_valid_bev_range = self.valid_bev_range[cls_id]
            # top, left, bottom, right
            lt_y, lt_x, rb_y, rb_x = cls_valid_bev_range
            if is_in_range(
                (bev_ct_int[1], bev_ct_int[0]),
                (lt_y, lt_x, rb_y - 1, rb_x - 1),
            ):
                insert_bev_hm = self.get_rotated_gaussian2D(
                    bev_dim, vcs_yaw + np.pi / 2, alpha=1.0
                )

                if self.N_steps_PSC_rot is not None:
                    phase_shift_targets = tuple(
                        np.cos(
                            self.psc_phase_factor * vcs_yaw
                            + 2 * np.pi * x / self.N_steps_PSC_rot
                        )
                        for x in range(self.N_steps_PSC_rot)
                    )
                else:
                    phase_shift_targets = (np.cos(vcs_yaw), np.sin(vcs_yaw))

                if self.smallobj_dilate_cfg:
                    if self.smallobj_dilate_cfg.get("hm_size_floor", None):
                        hm_alpha = self.smallobj_dilate_cfg.get(
                            "hm_alpha", 1.0
                        )
                        hm_size_floor = self.smallobj_dilate_cfg[
                            "hm_size_floor"
                        ]
                        hm_wh = (
                            max(bev_dim[0], hm_size_floor[0]),
                            max(bev_dim[1], hm_size_floor[1]),
                        )
                        insert_bev_hm = self.get_rotated_gaussian2D(
                            hm_wh, vcs_yaw + np.pi / 2, alpha=hm_alpha
                        )
                    dilated_cls_id = self.smallobj_dilate_cfg["dilate_cls_id"]
                    assert (
                        dilated_cls_id
                    ), f"please make {self.smallobj_dilate_cfg} is right."
                    dilated_kernel = self.smallobj_dilate_cfg["kernel"]
                    iterations = self.smallobj_dilate_cfg["iterations"]
                    if cls_id in dilated_cls_id:
                        kernel = np.ones(
                            (dilated_kernel[0], dilated_kernel[1]), np.uint8
                        )
                        insert_bev_hm = cv2.dilate(
                            insert_bev_hm, kernel, iterations
                        )
                insert_bev_wh = insert_bev_hm.shape[:2][::-1]
                insert_bev_reg_map_list = [
                    get_reg_map(insert_bev_wh, vcs_dim),
                    get_reg_map(insert_bev_wh, phase_shift_targets),
                    get_ctoff_map(insert_bev_wh, bev_ct),
                ]
                bev_reg_map_list = [
                    bev_discobj_wh,
                    bev_discobj_rot,
                    bev_discobj_ct_offset,
                ]
                draw_heatmap(
                    bev_discobj_hm[:, :, cls_id], insert_bev_hm, bev_ct_int
                )
                draw_heatmap(
                    bev_discobj_weight_hm,
                    insert_bev_hm,
                    bev_ct_int,
                    bev_reg_map_list,
                    insert_bev_reg_map_list,
                )

                count_id += 1
            else:
                anno["ignore"] = True

            if self.class_group_matcher:
                # filter ignore object
                if not anno["ignore"]:
                    if anno["cls_id"] == source_cls_id:
                        source_anno_idx.append(anno_idx)
                    if anno["cls_id"] == target_cls_id:
                        target_anno_idx.append(anno_idx)

        if self.class_group_matcher:
            # get matched group instance target,
            # size is (h, w, num_classes)
            bev_group_insts = self._group_class_map(
                annotations,
                source_anno_idx,
                target_anno_idx,
                source_cls_id,
                target_cls_id,
            )

        gt_bev_discrete_obj = {
            "bev_discobj_hm": bev_discobj_hm,
            "bev_discobj_wh": bev_discobj_wh,
            "bev_discobj_rot": bev_discobj_rot,
            "bev_discobj_ct_offset": bev_discobj_ct_offset,
            "bev_discobj_weight_hm": bev_discobj_weight_hm,
            "bev_discobj_ignore": bev_discobj_ignore,
        }
        if self.ignore_miss_cls:
            gt_bev_discrete_obj.update(
                {"bev_discobj_cls_ignore": bev_discobj_cls_ignore}
            )

        if self.class_group_matcher:
            gt_bev_discrete_obj.update(
                {"bev_discobj_instances": bev_group_insts}
            )

        # get per class valid range mask
        if self.valid_vcs_range_percls:
            cls_valid_range_mask = np.zeros(
                (*self.bev_size, self.num_classes), dtype=np.float32
            )
            for cls_id in range(self.num_classes):
                cls_valid_bev_range = self.valid_bev_range[cls_id]
                lt_y, lt_x, rb_y, rb_x = cls_valid_bev_range
                cls_valid_range_mask[lt_y:rb_y, lt_x:rb_x, cls_id] = 1.0
            gt_bev_discrete_obj.update(
                {"bev_discobj_cls_valid_mask": cls_valid_range_mask}
            )

        for i, anno in enumerate(annotations):
            if i >= self.max_objs:
                break
            vcs_discobj_loc[i] = anno["location"]
            vcs_discobj_dim[i] = anno["dimension"]
            vcs_discobj_ignore[i] = 1 if anno["ignore"] else 0
            vcs_discobj_rate_visible[i] = anno["rate_visible"]
            vcs_discobj_cls[i] = anno["cls_id"]
            vcs_discobj_yaw[i] = anno["yaw"]

        annos_bev_discrete_obj = {
            "vcs_discobj_loc": vcs_discobj_loc,
            "vcs_discobj_wh": vcs_discobj_dim,
            "vcs_discobj_ignore": vcs_discobj_ignore,
            "vcs_discobj_rate_visible": vcs_discobj_rate_visible,
            "vcs_discobj_cls": vcs_discobj_cls,
            "vcs_discobj_yaw": vcs_discobj_yaw,
        }
        data[f"gt_{self.res_key}"] = gt_bev_discrete_obj
        data[f"annos_{self.res_key}"] = annos_bev_discrete_obj
        return data

    def __repr__(self):
        return "ANCBevDiscreteTargetGenerator"


@OBJECT_REGISTRY.register
class ANCBevDiscreteWithClsTargetGenerator(ANCBevDiscreteTargetGenerator):
    """Generate ground truth labels for bev discrete objects.

    Use heatmap and class maps to replace haeatmap for each class.

    """

    def __init__(self, **kwargs):
        super(ANCBevDiscreteWithClsTargetGenerator, self).__init__(**kwargs)

    def __call__(self, data: dict) -> dict:
        """Generate bev discrete object labels, e.g., arrow, crosswalk, etc.

        Heatmap for all instances and class maps to do classification.
        Args:
            data: Type is ndarray
                The dict contains at least annotations

        """
        assert "gt_bev_discrete_raw" in data
        objs = data.get("gt_bev_discrete_raw")
        auxiliary_info = objs.get("Other_info", None)
        if self.ignore_miss_cls:
            assert 0 <= self.ignore_miss_clsid < self.num_classes
            bev_discobj_cls_ignore = self.get_class_ignore_mask(
                objs, self.ignore_miss_clsid
            )  # (1, 1, num_classes)

        ignore_mask = np.zeros(self.bev_size, np.uint8)
        timestamp = str(int(data["timestamp"][0] * 1000))
        annotations = []
        for name, label in self.name2label.items():
            if name not in objs:
                proto_objs = objs.get(self.name2group[name], {name: []})
            else:
                proto_objs = objs

            for obj in proto_objs.get(name, []):
                # filted by lidar
                if self.lidar_filter and auxiliary_info is not None:
                    obj = self.lidar_filter(obj, auxiliary_info, name)
                # The value is 1 when the vismask is used and
                #   the target is fully visible.
                # The value is 2 when no vismask is used or
                #   lidar is used to mask but the target box is visible
                rate_visible = (
                    obj.get("rate_visible", 2.0)
                    if isinstance(obj, dict)
                    else 2
                )
                if "type" in obj:
                    label = obj["type"]
                if "pts" in obj:
                    obj = obj["pts"]
                if len(obj) == 0:
                    continue
                cls_id = int(self.category2id_map.get(label, 0))
                anno = self.get_annotations(
                    obj, cls_id, timestamp, name, rate_visible
                )
                annotations.append(anno)

            ignore_objs = proto_objs.get("ignores", None)
            if ignore_objs:
                self.draw_ignore_mask(ignore_mask, ignore_objs)
                # ignore objs for validation.
                # Set cls_id as -2 for labeled ignore instances.
                if self.return_ignore_obj:
                    for obj in ignore_objs:
                        if "pts" in obj:
                            obj = obj["pts"]
                        if len(obj) == 0:
                            continue
                        anno = self.get_annotations(
                            obj, -2, timestamp, name, 2
                        )
                        anno["ignore"] = True
                        annotations.append(anno)

        if self.use_vis_mask:
            assert "bev_occlusion_mask" in data
            vis_mask_ = data.get("bev_occlusion_mask")
            if vis_mask_:
                vis_mask = compute_vismask(
                    vis_mask_,
                    visible_flag=[1],
                    occlusion_flag=[0, 1, 2, 3],
                )
                vismask_size = vis_mask.shape[:2]
                assert vismask_size in self.vis_mask_vcs_range_cfg
                vis_mask_vcs_range = self.vis_mask_vcs_range_cfg[vismask_size]
                # If it exceeds the vismask range, it will be
                # treated as visible
                vis_mask = get_roi_resize_data(
                    vis_mask,
                    vis_mask_vcs_range,
                    self.vcs_range,
                    self.bev_size,
                    pad_index=1,
                )
                self.gt_filter(
                    annotations, vis_mask, self.vis_mask_expand_scale
                )
                ignore_mask[vis_mask == 0] = 0

        bev_discobj_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev_discobj_hm_cls = np.zeros(
            (*self.bev_size, self.num_classes), dtype=np.float32
        )
        bev_discobj_wh = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # w, h
        bev_discobj_rot = np.zeros(
            (*self.bev_size, self.rot_encode_size), dtype=np.float32
        )  # cos, sin
        bev_discobj_ct_offset = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # (u, v)
        bev_discobj_weight_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev_discobj_ignore = np.array(ignore_mask, dtype=np.float32)

        # used for eval
        vcs_discobj_loc = np.zeros((self.max_objs, 2), dtype=np.float32)
        vcs_discobj_dim = np.zeros((self.max_objs, 2), dtype=np.float32)
        vcs_discobj_cls = np.zeros((self.max_objs), dtype=np.float32) - 1
        vcs_discobj_ignore = np.zeros((self.max_objs,), dtype=np.float32)
        vcs_discobj_rate_visible = (
            np.zeros((self.max_objs,), dtype=np.float32) - 1
        )
        vcs_discobj_yaw = np.zeros((self.max_objs,), dtype=np.float32)

        count_id = 0
        for anno_idx, anno in enumerate(annotations):  # noqa [B007]
            if anno["ignore"]:
                continue
            if count_id >= self.max_objs:
                break
            if (
                anno["area"] < self.vcs_bbox_area_thresh
                or anno.get("rate_visible", 2) < self.visible_threshold
            ):
                anno["ignore"] = True
                continue

            cls_id = anno["cls_id"]
            vcs_yaw = anno["yaw"]  # in rad
            vcs_dim = anno["dimension"]  # (w, h), w along yaw direction
            vcs_loc = anno["location"]  # (x, y) in vcs

            if self.ego_ignore_range is not None:
                if (
                    self.ego_ignore_range[0]
                    <= vcs_loc[0]
                    <= self.ego_ignore_range[2]
                    and self.ego_ignore_range[1]
                    <= vcs_loc[1]
                    <= self.ego_ignore_range[-1]
                ):
                    anno["ignore"] = True
                    continue

            bev_ct_int = self.vcs2bev_coord(vcs_loc)
            bev_ct = (
                (self.vcs_range[3] - vcs_loc[1]) / self.m_perpixel[1],
                (self.vcs_range[2] - vcs_loc[0]) / self.m_perpixel[0],
            )  # (u, v)
            bev_dim = [
                vcs_dim[0] / self.m_perpixel[0],
                vcs_dim[1] / self.m_perpixel[1],
            ]
            # get class valid bev range
            cls_valid_bev_range = self.valid_bev_range[cls_id]
            # top, left, bottom, right
            lt_y, lt_x, rb_y, rb_x = cls_valid_bev_range
            if is_in_range(
                (bev_ct_int[1], bev_ct_int[0]),
                (lt_y, lt_x, rb_y - 1, rb_x - 1),
            ):
                insert_bev_hm = self.get_rotated_gaussian2D(
                    bev_dim, vcs_yaw + np.pi / 2, alpha=1.0
                )

                if self.N_steps_PSC_rot is not None:
                    phase_shift_targets = tuple(
                        np.cos(
                            self.psc_phase_factor * vcs_yaw
                            + 2 * np.pi * x / self.N_steps_PSC_rot
                        )
                        for x in range(self.N_steps_PSC_rot)
                    )
                else:
                    phase_shift_targets = (np.cos(vcs_yaw), np.sin(vcs_yaw))

                if self.smallobj_dilate_cfg:
                    if self.smallobj_dilate_cfg.get("hm_size_floor", None):
                        hm_alpha = self.smallobj_dilate_cfg.get(
                            "hm_alpha", 1.0
                        )
                        hm_size_floor = self.smallobj_dilate_cfg[
                            "hm_size_floor"
                        ]
                        hm_wh = (
                            max(bev_dim[0], hm_size_floor[0]),
                            max(bev_dim[1], hm_size_floor[1]),
                        )
                        insert_bev_hm = self.get_rotated_gaussian2D(
                            hm_wh, vcs_yaw + np.pi / 2, alpha=hm_alpha
                        )
                    dilated_cls_id = self.smallobj_dilate_cfg["dilate_cls_id"]
                    assert (
                        dilated_cls_id
                    ), f"please make {self.smallobj_dilate_cfg} is right."
                    dilated_kernel = self.smallobj_dilate_cfg["kernel"]
                    iterations = self.smallobj_dilate_cfg["iterations"]
                    if cls_id in dilated_cls_id:
                        kernel = np.ones(
                            (dilated_kernel[0], dilated_kernel[1]), np.uint8
                        )
                        insert_bev_hm = cv2.dilate(
                            insert_bev_hm, kernel, iterations
                        )
                insert_bev_wh = insert_bev_hm.shape[:2][::-1]
                insert_bev_reg_map_list = [
                    get_reg_map(insert_bev_wh, vcs_dim),
                    get_reg_map(insert_bev_wh, 1.0),
                    get_reg_map(insert_bev_wh, phase_shift_targets),
                    get_ctoff_map(insert_bev_wh, bev_ct),
                ]
                bev_reg_map_list = [
                    bev_discobj_wh,
                    bev_discobj_hm_cls[:, :, cls_id],
                    bev_discobj_rot,
                    bev_discobj_ct_offset,
                ]
                draw_heatmap(bev_discobj_hm, insert_bev_hm, bev_ct_int)
                draw_heatmap(
                    bev_discobj_weight_hm,
                    insert_bev_hm,
                    bev_ct_int,
                    bev_reg_map_list,
                    insert_bev_reg_map_list,
                )

                count_id += 1
            else:
                if self.valid_vcs_range_percls:
                    # only one channel for all category, ignore mask
                    # is shared by all category
                    self.draw_ignore_mask(ignore_mask, anno["lines"])
                anno["ignore"] = True

        gt_bev_discrete_obj = {
            "bev_discobj_hm": bev_discobj_hm,
            "bev_discobj_hm_cls": bev_discobj_hm_cls,
            "bev_discobj_wh": bev_discobj_wh,
            "bev_discobj_rot": bev_discobj_rot,
            "bev_discobj_ct_offset": bev_discobj_ct_offset,
            "bev_discobj_weight_hm": bev_discobj_weight_hm,
            "bev_discobj_ignore": bev_discobj_ignore,
        }
        if self.ignore_miss_cls:
            gt_bev_discrete_obj.update(
                {"bev_discobj_cls_ignore": bev_discobj_cls_ignore}
            )

        if self.valid_vcs_range_percls:
            # get per class valid range mask
            cls_valid_range_mask = np.zeros(
                (*self.bev_size, self.num_classes), dtype=np.float32
            )
            for cls_id in range(self.num_classes):
                cls_valid_bev_range = self.valid_bev_range[cls_id]
                lt_y, lt_x, rb_y, rb_x = cls_valid_bev_range
                cls_valid_range_mask[lt_y:rb_y, lt_x:rb_x, cls_id] = 1.0
            gt_bev_discrete_obj.update(
                {"bev_discobj_cls_valid_mask": cls_valid_range_mask}
            )

        for i, anno in enumerate(annotations):
            if i > self.max_objs:
                break
            vcs_discobj_loc[i] = anno["location"]
            vcs_discobj_dim[i] = anno["dimension"]
            vcs_discobj_ignore[i] = 1 if anno["ignore"] else 0
            vcs_discobj_rate_visible[i] = anno["rate_visible"]
            vcs_discobj_cls[i] = anno["cls_id"]
            vcs_discobj_yaw[i] = anno["yaw"]

        annos_bev_discrete_obj = {
            "vcs_discobj_loc": vcs_discobj_loc,
            "vcs_discobj_wh": vcs_discobj_dim,
            "vcs_discobj_ignore": vcs_discobj_ignore,
            "vcs_discobj_rate_visible": vcs_discobj_rate_visible,
            "vcs_discobj_cls": vcs_discobj_cls,
            "vcs_discobj_yaw": vcs_discobj_yaw,
        }
        data[f"gt_{self.res_key}"] = gt_bev_discrete_obj
        data[f"annos_{self.res_key}"] = annos_bev_discrete_obj
        return data

    def __repr__(self):
        return "ANCBevDiscreteWithClsTargetGenerator"


@OBJECT_REGISTRY.register
class ANCOnlineMappingTargetGenerator(object):
    """Generate gound truth labels for online mapping.

    Args:
        head_groups: module output head infos.
        vcs_range: vcs range.(order is (bottom,right,top,left)).
        out_size: model output size.
        view_bev_size: the size of view bev results.
        roi_weight_cfg: all configurations related to ROI weights.
        roadedge_occ_cfg: config of filting gt with roadedge.
        map_dilate: the dialte radius of generating om gt.
        dilate_weight: weight dict of different dilate region.
            the dict keys must be ("background", "dilated", "foreground"),
            the dilated region means background near the foreground
            eg. {"background": 0.5, "dilated" : 1.0, "foreground": 1.5}
        split_close_roadedge: whether split U-shape or close roadedge.
        view_cols: the column of view img.
        view_sub_head: if view sub task head.
        visualize_output_dir: gt visualization output dir.
        global_ignore_index: global ignore class index.
        assign_near_instance_cfg: near instance gt assign config.
        merge_instance_cfg: cfg of merge instances that are physically
            connected but have different properties.
        block_warp_padding: order is (left,right,up,bottom).
        merge_crosspoint: If true, om segments can transfer to
            CrosspointTargetGenerator to filter outlier crosspoints.
            https://horizonrobotics.feishu.cn/wiki/XFtHw5c52im6pakNpEacCNmUn7d.
        om_short_filter_cfg: filter short instance config.
        om_horizontal_cfg: horizontal instance process cfg.
        merge_solid_dash_cfg: process solid dash line cfg.
        gt_name: gt name.

    Notice: please refer to this document for more details
        https://horizonrobotics.feishu.cn/docx/Az33dhC6dopgZ2x9rkQcHJhensg
    """

    def __init__(
        self,
        head_groups: dict,
        vcs_range: Sequence[float],
        out_size: Sequence[int],
        view_bev_size: Sequence[int],
        roi_weight_cfg: dict = None,
        roadedge_occ_cfg: dict = None,
        map_dilate: int = 0,
        dilate_weight: Dict[str, float] = None,
        split_close_roadedge: bool = False,
        view_cols: int = 3,
        view_sub_head: bool = True,
        visualize_output_dir: str = None,
        global_ignore_index: int = -1,
        assign_near_instance_cfg: dict = None,
        merge_instance_cfg: dict = None,
        block_warp_padding: Sequence[int] = None,
        merge_crosspoint: bool = False,
        om_short_filter_cfg: dict = None,
        om_horizontal_cfg: dict = None,
        merge_solid_dash_cfg: dict = None,
        gt_name: str = "om_target",
    ):
        self.out_size = out_size
        self.head_groups = head_groups
        self.vcs_range = vcs_range
        self.view_bev_size = view_bev_size
        self.roi_weight_cfg = roi_weight_cfg
        self.roadedge_occ_cfg = roadedge_occ_cfg
        self.map_dilate = map_dilate
        self.dilate_weight = dilate_weight
        if dilate_weight is None:
            self.map_dilate = 0
        self.split_close_roadedge = split_close_roadedge
        self.global_ignore_index = global_ignore_index
        self.assign_near_instance_cfg = assign_near_instance_cfg
        self.merge_instance_cfg = merge_instance_cfg
        self.om_horizontal_cfg = om_horizontal_cfg
        self.merge_solid_dash_cfg = merge_solid_dash_cfg

        self.view_cols = view_cols
        self.view_sub_head = view_sub_head
        self.visualize_output_dir = visualize_output_dir
        self.use_view = self.visualize_output_dir is not None
        self.gt_name = gt_name
        self.block_warp_padding = block_warp_padding
        self.merge_crosspoint = merge_crosspoint
        self.om_short_filter_cfg = om_short_filter_cfg

    def __call__(self, data):
        image_files = join_path(data["pack_dir"], data["img_paths"])
        data["image_files"] = image_files
        gt_online_mapping_ori = data["gt_online_mapping"]

        meta_info = None
        if self.use_view:
            if self.gt_name == "om_target":
                meta_info = data["meta_info"]
            elif self.gt_name == "om_target_small":
                meta_info = data["meta_info_small"]

        data[self.gt_name] = get_gt_online_mapping(
            gt_online_mapping_ori,
            head_groups=self.head_groups,
            out_size=self.out_size,
            origin_imgs=data["origin_imgs"] if self.use_view else None,
            roi_weight_cfg=self.roi_weight_cfg,
            roadedge_occ_cfg=self.roadedge_occ_cfg,
            meta_info=meta_info,
            map_dilate=self.map_dilate,
            dilate_weight=self.dilate_weight,
            view_bev_size=self.view_bev_size,
            vcs_range=self.vcs_range,
            split_close_roadedge=self.split_close_roadedge,
            image_files=image_files,
            view_cols=self.view_cols,
            view_sub_head=self.view_sub_head,
            visualize_output_dir=self.visualize_output_dir,
            global_ignore_index=self.global_ignore_index,
            assign_near_instance_cfg=self.assign_near_instance_cfg,
            merge_instance_cfg=self.merge_instance_cfg,
            block_warp_padding=self.block_warp_padding,
            merge_crosspoint=self.merge_crosspoint,
            om_short_filter_cfg=self.om_short_filter_cfg,
            om_horizontal_cfg=self.om_horizontal_cfg,
            merge_solid_dash_cfg=self.merge_solid_dash_cfg,
        )
        return data

    def __repr__(self):
        return "OnlineMappingTargetGenerator"


@OBJECT_REGISTRY.register
class ANCCrossPointTargetGenerator(object):
    """Generate gound truth labels for bev-crosspoint.

    Args:
        stride: Crosspoint output stride.
        target_categorys: Crosspoint output category.
        cls_group_map: Output categories of each group.
        vcs_range: Vcs range, order is (bottom,right,top,left).
        use_vismask: If use vismask.
        vismask_vcs_range_cfg: Raw offline vismask image size
            as key, and corresponding vcs range as value.
        bev_size: Bev fusion feature size, order is (h,w).
        ignore_index: Ignored cls_id in train and val stage.
        ignore_fileter_length: Ignore areas with shorter sides
            less than this value will be filtered, unit(m).
        gaussian_diameter: Gaussian diameter for different categories pts,
            order is (w,h).
        gaussian_sigma: Gaussian variance.
        use_om_aux_loss: If use online mapping auxiliary loss.
        cpts_use_occ: If use roadedge occlusion to filter cpts.
        gt_name: return gt name.
    """

    def __init__(
        self,
        stride: int,
        target_categorys: Dict[str, int],
        cls_group_map: Dict[str, Sequence[int]],
        vcs_range: Sequence[float],
        use_vismask: bool,
        vismask_vcs_range_cfg: Dict[str, Sequence[float]],
        bev_size: Sequence[float],
        ignore_index: int,
        ignore_filter_length: float,
        gaussian_diameter: Dict[str, Tuple[float]],
        gaussian_sigma: Sequence[float],
        use_om_aux_loss: bool = False,
        cpts_use_occ: bool = False,
        gt_name: str = "crosspoint_target",
    ):
        self.stride = stride
        self.target_categorys = target_categorys
        self.cls_group_map = cls_group_map
        self.vcs_range = vcs_range
        self.use_vismask = use_vismask
        self.bev_size = bev_size
        self.ignore_index = ignore_index
        self.ignore_filter_length = ignore_filter_length
        self.gaussian_diameter = gaussian_diameter
        self.gaussian_sigma = gaussian_sigma
        self.cpts_use_occ = cpts_use_occ
        for group in gaussian_diameter.keys():
            gaussian_name = f"gaussian_{group}"
            gaussian = get_gaussian2D(
                np.array(gaussian_diameter[group]),
                alpha=1,
                sigma=self.gaussian_sigma,
            )
            setattr(self, gaussian_name, gaussian)

        self.gt_name = gt_name
        self.use_om_aux_loss = use_om_aux_loss
        self.bev_h, self.bev_w = self.bev_size
        self.out_h = self.bev_h // self.stride
        self.out_w = self.bev_w // self.stride
        (
            self.bottom_offset,
            self.right_offset,
            self.top_offset,
            self.left_offset,
        ) = self.vcs_range
        self.scope_h = self.top_offset - self.bottom_offset
        self.scope_w = self.left_offset - self.right_offset
        self.meter_per_out_pixel_h = self.scope_h / self.out_h
        self.meter_per_out_pixel_w = self.scope_w / self.out_w

        if self.use_vismask:
            assert (
                vismask_vcs_range_cfg is not None
            ), "pls define vismask_vcs_range_cfg in config first."

            self.vismask_vcs_range_cfg = vismask_vcs_range_cfg
            self.last_vismask_img_h = -1
            self.last_vismask_img_w = -1
            self.vismask_crop_size = None

        kernal_width = int(
            (ignore_filter_length // self.meter_per_out_pixel_w // 2) * 2 + 1
        )
        kernal_height = int(
            (ignore_filter_length // self.meter_per_out_pixel_h) // 2 * 2 + 1
        )
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernal_width, kernal_height)
        )

    def __call__(self, data):
        image_files = join_path(data["pack_dir"], data["img_paths"])
        data["image_files"] = image_files
        data[self.gt_name] = self.get_crosspoint_gt(data)
        return data

    def __repr__(self):
        return "CrossPointTargetGenerator"

    def draw_ignore_mask(self, gt_point):
        ignore_contours = []
        ignore_regions = gt_point["ignores"]
        for region in ignore_regions:
            contour = []
            contour_pts = [pt[:2] for pt in region] + [region[-1][2:4]]
            for pt in contour_pts:
                positive_x = int(
                    (self.top_offset - pt[0]) / self.meter_per_out_pixel_h
                )
                positive_y = int(
                    (self.left_offset - pt[1]) / self.meter_per_out_pixel_w
                )
                contour.append([positive_y, positive_x])
            ignore_contours.append(np.array(contour))

        ignore_mask = np.zeros((self.out_h, self.out_w), np.uint8)
        ignore_mask = cv2.drawContours(
            ignore_mask,
            ignore_contours,
            -1,
            (self.ignore_index),
            -1,
        )
        ignore_mask = cv2.morphologyEx(
            ignore_mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1
        )
        return ignore_mask

    def _get_vismask(self, data):
        assert (
            "bev_occlusion_mask" in data
        ), "Make sure bev_occlusion_mask in data when use vismask."
        vis_mask = data["bev_occlusion_mask"]
        vis_mask = compute_vismask(
            vis_mask,
            visible_flag=[1, 2],
            occlusion_flag=[0, 1, 2, 3],
        )
        if vis_mask is None:
            return None
        vismask_img_h, vismask_img_w = vis_mask.shape

        # cal crop size at first and vismask size changed
        if (
            self.vismask_crop_size is None
            or vismask_img_h != self.last_vismask_img_h
            or vismask_img_w != self.last_vismask_img_w
        ):
            self.last_vismask_img_h = vismask_img_h
            self.last_vismask_img_w = vismask_img_w
            vismask_vcs_range = self.vismask_vcs_range_cfg[vis_mask.shape]
            self.vismask_crop_size = list(
                get_roi_vcs_range_box(
                    vis_mask.shape, vismask_vcs_range, self.vcs_range
                )
            )

            # when vcs range bigger than vismask range, need clip
            self.vismask_crop_size[0::2] = np.clip(
                self.vismask_crop_size[0::2], 0, vismask_img_h
            )
            self.vismask_crop_size[1::2] = np.clip(
                self.vismask_crop_size[1::2], 0, vismask_img_w
            )

            # update vismask vcs range after crop
            self.crop_vismask_vcs_range = list(vismask_vcs_range).copy()
            self.crop_vismask_vcs_range[:2] = np.maximum(
                vismask_vcs_range[:2], self.vcs_range[:2]
            )
            self.crop_vismask_vcs_range[2:] = np.minimum(
                vismask_vcs_range[2:], self.vcs_range[2:]
            )

            # calculate valid vismask size on output layer
            self.range_with_vismask = get_roi_vcs_range_box(
                (self.out_h, self.out_w),
                self.vcs_range,
                self.crop_vismask_vcs_range,
            )

        lt_x, lt_y, rb_x, rb_y = self.vismask_crop_size
        vis_mask = vis_mask[lt_x:rb_x, lt_y:rb_y]

        h, w = vis_mask.shape
        final_vismask_h = (
            self.range_with_vismask[2] - self.range_with_vismask[0]
        )
        final_vismask_w = (
            self.range_with_vismask[3] - self.range_with_vismask[1]
        )
        if (h, w) != (final_vismask_h, final_vismask_w):
            vis_mask = cv2.resize(
                vis_mask,
                (final_vismask_w, final_vismask_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return vis_mask

    def _cal_gt_stats(self, gt_stats, gt_point, group, occ_segments=None):
        """Cal cls, x, y gt value of each branch."""
        gaussian = getattr(self, f"gaussian_{group}")
        point_gt_origin = gt_point[group]
        for pt in point_gt_origin:
            cls = int(pt[0])
            x = pt[1]
            y = pt[2]
            if (
                x > self.top_offset
                or x <= self.bottom_offset
                or y > self.left_offset
                or y <= self.right_offset
            ):
                continue
            # cpts_use_occ is set to true and roadedge segments exist
            if self.cpts_use_occ and occ_segments is not None:
                status = 0
                if occ_segments.shape[0] != 0:
                    start_pt = (0, 0)
                    end_pt = (x, y)
                    intersections = multi_line_segment_intersection(
                        occ_segments, start_pt, end_pt
                    )
                    if np.any(intersections):
                        status = 1
                if status == 1:
                    continue

            positive_x = int(
                (self.top_offset - x) / self.meter_per_out_pixel_h
            )
            positive_y = int(
                (self.left_offset - y) / self.meter_per_out_pixel_w
            )
            draw_heatmap(
                gt_stats[group]["cls"][0, cls],
                gaussian,
                (positive_y, positive_x),
            )
            gt_stats[group]["x"][0, positive_x, positive_y] = (
                (self.top_offset - x) % self.meter_per_out_pixel_h
            ) / self.meter_per_out_pixel_h
            gt_stats[group]["y"][0, positive_x, positive_y] = (
                (self.left_offset - y) % self.meter_per_out_pixel_w
            ) / self.meter_per_out_pixel_w

    def get_crosspoint_gt(self, data):
        gt_point = data["gt_bev_crosspoint"]
        occ_segments = None
        if "om_target" in data:
            occ_segments = (
                data["om_target"].pop("occ_segments")
                if "occ_segments" in data["om_target"]
                else None
            )
        gt_stats = {}
        for key in self.cls_group_map.keys():
            gt_stats[key] = {}
            gt_stats[key]["x"] = np.zeros((1, self.out_h, self.out_w))
            gt_stats[key]["y"] = np.zeros((1, self.out_h, self.out_w))

            # data of version1.0 have no changepoints label
            # set ignore index to cls and skip x,y gt calculate
            if key == "changepoints" and gt_point["data_version"] == "v1.0":
                gt_stats[key]["cls"] = (
                    np.ones(
                        (
                            1,
                            len(self.cls_group_map[key]),
                            self.out_h,
                            self.out_w,
                        ),
                        dtype=np.float,
                    )
                    * self.ignore_index
                )
                continue
            else:
                gt_stats[key]["cls"] = np.zeros(
                    (1, len(self.cls_group_map[key]), self.out_h, self.out_w),
                    dtype=np.float,
                )

            self._cal_gt_stats(gt_stats, gt_point, key, occ_segments)

        # get ignore mask
        ignore_mask = self.draw_ignore_mask(gt_point)

        # set unvisable area as ignore
        if self.use_vismask:
            vis_mask = self._get_vismask(data)
            if vis_mask is not None:
                ignore_mask_with_vismask = ignore_mask[
                    self.range_with_vismask[0] : self.range_with_vismask[2],
                    self.range_with_vismask[1] : self.range_with_vismask[3],
                ]
                ignore_mask_with_vismask[vis_mask == 0] = self.ignore_index

        for key in self.cls_group_map.keys():
            ignore_mask_group = np.tile(
                np.expand_dims(ignore_mask, 0),
                (len(self.cls_group_map[key]), 1, 1),
            )
            gt_stats[key]["cls"][
                0, ignore_mask_group == self.ignore_index
            ] = self.ignore_index
        return gt_stats


@OBJECT_REGISTRY.register
class GetCalibParams(object):
    """Get calib params from annotations.

    Args:
        camera_view_names: Each of view name.
        view_shapes: Each of View image size.
        return_extra: Whether to return extra info of transformation matrix.
        homo_noise: The params for generating noise extrinsic
            parameter matrix for T_vcs2cam.
    """

    def __init__(
        self,
        camera_view_names: Optional[Sequence[str]],
        view_shapes: Optional[Dict[str, Sequence[int]]] = None,
        return_extra: bool = False,
        homo_noise: Dict[str, Any] = None,
    ):
        self.camera_view_names = camera_view_names
        self.view_shapes = view_shapes
        self.return_extra = return_extra
        if homo_noise is not None:
            assert "noise_range" in homo_noise, "missing noise_range"
            self.homo_noise = np.array(homo_noise["noise_range"])
            self.noise_type = homo_noise.get("noise_type", None)
            self.noise_view = homo_noise.get("noise_view", None)
            self.noise_prob = homo_noise.get("noise_prob", None)
        else:
            self.homo_noise = None
            self.noise_type = None
            self.noise_view = None
            self.noise_prob = 0.0

    def __call__(self, data):
        if random.random() < self.noise_prob:
            if self.noise_view == "random":
                noise_view = random.choice(self.camera_view_names)
            else:
                noise_view = self.noise_view
        else:
            noise_view = None

        label = data["meta"]
        calib_params = _get_calib_params_from_anno(
            label,
            self.camera_view_names,
            self.view_shapes,
            return_extra=self.return_extra,
            homo_noise=self.homo_noise,
            noise_type=self.noise_type,
            noise_view=noise_view,
        )
        data["calib_params"] = calib_params

        return data


@OBJECT_REGISTRY.register
class ANCHisOdometryCollector(object):
    """Collect history odometry information for target task.

    Args:
        max_his_odo_len: max length of odo info. If the history odo info is not
            enough, the first odo info will be repeated.

    """

    def __init__(
        self,
        max_his_odo_len: int = 1,
    ):
        self.max_his_odo_len = max_his_odo_len
        self.his_odo = []

    def __call__(self, data):

        assert "odo_info" in data
        if data["temporal_clr_flag"]:
            self.his_odo = []
        self.his_odo.append(data["odo_info"][np.newaxis, ...])
        self.his_odo = self.his_odo[-self.max_his_odo_len :]
        odo_info = copy.deepcopy(self.his_odo)
        if len(odo_info) < self.max_his_odo_len:
            odo_info = [odo_info[0]] * (
                self.max_his_odo_len - len(odo_info)
            ) + odo_info
        odo_info = np.concatenate(odo_info, axis=0)
        data["odo_info"] = torch.from_numpy(odo_info).float().unsqueeze(0)

        return data
