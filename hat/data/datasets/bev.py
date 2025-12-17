# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import fcntl
import json
import logging
import math
import os
import random
from typing import Mapping, Sequence, Tuple

import cv2
import fsspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation

from hat.callbacks.save_eval_results.save_consistency_result import align_ceil
from hat.core.affine import get_vcs2bev_img_mat
from hat.core.bev_elevation_utils import (
    HOMO_PAD_VALUE,
    decimal_div,
    reformat_homo_info,
    reformat_meta_info,
)
from hat.core.virtual_camera.camera_base import CameraParam, SetCameraParam
from hat.core.virtual_camera.utils import transform_euler2rotMat
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.trace import get_part_dict

try:
    from pytorch3d.transforms import euler_angles_to_matrix
except ImportError:
    euler_angles_to_matrix = None
import hashlib

logger = logging.getLogger(__name__)

__all__ = ["HomoGenerator"]

VALID_Z = 0.05
EPS = 1e-7


class HomoNoise(object):
    """Generate noise extrinsic parameter matrix for T_vcs2cam.

    There are three types of noise, containing:
    (a) 'random_cam', the independent random noise for each cameras. A noise
        matrix T_cam2noisecam is generated for each camera and a result matrix
        T_vcs2noisecam is returned.
    (b) 'specific_cam', the noise is a specific value compared to (a).
    (c) 'random_vcs', the random noise on vcs-xy plane for a bumpy vehicle.
        A noise matrix T_vcs2noisevcs acting on vcs-xy plane is generated and
        all cameras share this matrix. T_noisevcs2cam for each camera is
        returned.

    Args:
        noise_value: It is the exact value of generated noise when the noise
            type is 'specific_cam'. When the noise type is 'random_cam' or
            'random_vcs', it is the upper bound value of generated random
            noises. The format is (roll, pitch, yaw, x, y, z), the units are
            degrees and meters.
        noise_view_names: The camera view names with noise. It must be a
            subset of `camera_view_names` in "HomoGenerator".
        noise_type: The type of the noise should be in ['random_cam',
            'specific_cam', 'random_vcs'], representing the above three cases
            (a), (b), (c) respectively.
    """

    def __init__(
        self,
        noise_value: Sequence[float],
        noise_view_names: Sequence[str],
        noise_type: str = "random_cam",
    ):
        assert noise_type in ["random_cam", "specific_cam", "random_vcs"]
        self.noise_value = np.array(noise_value)
        self.noise_type = noise_type
        self.noise_view_names = noise_view_names
        self.view2noise = {}

    def _get_random_noise(self):
        """Get random noise according to the upper bound."""
        # angle -> radian
        d_rpy_limit = self.noise_value[:3] / 180 * math.pi
        d_xyz_limit = self.noise_value[3:6]

        # get random noise
        d_rpy = [
            random.random() * rad_limit * 2 - rad_limit
            for rad_limit in d_rpy_limit
        ]
        d_xyz = [
            random.random() * trans_limit * 2 - trans_limit
            for trans_limit in d_xyz_limit
        ]

        return d_rpy, d_xyz

    def get_noise_matrix(self, T_vcs2cam, homo_noise):
        """Get T_vcs2cam with noise."""
        if self.noise_type == "random_cam":
            noise_rpy, noise_xyz = homo_noise
            R_cam2noisecam = Rotation.from_euler(
                "xyz", noise_rpy, degrees=False
            ).as_matrix()

            T_cam2noisecam = np.zeros((4, 4))
            T_cam2noisecam[:3, :3] = R_cam2noisecam
            T_cam2noisecam[:3, 3] = noise_xyz
            T_cam2noisecam[3, 3] = 1
            T_vcs2noisecam = T_cam2noisecam @ T_vcs2cam

            return T_vcs2noisecam
        elif self.noise_type == "specific_cam":
            T_cam2vcs = np.linalg.inv(T_vcs2cam)
            T_camnew2cam = np.array(
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
            )
            T_camnew2vcs = T_cam2vcs @ T_camnew2cam

            R_camnew2vcs = T_camnew2vcs[:3, :3]
            xyz = T_camnew2vcs[:3, 3]

            # add noise
            T_noisecamnew2vcs = np.zeros((4, 4))
            noise_rpy, noise_xyz = homo_noise
            R_noise = Rotation.from_euler(
                "xyz", noise_rpy, degrees=False
            ).as_matrix()
            R_noisecam2vcs = R_noise @ R_camnew2vcs

            T_noisecamnew2vcs[:3, :3] = R_noisecam2vcs
            T_noisecamnew2vcs[:3, 3] = xyz + noise_xyz
            T_noisecamnew2vcs[3, 3] = 1

            T_cam2vcs = T_noisecamnew2vcs @ np.linalg.inv(T_camnew2cam)
            return np.linalg.inv(T_cam2vcs)
        elif self.noise_type == "random_vcs":
            T_cam2vcs = np.linalg.inv(T_vcs2cam)

            T_vcs2ground = np.zeros((4, 4))
            noise_rpy, noise_xyz = homo_noise
            R_vcs2ground = Rotation.from_euler(
                "xyz", noise_rpy, degrees=False
            ).as_matrix()
            T_vcs2ground[:3, :3] = R_vcs2ground
            T_vcs2ground[:3, 3] = noise_xyz
            T_vcs2ground[3, 3] = 1

            T_cam2ground = T_vcs2ground @ T_cam2vcs

            return np.linalg.inv(T_cam2ground)

    def get_homo_noise(self):
        if self.noise_type == "random_vcs":
            bump_noise = self._get_random_noise()
            for view in self.noise_view_names:
                self.view2noise[view] = bump_noise
        elif self.noise_type == "random_cam":
            for view in self.noise_view_names:
                random_noise = self._get_random_noise()
                self.view2noise[view] = random_noise
        else:
            rpy = list(self.noise_value[:3] / 180 * math.pi)
            xyz = list(self.noise_value[3:])
            for view in self.noise_view_names:
                self.view2noise[view] = rpy, xyz


class HomoGenerator(object):  # noqa: D205,D400
    """Generate homography by online compute or offline load.

    Args:
        calib_path: path of lidar and camera file, include calibration
            params files like,

            camera_front_left.yaml,
            camera_front_right.yaml,
            camera_front.yaml,
            camera_rear_left.yaml,
            camera_rear_right.yaml,
            camera_rear.yaml,
            log_trans,
            calibration.json,

            note that homography can be computed by the order:
            (1) attribute.json, generated by hde-m (https://gitlab.hobot.cc/ptd/experimental/alg/Honeybadger/horizondataengine/hde-manager), #noqa
                contains all camera intrinsic/extrinsic params, lidar2vcs and lidar2cam for compute homography
            (2) calibration.json file, calibration.json includes all camera
                intrinsic/extrinsic params, lidar2vcs and lidar2cam for
                compute homography. Refer to aidi data system for details of
                calibration.json file of different vehicle, e.g.,
                http://data.aidi.hobot.cc/data-label/common-page?commonPageName=query-list&showMenu=1&conditions=%7B%7D&commonPageTab=default&queryConditionTab=Visual #noqa
            (3) xxx.ymal + log_trans, each camera's intrinsic/extrinsic, and lidar2vcs can get from log_trans

            If all the above files provided, use attribute.json to compute homography.


        homo_path: path of homograph matrix npy file.
        calib_para: dict of calib para for cameras, each camera's para include
            T_vcs2cam, K, d_coef, calculated from the pack parameters.
        spatial_resolution: bev spatial resolution.(unit is meters)
        vcs_range: visbile range of bev, (bottom, right, top, left)
            in order.
        camera_view_names: sub directory name of each view,
        task_camera_view_names: the camera_view_name of the task.
            NOTE: 注意这里的task_camera_view_names 和 camera_view_names不是一个概念,
            task_camera_view_names表示实际任务决定的视角，
            camera_view_names是该dataset数据实际拥有的视角数量。
            task_camera_view_names可以是camera_view_names的子集或者差集，在
            此函数里面作为reformat_meta_info的输入进行meta_info的处理。
            e.g. camera_view_names是7v的dataset。
                 task_camera_view_names可以是7v，11v，6v等。
        per_view_shape: original img shape correspond to each camera view.  # noqa
        use_distorted_offset: return distorted homography offset or undistorted.  # noqa
        H_persp_view_scale : Indicates the scaling factor of the perspective view size in homo_mat.
            Assuming that the perspective view size corresponding to the homo_mat matrix is (Hpers, Wpers),
            now change it to (Hpers*scale, Wpers*scale), then set H_persp_view_scale to scale.
            During BEV training, H_persp_view_scale is usually set to 1/4,
            because the 1/4 scale features of BEV stage1 are used for spatial fusion
        homo_transforms: dict of homo transforms,
        Describing the changes from origin img to the input of Bevfusion.  # noqa
            each veiw in Homo tansform is like this
        .. code-block:: JSON

        {
            Resize: (H,W)
            Crop: (top,left,height,width)
            Pad: (left, top, right and bottom)
        }
        vcs_plane_heights: the heights of multi-height vcs planes.
            Default to (0, ), representing the single ground plane.
        return_offset_in_meta_info: whether return the offset in
            meta-info.
            if `True`: the meta_info will calculate the dist
            offset by `get_homo_offset` and return it on meta_info.
            if `False`: `homogenerator` will not cal offset on cpu,
            and the homo_offset will not in meta_info, offset will
            be calculated on gpu, this can save memory.
        offset_save_path: path to save homo_offset. if set
            `offset_save_path`, the homo_offset will be generate
            and save the this path, the name is determined by all
            the parameters.
    """

    def __init__(
        self,
        calib_path: str,
        homo_path: str,
        spatial_resolution: Sequence[float],
        vcs_range: Sequence[float],
        camera_view_names: Sequence[str],
        task_camera_view_names: Sequence[str],
        per_view_shape: Mapping,
        homo_transforms: Mapping,
        calib_para: dict = None,
        H_persp_view_scale: float = 1.0,
        use_distorted_offset: bool = False,
        homo_noise: dict = None,
        vcs_plane_heights: Sequence[float] = (0,),
        return_offset_in_meta_info: bool = True,
        offset_save_path: str = None,
    ):
        self.calib_path = calib_path
        self.homo_path = homo_path
        self.calib_para = calib_para
        self.spatial_resolution = spatial_resolution
        self.camera_view_names = camera_view_names
        self.per_view_shape = per_view_shape
        self.use_distorted_offset = use_distorted_offset
        self.homo_transforms = homo_transforms
        self.H_persp_view_scale = H_persp_view_scale
        self.vcs_range = vcs_range
        self.return_offset_in_meta_info = return_offset_in_meta_info
        self.height = int(
            decimal_div(
                abs(vcs_range[2] - vcs_range[0]), spatial_resolution[0]
            )
        )  # bev image height
        self.width = int(
            decimal_div(
                abs(vcs_range[3] - vcs_range[1]), spatial_resolution[1]
            )
        )  # bev image width

        vcs2bev = get_vcs2bev_img_mat(
            self.vcs_range, (self.height, self.width)
        )
        self.T_ipm2vcsgnd = np.linalg.inv(vcs2bev)
        self.vcs_plane_heights = vcs_plane_heights

        self.homo_noise = None
        if homo_noise is not None:
            self.homo_noise = HomoNoise(**homo_noise)
            self.homo_noise.get_homo_noise()

        # used to check diferent version calib files.
        # keys is hard code here, usually it not change.
        self.local_coor_param_key = (
            "roll",
            "pitch",
            "yaw",
            "camera_x",
            "camera_y",
            "camera_z",
            "vcs",
        )
        virtual_cam_param = SetCameraParam()
        self.poseMat_adascam2cam = np.linalg.inv(
            virtual_cam_param.poseMat_cam2adascam
        )
        self.task_camera_view_names = task_camera_view_names
        if offset_save_path is not None:
            if not os.path.exists(offset_save_path):
                os.makedirs(offset_save_path, exist_ok=True)
        self.offset_save_path = offset_save_path
        self.meta_info = self.get_meta_info()

    @staticmethod
    def _get_T_lidar2vcs(
        lidar_param_file: str,
        camera_param_file: str,
    ):
        """Get lidar to vcs transform.

        Lidar param file is log_tran file and camera param file is yaml file.
        If both param files contain lidar2vcs, prefer to use lidar2vcs in
        camera param file.
        """
        with fsspec.open(camera_param_file, "r") as fid:
            content = fid.read()
        camera_params = yaml.safe_load(content)
        if "lidar2vcs" in camera_params.keys():
            lidar2vcs = np.array(camera_params["lidar2vcs"]).reshape((2, 3))
        else:
            with open(lidar_param_file, "r") as f:
                lidar2vcs = f.readlines()
                for id, v in enumerate(lidar2vcs):
                    if "lidar2vcs" in v:
                        rpy_file = lidar2vcs[id + 1]
                        xyz_file = lidar2vcs[id + 2]
                        break

            rpy = rpy_file.split()
            xyz = xyz_file.split()
            lidar2vcs = np.zeros((2, 3))
            for idx, e in enumerate(rpy[1:]):
                lidar2vcs[0, idx] = float(e.split(",")[0])
            for idx, e in enumerate(xyz[1:]):
                lidar2vcs[1, idx] = float(e.split(",")[0])

        return lidar2vcs

    def _get_calib_parameters(
        self,
        camera_view_name: str,
        lidar_param_file: str = None,
        camera_param_file: str = None,
        calib_param_file: str = None,
        attribute_param_file: str = None,
    ):
        """Get calib params such as lidar extrinsics and camera intrinsics.

        Note that lidar extrinsics include lidar2vcs/lidar2cam transforms,
        camera intrinsics include K and distort coefficients.

        Args:
            camera_view_name: camera view name, e.g., "camera_front".
            lidar_param_file: params file for lidar, e.g.,
                "path/to/log_trans".
            camera_param_file: params file for camera, e.g.,
                "path/to/camera_front.yaml".
            calib_param_file: a calibration params json file, e.g.,
                "xxx/calibration.json", contains calibration parameters
                for lidar and cameras, refer to calibration params version
                in aidi data system.
            attribute_param_file: a attribute json, generated from the HDE-M,
                contains the pack's calibration and sync infomation.

        Returns:
            list array: vcs2cam, camera K, distort coeffs, cam2local eular,
                    cam2local_trans and T_local2vcs
        """
        if attribute_param_file is not None and os.path.exists(
            attribute_param_file
        ):
            # unify camera name in calib_param_file

            with open(attribute_param_file, "r") as f:
                attribute_param = json.load(f)
                calib_param = attribute_param["calibration"]
                if "lidar_top_2_chassis" in calib_param:
                    T_lidar2vcs = np.array(
                        calib_param["lidar_top_2_chassis"], dtype=np.float32
                    )
                    R_lidar2vcs = T_lidar2vcs[:3, :3].copy()
                    t_lidar2vcs = T_lidar2vcs[:3, 3].copy()
                else:
                    T_lidar2vcs = None
                    R_lidar2vcs = None
                    t_lidar2vcs = None

            # getting local param
            local_param = calib_param[camera_view_name]
            K = np.array(local_param["K"], dtype=np.float32)
            d_coef = np.array(local_param["d"], dtype=np.float32)

            if calib_param.get(f"lidar_top_2_{camera_view_name}", None):
                T_lidar2cam = np.array(
                    calib_param[f"lidar_top_2_{camera_view_name}"],
                    dtype=np.float32,
                )
            else:
                # use np.linalg.inv(T_local2vcs @ T_cam2local）to get T_vcs2cam
                # if "lidar_top_2_{camera_view_name}" not in attribute.json.
                calib_getter = CameraParam.init_cam_param_by_dict(local_param)
                T_vcs2cam = calib_getter.poseMat_vcs2cam
                T_lidar2vcs = calib_getter.poseMat_lidar2vcs
                T_local2vcs = calib_getter.poseMat_local2vcs

                cam2local_eular = np.array(
                    [
                        local_param["roll"],
                        local_param["pitch"],
                        local_param["yaw"],
                    ]
                )
                cam2local_trans = np.array(
                    [
                        local_param["camera_x"],
                        local_param["camera_y"],
                        local_param["camera_z"],
                    ]
                )
                return (
                    T_vcs2cam,
                    K,
                    d_coef,
                    cam2local_eular,
                    cam2local_trans,
                    T_local2vcs,
                    T_lidar2vcs,
                )

        elif calib_param_file is not None and os.path.exists(calib_param_file):
            # unify camera name in calib_param_file
            camera_view_rename = {
                "camera_front_left": "camera_frontleft",
                "camera_front": "camera_front",
                "camera_front_right": "camera_frontright",
                "camera_rear_left": "camera_rearleft",
                "camera_rear": "camera_rear",
                "camera_rear_right": "camera_rearright",
                "fisheye_front": "camera_fisheye_front",
                "fisheye_rear": "camera_fisheye_rear",
                "fisheye_left": "camera_fisheye_left",
                "fisheye_right": "camera_fisheye_right",
                "camera_front_30fov": "camera_front_30fov",
            }
            with open(calib_param_file, "r") as f:
                calib_param = json.load(f)

            # The calibration.json file may contains only "lidar_top_2_vcs"
            # or "lidar_2_vcs" or both.
            # (1) if "lidar_2_vcs" only, only use "lidar_2_vcs"
            # (2) if "lidar_2_vcs" and "lidar_top_2_vcs" both exists, the
            #     xyz in both is different, we use "lidar_top_2_vcs" to
            #     keep same to attribute.json file.
            lidar2vcs = calib_param.get("lidar_top_2_vcs", "lidar_2_vcs")
            R_lidar2vcs = (
                Rotation.from_euler(
                    "xyz",
                    lidar2vcs["rpy"],
                )
            ).as_matrix()
            t_lidar2vcs = np.array(
                lidar2vcs["xyz"],
            )
            T_lidar2cam = np.array(
                calib_param[
                    f"lidar_top_2_{camera_view_rename[camera_view_name]}"
                ]
            )
            K = np.array(
                calib_param[camera_view_rename[camera_view_name]]["K"]
            )
            d_coef = np.array(
                calib_param[camera_view_rename[camera_view_name]]["d"]
            )
            # getting local param
            local_param = calib_param[
                f"{camera_view_rename[camera_view_name]}_json"
            ]

        else:
            assert os.path.exists(lidar_param_file), lidar_param_file
            assert os.path.exists(camera_param_file), camera_param_file

            T_lidar2vcs_file = self._get_T_lidar2vcs(
                lidar_param_file, camera_param_file
            )
            with open(camera_param_file, "r") as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            R_lidar2vcs = (
                Rotation.from_euler(
                    "xyz",
                    [
                        T_lidar2vcs_file[0, 0],
                        T_lidar2vcs_file[0, 1],
                        T_lidar2vcs_file[0, 2],
                    ],
                )
            ).as_matrix()
            t_lidar2vcs = np.array(
                [
                    T_lidar2vcs_file[1, 0],
                    T_lidar2vcs_file[1, 1],
                    T_lidar2vcs_file[1, 2],
                ]
            )

            # lidar to camera
            R0_lidar2cam = params["extrinsic"]["rotation0"]
            t0_lidar2cam = params["extrinsic"]["translation0"]

            R_lidar2cam = Rotation.from_quat(
                [
                    R0_lidar2cam["q.x"],
                    R0_lidar2cam["q.y"],
                    R0_lidar2cam["q.z"],
                    R0_lidar2cam["q.w"],
                ]
            ).as_matrix()
            t_lidar2cam = np.array(
                [t0_lidar2cam["x"], t0_lidar2cam["y"], t0_lidar2cam["z"]]
            )
            T_lidar2cam = np.zeros((4, 4))
            T_lidar2cam[:3, :3] = R_lidar2cam
            T_lidar2cam[3, 3] = 1
            T_lidar2cam[:3, 3] = t_lidar2cam

            K0 = params["intrinsic"]["cameraMatrix0"]
            d_coef = np.array(K0["distCoeffs"])
            K = np.array(
                [
                    [K0["ax"], 0, K0["ux"]],
                    [0, K0["ay"], K0["uy"]],
                    [0, 0, 1],
                ]
            )
            local_param = None

        assert R_lidar2vcs is not None and t_lidar2vcs is not None
        T_lidar2vcs = np.zeros((4, 4))
        T_lidar2vcs[:3, :3] = R_lidar2vcs
        T_lidar2vcs[:3, 3] = t_lidar2vcs
        T_lidar2vcs[3, 3] = 1

        T_vcs2cam = np.matmul(T_lidar2cam, np.linalg.inv(T_lidar2vcs))

        # Generate local coor calib param. if `attribute_param_file`
        # or `calib_param_file` exists, and `local_coor_param_key`
        # in file, we can get the local coord calib from it.
        # otherwise, we calculate the local param from `T_vcs2cam`
        if local_param is not None and set(self.local_coor_param_key).issubset(
            set(local_param.keys())
        ):
            # all the local_coor_param_key are contains in the
            # local param
            cam2local_eular = np.array(
                [local_param["roll"], local_param["pitch"], local_param["yaw"]]
            )
            cam2local_trans = np.array(
                [
                    local_param["camera_x"],
                    local_param["camera_y"],
                    local_param["camera_z"],
                ]
            )
            T_local2vcs = np.eye(4, dtype=np.float)
            rot_local2vcs = transform_euler2rotMat(
                np.array(local_param["vcs"]["rotation"])
            )
            T_local2vcs[:3, :3] = rot_local2vcs
            T_local2vcs[0:3, 3] = np.array(local_param["vcs"]["translation"]).T
        else:
            # in this case, the calib file don't contain the local coor,
            # we should calculate it by vcs2cam
            # below part code is refer to
            # `poseMat_local2vcs` in `HAT/hat/core/virtual_camera/camera_base.py` # noqa
            # in this way, we use adascam2vcs to calculat the cam2local, so the yaw of # noqa
            # cam2local we will set to 0, since we cannot calculate it from adascam2vcs. # noqa
            T_cam2vcs = np.linalg.inv(T_vcs2cam)

            T_adascam2vcs = T_cam2vcs @ self.poseMat_adascam2cam
            rot_adascam2vcs = Rotation.from_matrix(T_adascam2vcs[:3, :3])
            ypr_adascam2vcs = rot_adascam2vcs.as_euler("zyx")

            translation_local2vcs = T_adascam2vcs[:3, 3].copy()
            translation_local2vcs[2] = 0
            rpy_local2vcs = np.array([0, 0, ypr_adascam2vcs[0]])
            rot_local2vcs = transform_euler2rotMat(rpy_local2vcs)
            T_local2vcs = np.eye(4)
            T_local2vcs[:3, :3] = rot_local2vcs
            T_local2vcs[0:3, 3] = np.array(translation_local2vcs).T

            cam2local_eular = np.zeros(3)
            cam2local_eular[0] = ypr_adascam2vcs[2]
            cam2local_eular[1] = ypr_adascam2vcs[1]
            cam2local_trans = np.zeros(3)
            cam2local_trans[2] = float(T_adascam2vcs[2, 3])

        return (
            T_vcs2cam,
            K,
            d_coef,
            cam2local_eular,
            cam2local_trans,
            T_local2vcs,
            T_lidar2vcs,
        )

    def compute_homography(
        self,
        K,
        T_vcs2cam,
    ):
        """Compute homography by calibration params for one camera view.

        Args:
            K array: camera intrinsic K matrix.
            T_vcs2cam array: transform of vcs to camera.

        Returns:
            numpy array: homography.
        """
        # multi-height vcs planes
        T_vcsmulti2vcs = np.eye(4, dtype="float32")[None].repeat(
            len(self.vcs_plane_heights), axis=0
        )
        T_vcsmulti2vcs[:, 2, 3] = self.vcs_plane_heights
        T_vcsmulti2cam = T_vcs2cam @ T_vcsmulti2vcs
        T_vcsmulti2img = K @ T_vcsmulti2cam[..., :3, :]
        T_vcsgnd2img = T_vcsmulti2img[..., (0, 1, 3)]
        H_ipm2img = T_vcsgnd2img @ self.T_ipm2vcsgnd  # original image
        return H_ipm2img

    @staticmethod
    def _view_ipm_mask(
        camera_name,
        output_size,
        vcs_range,
        pixel_per_meter,
        camera_yaws,
    ):

        yaw = camera_yaws[camera_name] + 90  # yaw relative to arctan2 theta
        mask = np.zeros((output_size[0], output_size[1], 3), dtype=np.bool_)
        for i in range(output_size[1]):
            for j in range(output_size[0]):
                theta = np.rad2deg(
                    np.arctan2(
                        -j + vcs_range[2] * pixel_per_meter[0],
                        i - vcs_range[3] * pixel_per_meter[1],
                    )
                )

                if abs(theta - yaw) > 90 and abs(theta - yaw) < 270:
                    mask[j, i, :] = True
        return mask

    @staticmethod
    def _remap_image(img, x, y, target_img, u, v, use_opencv=False):
        if use_opencv:
            map1 = np.reshape(x, (target_img.shape[0], target_img.shape[1]))
            map2 = np.reshape(y, (target_img.shape[0], target_img.shape[1]))
            target_img = cv2.remap(
                img,
                map1.astype(np.float32),
                map2.astype(np.float32),
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            valid_index = (
                (x >= 0) * (x < img.shape[1]) * (y >= 0) * (y < img.shape[0])
            )
            x = x[valid_index]
            y = y[valid_index]
            u = u[valid_index]
            v = v[valid_index]
            target_img[v, u, :] = img[y, x, :]

        return target_img

    def _coords_before_after_ipm(
        self,
        height: int,
        width: int,
        ipm: np.ndarray,
        fov_range: Tuple[float, float] = None,
    ):
        """Two coordinates corresponding to before and after ipm.

        Args:
            height: height of image after ipm.
            width: width of image after ipm.
            ipm: homography mat, shape (N, 3, 3).
            fov_range: tuple(min_fov, max_fov), horizon fov range.
                default: None.
        """
        x, y = np.meshgrid(range(width), range(height), indexing="xy")
        id_coords = np.stack([x, y], axis=0).astype("float32")
        ones = np.ones((1, height, width), dtype="float32")
        pix_coords = np.concatenate([id_coords, ones], axis=0).reshape((3, -1))

        cam_points = np.matmul(ipm, pix_coords)

        if fov_range is not None:  # pinhole camera
            azimuth_x = np.arctan2(cam_points[:, 0, :], cam_points[:, 2, :])
            valid_pts_mask = np.full(azimuth_x.shape, False)
            valid_pts_mask[
                np.logical_and(
                    azimuth_x > fov_range[0],  # fov_range[0] is min fov
                    azimuth_x < fov_range[1],  # fov_range[1] is max fov
                )
            ] = True
        else:  # for fisheye, fov_range is none, remain valid_z
            valid_pts_mask = None
        cam_points[:, 2:3, :] = np.clip(
            cam_points[:, 2:3, :], a_min=VALID_Z, a_max=None
        )
        new_pix_coords = cam_points / (cam_points[:, 2:3, :])
        return pix_coords, new_pix_coords, valid_pts_mask

    def _get_ipm_image(self, img_ud, ipm_size, ipm):
        """Get the ipm fusion image from 6v image."""
        height = ipm_size[0]
        width = ipm_size[1]
        x, y = np.meshgrid(range(width), range(height), indexing="xy")
        _, XYW, _ = self._coords_before_after_ipm(height, width, ipm)

        XYW = XYW[0]  # pick the ground plane
        u = XYW[0, :]
        v = XYW[1, :]
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        img_ipm = self._remap_image(
            img_ud, u, v, np.zeros((height, width, 3)), x, y, use_opencv=True
        )
        return img_ipm

    def _get_transform_mat_and_size(self, sub_dir):
        """Generate a transformation matrix to corrent camera intrinsics.

        and get the ipm img size.

        The order of img changes:
        resize->crop->padding->scale_resize

        """
        transform_list = self.homo_transforms[sub_dir]
        resize_mat = np.eye(3, dtype="float32")
        crop_mat = np.eye(3, dtype="float32")
        padding_mat = np.eye(3, dtype="float32")
        scale_mat = np.eye(3, dtype="float32")

        height, width = self.per_view_shape[sub_dir]
        if "Resize" in transform_list.keys():
            resize_mat[0, 0] = transform_list["Resize"][1]
            resize_mat[1, 1] = transform_list["Resize"][0]
            height = transform_list["Resize"][0]
            width = transform_list["Resize"][1]
        if "Crop" in transform_list.keys():
            crop_mat[1, 2] = -transform_list["Crop"][0]  # top
            crop_mat[0, 2] = -transform_list["Crop"][1]  # left
            height = transform_list["Crop"][2]
            width = transform_list["Crop"][3]
        if "Pad" in transform_list.keys():
            # left, top, right and bottom
            padding_mat[1, 2] = transform_list["Pad"][1]  # top
            padding_mat[0, 2] = transform_list["Pad"][0]  # left
            height = (
                height + transform_list["Pad"][1] + transform_list["Pad"][3]
            )  # height + top + bottom
            width = (
                width + transform_list["Pad"][0] + transform_list["Pad"][2]
            )  # height + left + right
        if self.H_persp_view_scale:
            scale_mat[0, 0] = self.H_persp_view_scale
            scale_mat[1, 1] = self.H_persp_view_scale
            height *= self.H_persp_view_scale
            width *= self.H_persp_view_scale
        trans_mat = scale_mat @ padding_mat @ crop_mat @ resize_mat
        return trans_mat, (height, width)

    def visualize_bev_homography(
        self, sync_imgs: dict, apply_view_mask: bool = True
    ):
        """Visualize ipm fusion image under bev.

        If provide calib_path or calib_para, visualize ipm result for
        undistored 6v image.
        Otherwise, only visualize ipm result for distored 6v image, since
        homo_path only proive calculated homography without undistortion
        params K and d. Also, ipm fusion order must be
        camera_front->camera_rear_left->camera_rear_right->
        camera_front_left->camera_front_right-> camera_rear.

        Args:
            sync_imgs: map image names to 6 camera views, e.g.,

                camera_front: 'path/to/xxx.jpg',
                camera_front_left: 'path/to/xxx.jpg',
                camera_front_right: 'path/to/xxx.jpg',
                camera_rear: 'path/to/xxx.jpg',
                camera_rear_left: 'path/to/xxx.jpg',
                camera_rear_right: 'path/to/xxx.jpg',
            apply_view_mask: whether apply view mask on ipm result.

        Returns:
            numpy array: image after ipm.
        """
        pixel_per_meter = [
            1 / self.spatial_resolution[0],
            1 / self.spatial_resolution[1],
        ]
        ipm_size = (self.height, self.width, 3)
        birdsEyeView = np.zeros(ipm_size)

        # correspond to 6v cameras
        camera_yaws = {
            "camera_front": 0,
            "camera_rear_left": 130,
            "camera_rear_right": -130,
            "camera_front_left": 50,
            "camera_front_right": -50,
            "camera_rear": 180,
        }
        raw_imgs = {}
        for camera_view in camera_yaws:
            if self.calib_path is not None:
                lidar_param_file = os.path.join(self.calib_path, "log_trans")
                camera_param_file = os.path.join(
                    self.calib_path, camera_view + ".yaml"
                )
                calib_param_file = os.path.join(
                    self.calib_path, "calibration.json"
                )

                attribute_param_file = os.path.join(
                    self.calib_path, "attribute.json"
                )

                (
                    T_vcs2cam,
                    K,
                    d_coef,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_calib_parameters(
                    camera_view,
                    lidar_param_file,
                    camera_param_file,
                    calib_param_file,
                    attribute_param_file,
                )

                if (
                    self.homo_noise is not None
                    and camera_view in self.homo_noise.view2noise
                ):
                    T_vcs2cam = self.homo_noise.get_noise_matrix(
                        T_vcs2cam, self.homo_noise.view2noise[camera_view]
                    )
                homo = self.compute_homography(
                    K,
                    T_vcs2cam,
                )
            elif self.calib_para is not None:

                assert camera_view in self.calib_para
                K = self.calib_para[camera_view]["K"]
                d_coef = self.calib_para[camera_view]["d_coef"]
                T_vcs2cam = self.calib_para[camera_view]["T_vcs2cam"]

                if (
                    self.homo_noise is not None
                    and camera_view in self.homo_noise.view2noise
                ):
                    T_vcs2cam = self.homo_noise.get_noise_matrix(
                        T_vcs2cam, self.homo_noise.view2noise[camera_view]
                    )
                homo = self.compute_homography(
                    K,
                    T_vcs2cam,
                )
            else:
                hom_file = os.path.join(self.homo_path, camera_view + ".npy")
                assert os.path.exists(hom_file), hom_file
                homo = np.load(hom_file)
                # the shape of multi-planes homo must be (N, 3, 3).
                if len(homo.shape) == 2:
                    homo = homo[None]
                K, d_coef = None, None
            image = cv2.imread(sync_imgs[camera_view])
            if K is not None and d_coef is not None:
                undist_img = cv2.undistort(image[:, :, ::-1], K, d_coef)
            else:
                undist_img = image[:, :, ::-1]

            img_ipm = self._get_ipm_image(undist_img, ipm_size, homo)

            if apply_view_mask:
                mask = self._view_ipm_mask(
                    camera_view,
                    ipm_size,
                    self.vcs_range,
                    pixel_per_meter,
                    camera_yaws,
                )
            else:
                mask = np.zeros((ipm_size[0], ipm_size[1], 3), dtype=np.bool_)
            img_ipm_masked = (img_ipm * (~mask)).astype("uint8")
            mask = np.any(img_ipm_masked != (0, 0, 0), axis=-1)
            birdsEyeView[mask] = img_ipm_masked[mask]

            raw_imgs[camera_view] = cv2.resize(image, (512, 320))
        top_img = np.hstack(
            [
                raw_imgs["camera_front_left"],
                raw_imgs["camera_front"],
                raw_imgs["camera_front_right"],
            ]
        )
        middle_img = np.hstack(
            [
                raw_imgs["camera_rear_left"],
                raw_imgs["camera_rear"],
                raw_imgs["camera_rear_right"],
            ]
        )
        pad_left = (512 * 3 - ipm_size[1]) // 2
        pad_right = 512 * 3 - ipm_size[1] - pad_left
        bottom_img = np.hstack(
            [
                np.zeros((ipm_size[0], pad_left, 3)),
                birdsEyeView,
                np.zeros((ipm_size[0], pad_right, 3)),
            ]
        )
        img_all = np.vstack([top_img, middle_img, bottom_img])

        return img_all

    def compute_homo_offset(self, homography):
        """Compute homography offset by homography."""
        pix_coords, new_pix_coords, valid_mask = self._coords_before_after_ipm(
            self.height, self.width, homography
        )
        num_planes = len(self.vcs_plane_heights)
        homo_offset = new_pix_coords[:, :2] - pix_coords[:2]  # (N, 2, -1)
        homo_offset = homo_offset.reshape(
            num_planes, 2, self.height, self.width
        )
        homo_offset = np.transpose(homo_offset, (0, 2, 3, 1))
        return homo_offset

    def get_homography(self):  # noqa: D205,D400
        """Get homography by load homography in homo_path or compute
        homography according calibration params in calib_path or calib_para
        for each camera view.

        Note that compute homography by calibration params if calib_path
        or calib_para is provided, otherwise load homography saved in
        homo_path. Also, f both calibration.json and log_trans files in
        calib_path, use calibration.json to compute homography.

        Returns:
            numpy array: homography matrix.
        """
        assert not (
            self.calib_path is None
            and self.calib_para is None
            and self.homo_path is None
        ), "please provide calib path or calib_para or homo path"

        assert len(self.camera_view_names) == len(self.per_view_shape), (
            "please provide ori image shape of each view"
            "to normalize homography matrix."
        )

        H = {}
        for sub_dir in self.camera_view_names:
            if self.calib_path is not None:
                assert self.spatial_resolution is not None
                assert self.vcs_range is not None
                lidar_param_file = os.path.join(self.calib_path, "log_trans")
                camera_param_file = os.path.join(
                    self.calib_path, sub_dir + ".yaml"
                )
                calib_param_file = os.path.join(
                    self.calib_path, "calibration.json"
                )
                attribute_param_file = os.path.join(
                    self.calib_path, "attribute.json"
                )

                (
                    T_vcs2cam,
                    K,
                    d_coef,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_calib_parameters(
                    sub_dir,
                    lidar_param_file,
                    camera_param_file,
                    calib_param_file,
                    attribute_param_file,
                )

                if (
                    self.homo_noise is not None
                    and sub_dir in self.homo_noise.view2noise
                ):
                    T_vcs2cam = self.homo_noise.get_noise_matrix(
                        T_vcs2cam, self.homo_noise.view2noise[sub_dir]
                    )
                ori_homo = self.compute_homography(
                    K,
                    T_vcs2cam,
                )
            elif self.calib_para is not None:

                assert sub_dir in self.calib_para
                K = self.calib_para[sub_dir]["K"]
                T_vcs2cam = self.calib_para[sub_dir]["T_vcs2cam"]

                if (
                    self.homo_noise is not None
                    and sub_dir in self.homo_noise.view2noise
                ):
                    T_vcs2cam = self.homo_noise.get_noise_matrix(
                        T_vcs2cam, self.homo_noise.view2noise[sub_dir]
                    )
                ori_homo = self.compute_homography(
                    K,
                    T_vcs2cam,
                )

            else:
                hom_file = os.path.join(self.homo_path, sub_dir + ".npy")
                assert os.path.exists(hom_file), hom_file
                ori_homo = np.load(hom_file)
                # the shape of multi-planes homo must be (N, 3, 3).
                if len(ori_homo.shape) == 2:
                    ori_homo = ori_homo[None]

            norm_mat = np.array(
                [
                    [1 / self.per_view_shape[sub_dir][1], 0, 0],
                    [0, 1 / self.per_view_shape[sub_dir][0], 0],
                    [0, 0, 1],
                ],
                dtype="float32",
            ).reshape(3, 3)
            norm_homo = norm_mat @ ori_homo
            transfomed_mat, _ = self._get_transform_mat_and_size(sub_dir)
            cur_homo = transfomed_mat @ norm_homo
            H[sub_dir] = cur_homo.astype("float32")
        return H

    def get_homo_offset(
        self,
    ):

        if self.use_distorted_offset:
            return self.get_dist_homo_offset()
        else:
            return self.get_undist_homo_offset()

    def get_undist_homo_offset(self):
        """Get homography offset for each camera view."""
        H = self.get_homography()  # (6, 3, 3)
        homo_offset = {}
        for camera_name in self.camera_view_names:
            homo_offset[camera_name] = self.compute_homo_offset(H[camera_name])
        return homo_offset

    def calculate_fov_range(
        self, img_size: tuple, k: np.ndarray, distort: np.ndarray
    ):
        """Calculate fov by intrinsic params, only for pinhole cam.

            copy from "calculate_fov_range" of
            "HAT/hat/core/virtual_camera/camera_base.py".

        Args:
            img_size: img size of ipm img.
            k: intrinsics param.
            distort: distort param.

        """
        h, w = img_size
        points = np.array(
            [[0, h // 2], [w - 1, h // 2], [w // 2, 0], [w // 2, h - 1]]
        )
        points = np.expand_dims(points, axis=1)
        # project img_pts
        un_pixel_points = cv2.undistortPoints(
            points,
            k,
            distort,
            None,
            k,
        )  # noqa
        un_pixel_points = un_pixel_points.reshape(-1, 2)
        lens_points = un_pixel_points - k[:2, 2]
        lens_points = np.divide(
            lens_points,
            [k[0, 0], k[1, 1]],
            out=lens_points,
        )
        zs = np.ones_like(lens_points[:, 0])
        cam_points = np.hstack((lens_points, zs[:, np.newaxis]))
        azimuth_x = np.arctan2(cam_points[:, 0], cam_points[:, 2])
        hfov_range = np.sort(np.array([azimuth_x[0], azimuth_x[1]]))
        return hfov_range

    def get_dist_homo_offset(self):
        assert self.calib_path is not None or self.calib_para is not None
        assert len(self.camera_view_names) == len(self.per_view_shape), (
            "please provide ori image shape of each view"
            "to normalize homography matrix."
        )
        dist_homo_offset = {}
        for _, sub_dir in enumerate(self.camera_view_names):
            assert self.spatial_resolution is not None
            assert self.vcs_range is not None
            if self.calib_path is not None:
                lidar_param_file = os.path.join(self.calib_path, "log_trans")
                camera_param_file = os.path.join(
                    self.calib_path, sub_dir + ".yaml"
                )
                calib_param_file = os.path.join(
                    self.calib_path, "calibration.json"
                )
                attribute_param_file = os.path.join(
                    self.calib_path, "attribute.json"
                )

                (
                    T_vcs2cam,
                    K,
                    d_coef,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_calib_parameters(
                    sub_dir,
                    lidar_param_file,
                    camera_param_file,
                    calib_param_file,
                    attribute_param_file,
                )

                if (
                    self.homo_noise is not None
                    and sub_dir in self.homo_noise.view2noise
                ):
                    T_vcs2cam = self.homo_noise.get_noise_matrix(
                        T_vcs2cam, self.homo_noise.view2noise[sub_dir]
                    )
                ori_homo = self.compute_homography(
                    K,
                    T_vcs2cam,
                )
            else:
                assert sub_dir in self.calib_para
                K = self.calib_para[sub_dir]["K"]
                d_coef = self.calib_para[sub_dir]["d_coef"]
                T_vcs2cam = self.calib_para[sub_dir]["T_vcs2cam"]

                if (
                    self.homo_noise is not None
                    and sub_dir in self.homo_noise.view2noise
                ):
                    T_vcs2cam = self.homo_noise.get_noise_matrix(
                        T_vcs2cam, self.homo_noise.view2noise[sub_dir]
                    )
                ori_homo = self.compute_homography(
                    K,
                    T_vcs2cam,
                )
            norm_mat = np.array(
                [
                    [1 / self.per_view_shape[sub_dir][1], 0, 0],
                    [0, 1 / self.per_view_shape[sub_dir][0], 0],
                    [0, 0, 1],
                ],
                dtype="float32",
            ).reshape(3, 3)
            norm_homo = norm_mat @ ori_homo
            norm_K = norm_mat @ K
            transfomed_mat, ipm_img_size = self._get_transform_mat_and_size(
                sub_dir
            )
            cur_K = transfomed_mat @ norm_K
            cur_homo = transfomed_mat @ norm_homo

            cur_bev2cam = np.linalg.inv(cur_K).astype("float32") @ cur_homo

            if "fisheye" in sub_dir:
                hfov_range = None
            else:
                hfov_range = self.calculate_fov_range(
                    ipm_img_size, cur_K, d_coef
                )
            (
                pix_coords,
                camera_coords,
                valid_mask,
            ) = self._coords_before_after_ipm(
                self.height, self.width, cur_bev2cam, hfov_range
            )

            rvec = np.zeros(shape=(3, 1), dtype=np.float32)
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)

            camera_coords = camera_coords.transpose(0, 2, 1)  # (N, -1, 3)

            if "fisheye" in sub_dir:
                camera_coords = camera_coords[..., :2]
                if len(d_coef) > 4 and sum(d_coef[4:]) == 0:
                    # first 4 params are valid for fisheye distort
                    d_coef = d_coef[:4]
                pts_distorted = cv2.fisheye.distortPoints(
                    camera_coords, cur_K, D=d_coef
                )
            else:
                pts_distorted, _ = cv2.projectPoints(
                    camera_coords.reshape(-1, 3), rvec, tvec, cur_K, d_coef
                )
            num_planes = len(self.vcs_plane_heights)
            pts_distorted = pts_distorted.reshape(num_planes, -1, 2)
            d_homo_offset = pts_distorted - pix_coords.transpose(1, 0)[:, 0:2]
            d_homo_offset = d_homo_offset.reshape(
                num_planes, self.height, self.width, 2
            )
            if valid_mask is not None:
                valid_mask = (~valid_mask).reshape(
                    num_planes, self.height, self.width
                )
                d_homo_offset[valid_mask] = HOMO_PAD_VALUE
            # Clip homo_offset to prevent loss from being nan
            d_homo_offset = np.clip(
                d_homo_offset, a_min=-HOMO_PAD_VALUE, a_max=HOMO_PAD_VALUE
            )
            dist_homo_offset[sub_dir] = d_homo_offset

        return dist_homo_offset

    def get_meta_info(
        self,
    ):
        """Get calibration param and reformat the calib."""
        homo_mat = self.get_homography()
        T_vcs2cams = {}
        intrinsics = {}
        distort_coeffs = {}
        transformats = {}
        ipm_img_sizes = {}
        img_shape = {}
        cam2local_rot = {}
        cam2local_translation = {}
        T_local2vcs = {}
        T_lidar2vcs = np.eye(4)

        assert self.calib_path is not None or self.calib_para is not None
        assert len(self.camera_view_names) == len(self.per_view_shape), (
            "please provide ori image shape of each view"
            "to normalize homography matrix."
        )
        for sub_dir in self.camera_view_names:
            assert self.spatial_resolution is not None
            assert self.vcs_range is not None
            if self.calib_path is not None:
                lidar_param_file = os.path.join(self.calib_path, "log_trans")
                camera_param_file = os.path.join(
                    self.calib_path, sub_dir + ".yaml"
                )
                calib_param_file = os.path.join(
                    self.calib_path, "calibration.json"
                )
                attribute_param_file = os.path.join(
                    self.calib_path, "attribute.json"
                )

                (
                    T_vcs2cam,
                    K,
                    d_coef,
                    cam2local_eular,
                    cam2local_trans,
                    local2vcs,
                    T_lidar2vcs,
                ) = self._get_calib_parameters(
                    sub_dir,
                    lidar_param_file,
                    camera_param_file,
                    calib_param_file,
                    attribute_param_file,
                )
            else:
                assert sub_dir in self.calib_para
                K = self.calib_para[sub_dir]["K"]
                d_coef = self.calib_para[sub_dir]["d_coef"]
                T_vcs2cam = self.calib_para[sub_dir]["T_vcs2cam"]
                cam2local_eular = self.calib_para[sub_dir]["cam2local_eular"]
                cam2local_trans = self.calib_para[sub_dir]["cam2local_trans"]
                local2vcs = self.calib_para[sub_dir]["T_local2vcs"]
                T_lidar2vcs = self.calib_para.get(
                    "T_lidar2vcs", np.eye(4)
                )  # make sure this is not identity matrix if you need this transformation. # noqa

            if (
                self.homo_noise is not None
                and sub_dir in self.homo_noise.view2noise
            ):
                T_vcs2cam = self.homo_noise.get_noise_matrix(
                    T_vcs2cam, self.homo_noise.view2noise[sub_dir]
                )

            transfomed_mat, ipm_img_size = self._get_transform_mat_and_size(
                sub_dir
            )
            T_vcs2cams[sub_dir] = T_vcs2cam[np.newaxis, ...].astype("float64")
            intrinsics[sub_dir] = K[np.newaxis, ...].astype("float64")
            if isinstance(d_coef, np.ndarray):
                # NOTE: here is a tricky implement for dostort_coeff paramter
                # the dostort_coeff in horizon calib is 4 or 8 params.
                # for batch collate and calculate, we pad 4 -> 8, in actually
                # calculate the pad `0` will not affect the calculation.
                assert d_coef.shape[0] in (4, 8)
                if d_coef.shape[0] == 4:
                    pad_d_coef = np.zeros((4,))
                    d_coef = np.concatenate((d_coef, pad_d_coef))
            distort_coeffs[sub_dir] = d_coef.astype("float64")
            transformats[sub_dir] = transfomed_mat[np.newaxis, ...].astype(
                "float64"
            )
            ipm_img_sizes[sub_dir] = np.array(list(ipm_img_size)).astype(
                "float64"
            )
            img_shape[sub_dir] = np.array(list(self.per_view_shape[sub_dir]))[
                np.newaxis, ...
            ].astype("float64")

            cam2local_rot[sub_dir] = cam2local_eular[np.newaxis, ...].astype(
                "float64"
            )
            cam2local_translation[sub_dir] = cam2local_trans[
                np.newaxis, ...
            ].astype("float64")
            T_local2vcs[sub_dir] = local2vcs[np.newaxis, ...].astype("float64")

        # all meta info are double data_type
        meta_info = {
            "T_vcs2cam": T_vcs2cams,  # pose mat of the vcs -> camera
            "intrinsics": intrinsics,  # camera intrisics
            "distort_coeffs": distort_coeffs,  # camera distort coeffs
            "transformats": transformats,  # the transformat after the img transforms (resize + crop + pad) # noqa
            "ipm_img_sizes": ipm_img_sizes,  # the ipm image size, now usually 1/4 of the input size # noqa
            "img_shape": img_shape,  # original input image size
            "homo_mat": homo_mat,  # homo_graphy 3x3 mat
            "homo_transforms": self.homo_transforms,  # dict of homo transforms, describing the changes from origin img to the input of Bevfusion # noqa
            "cam2local_rot": cam2local_rot,  #  rotation(eular angle) of cam coor to local coor # noqa
            "cam2local_translation": cam2local_translation,  # translation of adas_cam to local coor # noqa
            "T_local2vcs": T_local2vcs,  # the posemat of local coor to vcs coor # noqa
            "persp_view_scale": self.H_persp_view_scale,  # the scale of feature to input img. # noqa
            "T_lidar2vcs": T_lidar2vcs,  # Unavalibily to access lidar calibration will lead to identity matrix, make sure this makes sense in your condition. # noqa
        }
        if self.return_offset_in_meta_info:
            meta_info["homo_offset"] = self.get_homo_offset()
        if self.calib_path is not None:
            meta_info["calib_path"] = self.calib_path
        meta_info = reformat_meta_info(
            meta_info, self.task_camera_view_names, len(self.vcs_plane_heights)
        )
        if self.offset_save_path:
            # offset的保存名字依据homo_generator里面的成员变量的值决定
            offset_hash = self.get_offset_hash()
            offset_save_file = os.path.join(
                self.offset_save_path, f"{offset_hash}.npy"
            )
            if not os.path.exists(offset_save_file):
                if "homo_offset" in meta_info:
                    homo_offset = meta_info.pop("homo_offset")
                else:
                    homo_offset = self.get_homo_offset()
                    homo_offset = reformat_homo_info(
                        homo_offset,
                        self.task_camera_view_names,
                        HOMO_PAD_VALUE,
                    )
                with open(offset_save_file, "wb") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    np.save(f, homo_offset)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            meta_info["offset_save_file"] = offset_save_file
        return meta_info

    def get_offset_hash(
        self,
    ):
        """Get offset hash according self param."""
        cls_attr = copy.deepcopy(self.__dict__)
        if "return_offset_in_meta_info" in cls_attr:
            cls_attr.pop("return_offset_in_meta_info")
        if "offset_save_path" in cls_attr:
            cls_attr.pop("offset_save_path")

        cls_attr = str(cls_attr)
        offset_hash = hashlib.sha256(str(cls_attr).encode("utf-8")).hexdigest()
        return offset_hash

    def load_homo_offset(
        self,
    ):
        """Load homo_offset and return meta_info."""
        meta_info = copy.deepcopy(self.meta_info)
        if "offset_save_file" in self.meta_info:
            offset_save_file = self.meta_info["offset_save_file"]
            try:
                homo_offset = np.load(offset_save_file, allow_pickle=True)
            except Exception as e:
                logger(e)
                homo_offset = self.get_homo_offset()
                homo_offset = reformat_homo_info(
                    homo_offset, self.task_camera_view_names, HOMO_PAD_VALUE
                )
                with open(offset_save_file, "wb") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    np.save(offset_save_file, homo_offset)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            meta_info["homo_offset"] = homo_offset
        return meta_info


@OBJECT_REGISTRY.register
class ANCGenerateBEVHomOffset(nn.Module):
    """Module for BEV offset generate on gpu.

    This module is a torch vesion of `get_homo_offset`
    in `HomoGenerator` which can save memory.
    The calculte pipeline is same to the `HomoGenerator`.
    if want to use this class, the meta info should be
    collect from the  `get_meta_info` of `HomoGenerator`

    .. note:: the offset calculate from `GenerateBEVHomOffset`
        and `HomoGenerator` have computational accuracy.
        actually ~1e-3.

    Args:
        vcs_range: visbile range of bev, (bottom, right, top, left) in order.
        spatial_resolution: bev spatial resolution.(unit is meters).
        vcs_plane_heights: the heights of multi-height vcs planes,
            Default to (0, ), representing the single ground plane.
    """

    def __init__(
        self,
        vcs_range: Sequence[float],
        spatial_resolution: Sequence[float],
        vcs_plane_heights: Sequence[float] = (0,),
    ) -> None:
        super(ANCGenerateBEVHomOffset, self).__init__()
        self.vcs_range = vcs_range
        self.vcs_plane_heights = vcs_plane_heights

        self.height = int(
            decimal_div(
                abs(vcs_range[2] - vcs_range[0]), spatial_resolution[0]
            )
        )  # bev image height
        self.width = int(
            decimal_div(
                abs(vcs_range[3] - vcs_range[1]), spatial_resolution[1]
            )
        )  # bev image width

        vcs2bev = get_vcs2bev_img_mat(
            self.vcs_range, (self.height, self.width)
        )
        self.T_ipm2vcsgnd = torch.from_numpy(
            np.linalg.inv(vcs2bev)[np.newaxis, ...]
        )

        T_vcsmulti2vcs = np.eye(4)[None].repeat(
            len(self.vcs_plane_heights), axis=0
        )
        T_vcsmulti2vcs[:, 2, 3] = self.vcs_plane_heights
        self.T_vcsmulti2vcs = torch.from_numpy(
            T_vcsmulti2vcs.astype("float64")[np.newaxis, ...]
        )

        x, y = np.meshgrid(
            range(self.width), range(self.height), indexing="xy"
        )
        id_coords = np.stack([x, y], axis=0).astype("float32")
        ones = np.ones((1, self.height, self.width), dtype="float32")
        pix_coords = np.concatenate([id_coords, ones], axis=0).reshape((3, -1))
        self.pix_coords = torch.from_numpy(pix_coords.astype("float64"))

        self.num_planes = len(self.vcs_plane_heights)

    @staticmethod
    def _stack_data_by_view(data: Mapping, view_idxs: Sequence = None):
        """Stack data according to views according batch_size.

        Args:
            data: data need to stack. e.g. T_vcs2cam, intrinsice.
            view_idxs: views_idxs to stack the data
                [type:1]: 1 -> [1];
                [type:2]: [1,2,3];
                [type:3]: None (means stack all); Defaults to None.

        for example, bs=2, cam_num=4, view_idxs=[0,1]:
            data["T_vcs2cam"] = \
                [
                    (2, 1, 4, 4),  # cam0
                    (2, 1, 4, 4),  # cam1
                    (2, 1, 4, 4),  # cam2
                    (2, 1, 4, 4),  # cam3
                ]
            step1: split and stack as batch
            step2: stack as the cam view idx[0,1]
                return_data = \
                [
                    (1, 1, 4, 4),  # cam0
                    (1, 1, 4, 4),  # cam1

                    (1, 1, 4, 4),  # cam0
                    (1, 1, 4, 4),  # cam1
                ]-> [
                    (2, 1, 4, 4),  # bs0(cam0+cam1)
                    (2, 1, 4, 4),  # bs1(cam0+cam1)
                ]->
                    (4, 1, 4, 4)   # bs*cam, 1, 4, 4
        """

        meta_key_0 = list(data.keys())[0]
        bs = data[meta_key_0][0].shape[0]
        if view_idxs is not None:
            view_idxs = _as_list(view_idxs)
        else:
            view_idxs = list(range(len(data[meta_key_0])))

        meta_info_by_view = {}
        for meta_name, meta in data.items():
            # each meta contains batch_size data, should split them and stack as bs. # noqa
            stack_meta = [
                torch.stack([meta[_idx][_bs] for _idx in view_idxs])
                for _bs in range(bs)
            ]
            meta_info_by_view[meta_name] = torch.cat(stack_meta, dim=0)
        return meta_info_by_view

    def compute_homography(
        self,
        K: torch.Tensor,
        T_vcs2cam: torch.Tensor,
    ) -> torch.Tensor:
        """Compute homography by calibration params for one camera view.

        Args:
            K: camera intrinsic K matrix, shape: (bs*num_views, 1, 3, 3).
            T_vcs2cam: transform of vcs to camera,
                shape: ((bs*num_views, 1, 4, 4)).

        Returns:
            torch.Tensor: homography.
        """
        # multi-height vcs planes
        T_vcsmulti2cam = (
            T_vcs2cam @ self.T_vcsmulti2vcs
        )  # (bs*num_views, 1, 4, 4) * (1, 4, 4, 4) -> (bs*num_views, 1, 4, 4)
        T_vcsmulti2img = (
            K @ T_vcsmulti2cam[..., :3, :]
        )  # (bs*num_views, 1, 3, 3) @ (bs*num_views, 4, 2, 4)
        T_vcsgnd2img = T_vcsmulti2img[..., (0, 1, 3)]
        H_ipm2img = T_vcsgnd2img @ self.T_ipm2vcsgnd  # (bs*num_views, 4, 3, 3)
        return H_ipm2img

    def calculate_fov_range(
        self, img_size: torch.Tensor, k: torch.Tensor, distort: torch.Tensor
    ):
        """Calculate fov by intrinsic params, only for pinhole cam.

            copy from "calculate_fov_range" of
            "HAT/hat/core/virtual_camera/camera_base.py".

        Args:
            img_size: img size of ipm img.
            k: intrinsics param.
            distort: distort param.
        """

        bs_views = img_size.shape[0]  # batch_size * num_views
        h, w = img_size[:, 0], img_size[:, 1]  # (bs*num_views, 2)

        points = torch.zeros((bs_views, 4, 2), dtype=torch.double)
        points[:, 0, 1] = torch.div(h, 2)
        points[:, 1, 0] = w - 1
        points[:, 1, 1] = torch.div(h, 2)
        points[:, 2, 0] = torch.div(w, 2)
        points[:, 3, 0] = torch.div(w, 2)
        points[:, 3, 1] = h - 1
        points = points.permute(0, 2, 1).to(k.device)  # (bs*num_views, 2, 4)

        cam_points = self._pinhole_undistortion(points, distort, k.squeeze(1))
        azimuth_x = torch.atan2(cam_points[:, 0, :], cam_points[:, -1, :])
        hfov_range, _ = torch.sort(azimuth_x[:, :2], -1)
        return hfov_range

    def generate_dist_homo_offset(
        self, meta_info: dict, fisheye: bool = False
    ):
        """Generate dist homo offset.

        Args:
            meta_info: dict contains calibration for homooffset calculate.
            fisheye (bool, optional): whether procesing fisheye cam.
                Defaults to False.

        """
        bs_views = meta_info["intrinsics"].shape[
            0
        ]  # batch_size * num_cam_views
        device = meta_info["intrinsics"].device

        # generate homography
        ori_homo = self.compute_homography(
            meta_info["intrinsics"], meta_info["T_vcs2cam"]
        )  # (bs*num_views, 4, 3, 3)

        # get normalize mat
        norm_mat = (
            torch.eye(3).repeat(bs_views, 1, 1, 1).to(device).double()
        )  # (bs*num_views, 1, 3, 3)
        norm_mat[..., 0, 0] = 1 / meta_info["img_shape"][..., 1]
        norm_mat[..., 1, 1] = 1 / meta_info["img_shape"][..., 0]
        norm_homo = norm_mat @ ori_homo
        norm_K = norm_mat @ meta_info["intrinsics"]
        # calcalate K and homo after transform
        cur_K = meta_info["transformats"] @ norm_K
        cur_homo = meta_info["transformats"] @ norm_homo
        cur_bev2cam = (
            torch.inverse(cur_K) @ cur_homo
        )  # (bs*num_views, 4, 3, 3)

        if fisheye:
            hfov_range = None
        else:
            hfov_range = self.calculate_fov_range(  # (B*views, 2)
                meta_info["ipm_img_sizes"], cur_K, meta_info["distort_coeffs"]
            )

        camera_coords, valid_mask = self._coords_before_after_ipm(
            cur_bev2cam, hfov_range
        )  # (B, 4, 3, H*W)

        if fisheye:
            pts_distorted = self._fisheye_distortion(
                camera_coords,
                meta_info["distort_coeffs"].unsqueeze(1),
                cur_K,
            )
        else:
            pts_distorted = self._pinhole_distortion(
                camera_coords,
                meta_info["distort_coeffs"].unsqueeze(1),
                cur_K,
            )

        pts_distorted = pts_distorted.permute(0, 1, 3, 2)[..., :2]
        d_homo_offset = pts_distorted - self.pix_coords.permute(1, 0)[:, 0:2]
        d_homo_offset = d_homo_offset.view(-1, self.height, self.width, 2)
        if valid_mask is not None:
            valid_mask = (~valid_mask).reshape(-1, self.height, self.width)
            d_homo_offset[valid_mask] = HOMO_PAD_VALUE
        # Clip homo_offset to prevent loss from being nan
        d_homo_offset = torch.clip(
            d_homo_offset, min=-HOMO_PAD_VALUE, max=HOMO_PAD_VALUE
        )
        return d_homo_offset

    def _coords_before_after_ipm(
        self,
        ipm: torch.Tensor,
        fov_range: torch.Tensor = None,
    ):
        """Two coordinates corresponding to before and after ipm.

        Args:
            ipm: homography mat, shape (B, num_planes, 3, 3).
            fov_range: horizon fov range.
                default: None.
        """

        cam_points = torch.matmul(ipm, self.pix_coords)
        if fov_range is not None:  # pinhole camera
            fov_range = fov_range.unsqueeze(
                1
            )  # (bs * num_views, 2) -> (bs * num_views, 1, 2)
            azimuth_x = torch.atan2(
                cam_points[..., 0, :], cam_points[..., 2, :]
            )
            valid_pts_mask = torch.full(azimuth_x.shape, False)
            valid_pts_mask[
                (
                    (azimuth_x > fov_range[..., 0:1])
                    & (
                        azimuth_x < fov_range[..., 1:2]
                    )  # fov_range[0] is min fov
                )
            ] = True
        else:  # for fisheye, fov_range is none, remain valid_z
            valid_pts_mask = None
        cam_points[..., 2:3, :] = torch.clip(
            cam_points[..., 2:3, :], min=VALID_Z, max=None
        )
        new_pix_coords = cam_points / (cam_points[..., 2:3, :])
        return new_pix_coords, valid_pts_mask

    @staticmethod
    def _fisheye_distortion(
        cam_pts: torch.Tensor,
        distort_coef: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        """Fisheye distortion implement using torch.

        the results is same to `cv2.fisheye.distortPoints`
        """
        if distort_coef.shape[-1] == 8:
            k1, k2, k3, k4, _, _, _, _ = torch.split(
                distort_coef, [1] * 8, dim=2
            )
        elif distort_coef.shape[-1] == 4:
            k1, k2, k3, k4 = torch.split(distort_coef, [1] * 4, dim=2)

        cam_x = cam_pts[..., 0, :]  # (B, H*W)
        cam_y = cam_pts[..., 1, :]  # (B, H*W)

        r2 = cam_x * cam_x + cam_y * cam_y
        r = torch.sqrt(r2)
        theta = torch.atan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta2 * theta4
        theta8 = theta4 * theta4

        theta_d = theta * (
            1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
        )

        cam_x_distort = (theta_d / r) * cam_x
        cam_y_distort = (theta_d / r) * cam_y

        cam_distort = torch.stack(
            (cam_x_distort, cam_y_distort, torch.ones_like(cam_x_distort)),
            dim=2,
        )
        pixel_point = torch.matmul(intrinsics, cam_distort)
        return pixel_point

    @staticmethod
    def _pinhole_distortion(
        cam_pts: torch.Tensor,
        distort_coef: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Pinhole distortion implement using torch.

        the results is same to `cv2.projectPoints`
        """
        if distort_coef.shape[-1] == 8:
            k1, k2, p1, p2, k3, k4, k5, k6 = torch.split(
                distort_coef, [1] * 8, dim=2
            )
        elif distort_coef.shape[-1] == 4:
            k1, k2, p1, p2 = torch.split(distort_coef, [1] * 4, dim=2)
            k3, k4, k5, k6 = 0, 0, 0, 0

        cam_x = cam_pts[..., 0, :]  # (B, H*W)
        cam_y = cam_pts[..., 1, :]  # (B, H*W)

        r2 = cam_x * cam_x + cam_y * cam_y
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2 * cam_x * cam_y
        a2 = r2 + 2 * cam_x * cam_x
        a3 = r2 + 2 * cam_y * cam_y
        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
        icdist2 = 1.0 / (1 + k4 * r2 + k5 * r4 + k6 * r6)

        cam_x_distort = cam_x * cdist * icdist2 + p1 * a1 + p2 * a2
        cam_y_distort = cam_y * cdist * icdist2 + p1 * a3 + p2 * a1

        cam_distort = torch.stack(
            (cam_x_distort, cam_y_distort, torch.ones_like(cam_x_distort)),
            dim=2,
        )
        pixel_point = torch.matmul(intrinsics, cam_distort)
        return pixel_point

    @staticmethod
    def _pinhole_undistortion(
        pixel_point: torch.Tensor,
        distort_coef: torch.Tensor,
        intrinsics: torch.Tensor,
        iteration: int = 5,
    ) -> torch.Tensor:
        """Pinhole undistortion implement using torch.

        the results is same to `cv2.undistortPoints`
        """

        ones = torch.ones_like(pixel_point[:, :1, :])
        pixel_point = torch.cat((pixel_point, ones), dim=1)
        inv_intrinsics = torch.inverse(intrinsics)

        # step1: convert to cam plane
        cam_pts = torch.matmul(inv_intrinsics, pixel_point)
        cam_x = cam_pts[:, 0, :]
        cam_y = cam_pts[:, 1, :]

        # step2: convert from distort -> undistort
        if distort_coef.shape[-1] == 8:
            k1, k2, p1, p2, k3, k4, k5, k6 = torch.split(
                distort_coef, [1] * 8, dim=1
            )
        elif distort_coef.shape[-1] == 4:
            k1, k2, p1, p2 = torch.split(distort_coef, [1] * 4, dim=1)
            k3, k4, k5, k6 = 0, 0, 0, 0

        cam_x_0 = cam_x.clone()
        cam_y_0 = cam_y.clone()

        for _ in range(iteration):
            r2 = cam_x * cam_x + cam_y * cam_y
            r4 = r2 * r2
            r6 = r4 * r2
            icdist = (1 + k4 * r2 + k5 * r4 + k6 * r6) / (
                1 + k1 * r2 + k2 * r4 + k3 * r6
            )
            a1 = 2 * cam_x * cam_y
            a2 = r2 + 2 * cam_x * cam_x
            a3 = r2 + 2 * cam_y * cam_y
            cam_x = (cam_x_0 - (p1 * a1 + p2 * a2)) * icdist
            cam_y = (cam_y_0 - (p1 * a3 + p2 * a1)) * icdist

        cam_undistort = torch.stack((cam_x, cam_y, ones[:, 0, :]), dim=1)
        return cam_undistort

    @staticmethod
    def repeat_data(
        data: torch.Tensor,
        repeat_time: int,
        split_num: int,
    ):
        """Repeat data for temporal bev.

        .. note:: the data must collate on batch
            if repeat_time > 1 in temporal bev, the data should
            repeat as below:
            (1) split data as split_num.
            (2) repeat each batch's data on dim=0.
            (3) concat repeated offset on batch.

        Args:
            data: data used for repeat.
            .. note::
                homo_offset:
                    input shape: (b * v * num_plane , h , w ,2)
                    return shape: (b * repeat_time * v * num_plane, h , w , 2)

                uv_map:
                    input shape: (b *  view , h , w ,2)
                    return shape: (b * repeat_time *  view, h , w ,2)

            repeat_time : the repeat time for data.
            split_num : the split num for data.

        """
        if repeat_time > 1:
            data = data.split(split_num)
            data = torch.cat(
                [_data.repeat((repeat_time, 1, 1, 1)) for _data in data]
            )
        return data

    def forward(
        self, meta_info: dict, temporal_info: dict, views_domain2nums: dict
    ):
        """Calculate homooffset.

        Args:
            meta_info: meta_info contains the calibration
                for offset calculation.

                .. code-block:: none

                    meta_info (batch_size=2, length is camera_view_nums):
                        `T_vcs2cam`:
                            [(bs, 1, 4, 4), (bs, 1, 4, 4), (bs, 1, 4, 4),...]
                        `intrinsics`:
                            [(bs, 1, 3, 3), (bs, 1, 3, 3), (bs, 1, 3, 3),...]
                        `distort_coeffs`:
                            [(bs, 8), (bs, 8), (bs, 8),...]
                        `transformats`:
                            [(bs, 1, 3, 3), (bs, 1, 3, 3), (bs, 1, 3, 3),...]
                        `ipm_img_sizes`
                            (ipm image size, usually 1/4 of input size):
                            [(bs,2), (bs, 2), (bs, 2),...]  (h,w)
                        `img_shape`:
                            [(bs, 1, 2), (bs, 1, 2), (bs, 1, 2),...]
                        `fake_homo_flag`(for pad homooffset, 1: fake, 0: true):
                            bs, num_cam_views * num_planes

            temporal_info: temporal information for data clip.
                example (batch_size=2, num_frames_per_iter=4):

                .. code-block:: none

                    {temporal_info: {`num_frames_per_iter`: (4, 4)}}

            views_domain2nums: A batch collated dict to map view domains to
                corresponding view nums, specifying how to organize data.
                The format is:

                .. code-block:: none

                    {
                        "front": bs * (front_view_num, ),
                        "side": bs * (side_view_num, ),
                        "round: bs * (round_view_num, ),
                        "narrow": bs * (narrow_view_num, ),
                    }

        """
        aug_flag = meta_info.pop("aug_flag")[0]
        if not aug_flag:
            if "homo_offset" in meta_info:
                return meta_info.pop("homo_offset")
        else:
            meta_info.pop("homo_offset", None)

        views = list(list(zip(*views_domain2nums.values()))[0])
        start_view_idx_list = np.cumsum([0] + views[:-1])
        end_view_idx_list = np.cumsum(views)

        bs = len(views_domain2nums["front"])
        all_homo_offset = [[] for _ in range(bs)]
        fake_homo_flag = meta_info.pop("fake_homo_flag")
        num_cam_views_and_planes = fake_homo_flag.shape[1]
        # repeat_times equals to the number of frames.
        repeat_times = temporal_info.pop("num_frames_per_iter")[0]
        # below three param should first on gpu
        self.T_ipm2vcsgnd = self.T_ipm2vcsgnd.to(fake_homo_flag)
        self.T_vcsmulti2vcs = self.T_vcsmulti2vcs.to(fake_homo_flag)
        self.pix_coords = self.pix_coords.to(fake_homo_flag)

        for view_domain, start_view_idx, end_view_idx in zip(
            views_domain2nums.keys(), start_view_idx_list, end_view_idx_list
        ):
            if views_domain2nums[view_domain][0] == 0:
                continue
            img_meta = self._stack_data_by_view(
                meta_info,
                view_idxs=list(range(start_view_idx, end_view_idx)),
            )
            # bs*views, h,w,2
            img_homo = self.generate_dist_homo_offset(
                img_meta,
                fisheye=(view_domain == "round"),
            )
            # [(views, h,w,2), (views, h,w,2), ... ]
            img_homo_list = torch.split(
                img_homo, self.num_planes * views_domain2nums[view_domain][0]
            )  # split to batch
            for i, _homo in enumerate(img_homo_list):
                all_homo_offset[i].append(_homo)

        # all_homo_offset: (b * v * num_plane , h , w ,2)
        # all_homo_offset[0]: bs=0, cam_0, plane0's offset
        all_homo_offset = torch.cat(
            [torch.cat(_homo) for _homo in all_homo_offset]
        ).float()  # conver from float32 -> float64
        # NOTE：目前SD涉及到不同视角数据集的复用，因此会对homooffset进行fake的赋值。
        fake_homo_flag = fake_homo_flag.view(-1)
        assert fake_homo_flag.shape[0] == all_homo_offset.shape[0]
        all_homo_offset[torch.nonzero(fake_homo_flag)] = HOMO_PAD_VALUE

        # all_homo_offset: (b * repeat_time * v * num_plane , h , w ,2)
        all_homo_offset = self.repeat_data(
            all_homo_offset, repeat_times, num_cam_views_and_planes
        )

        return all_homo_offset


@OBJECT_REGISTRY.register
class ANCCamPrenorm(object):
    """Module for generate the uv_map for cam prenorm.

    .. note:: The uv_map generate by CamPrenorm is used in BPU train/inference.

    Args:
        camera_prenorm_setting: camera setting to do prenormalize.

        .. code-block:: python

        # [front, front_left, front_right,... rear]
        # if you want to do front_left and front_right, set as below
        camera_front_left = [0, 0, 0]  # [0,0,0] means r p y (rad)
        camera_front_right = [0, 0, 0]
        # check if cached
        data_keys = sorted(data[sub_dir].keys())
        calib_mat_str = {
            k: np.array(data[sub_dir][k], np.float64).tolist()
            for k in data_keys
        }

        camera_view_names: sub directory name of each view.
    """

    def __init__(
        self,
        camera_prenorm_setting: Mapping,
        camera_view_names: Sequence,
    ):
        super(ANCCamPrenorm, self).__init__()
        self.calib_keys = ANCRpyParamAug.calib_keys
        virtual_cam_param = SetCameraParam()
        self.cam2adascam = torch.from_numpy(
            virtual_cam_param.poseMat_cam2adascam
        ).double()

        # get prenorm idx
        cam_idx_prenorm_setting = {
            camera_view_names.index(cam_name): rpy
            for cam_name, rpy in camera_prenorm_setting.items()
        }

        cam_idx_prenorm_setting = {
            k: cam_idx_prenorm_setting[k]
            for k in sorted(cam_idx_prenorm_setting.keys())
        }
        self.cam_idx_prenorm_setting = cam_idx_prenorm_setting
        self.camera_prenorm_idx = set(cam_idx_prenorm_setting.keys())

    def generate_identity_uv_map(self, meta_info, persp_view_scale):
        """Generate the identity grid_sample."""
        bs_views = meta_info["intrinsics"].shape[
            0
        ]  # batch_size * num_cam_views
        device = meta_info["intrinsics"].device
        inv_persp_view_scale = 1 / persp_view_scale
        img_size = meta_info["ipm_img_sizes"] * inv_persp_view_scale
        dst_height, dst_width = int(img_size[0, 0].item()), int(
            img_size[0, 1].item()
        )
        return torch.zeros((bs_views, dst_height, dst_width, 2), device=device)

    def __call__(self, data):
        """Detail pipeline of the rpy augmentation."""
        with torch.no_grad():
            # do prenorm as th index
            assert (
                "meta_info" in data or "meta_info_small" in data
            ), "At least one of meta_info and meta_info_small is in data."
            if "meta_info" in data:
                meta_info = copy.deepcopy(data["meta_info"])
            elif "meta_info_small" in data:
                meta_info = copy.deepcopy(data["meta_info_small"])
            meta_info = get_part_dict(meta_info, self.calib_keys)
            persp_view_scale = meta_info.pop("persp_view_scale", [1.0])[0]
            device = meta_info["cam2local_rot"][0].device
            bs = meta_info["cam2local_rot"][0].shape[0]
            # repeat_times equals to the number of frames.
            repeat_times = data["temporal_info"]["num_frames_per_iter"][0]
            if self.cam2adascam.device != device:
                self.cam2adascam = self.cam2adascam.to(device)

            views_domain2nums = data["view"]
            uv_map_names = []
            for view_domain in views_domain2nums.keys():
                uv_map_name = (
                    view_domain + "_img_uv_map"
                    if view_domain != "front"
                    else "img_uv_map"
                )
                uv_map_names.append(uv_map_name)
            views = list(list(zip(*views_domain2nums.values()))[0])
            start_view_idx_list = np.cumsum([0] + views[:-1])
            end_view_idx_list = np.cumsum(views)
            for view_num, start_view_idx, end_view_idx, uv_map_name in zip(
                views, start_view_idx_list, end_view_idx_list, uv_map_names
            ):
                if view_num == 0:
                    continue
                camera_type_idxs = list(range(start_view_idx, end_view_idx))
                prenorm_cam_idx = list(
                    set(camera_type_idxs).intersection(self.camera_prenorm_idx)
                )
                if len(prenorm_cam_idx) == 0:
                    continue
                prenorm_cam_idx.sort()
                # Assign the rot
                for cam_idx in prenorm_cam_idx:
                    meta_info["cam2local_rot"][cam_idx] = (
                        torch.tensor(self.cam_idx_prenorm_setting[cam_idx])
                        .repeat(bs, 1, 1)
                        .to(device)
                    )

                # get prenorm uv_map
                prenorm_view_nums = len(prenorm_cam_idx)
                prenorm_img_meta = ANCGenerateBEVHomOffset._stack_data_by_view(
                    meta_info,
                    view_idxs=prenorm_cam_idx,
                )

                prenorm_img_uv_map, T_vcs2cam = ANCRpyParamAug.generate_uv_map(
                    prenorm_img_meta,
                    self.cam2adascam,
                    persp_view_scale,
                    return_uv_map_offset=True,
                )

                prenorm_img_uv_map = ANCGenerateBEVHomOffset.repeat_data(
                    prenorm_img_uv_map, repeat_times, prenorm_view_nums
                )
                # prenorm_cam_idx：    cam在所有cam_view里面的idx
                # prenorm_cam_reindex： prenorm_cam_idx与生成的prenorm_img_uv_map的idx mapping # noqa
                prenorm_cam_reindex = {
                    inter_i: i for i, inter_i in enumerate(prenorm_cam_idx)
                }

                # assign the prenorm T_vcs2cam
                for idx in prenorm_cam_idx:
                    meta_info["T_vcs2cam"][idx] = T_vcs2cam[
                        prenorm_cam_reindex[idx] :: prenorm_view_nums
                    ]

                # get non-prenorm uv_map, using identity mapping.
                non_prenorm_cam_idx = list(
                    set(camera_type_idxs) - set(prenorm_cam_idx)
                )
                non_prenorm_view_nums = len(non_prenorm_cam_idx)
                if non_prenorm_view_nums > 0:
                    non_prenorm_img_meta = (
                        ANCGenerateBEVHomOffset._stack_data_by_view(
                            meta_info,
                            view_idxs=non_prenorm_cam_idx,
                        )
                    )
                    non_prenorm_img_uv_map = self.generate_identity_uv_map(
                        non_prenorm_img_meta, persp_view_scale
                    )

                    non_prenorm_img_uv_map = (
                        ANCGenerateBEVHomOffset.repeat_data(
                            non_prenorm_img_uv_map,
                            repeat_times,
                            non_prenorm_view_nums,
                        )
                    )

                    non_prenorm_cam_reindex = {
                        diff_i: i
                        for i, diff_i in enumerate(non_prenorm_cam_idx)
                    }

                    # reorder the uv_map as bs*views
                    prenorm_img_uv_map = prenorm_img_uv_map.split(
                        prenorm_view_nums
                    )
                    non_prenorm_img_uv_map = non_prenorm_img_uv_map.split(
                        non_prenorm_view_nums
                    )
                    uv_maps = []
                    for _bs in range(bs):
                        bs_uv_maps = []
                        for i in camera_type_idxs:
                            bs_uv_maps.append(
                                non_prenorm_img_uv_map[_bs][
                                    non_prenorm_cam_reindex[i]
                                ]
                                if i in non_prenorm_cam_idx
                                else prenorm_img_uv_map[_bs][
                                    prenorm_cam_reindex[i]
                                ]
                            )
                        uv_maps.append(torch.stack(bs_uv_maps))
                    uv_maps = torch.cat(uv_maps)
                else:
                    uv_maps = prenorm_img_uv_map

                data[uv_map_name] = [uv_maps]

            if "meta_info" in data:
                data["meta_info"]["T_vcs2cam"] = meta_info["T_vcs2cam"]
                data["meta_info"]["cam2local_rot"] = meta_info["cam2local_rot"]
            if "meta_info_small" in data:
                data["meta_info_small"]["T_vcs2cam"] = meta_info["T_vcs2cam"]
                data["meta_info_small"]["cam2local_rot"] = meta_info[
                    "cam2local_rot"
                ]

        return data


@OBJECT_REGISTRY.register
class ANCRpyParamAug(object):
    """Module for rpy random perturb and aug img generate.

    Args:
        prob: the prob to perturb eular angle.
        aug_degree: the aug_degeree used add
            to the cam2local rpy, if set to
            `1`, means random-> [-1,1].
        mode: mode for grid_sample, here use torch.grid_sample.
        padding_mode: padding_mode for grid_sample.
        camera_view_names: sub directory name of each view
        filter_cameras: camera not do rpy aug.
        empty_cache: Whether to execute torch.cuda.empty_cache().
    """

    calib_keys = [
        "T_vcs2cam",
        "intrinsics",
        "distort_coeffs",
        "transformats",
        "ipm_img_sizes",
        "img_shape",
        "cam2local_rot",
        "cam2local_translation",
        "T_local2vcs",
        "persp_view_scale",
    ]

    def __init__(
        self,
        prob: float = 0.2,
        aug_degree: float = 1,  # deg
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        camera_view_names: Sequence = None,
        filter_cameras: Sequence = None,
        empty_cache: bool = False,
    ):
        super(ANCRpyParamAug, self).__init__()
        self.prob = [prob]
        self.aug_degree = [aug_degree]
        self.mode = mode
        self.padding_mode = padding_mode

        # The meta info may contains different type of data, in `RpyParamAug` # noqa
        # we only use necessary calib param, therefore we set the `calib_keys`
        # to generate the param from meta_info, here we hard code since the
        # keys will not be change.
        virtual_cam_param = SetCameraParam()
        self.cam2adascam = torch.from_numpy(
            virtual_cam_param.poseMat_cam2adascam
        ).double()

        # getting the filter_cam_idx by camera_view_names and filter_cameras # noqa
        if filter_cameras:
            assert camera_view_names is not None
            filter_cam_idx = [
                camera_view_names.index(cam) for cam in filter_cameras
            ]
        else:
            filter_cam_idx = None
        self.filter_cam_idx = filter_cam_idx
        self.empty_cache = empty_cache

    @staticmethod
    def generate_uv_map(
        meta_info,
        cam2adascam,
        persp_view_scale=1,
        return_uv_map_offset=False,
    ):
        """Generate uv map based on the meta_info.

        NOTE: only support pinhole now.

        Args:
            meta_info: dict contains calibration for homooffset calculate.
            cam2adascam: transformat of opencv_cam to horizon_adas_cam.
            persp_view_scale: the feature scale of input imgs, since the
                `transformats` and `ipm_img_sizes` in meta_info is resize
                by the `persp_view_scale`, use (1/persp_view_scale) to
                rescale the transformats and ipm_img_sizes.
            return_uv_map_offset: whether return the uv offset for
                horizon_grid_sample. default: False.
                True: return the uv_map offset.
                False: return the uv_map grid, and normalize to [-1,1]

        """
        bs_views = meta_info["intrinsics"].shape[
            0
        ]  # batch_size * num_cam_views
        device = meta_info["intrinsics"].device

        dst_adascam2local = (
            torch.eye(4, dtype=torch.double)
            .repeat(bs_views, 1, 1, 1)
            .to(device)
        )
        if euler_angles_to_matrix is None:
            raise ModuleNotFoundError(
                "Need install pytorch3d according requirement."
            )
        rot_mat = euler_angles_to_matrix(
            meta_info["cam2local_rot"], "XYZ"
        )  # (bs_view,1,3,3)
        dst_adascam2local[..., :3, :3] = rot_mat
        dst_adascam2local[..., 0:3, 3] = meta_info["cam2local_translation"]
        T_dstcam2vcs = (
            meta_info["T_local2vcs"] @ dst_adascam2local @ cam2adascam
        )
        T_vcs2dstcam = torch.inverse(T_dstcam2vcs)

        # rescale the transformats and ipm_img_sizes by (1 / persp_view_scale)
        inv_persp_view_scale = 1 / persp_view_scale
        meta_info["transformats"][..., 0, 0] *= inv_persp_view_scale
        meta_info["transformats"][..., 1, 1] *= inv_persp_view_scale
        img_size = meta_info["ipm_img_sizes"] * inv_persp_view_scale

        # get normalize mat
        norm_mat = (
            torch.eye(3).repeat(bs_views, 1, 1, 1).to(device).double()
        )  # (bs*num_views, 1, 3, 3)
        norm_mat[..., 0, 0] = 1 / meta_info["img_shape"][..., 1]
        norm_mat[..., 1, 1] = 1 / meta_info["img_shape"][..., 0]

        # calcalate K and homo after transform
        norm_K = norm_mat @ meta_info["intrinsics"]
        cur_K = meta_info["transformats"] @ norm_K

        # generate dst meshgrid
        dst_height, dst_width = int(img_size[0, 0].item()), int(
            img_size[0, 1].item()
        )
        dst_meshgrid = torch.meshgrid(
            torch.arange(dst_width), torch.arange(dst_height), indexing="xy"
        )
        dst_pix_coords = torch.stack(dst_meshgrid, axis=0).double()  # (2,h,w)
        dst_pix_coords = (
            dst_pix_coords.repeat(bs_views, 1, 1, 1)
            .view(bs_views, 2, -1)
            .to(device)
        )
        # project dst pixel coor to dist cam coor
        dst_cam_points = ANCGenerateBEVHomOffset._pinhole_undistortion(
            dst_pix_coords, meta_info["distort_coeffs"], cur_K.squeeze(1)
        )
        ones = torch.ones((bs_views, 1, dst_height * dst_width)).to(device)
        dst_cam_points = torch.cat((dst_cam_points, ones), dim=1).unsqueeze(1)

        # i. project dst cam to vcs
        vcs_point = T_dstcam2vcs @ dst_cam_points
        # ii project vsc to src cam
        T_vcs2srccam = meta_info.pop("T_vcs2cam")
        src_cam_points = T_vcs2srccam @ vcs_point
        src_cam_points = src_cam_points / (src_cam_points[..., 2:3, :] + EPS)
        # iii, projct src cam to pixel
        src_pixel_point = ANCGenerateBEVHomOffset._pinhole_distortion(
            src_cam_points[..., :3, :],
            meta_info["distort_coeffs"].unsqueeze(1),
            cur_K,
        )
        uv_map = src_pixel_point.permute(0, 1, 3, 2)[..., :2]
        uv_map = uv_map.reshape((bs_views, dst_height, dst_width, 2)).float()

        if return_uv_map_offset:
            pixel_grid = (
                torch.from_numpy(
                    np.indices(
                        (dst_width, dst_height), dtype=np.float32
                    ).transpose(2, 1, 0)
                )
                .to(device)
                .repeat(bs_views, 1, 1, 1)
            )
            uv_map = uv_map - pixel_grid
        else:  # normalize [-1,1], using F.grid_sample
            uv_map[..., 0] = uv_map[..., 0] * 2 / (dst_width - 1) - 1
            uv_map[..., 1] = uv_map[..., 1] * 2 / (dst_height - 1) - 1

        return uv_map, T_vcs2dstcam

    def get_idx_intersection(self, idx_1, idx_2):
        if idx_1 is not None and idx_2 is not None:
            return list(set(idx_1).intersection(set(idx_2)))
        else:
            return []

    def get_aug_output(
        self,
        meta_info,
        img,
        view_idxs,
        filter_view_idxs,
        persp_view_scale,
        repeat_times,
    ):
        """Calculate the T_vcs2cam and image by rpy aug."""
        view_nums = len(view_idxs)
        batch_size = meta_info["T_vcs2cam"][0].shape[0]
        # step1: get filter idx and view_rpy_idx for data reorganize.
        filter_idxs = self.get_idx_intersection(view_idxs, filter_view_idxs)
        filter_idxs.sort()

        view_rpy_idx = copy.copy(view_idxs)
        if len(filter_idxs) > 0:
            view_rpy_idx = list(set(view_idxs) - set(filter_view_idxs))
        view_rpy_idx.sort()
        view_rpy_data_idx = [i - view_idxs[0] for i in view_rpy_idx]

        if len(view_rpy_idx) == 0:
            return [meta_info["T_vcs2cam"][i] for i in view_idxs], img

        # step2: get all view_idxs augmatated T_vcs2cam and warped img
        img_meta = ANCGenerateBEVHomOffset._stack_data_by_view(
            meta_info,
            view_idxs=view_idxs,
        )

        img_uv_map, T_vcs2dstcam = self.generate_uv_map(
            img_meta, self.cam2adascam, persp_view_scale
        )

        img_uv_map = ANCGenerateBEVHomOffset.repeat_data(
            img_uv_map, repeat_times, view_nums
        )

        warped_img = [
            F.grid_sample(
                _img,
                img_uv_map,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=False,
            )
            for _img in img
        ]

        # step3: reorganize the T_vcs2cam and warped_img
        if len(filter_idxs) > 0:
            reorder_warped_img = []
            # !get img
            for _img, _warped_img in zip(img, warped_img):
                _img = _img.split(view_nums)
                _warped_img = _warped_img.split(view_nums)
                bs_img = []
                for _bs in range(batch_size):
                    for _idx in range(view_nums):
                        if _idx in view_rpy_data_idx:
                            bs_img.append(_warped_img[_bs][_idx])
                        else:
                            bs_img.append(_img[_bs][_idx])
                reorder_warped_img.append(torch.stack(bs_img))
            warped_img = reorder_warped_img
            # !get vcs2cam

            T_vcs2cam = []
            for _idx in view_idxs:
                if _idx in view_rpy_idx:
                    index = view_rpy_idx.index(_idx)
                    data_idx = view_rpy_data_idx[index]
                    T_vcs2cam.append(T_vcs2dstcam[data_idx::view_nums])
                else:
                    T_vcs2cam.append(meta_info["T_vcs2cam"][_idx])
        else:
            T_vcs2cam = [T_vcs2dstcam[i::view_nums] for i in range(view_nums)]

        return T_vcs2cam, warped_img

    def __call__(self, data):
        """Detail pipeline of the rpy augmentation.

        step1: determine whether do rpy aug, according following condition:
            (1) this batch(task) do augtransforms
            (2) this batch(task) do rpy aug
            (3) the random_prob < prob

        step2: generate the perturb random rpy value as `aug_degree`,
            and update the `cam2local_rot` to calculate `dst_adascam2local`.

        step3: according dst_adascam2local and local2vcs and cam2adascam
            can calculate the T_vcs2dstcam.

        step4: generate the uv_map from dstcam2srccam
            (1) initialize the dst_meshgrid
            (2) project dst_meshgrid to dst_cam_points
            (3) project dst_cam_points to vcs using T_dstcam2vcs
            (4) project vcs to ori(src)_cam_pts using T_vcs2srccam
            (5) getting the src src_pixel_point as the uv map
            (6) using F.gridsample to generate the dst img.
        """
        with torch.no_grad():
            if not (
                "aug_transforms" in data
                and "rpy_augmentation" in data["aug_transforms"]
            ):
                return data

            # NOTE: The `prob` and `aug_degree` will use
            # the batch data firstly (different task may
            # use different prob or aug_degree), if not
            # set, will use the default param.
            prob = data["aug_transforms"]["rpy_augmentation"].get(
                "prob", self.prob
            )[0]
            aug_degree = data["aug_transforms"]["rpy_augmentation"].get(
                "aug_degree", self.aug_degree
            )[0]

            if torch.rand(1) > prob:
                return data

            # deepcopy meta_info
            assert (
                "meta_info" in data or "meta_info_small" in data
            ), "At least one of meta_info and meta_info_small is in data."
            if "meta_info" in data:
                meta_info = copy.deepcopy(data["meta_info"])
            elif "meta_info_small" in data:
                meta_info = copy.deepcopy(data["meta_info_small"])
            meta_info = get_part_dict(meta_info, ANCRpyParamAug.calib_keys)
            # repeat_times equals to the number of frames
            repeat_times = data["temporal_info"]["num_frames_per_iter"][0]
            # do rpy augmentation
            device = meta_info["cam2local_rot"][0].device
            bs = meta_info["cam2local_rot"][0].shape[0]
            num_views = len(meta_info["cam2local_rot"])

            if self.cam2adascam.device != device:
                self.cam2adascam = self.cam2adascam.to(device)

            # normalize -> (-aug_degree, +aug_degree)
            rpy_random = (
                2
                * aug_degree
                * torch.rand(
                    (num_views * bs, 1, 3), dtype=torch.double, device=device
                )
                - aug_degree
            )
            # convert to rad.
            rpy_random = torch.deg2rad(rpy_random)

            # the rpy_random of filter_idx is zeros
            if self.filter_cam_idx:
                for idx in self.filter_cam_idx:
                    rpy_random[idx * bs : (idx + 1) * bs] = torch.zeros_like(
                        rpy_random[0]
                    )

            meta_info["cam2local_rot"] = torch.cat(meta_info["cam2local_rot"])
            # in-place modify the cam2local_rot
            meta_info["cam2local_rot"] += rpy_random
            cam2local_rot = list(meta_info["cam2local_rot"].split(bs, dim=0))
            meta_info["cam2local_rot"] = cam2local_rot
            if "meta_info" in data:
                data["meta_info"]["cam2local_rot"] = cam2local_rot
            if "meta_info_small" in data:
                data["meta_info_small"]["cam2local_rot"] = cam2local_rot

            # generate img and T_vcs2dstcam.
            views_domain2nums = data["view"]
            views = list(list(zip(*views_domain2nums.values()))[0])
            start_view_idx_list = np.cumsum([0] + views[:-1])
            end_view_idx_list = np.cumsum(views)
            img_keys = []
            for view_domain in views_domain2nums.keys():
                # For "front" domain, corresponding image key is "img",
                # for other domain, corresponding image key is domain+"_img".
                img_key = (
                    view_domain + "_img" if view_domain != "front" else "img"
                )
                img_keys.append(img_key)

            # meta_info = get_part_dict(meta_info, self.calib_keys)
            persp_view_scale = meta_info.pop("persp_view_scale", [1.0])[0]
            T_vcs2dstcam = []
            for view_num, start_view_idx, end_view_idx, img_key in zip(
                views, start_view_idx_list, end_view_idx_list, img_keys
            ):
                if view_num == 0:
                    continue
                # NOTE: fisheye not support rpy aug now.
                if img_key == "round_img":
                    T_vcs2dstcam.extend(
                        meta_info["T_vcs2cam"][start_view_idx:end_view_idx]
                    )
                    continue
                view_idxs = list(range(start_view_idx, end_view_idx))
                T_vcs2cam, img = self.get_aug_output(
                    meta_info,
                    data[img_key],
                    view_idxs,
                    self.filter_cam_idx,
                    persp_view_scale,
                    repeat_times,
                )
                T_vcs2dstcam.extend(T_vcs2cam)
                data[img_key] = img

            # update the "T_vcs2cam" in data["meta_info"]
            # set the aug_flag in meta_info, and re-caculate the
            # offset in `GenerateBEVHomOffset`.
            if "meta_info" in data:
                data["meta_info"]["T_vcs2cam"] = T_vcs2dstcam
                data["meta_info"]["aug_flag"] = [True]
            if "meta_info_small" in data:
                data["meta_info_small"]["T_vcs2cam"] = T_vcs2dstcam
                data["meta_info_small"]["aug_flag"] = [True]
        if self.empty_cache:
            torch.cuda.empty_cache()
        return data


@OBJECT_REGISTRY.register
class ANCConvertSoftwareOffset(object):
    """Generate homography by online compute or offline load.

    Args:
        offset_path: path of offset that software provides.
        warp_sizes: img sizes after warped.
        ipm_output_size: img sizes of ipm output.
        vcs_plane_heights: The heights of multi-height vcs planes.
        block_warp_padding: Order is (left,right,up,bottom).
        grid_quant_scale: Quantize scale.

    """

    def __init__(
        self,
        offset_path: str = None,
        warp_sizes: list = None,
        ipm_output_size: tuple = None,
        vcs_plane_heights: tuple = None,
        block_warp_padding: list = None,
        grid_quant_scale: float = None,
    ):
        self.offset_path = offset_path
        self.ipm_output_size = ipm_output_size
        self.warp_sizes = warp_sizes
        self.vcs_plane_heights = vcs_plane_heights
        self.block_warp_padding = block_warp_padding
        self.grid_quant_scale = grid_quant_scale

    def __call__(self):

        num_plane = len(self.vcs_plane_heights)
        num_view = len(self.warp_sizes)
        num_offset = num_plane * num_view

        offsets = []

        for i in range(num_offset):
            # load
            offset_path_i = os.path.join(
                self.offset_path, f"homo_offset_{i}.bin"
            )
            assert os.path.exists(offset_path_i)
            offset_i = np.fromfile(open(offset_path_i, "rb"), dtype=np.int16)
            # reshape
            warp_size_i = self.warp_sizes[i // num_plane]
            pad_warp_size_i = (
                warp_size_i[0],
                align_ceil(warp_size_i[1], 256, 2, [0, 16, 32, 64, 128]),
            )
            offset_i = offset_i.reshape(
                2, pad_warp_size_i[0], pad_warp_size_i[1]
            )
            offset_i = offset_i[:, : warp_size_i[0], : warp_size_i[1]]
            offset_i = offset_i.transpose(1, 2, 0)
            # dequant
            offset_i = offset_i.astype(np.float64)
            offset_i *= self.grid_quant_scale
            # padding offset
            ori_offset = np.ones(
                (self.ipm_output_size[0], self.ipm_output_size[1], 2),
                dtype=np.float32,
            )
            # reverse warp padding
            pad_l, pad_r, pad_u, pad_b = self.block_warp_padding[
                i // num_plane
            ]
            offset_i[:, :, 0] -= pad_l
            offset_i[:, :, 1] -= pad_u
            ori_offset[
                pad_u : self.ipm_output_size[0] - pad_b,
                pad_l : self.ipm_output_size[1] - pad_r,
                :,
            ] = offset_i

            offsets.append(ori_offset[np.newaxis, :])

        return np.concatenate(offsets, axis=0)
