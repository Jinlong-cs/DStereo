# Copyright (c) Horizon Robotics. All rights reserved.
import json
from typing import Dict, List

import cv2
import numpy as np
import torch.utils.data as data
from scipy.spatial.transform import Rotation

from hat.data.transforms.eye3d_pose import Eye3dPoseTransformList
from hat.registry import OBJECT_REGISTRY

EPS = 1e-6

__all__ = ["Eye3dPoseCarDataset", "Eye3dPoseDataset"]


class SyntheticDataList:
    """Synthetic data generator.

    Args:
        length: num synthetic data each epoch
        face_model: path to 3d facemodels txt
        range_list: pose range and 3d range to generate data
    """

    def __init__(self, length: int, face_model: str, range_list: list):
        self._len = length
        self._K = np.eye(3) * 1000
        self._K[-1, -1] = 1
        self._mtx_4 = np.array([1000, 1000, 0, 0])
        self._mtx = np.eye(3) * 1000.0
        self._mtx[-1, -1] = 1
        self._dist = np.zeros(shape=(1, 5))
        self.facemodels = self._get_facemodel(face_model)
        self._parse_range(range_list)

    def _parse_range(self, range_list):
        ratio = [_["ratio"] for _ in range_list]
        ratio = np.array(ratio) / sum(ratio)
        self.ratio = np.array(
            [ratio[: i + 1].sum() for i in range(len(ratio))]
        )
        self.range_list = []
        for d in range_list:
            self.range_list.append(
                [
                    np.array(d["mean"]),
                    np.array(d["scale"]),
                    np.array(d["pose_mean"]),
                    np.array(d["pose_scale"]),
                ]
            )

    def _get_facemodel(self, path):
        with open(path) as f:
            lines = [json.loads(_) for _ in f.read().split("\n") if _]
        for i in range(len(lines)):
            lines[i]["face"] = np.array(lines[i]["face"]).reshape(
                -1, 3
            ) - np.array([[0, 0, 75]])
        return tuple(lines)

    def project(self, points):
        p2d = points @ self._K.T
        p2d = p2d[:, :2] / p2d[:, 2:]
        return p2d

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        range_idx = (np.random.random() < self.ratio).sum() - 1
        t_mean, t_scale, pose_mean, pose_scale = self.range_list[range_idx]
        rpy = np.random.uniform(-1, 1, 3) * pose_scale + pose_mean
        T = np.random.uniform(-1, 1, 3) * t_scale + t_mean
        R = euler2matrix_zxy(rpy)

        id_index = np.random.randint(low=0, high=len(self.facemodels))
        points_cam = self.facemodels[id_index]["face"] @ R.T + T.reshape(1, 3)
        ldmks = self.project(points_cam)
        facecenter = self.project(T.reshape(1, 3)).reshape(-1)
        data = {}
        data["undistort_points_70"] = np.zeros([70, 2], np.float32)
        data["undistort_points_70"][2:] = ldmks
        data["mtx_4"] = self._mtx_4.copy()
        data["mtx"] = self._mtx.copy()
        data["rpy_norm"] = pose_norm(R, facecenter, self._K, self._dist)
        data["left_eye"] = (points_cam[36] + points_cam[39]) / 2
        data["right_eye"] = (points_cam[42] + points_cam[45]) / 2
        data["id_info"] = self.facemodels[id_index]["id_info"]
        return data


def calc_rotate_cam2norm(face_center: np.array, mtx: np.array, dist: np.array):
    """Calc rotate matrix to norm face.

    Args:
        face_center: face center, with size 2
        mtx: intrinsic matrix
        dist: distortion matrix

    Returns:
        rotate matrix
    """
    face_undist = cv2.undistortPoints(
        face_center.reshape(-1, 1, 2).astype(np.float32),
        mtx,
        dist,
        None,
        P=mtx,
    )[0, 0, :]
    face_undist_ex = np.array([face_undist[0], face_undist[1], 1.0]).reshape(
        3, 1
    )

    face_line = (np.linalg.inv(mtx) @ face_undist_ex).reshape(-1)
    cam_line = np.array([0, 0, 1])

    axis = np.cross(face_line, cam_line)
    if np.linalg.norm(axis) < EPS:
        return np.eye(3)
    if np.linalg.norm(face_line) < EPS:
        return np.eye(3)

    angle = np.dot(face_line, cam_line) / np.linalg.norm(face_line)
    angle = np.arccos(angle)
    rvec = axis / np.linalg.norm(axis) * angle
    return cv2.Rodrigues(rvec)[0]


def euler2matrix_zxy(rpy: np.array):
    """Convert euler angle in degree to rotation matrix.

    Args:
        rpy: (3,) array, predefined euler angle in degree.

    Returns:
        rotation matrix: (3,3) array
    """
    r, p, y = rpy.reshape(-1)
    return Rotation.from_euler("zxy", [r, -p, y], True).as_matrix()


def matrix2euler_zxy(R: np.array):
    """Convert rotation matrix to euler angle in degree.

    Args:
        rotation matrix: (3,3) array

    Returns:
        rpy: (3,) array, euler angle in degree, with order zxy.
    """
    return Rotation.from_matrix(R).as_euler("zxy", True).tolist()


def pose_norm(
    pose_matrix: np.array,
    face_center: np.array,
    mtx_panel: np.array,
    dist_panel: np.array,
):
    """Convert head pose from camera coordinate to normed coordinate.

    Args:
        pose_matrix: (3,3), head pose rotation matrix
        face_center: (2,)
        mtx_panel: (3,3), camera intrinsics
        dist_panel: (5,1), camera distortions

    Returns:
        rpy: (3,), euler angle in degree.
    """
    matrix_cam2norm = calc_rotate_cam2norm(face_center, mtx_panel, dist_panel)
    pose_matrix_face = matrix_cam2norm @ pose_matrix
    r, p, y = matrix2euler_zxy(pose_matrix_face)
    return np.array([r, -p, y])


@OBJECT_REGISTRY.register
class Eye3dPoseCarDataset(data.Dataset):
    """A dataset wrapping over txt file for real data or synthetic data.

    Each sample is a json string.

    Args:
        txtpath: path to real data txt. Defaults to "".
        synthetic_argv: argv for synthetic data. Defaults to None.
        transforms: data transforms. Defaults to None.
    """

    def __init__(
        self,
        txtpath: str = "",
        synthetic_argv: dict = None,
        transforms: Eye3dPoseTransformList = None,
    ):
        super(Eye3dPoseCarDataset, self).__init__()
        if txtpath and txtpath.endswith("txt"):
            self._datas = self._prepare_data(txtpath)
        else:
            assert isinstance(synthetic_argv, dict)
            self._datas = SyntheticDataList(**synthetic_argv)
        self._len = len(self._datas)
        self.transforms = transforms

    def _prepare_data(self, txtpath):
        datas = []
        keys = ["rpy_norm", "rvec_norm", "rvec_cam", "right_eye", "left_eye"]
        with open(txtpath) as f:
            lines = [_ for _ in f.read().split("\n") if _]
        for line in lines:
            d = json.loads(line)
            for k in keys:
                if k in d.keys():
                    d[k] = np.array(d[k], np.float32).reshape([-1, 1, 1])
            d["undistort_points_70"] = np.array(
                d["undistort_points_70"]
            ).reshape([-1, 2])
            mtx = np.zeros([3, 3], dtype=np.float64)
            mtx[0, 0] = d["mtx_4"][0]
            mtx[1, 1] = d["mtx_4"][1]
            mtx[2, 2] = 1
            mtx[0, 2] = d["mtx_4"][2]
            mtx[1, 2] = d["mtx_4"][3]
            d["mtx"] = mtx
            d["mtx_4"] = np.array(d["mtx_4"])
            datas.append(d)
        return datas

    def __getitem__(self, idx):
        data = self._datas[idx]
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self):
        return self._len

    def __getstate__(self):
        state = self.__dict__.copy()
        return state


@OBJECT_REGISTRY.register
class Eye3dPoseDataset(data.ConcatDataset):
    """A dataset concat multi eye3dpose dataset.

    Args:
        argvs: witch contains params to
            init Eye3dPoseCarDataset.
        transforms: data transforms. Defaults to None.
    """

    def __init__(
        self, argvs: List[Dict], transforms: Eye3dPoseTransformList = None
    ):
        datasets = [
            Eye3dPoseCarDataset(**_, transforms=transforms) for _ in argvs
        ]
        super().__init__(datasets)
