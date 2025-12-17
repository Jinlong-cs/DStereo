from typing import Any

import cv2
import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY

MEAN_FACE_MODEL = np.zeros([68, 3])
MEAN_FACE_MODEL_IDX = np.array(
    [36, 39, 45, 42, 30, 48, 54, 19, 21, 22, 24, 51, 57]
)
MEAN_FACE_MODEL[MEAN_FACE_MODEL_IDX] = np.array(
    [
        [-39.945, -29.992, 15.162],
        [-15.535, -27.362, 7.138],
        [48.48, -30.005, 15.483],
        [24.712, -27.328, 6.828],
        [0, 0, 0],
        [-20.147, 41.787, 10.324],
        [26.07, 41.539, 11.975],
        [-30.809, -52.5, 6.369],
        [-12.724, -48.967, 2.692],
        [20.901, -48.285, 0.376],
        [41.087, -53.016, 8.33],
        [1.829, 30.423, 3.142],
        [1.941, 51.462, 7.127],
    ]
)

__all__ = ["Eye3dPoseTransformList"]


class Eye3D:
    """Calculate eye 3d position with rvec and tvec from solvepnp.

    Args:
        face_model: 3d facemodel used in solvepnp with 68 3d points

    """

    def __init__(self, face_model: np.array):
        assert face_model.shape == (68, 3)
        self.facemodel = face_model[np.array([36, 39, 45, 42])]
        self.lefteye = (self.facemodel[0] + self.facemodel[1]) / 2
        self.lefteye = self.lefteye.reshape([3, 1])
        self.righteye = (self.facemodel[2] + self.facemodel[3]) / 2
        self.righteye = self.righteye.reshape([3, 1])

    def __call__(self, tvec: np.array, rvec: np.array):
        mat = cv2.Rodrigues(rvec)[0]
        left = mat @ self.lefteye + tvec.reshape([3, 1])
        right = mat @ self.righteye + tvec.reshape([3, 1])
        if tvec.reshape(-1)[-1] < 0:
            return -left.reshape(-1), -right.reshape(-1)
        return left.reshape(-1), right.reshape(-1)


class LdmkAug:
    """Aug ldmks with normal distribution or uniform distribution.

    Args:
        params: data augmentation parameters from train config

    """

    def __init__(self, params: dict):
        self.params = params
        self.distribution = params["distribution"]
        if self.distribution == "normal":
            self.scale = params["params"]["std"]
            self.clip = (
                params["params"]["clip"] / self.scale
                if self.scale
                else 10 ** 5
            )
            self.call_func = self.normal_call
        elif self.distribution == "uniform":
            self.scale = params["params"]["scale"]
            self.call_func = self.uniform_call
        else:
            raise NotImplementedError

    def __call__(self, ldmk: np.array):
        """Aug 2d ldmk.

        Args:
            ldmk: ldmks 2d, the first 4 ldmk index in 68 face points
                should be [36, 39, 45, 42].

        Returns:
            ldmks 2d with noise.
        """
        return self.call_func(ldmk)

    def normal_call(self, ldmk: np.array):
        rand_scale = (
            np.random.normal(0, 1, ldmk.shape).clip(-self.clip, self.clip)
            * self.scale
        )
        base_scale = np.linalg.norm(ldmk[0] - ldmk[2], ord=2)
        ldmk = ldmk + base_scale * rand_scale
        return ldmk

    def uniform_call(self, ldmk: np.array):
        rand_scale = np.random.uniform(-self.scale, self.scale, ldmk.shape)
        base_scale = np.linalg.norm(ldmk[0] - ldmk[2], ord=2)
        ldmk = ldmk + base_scale * rand_scale
        return ldmk


@OBJECT_REGISTRY.register
class Eye3dPoseTransformList:
    """Merge outputs from multiple different sub-transforms.

    Args:
        data_list: data info list
        aug_params: aug params. Defaults to None.
        group_channel: group channel. Defaults to 8.

    """

    def __init__(
        self,
        data_list: list,
        aug_params: dict = None,
        group_channel=8,
        ldmk_mean_scale: dict = None,
    ):
        self.data_list = data_list
        self.process_list = []
        self.ldmk_index = []
        self.ldmk_mean_scale = []
        for d in data_list:
            ldmk_idx = np.array(d["ldmk_idx"])
            assert ldmk_idx[:4].tolist() == [36, 39, 45, 42]
            self.process_list.append(Eye3dPoseTransform(**d["transform_args"]))
            self.ldmk_index.append(np.array(ldmk_idx))
            if ldmk_mean_scale:
                self.ldmk_mean_scale.append(
                    {
                        "ldmk_mean": np.array(
                            [
                                ldmk_mean_scale["ldmk_mean"][str(_)]
                                for _ in ldmk_idx
                            ]
                        ),
                        "ldmk_scale": np.array(
                            [
                                ldmk_mean_scale["ldmk_scale"][str(_)]
                                for _ in ldmk_idx
                            ]
                        ),
                    }
                )
            else:
                self.ldmk_mean_scale.append({})
            if (
                not d["transform_args"].get("face_model_path", False)
                and d["transform_args"]["use_pnp_eye3d"]
            ):
                for _ in ldmk_idx:
                    assert _ in MEAN_FACE_MODEL_IDX
        self.max_input_channel = max([_["input_shape"][0] for _ in data_list])
        if self.max_input_channel % group_channel:
            self.max_input_channel = group_channel * (
                self.max_input_channel // group_channel + 1
            )
        if aug_params:
            self.set_aug(aug_params)

    def set_aug(self, aug_params: dict):
        """Set data augmentation for each sub-transform.

        Args:
            aug_params: aug params
        """
        for _ in self.process_list:
            _.set_aug(aug_params)

    def __call__(self, data):
        return self.preprocess(data)

    def preprocess(self, data):
        ldmks = []
        rpys = []
        tvecs = []
        eye3ds = []
        rots = []
        for i, process in enumerate(self.process_list):
            ldmk_index = self.ldmk_index[i]
            ldmk_mean_scale = self.ldmk_mean_scale[i]
            ldmk, rpy, tvec, eye3d, rot = process.preprocess(
                data, ldmk_index, **ldmk_mean_scale
            )
            ldmk_tmp = torch.zeros([self.max_input_channel, 1, 1])
            len_ft = len(ldmk)
            ldmk_tmp[:len_ft] = ldmk
            ldmks.append(ldmk_tmp)
            rpys.append(rpy)
            tvecs.append(tvec)
            eye3ds.append(eye3d)
            rots.append(rot)
        return {
            "img": torch.cat(ldmks, dim=0),
            "gt_eye3d_pose": {
                "gt_pose": torch.cat(rpys, dim=0),
                "gt_eye3d": torch.cat(tvecs, dim=0),
                "gt_eye3d_xyz": torch.cat(eye3ds, dim=0),
                "gt_norm_rot": torch.cat(rots, dim=0),
            },
            "layout": "chw",
        }

    def postprocess(self, data, *model_output):
        result = []
        for i, process in enumerate(self.process_list):
            output = [_[:, 3 * i : 3 * (i + 1)] for _ in model_output]
            ldmk_index = self.ldmk_index[i]
            result.append(
                process.postprocess(data, *output, ldmk_index=ldmk_index)
            )
        return result


class Eye3dPoseTransform:
    """Get network inputs and transformed gts from data dicts.

    Args:
        t_mean: mean value to normalize eye3d depth
        t_std: std value to normalize eye3d depth
        use_pnp_eye3d: use eye3d calculated by solvepnp as gt for
            training if use_pnp_eye3d is True, use eye3d from data
            dict as gt for training if use_pnp_eye3d is False. Defaults
            to False.
        use_pnp_eye3d_test: use eye3d calculated by solvepnp as gt for
            testing if use_pnp_eye3d_test==True, use eye3d from data dict
            as gt for testing if use_pnp_eye3d==False. Defaults to False.
        use_gender: last value in output will be -1 for famale, 1 for
            male and 0 for unknown if use_gender==True, 0 for
            use_gender==False. Defaults to False.
        use_face_center: face center will be encoded into network input if
            use_face_center==True. Defaults to False.
        face_model_path: mean face model npy path for solvepnp with 68 3d
            points. default face model will be used if path is "".
            Defaults to "".
    """

    def __init__(
        self,
        t_mean: float,
        t_std: float,
        use_pnp_eye3d: bool = False,
        use_pnp_eye3d_test: bool = False,
        use_gender: bool = False,
        use_face_center: bool = False,
        face_model_path: str = "",
    ):
        self.t_mean = t_mean
        self.t_std = t_std
        self.use_pnp_eye3d = use_pnp_eye3d
        self.use_pnp_eye3d_test = use_pnp_eye3d_test
        if face_model_path:
            self.facemodel = np.load(face_model_path)
        else:
            self.facemodel = MEAN_FACE_MODEL
        self.train_aug = None
        self.use_gender = use_gender
        self.use_face_center = use_face_center
        self.eye3d = Eye3D(MEAN_FACE_MODEL)

    def set_aug(self, aug_params):
        self.train_aug = LdmkAug(aug_params)

    def _calc_rat_mat(self, normed_ldmks):
        optical_center_unitVec = np.array([0, 0, 1], np.float32)
        facecenter = normed_ldmks[1:4:2].mean(axis=0)
        face_center_ex = np.array([0, 0, 1], np.float32)
        face_center_ex[:2] = facecenter
        face_center_unitVec = face_center_ex / np.linalg.norm(
            face_center_ex, ord=2
        )
        theta = np.arccos(face_center_unitVec[-1])  # simplified dot prod
        unit_vec = np.cross(face_center_unitVec, optical_center_unitVec)
        unit_vec = unit_vec / np.linalg.norm(unit_vec, ord=2)
        rvec = theta * unit_vec
        return cv2.Rodrigues(rvec)[0]

    def preprocess(
        self,
        data: dict,
        ldmk_index: np.array,
        ldmk_mean: Any = 0,
        ldmk_scale: Any = 10,
    ):
        """Preprocess of transform.

        Args:
            data: data dict from dataset.
            ldmk_index: ldmk index

        Returns:
            list including network input as np.array with shape
            ((2*num_ldmk + 2[face ctr] + 1[gender]) x 1 x 1), pose gt
            with shape (3 x 1 x 1) and eye gt with shape (3 x 1 x 1).
        """
        # 1. get ldmks
        ldmks_unnorm = data["undistort_points_70"].reshape(-1, 2)
        ldmks_unnorm = ldmks_unnorm[2:][ldmk_index]
        # 2. add noise if necessary
        if self.train_aug:
            ldmks_unnorm = self.train_aug(ldmks_unnorm)
        if self.use_pnp_eye3d:
            retval, rvecs, tvecs = cv2.solvePnP(
                self.facemodel[ldmk_index],
                ldmks_unnorm,
                data["mtx"],
                np.zeros([1, 5]),
                cv2.SOLVEPNP_DLS,
            )
            lefteye, righteye = self.eye3d(tvecs, rvecs)
        # 3. norm ldmk with cx/cy and fx/fy
        ldmks = (ldmks_unnorm - data["mtx_4"][2:].reshape([1, 2])) / data[
            "mtx_4"
        ][:2].reshape([1, 2])
        facecenter = ldmks[1:4:2].mean(axis=0)
        # 4. norm with perspective
        R = self._calc_rat_mat(ldmks)
        ldmks_ex = np.ones(shape=(len(ldmks), 3), dtype=np.float32)
        ldmks_ex[:, :2] = ldmks
        ldmks_rotated = ldmks_ex @ R.T
        ldmks = ldmks_rotated[:, :2] / ldmks_rotated[:, 2:]
        ldmks = ldmks * ldmk_scale - ldmk_mean
        # 5. add face center
        if not self.use_face_center:
            facecenter = ldmks[1:4:2].mean(axis=0)
        ldmks = np.append(ldmks.reshape(-1), facecenter.reshape(-1))
        # 6. add gender
        gender = 0
        if self.use_gender and data.get("id_info", None):
            id_info = data["id_info"]
            gender = 1 if id_info["subject_info"]["gender"] == "male" else -1
        ldmks = np.append(ldmks, gender)
        # get normed pose gt
        rpy = data["rpy_norm"].reshape(-1) / 90
        # get normed T gt
        if not self.use_pnp_eye3d:
            lefteye = data["left_eye"].reshape(1, -1)
            righteye = data["right_eye"].reshape(1, -1)
        eye3ds = np.array([(_ @ R.T).reshape(-1) for _ in (lefteye, righteye)])
        z1, z2 = eye3ds[:, -1]
        z3 = self.t_mean
        tvec = np.array([z1, z2, z3])
        position = (tvec - self.t_mean) / self.t_std

        return [
            torch.from_numpy(_.reshape([-1, 1, 1]))
            for _ in [ldmks, rpy, position, eye3ds, R]
        ]  # noqa

    def postprocess(self, data: dict, rpy, depth, ldmk_index):
        """Postprocess of transform.

        Args:
            data: data dict from dataset.
            rpy: head pose info predicted by network
            depth: eye 3d position info predicted by network
            ldmk_index: ldmk index

        Returns:
            pose and eye3d predicted by network and gts
        """
        rpy = rpy.reshape(-1) * 90
        depth = (depth * self.t_std) + self.t_mean

        # recalculate R and ldmks
        ldmks_unnorm = data["undistort_points_70"].reshape(-1, 2)
        ldmks_unnorm = ldmks_unnorm[2:][ldmk_index]

        if self.use_pnp_eye3d_test:
            retval, rvecs, tvecs = cv2.solvePnP(
                self.facemodel[ldmk_index],
                ldmks_unnorm,
                data["mtx"],
                np.zeros([1, 5]),
                cv2.SOLVEPNP_DLS,
            )
            left_eye_gt, right_eye_gt = self.eye3d(tvecs, rvecs)
        ldmks = (ldmks_unnorm - data["mtx_4"][2:].reshape([1, 2])) / data[
            "mtx_4"
        ][:2].reshape([1, 2])
        R = self._calc_rat_mat(ldmks)
        ldmks_ex = np.ones(shape=(len(ldmks), 3), dtype=np.float32)
        ldmks_ex[:, :2] = ldmks
        ldmks_rotated = ldmks_ex @ R.T
        ldmks = ldmks_rotated[:, :2] / ldmks_rotated[:, 2:]
        ldmks = ldmks

        z = depth.reshape(-1)[0]
        eye_point_ex = np.array([0, 0, 1.0])
        eye_point_ex[:2] = (ldmks[0] + ldmks[1]) / 2
        eye_line_vec = eye_point_ex.reshape(-1)
        x = eye_line_vec[0] * z / eye_line_vec[-1]
        y = eye_line_vec[1] * z / eye_line_vec[-1]
        left_eye = R.T @ np.array([x, y, z]).reshape(3, 1)
        left_eye = left_eye.reshape(-1)

        z = depth.reshape(-1)[1]
        eye_point_ex = np.array([0, 0, 1.0])
        eye_point_ex[:2] = (ldmks[2] + ldmks[3]) / 2
        eye_line_vec = eye_point_ex.reshape(-1)
        x = eye_line_vec[0] * z / eye_line_vec[-1]
        y = eye_line_vec[1] * z / eye_line_vec[-1]
        right_eye = R.T @ np.array([x, y, z]).reshape(3, 1)
        right_eye = right_eye.reshape(-1)

        rpy_gt = data["rpy_norm"].reshape(rpy.shape)
        if not self.use_pnp_eye3d_test:
            left_eye_gt = data["left_eye"].reshape(left_eye.shape)
            right_eye_gt = data["right_eye"].reshape(left_eye.shape)
        return rpy, left_eye, right_eye, rpy_gt, left_eye_gt, right_eye_gt
