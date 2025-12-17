import random

import cv2
import numpy as np
import torch

from hat.data.transforms.detection import ToTensor
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ActionKpsReshape",
    "ActionKpsScoreNormalize",
    "ActionKpsSmooth",
    "ActionKpsNormalize",
    "ActionKpsJitter",
    "ActionKpsRandScale",
    "ActionKpsToTensor",
]


@OBJECT_REGISTRY.register
class ActionKpsReshape(object):
    """Reshape hand kps for gesture task.

    Args:
        num_kps: num of kps, such as 21 for hand.
            Defaults to 21.
        feat_ch: kps channel. Defaults to 3.
    """

    def __init__(
        self,
        num_kps: int = 21,
        feat_ch: int = 3,
    ):

        self.num_kps = num_kps
        self.feat_ch = feat_ch

    def __call__(self, data):
        if "clip_keypoints" in data:
            data["clip_keypoints"] = data["clip_keypoints"].reshape(
                (-1, self.num_kps, self.feat_ch)
            )
        else:
            raise ValueError("clip_keypoints not in data")
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"num_kps={self.num_kps}"
        repr_str += f"feat_ch={self.feat_ch}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionKpsScoreNormalize(object):
    """Normalize kps confidence score.

    Args:
        use_kps_score: whether to use score of kps.
            Defaults to True.
        disable_kps_conf: whether to change kps confidence score to zero.
            Defaults to False.
        use_3d_kps: whether to use 3d kps of hand.
            Defaults to False.
        kps_score_norm_scale: scale of normalize.
            Defaults to 4.897640403536304.
    """

    def __init__(
        self,
        use_kps_score: bool = True,
        disable_kps_conf: bool = False,
        use_3d_kps: bool = True,
        kps_score_norm_scale: float = 4.897640403536304,
    ):

        self.use_kps_score = use_kps_score
        self.disable_kps_conf = disable_kps_conf
        self.kps_score_norm_scale = kps_score_norm_scale
        self.use_3d_kps = use_3d_kps
        self._norm_kps_conf = False

    def __call__(self, data):
        if self.use_kps_score:
            score_idx = 3 if self.use_3d_kps else 2  # xyzs or xys
            if self.disable_kps_conf:
                data["clip_keypoints"][:, :, score_idx] = 0
            else:
                if data["clip_keypoints"][:, :, score_idx].max() > 2.0:
                    self._norm_kps_conf = True
                if self._norm_kps_conf:
                    data["clip_keypoints"][:, :, score_idx] = (
                        data["clip_keypoints"][:, :, score_idx]
                        / self.kps_score_norm_scale
                    )  # noqa
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"use_kps_score={self.use_kps_score}"
        repr_str += f"disable_kps_conf={self.disable_kps_conf}"
        repr_str += f"kps_score_norm_scale={self.kps_score_norm_scale}"
        repr_str += f"use_3d_kps={self.use_3d_kps}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionKpsSmooth(object):
    """Smooth kps for action task.

    Args:
        p_smooth_kps:  smooth kps probability, range between [0, 1].
            Defaults to 0.
        smooth_method (str, optional): method of smooth kps,
            support 'blur', 'GaussianBlur'.
            Defaults to 'blur'.
    """

    def __init__(
        self,
        p_smooth_kps: float = 0,
        smooth_method: str = "blur",
    ):
        self._smooth = p_smooth_kps
        self._smooth_method = smooth_method

    def __call__(self, data):
        if np.random.choice([False, True], p=[1 - self._smooth, self._smooth]):
            if "clip_keypoints" in data:
                if self._smooth_method == "blur":
                    data["clip_keypoints"] = cv2.blur(
                        data["clip_keypoints"], ksize=(1, 3)
                    )
                elif self._smooth_method == "GaussianBlur":
                    data["clip_keypoints"] = cv2.GaussianBlur(
                        data["clip_keypoints"],
                        ksize=(1, 3),
                        sigmaX=0.0,
                        sigmaY=1.0,
                    )
                else:
                    raise ValueError(
                        f"Smooth method support blur,GaussianBlur.  \
                        this method '{self._smooth_method}' is not supported"
                    )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"use_smooth_kps={self._smooth}"
        repr_str += f"smooth_method={self._smooth_method}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionKpsNormalize(object):
    """Normalize kps to [-1,1] for gesture task.

    Args:
        use_3d_kps: whether to use 3d kps of hand.
            Defaults to False.
        z_norm_type: norm type of z axis for 3d kps.
            Support "z_abs_max", "follow_xy"
            Defaults to "z_abs_max".
        use_kps_score: whether to use score of kps.
            Defaults to True.
        norm_type: norm type of x,y axis for kps.
            Support "center_norm", "center_norm1",
            "center_norm_resp", "diff_norm".
            Defaults to "center_norm".
        center_idxs: center normalize point indexes.
            Defaults to [9].
        center_type: method of getting center coordinate.
            Support "fixed", "median", "fixed_all", "median_all".
            Defaults to "fixed".
        multiscale_norm_ratio_list: list of multi norm ratio for kps.
            Defaults to [1.0].
        multi_ratio_channel: apply multis scale for channel 'xy' or 'xyz'.
            Defaults to 'xyz'.
        kps_num_list: kps num for two hand gesture task.
            Defaults to [21].
        fix_norm_len: method of getting max border.
            Support 'height', 'width' for two hand task.
            For one hand gesture task, must set to None.
            Defaults to None.
    """

    def __init__(
        self,
        use_3d_kps: bool = False,
        z_norm_type: str = "z_abs_max",
        use_kps_score: bool = True,
        norm_type: str = "center_norm",
        center_idxs: list = (9,),
        center_type: str = "fixed",
        multiscale_norm_ratio_list: list = (1.0,),
        multi_ratio_channel: str = "xyz",
        kps_num_list: list = (21,),
        fix_norm_len: str = None,
    ):
        self.use_3d_kps = use_3d_kps
        self.use_kps_score = use_kps_score
        self.norm_type = norm_type
        self.z_norm_type = z_norm_type
        self.center_type = center_type
        self.fix_norm_len = fix_norm_len
        self.multiscale_norm_ratio_list = multiscale_norm_ratio_list
        self.multi_ratio_channel = multi_ratio_channel
        self.kps_num_list = kps_num_list

        self.center_idxs = np.array(center_idxs)
        if len(self.center_idxs.shape) == 1:
            self.center_idxs = self.center_idxs.reshape((1, -1))
        assert len(self.center_idxs.shape) == 2
        assert len(self.kps_num_list) == self.center_idxs.shape[0]

    def center_coord(self, points, points_idxs, reduce_type="mean"):
        assert (reduce_type == "mean") or (reduce_type == "median")
        ndim = len(points.shape)
        if points_idxs is not None:
            num = len(points_idxs)
            assert num > 0 and num < len(points), (
                "points indexes number should be"
                "greater than 0 and less than {}".format(len(points))
            )
            if not isinstance(points_idxs, np.ndarray):
                points_idxs = np.array(points_idxs)
        if ndim == 2:
            assert points_idxs is not None
            center = points[points_idxs]
            x_c = center[:, 0].sum() / num
            y_c = center[:, 1].sum() / num
        elif ndim == 3:
            if points_idxs is not None:
                # mean or median across all frames for selected points
                center = points[:, points_idxs]
                if reduce_type == "mean":
                    x_c = center[:, 0, 0].mean()
                    y_c = center[:, 0, 1].mean()
                else:
                    x_c = np.median(center[:, 0, 0])
                    y_c = np.median(center[:, 0, 1])
            else:
                assert reduce_type == "median"
                # use median of all points
                x_c = np.median(points[:, :, 0])
                y_c = np.median(points[:, :, 1])
        else:
            raise ValueError
        return (x_c, y_c)

    def kps_center_norm(
        self,
        keypoints,
        kps_num_list,
        boxes,
        center_idxs,
        center_type,
        fix_norm_len,
        img_size,
        norm_type,
        use_3d_kps,
        z_norm_type,
        use_kps_score,
        norm_ratio_list,
        multi_ratio_channel,
        ignore,
    ):
        assert len(kps_num_list) * 4 == boxes.shape[1]
        boxes = boxes.reshape((-1, len(kps_num_list), 4))
        keypoints_final_list = []
        accum_idx = 0

        for cur_idx, kps_num in enumerate(kps_num_list):
            assert kps_num > 0
            keypoints_i = keypoints[:, accum_idx : accum_idx + kps_num, :]
            boxes_i = boxes[:, cur_idx, :]
            accum_idx += kps_num

            center_idx_i = center_idxs[cur_idx]
            if len(center_idx_i) == 1 and center_idx_i[0] < 0:
                # for two hands hgr human body normalization
                root = (
                    (boxes_i[0, 0] + boxes_i[0, 2]) / 2,
                    (boxes_i[0, 1] + boxes_i[0, 3]) / 2,
                )
            elif center_type == "fixed":
                root = self.center_coord(keypoints_i[0], center_idx_i)
            elif center_type == "median":
                root = self.center_coord(keypoints_i, center_idx_i, "median")
            elif center_type == "fixed_all":
                root = self.center_coord(keypoints_i, center_idx_i)
            elif center_type == "median_all":
                root = self.center_coord(keypoints_i, None, "median")
            else:
                raise NotImplementedError
            keypoints_i[:, :, :2] = keypoints_i[:, :, :2] - root

            # norm keypoints value according to max border value
            if fix_norm_len == "height" and img_size[0] > 1:
                max_border = float(img_size[0]) / 2
            elif fix_norm_len == "width" and img_size[1] > 1:
                max_border = float(img_size[1]) / 2
            else:
                if norm_type == "center_norm1":
                    enclose_width = boxes_i[:, 2].max() - boxes_i[:, 0].min()
                    enclose_height = boxes_i[:, 3].max() - boxes_i[:, 1].min()
                    max_border = max(enclose_width, enclose_height)
                elif norm_type == "center_norm":
                    max_width = (boxes_i[:, 2] - boxes_i[:, 0]).max()
                    max_height = (boxes_i[:, 3] - boxes_i[:, 1]).max()
                    max_border = max(max_width, max_height)
                else:
                    raise NotImplementedError(
                        "unrecognized norm_type {}".format(norm_type)
                    )

            if max_border < 1:
                # use image height
                max_border = max(100, float(img_size[0]) / 2)
                ignore = True

            if use_3d_kps:
                if z_norm_type == "z_abs_max":
                    max_z_value = np.max(np.abs(keypoints_i[:, :, 2]))
                elif z_norm_type == "follow_xy":
                    max_z_value = max_border
                else:
                    assert z_norm_type.isnumeric()
                    max_z_value = int(z_norm_type)

            xyz_len = (
                (keypoints_i.shape[2] - 1)
                if use_kps_score
                else keypoints_i.shape[2]
            )
            assert xyz_len == (3 if use_3d_kps else 2)
            # multi_ratio apply to xyz coordinate  # noqa
            keypoints_trans_len = xyz_len * len(norm_ratio_list)
            if use_3d_kps and multi_ratio_channel == "xy":
                # multi_ratio not apply to z coordinate
                keypoints_trans_len -= len(norm_ratio_list) - 1
            if use_kps_score:
                keypoints_trans_len += 1  # kps score used
            keypoints_after_trans = np.zeros(
                (
                    keypoints_i.shape[0],
                    keypoints_i.shape[1],
                    keypoints_trans_len,
                ),
                dtype=np.float32,
            )
            cur_idx = 0
            for k in range(len(norm_ratio_list)):
                max_border_xy = max_border * norm_ratio_list[k]
                keypoints_after_trans[:, :, cur_idx] = (
                    keypoints_i[:, :, 0] / max_border_xy
                )
                keypoints_after_trans[:, :, cur_idx + 1] = (
                    keypoints_i[:, :, 1] / max_border_xy
                )
                cur_idx += 2
                if use_3d_kps and (
                    multi_ratio_channel == "xyz" or norm_ratio_list[k] == 1.0
                ):
                    keypoints_after_trans[:, :, cur_idx] = keypoints_i[
                        :, :, 2
                    ] / (max_z_value * norm_ratio_list[k])
                    cur_idx += 1

            if use_kps_score:
                assert (cur_idx + 1) == keypoints_after_trans.shape[2]
                keypoints_after_trans[:, :, -1] = keypoints_i[:, :, -1]
            else:
                assert cur_idx == keypoints_after_trans.shape[2]

            keypoints_final_list.append(keypoints_after_trans)
        keypoints = np.concatenate(keypoints_final_list, axis=1)
        return keypoints, max_border, ignore

    def kps_center_norm_resp(
        self, center_idxs, keypoints, boxes, img_size, ignore
    ):
        # center align each frame with its center points
        # keypoints shape must be (num_rois, num_kps, 3)
        for idx in range(keypoints.shape[0]):
            root = self.center_coord(keypoints[idx], center_idxs)
            keypoints[idx, :, :2] = keypoints[idx, :, :2] - root[:]

        # norm keypoints value according to max border value
        max_width = (boxes[:, 2] - boxes[:, 0]).max()
        max_height = (boxes[:, 3] - boxes[:, 1]).max()
        max_border = max(max_width, max_height)
        if max_border < 1:
            # use image height
            max_border = max(100, float(img_size[0]) / 2)
            ignore = True
        keypoints[:, :, 0] = keypoints[:, :, 0] / max_border
        keypoints[:, :, 1] = keypoints[:, :, 1] / max_border
        return keypoints, max_border, ignore

    def kps_diff_norm(self, keypoints, boxes, img_size, ignore):
        # center align each frame with its center points
        # keypoints shape must be (num_rois, num_kps, 3)
        keypoints_copy = keypoints.copy()
        for idx in range(1, keypoints.shape[0]):
            keypoints[idx, :, :2] = (
                keypoints_copy[idx, :, :2] - keypoints_copy[idx - 1, :, :2]
            )  # noqa
        keypoints[0, :, :] = keypoints[1, :, :]

        # norm keypoints value according to max border value
        max_width = (boxes[:, 2] - boxes[:, 0]).max()
        max_height = (boxes[:, 3] - boxes[:, 1]).max()
        max_border = max(max_width, max_height)
        if max_border < 1:
            # use image height
            max_border = max(100, float(img_size[0]) / 2)
            ignore = True
        keypoints[:, :, 0] = keypoints[:, :, 0] / max_border
        keypoints[:, :, 1] = keypoints[:, :, 1] / max_border
        return keypoints, max_border, ignore

    def act_kps_input_norm(
        self,
        keypoints,
        boxes,
        img_size,
        norm_type,
        z_norm_type,
        center_idxs,
        center_type,
        fix_norm_len,
        multiscale_norm_ratio_list,
        use_3d_kps,
        use_kps_score,
        multi_ratio_channel,
        kps_num_list,
    ):
        ignore = False
        max_border = 1e5
        if norm_type == "center_norm" or norm_type == "center_norm1":
            keypoints, max_border, ignore = self.kps_center_norm(
                keypoints,
                kps_num_list,
                boxes,
                center_idxs,
                center_type,
                fix_norm_len,
                img_size,
                norm_type,
                use_3d_kps,
                z_norm_type,
                use_kps_score,
                multiscale_norm_ratio_list,
                multi_ratio_channel,
                ignore,
            )
        elif norm_type == "center_norm_resp":
            keypoints, max_border, ignore = self.kps_center_norm_resp(
                center_idxs, keypoints, boxes, img_size, ignore
            )
        elif norm_type == "diff_norm":
            keypoints, max_border, ignore = self.kps_diff_norm(
                center_idxs, keypoints, boxes, img_size, ignore
            )
        else:
            raise ValueError(
                f"kps norm type just support  \
                'center_norm','center_norm1','center_norm_resp','diff_norm', \
                not support {norm_type}"
            )
        return keypoints, max_border, ignore

    def __call__(self, data):
        if "clip_keypoints" in data:
            (
                data["clip_keypoints"],
                data["max_border"],
                data["is_ignore"],
            ) = self.act_kps_input_norm(  # noqa
                data["clip_keypoints"],
                data["clip_boxes"],
                data["img_shape"],
                self.norm_type,
                z_norm_type=self.z_norm_type,
                center_idxs=self.center_idxs,
                center_type=self.center_type,
                fix_norm_len=self.fix_norm_len,
                multiscale_norm_ratio_list=self.multiscale_norm_ratio_list,
                use_3d_kps=self.use_3d_kps,
                use_kps_score=self.use_kps_score,
                multi_ratio_channel=self.multi_ratio_channel,
                kps_num_list=self.kps_num_list,
            )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"use_3d_kps={self.use_3d_kps}"
        repr_str += f"use_kps_score={self.use_kps_score}"
        repr_str += f"norm_type={self.norm_type}"
        repr_str += f"z_norm_type={self.z_norm_type}"
        repr_str += f"center_idxs={self.center_idxs}"
        repr_str += f"center_type={self.center_type}"
        repr_str += f"fix_norm_len={self.fix_norm_len}"
        repr_str += (
            f"multiscale_norm_ratio_list={self.multiscale_norm_ratio_list}"
        )
        repr_str += f"multi_ratio_channel={self.multi_ratio_channel}"
        repr_str += f"kps_num_list={self.kps_num_list}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionKpsJitter(object):
    """Add jitter noise to keypoints.

    Args:
        p_point_jitter: point jitter probability.
            Defaults to 0.5.
        jitter_std_value: standard deviation of noise.
            Defaults to 0.03.
    """

    def __init__(
        self,
        p_point_jitter: float = 0.5,
        jitter_std_value: float = 0.03,
    ):

        self._point_jitter = p_point_jitter
        self.jitter_std_value = jitter_std_value

    def do_point_jitter(self, points, std_value):
        shape = points.shape
        for i in range(shape[2]):
            scale_i = min(np.abs(points[:, :, i]).mean(), 1.5)
            noise = np.random.normal(
                loc=0.0, scale=std_value * scale_i, size=shape[:2]
            )
            points[:, :, i] = np.add(points[:, :, i], noise)
        return points

    def __call__(self, data):
        if np.random.choice(
            [False, True], p=[1 - self._point_jitter, self._point_jitter]
        ):
            if "clip_keypoints" in data:
                data["clip_keypoints"] = self.do_point_jitter(
                    data["clip_keypoints"], std_value=self.jitter_std_value
                )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"point_jitter={self._point_jitter}"
        repr_str += f"jitter_std_value={self.jitter_std_value}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionKpsRandScale(object):
    """Rand scale for keypoints.

    Args:
        p_rand_scale: keypoints scale probability.
            Defaults to 1.0.
        rand_scale_mu: the mean of gaussian distribution.
            Defaults to 1.0.
        rand_scale_sigma: the standard deviation of gaussian distribution.
            Defaults to 0.15.
    """

    def __init__(
        self,
        p_rand_scale: float = 1.0,
        rand_scale_mu: float = 1.0,
        rand_scale_sigma: float = 0.1,
    ):
        self._rand_scale = p_rand_scale
        self.rand_scale_mu = rand_scale_mu
        self.rand_scale_sigma = rand_scale_sigma

    def __call__(self, data):
        if np.random.choice(
            [False, True], p=[1 - self._rand_scale, self._rand_scale]
        ):
            if "clip_keypoints" in data:
                scale = random.gauss(self.rand_scale_mu, self.rand_scale_sigma)
                data["clip_keypoints"][:, :, :-1] *= scale
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"rand_scale={self._rand_scale}"
        repr_str += f"rand_scale_mu={self.rand_scale_mu}"
        repr_str += f"rand_scale_sigma={self.rand_scale_sigma}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionKpsToTensor(ToTensor):
    """Convert objects of various python types to torch.

    Args:
        tensor_layout: layout of tensor, support 'chw','hwc'.
            Defaults to 'chw'.
        use_3d_kps: whether to use 3d kps of hand.
            Defaults to False.
        hand_orient_concat: whether to concat hand orient to kps,
            only when use 3d kps is True.
            Defaults to False.
        xy_diff_concat: whether to concat xy diff to kps.
            only when use 3d kps is True.
            Defaults to False.
    """

    def __init__(
        self,
        tensor_layout: str = "chw",
        use_3d_kps: bool = False,
        hand_orient_concat: bool = False,
        xy_diff_concat: bool = False,
    ):
        super(ActionKpsToTensor, self).__init__()
        self.tensor_layout = tensor_layout
        self.use_3d_kps = use_3d_kps
        self.hand_orient_concat = hand_orient_concat
        self.xy_diff_concat = xy_diff_concat

        if self.hand_orient_concat or self.xy_diff_concat:
            assert (
                self.use_3d_kps
            ), "Concat hand_orient/xy_diff to Kps only when using 3d kps"

    def __call__(self, data):
        # step1: convert the layout from hwc to chw
        if self.tensor_layout == "chw":
            if "kps_layout" not in data:
                data["kps_layout"] = "hwc"
            if "clip_keypoints" in data:
                data["clip_keypoints"], _ = self._convert_layout(
                    data["clip_keypoints"], data["kps_layout"]
                )

            if "hand_orients" in data:
                if data["hand_orients"] is not None:
                    data["hand_orients"], _ = self._convert_layout(
                        data["hand_orients"], data["kps_layout"]
                    )

            if "xy_diffs" in data:
                if data["xy_diffs"] is not None:
                    data["xy_diffs"], _ = self._convert_layout(
                        data["xy_diffs"], data["kps_layout"]
                    )
            if "box_center" in data:
                data["box_center"] = data["box_center"].transpose((0, 3, 1, 2))
            data["kps_layout"] = "chw"

        # step2: convert to tensor
        if "clip_keypoints" in data:
            data["clip_keypoints"] = self._to_tensor(data["clip_keypoints"])
        if "hand_orients" in data:
            if data["hand_orients"] is not None:
                data["hand_orients"] = self._to_tensor(data["hand_orients"])
        if "xy_diffs" in data:
            if data["xy_diffs"] is not None:
                data["xy_diffs"] = self._to_tensor(data["xy_diffs"])
        if "box_center" in data:
            data["box_center"] = self._to_tensor(data["box_center"])
        if "finger_enc" in data:
            if data["finger_enc"] is not None:
                data["finger_enc"] = self._to_tensor(data["finger_enc"])
        if "time_enc" in data:
            if data["time_enc"] is not None:
                data["time_enc"] = self._to_tensor(data["time_enc"])

        # step3: concat feaure to kps
        if self.use_3d_kps:
            # be careful. not test
            if self.hand_orient_concat and data["hand_orients"] is not None:
                data["clip_keypoints"] = torch.cat(
                    (data["clip_keypoints"], data["hand_orients"]), dim=1
                )
                data.pop("hand_orients")
            if self.xy_diff_concat and data["xy_diffs"] is not None:
                data["clip_keypoints"] = torch.cat(
                    (data["clip_keypoints"], data["xy_diffs"]), dim=1
                )
                data.pop("xy_diffs")
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"to_tensor={self.to_tensor}"
        repr_str += f"tensor_layout={self.tensor_layout}"
        repr_str += f"use_3d_kps={self.use_3d_kps}"
        repr_str += f"hand_orient_concat={self.hand_orient_concat}"
        repr_str += f"xy_diff_concat={self.xy_diff_concat}"
        return repr_str
