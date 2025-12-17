# Copyright (c) Horizon Robotics. All rights reserved.
import logging

import numpy as np

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ActionAddHandOrient",
    "ActionAddXYDiff",
    "ActionAddMotionInRgbBranch",
    "ActionKpsAddFingerEncoding",
    "ActionKpsAddTemporalEncoding",
]
logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ActionAddHandOrient(object):
    """Get hand orientation based on keypoints.

    Args:
        seq_len: Sequence length for clip.
            Default to 32.
        num_kps: num of kps, such as 21 for hand.
            Defaults to 21.
        use_3d_kps: whether to use 3d kps of hand.
            Defaults to False.
        use_kps_score: whether to use score of kps.
            Defaults to True.
        use_hand_orient: whether to use hand orient feature.
            Defaults to False.
        hand_orient_type: type of hand orient, support 'x', 'y', 'z',
            'xy', 'xz', 'yz', 'xyz'
            Defaults to 'z'.
    """

    def __init__(
        self,
        seq_len: float = 32,
        num_kps: float = 21,
        use_3d_kps: bool = False,
        use_kps_score: bool = True,
        use_hand_orient: bool = False,
        hand_orient_type: str = "z",
    ):
        self.num_kps = num_kps
        self.seq_len = seq_len
        self.use_3d_kps = use_3d_kps
        self.use_kps_score = use_kps_score
        self.use_hand_orient = use_hand_orient
        assert hand_orient_type in ["x", "y", "z", "xy", "xz", "yz", "xyz"]
        self.hand_orient_type = hand_orient_type

    def get_hand_orient(self, keypoint, is_left_hand, hand_orient_type):

        eps = 1e-5
        assert keypoint.shape == (21, 3)
        wrist_root = keypoint[0, :]
        index_finger_root = keypoint[5, :]
        little_finger_root = keypoint[17, :]

        # the vector from the wrist to the index
        # finger metacarpophalangeal joints
        orient_base1 = index_finger_root - wrist_root
        base1_norm_val = np.linalg.norm(orient_base1, axis=0, keepdims=True)
        orient_base1 = orient_base1 / (base1_norm_val + eps)

        # the vector from the wrist to the pinky
        # finger metacarpophalangeal joints
        orient_base2 = little_finger_root - wrist_root
        base2_norm_val = np.linalg.norm(orient_base2, axis=0, keepdims=True)
        orient_base2 = orient_base2 / (base2_norm_val + eps)

        orient_list = []
        if "y" in hand_orient_type or "x" in hand_orient_type:
            # the vector of y axis
            orient_y = orient_base1 + orient_base2
            y_norm_val = np.linalg.norm(orient_y, axis=0, keepdims=True)
            orient_y = orient_y / (y_norm_val + eps)
            if "y" in hand_orient_type:
                orient_list.append(orient_y)

        if "z" in hand_orient_type or "x" in hand_orient_type:
            # the vector of z axis
            orient_z = np.cross(orient_base1, orient_base2)
            z_norm_val = np.linalg.norm(orient_z, axis=0, keepdims=True)
            orient_z = orient_z / (z_norm_val + eps)
            # change z-direction according to the
            # left-right attribute of the hand
            if is_left_hand == 1:
                orient_z = -orient_z
            if "z" in hand_orient_type:
                orient_list.append(orient_z)

        if "x" in hand_orient_type:
            # the vector of x axis
            orient_x = np.cross(orient_y, orient_z)
            x_norm_val = np.linalg.norm(orient_x, axis=0, keepdims=True)
            orient_x = orient_x / (x_norm_val + eps)
            orient_list.append(orient_x)

        res = np.stack(orient_list, axis=0)
        return res

    def __call__(self, data):
        # todo: need to check and test
        if "clip_keypoints" in data:
            hand_orients = None
            if self.use_3d_kps:
                if self.use_kps_score:
                    only_xyz_keypoints = data["clip_keypoints"][:, :, :3]
                else:
                    only_xyz_keypoints = data["clip_keypoints"].copy()

                if self.use_hand_orient:
                    is_left_hand = data["is_left_hand"]
                    hand_orient_tmp_list = []
                    for t_step in range(self.seq_len):
                        hand_orient_tmp = self.get_hand_orient(
                            only_xyz_keypoints[t_step, :, :],
                            is_left_hand,
                            self.hand_orient_type,
                        )
                        hand_orient_tmp = hand_orient_tmp.reshape((-1,))
                        hand_orient_all_kps = np.expand_dims(
                            hand_orient_tmp, 0
                        ).repeat(self.num_kps, axis=0)
                        hand_orient_tmp_list.append(hand_orient_all_kps)
                    hand_orients = np.stack(hand_orient_tmp_list, axis=0)
            else:
                logger.warning(
                    "ActionAddHandOrient just support 'use_3d_kps is True'"
                )

            data["hand_orients"] = hand_orients
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"seq_len={self.seq_len}"
        repr_str += f"num_kps={self.num_kps}"
        repr_str += f"use_3d_kps={self.use_3d_kps}"
        repr_str += f"use_kps_score={self.use_kps_score}"
        repr_str += f"use_hand_orient={self.use_hand_orient}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionAddXYDiff(object):
    """Get kps xy diff feature based on keypoints.

    Args:
        use_3d_kps: whether to use 3d kps of hand.
            Defaults to False.
        use_kps_score: whether to use score of kps.
            Defaults to True.
        use_xy_diff: whether to use xy diff feature of kps.
            Defaults to False.
        xy_diff_step: step of xy diff, must low than seq_len.
            Defaults to 2.
    """

    def __init__(
        self,
        use_3d_kps: bool = False,
        use_kps_score: bool = True,
        use_xy_diff: bool = False,
        xy_diff_step: int = 2,
    ):

        self.use_3d_kps = use_3d_kps
        self.use_kps_score = use_kps_score
        self.use_xy_diff = use_xy_diff
        self.xy_diff_step = xy_diff_step

    def __call__(self, data):
        # todo: need to check and test
        if "clip_keypoints" in data:
            xy_diffs = None
            if self.use_xy_diff:
                if self.use_3d_kps:
                    if self.use_kps_score:
                        only_xyz_keypoints = data["clip_keypoints"][:, :, :3]
                    else:
                        only_xyz_keypoints = data["clip_keypoints"].copy()

                    xy_diffs = np.zeros(
                        (
                            only_xyz_keypoints.shape[0],
                            only_xyz_keypoints.shape[1],
                            2,
                        ),
                        dtype=only_xyz_keypoints.dtype,
                    )
                    xy_diffs[: -self.xy_diff_step] = (
                        only_xyz_keypoints[self.xy_diff_step :, :, :2]
                        - only_xyz_keypoints[: -self.xy_diff_step, :, :2]
                    )

                    # xy_diff norm
                    boxes = data["clip_boxes"]
                    max_width = (boxes[:, 2] - boxes[:, 0]).max()
                    max_height = (boxes[:, 3] - boxes[:, 1]).max()
                    xy_diff_norm = max(max_width, max_height)
                    xy_diffs = xy_diffs / xy_diff_norm
                else:
                    logger.warning(
                        "ActionAddXYDiff just support 'use_3d_kps is True'"
                    )

            data["xy_diffs"] = xy_diffs
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"use_3d_kps={self.use_3d_kps}"
        repr_str += f"use_kps_score={self.use_kps_score}"
        repr_str += f"use_xy_diff={self.use_xy_diff}"
        repr_str += f"xy_diff_step={self.xy_diff_step}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionAddMotionInRgbBranch(object):
    """Add motion(hand bbox center motion) in rgb branch.

    Args:
        add_motion_in_rgb_branch: whether to add motion in rgb branch.
            Defaults to True.
    """

    def __init__(self, add_motion_in_rgb_branch: bool = True):
        self.add_motion_in_rgb_branch = add_motion_in_rgb_branch

    def __call__(self, data):
        if "frames" in data and self.add_motion_in_rgb_branch:
            # rgb branch
            if data["frames"] is not None:
                boxes = data["clip_boxes"]
                box_max_border = data["max_border"]
                diff_box_x = boxes[:, 2] - boxes[:, 0]
                diff_box_y = boxes[:, 3] - boxes[:, 1]
                valid_flag = (diff_box_x > 0) * (diff_box_y > 0)
                invalid_flag = (diff_box_x <= 0) + (diff_box_y <= 0)
                box_center_x = (boxes[:, 0] + boxes[:, 2]) / 2
                box_center_y = (boxes[:, 1] + boxes[:, 3]) / 2
                if valid_flag.sum() > 0:
                    min_x = boxes[valid_flag][:, 0].min()
                    min_y = boxes[valid_flag][:, 1].min()
                    box_center_x -= min_x
                    box_center_x /= box_max_border
                    box_center_y -= min_y
                    box_center_y /= box_max_border
                box_center_x[invalid_flag] = 0
                box_center_y[invalid_flag] = 0

                box_center = np.concatenate((box_center_x, box_center_y))
                box_center = box_center[np.newaxis, :].repeat(
                    len(data["frames"]), axis=0
                )  # noqa
                box_center = box_center[:, np.newaxis, np.newaxis, :]
                data["box_center"] = box_center
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"add_motion_in_rgb_branch={self.add_motion_in_rgb_branch}"
        return repr_str


def copy_keypoints_to_dst(src, dst, score_in_src=True):
    dst_cp = dst.copy()
    # score info cp
    if score_in_src:
        dst_cp[:, :, -1] = src[:, :, -1]
        end_idx = src.shape[2] - 1
    else:
        end_idx = src.shape[2]
    # other dims
    dst_cp[:, :, :end_idx] = src[:, :, :end_idx]
    return dst_cp


@OBJECT_REGISTRY.register
class ActionKpsAddFingerEncoding(object):
    """Add finger encoding for gesture task.

    For more details, refer to wiki.
    http://wiki.hobot.cc/pages/viewpage.action?pageId=170616630

    Args:
        use_finger_encoding: whether to use finger encoding.
            Defaults to True.
        encoding_type: method of finger encoding.
            Support "naive", "naive_onehot", "finger_root_onehot",
            "joints_decompose_onehot"
            Defaults to "finger_root_onehot".
        as_separate_input: whether to use finger encoding as separate_input.
            Defaults to False.
        num_kps: num of kps, such as 21 for hand.
            Defaults to 21.
        use_kps_score: whether to use score of kps.
            Defaults to True.
    """

    def __init__(
        self,
        use_finger_encoding: bool = True,
        encoding_type: str = "finger_root_onehot",
        as_separate_input: bool = False,
        num_kps: float = 21,
        use_kps_score: bool = True,
    ):

        self.use_finger_encoding = use_finger_encoding
        self.encoding_type = encoding_type
        self.as_separate_input = as_separate_input
        self.num_kps = num_kps
        self.use_kps_score = use_kps_score

    def add_finger_encoding(
        self,
        keypoints,
        finger_encoding_type,
        finger_points_num=21,
        score_in_keypoints=True,
    ):
        assert finger_encoding_type in [
            "naive",
            "naive_onehot",
            "finger_root_onehot",
            "joints_decompose_onehot",
        ]
        assert keypoints.shape[1] == finger_points_num
        assert finger_points_num == 21
        orig_coding_dim = keypoints.shape[2]
        added_coding_dim = -1
        if finger_encoding_type == "naive":
            keypoints_coding = np.zeros(
                (keypoints.shape[0], keypoints.shape[1], orig_coding_dim + 1)
            )
            for k in range(finger_points_num):
                keypoints_coding[
                    :, k, orig_coding_dim - score_in_keypoints
                ] = (
                    k * 1.0 / finger_points_num
                )  # noqa
            added_coding_dim = 1
        elif finger_encoding_type == "naive_onehot":
            keypoints_coding = np.zeros(
                (
                    keypoints.shape[0],
                    keypoints.shape[1],
                    orig_coding_dim + finger_points_num,
                )
            )
            # one hot coding
            for k in range(finger_points_num):
                keypoints_coding[
                    :, k, orig_coding_dim - score_in_keypoints + k
                ] = 1
            added_coding_dim = finger_points_num
        elif (
            finger_encoding_type == "finger_root_onehot"
            or finger_encoding_type == "joints_decompose_onehot"
        ):
            finger2root_index_list = [
                [0],
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ]
            root_num = len(finger2root_index_list)
            added_coding_dim = root_num
            if finger_encoding_type == "joints_decompose_onehot":
                # each person has 4 finger knots, and the palm
                # can be counted as another type
                added_coding_dim += 5
            keypoints_coding = np.zeros(
                (
                    keypoints.shape[0],
                    keypoints.shape[1],
                    orig_coding_dim + added_coding_dim,
                )
            )
            finger2root_index_map = {}
            for root_idx in range(root_num):
                for finger_idx in finger2root_index_list[root_idx]:
                    finger2root_index_map[finger_idx] = root_idx

            # root coding
            for k in range(finger_points_num):
                assert k in finger2root_index_map
                root_idx = finger2root_index_map[k]
                keypoints_coding[
                    :, k, orig_coding_dim - score_in_keypoints + root_idx
                ] = 1  # noqa
            if finger_encoding_type == "joints_decompose_onehot":
                fingerknot2index_list = [
                    0,
                    1,
                    2,
                    3,
                    4,
                    1,
                    2,
                    3,
                    4,
                    1,
                    2,
                    3,
                    4,
                    1,
                    2,
                    3,
                    4,
                    1,
                    2,
                    3,
                    4,
                ]
                for k in range(finger_points_num):
                    knot_idx = fingerknot2index_list[k]
                    keypoints_coding[
                        :,
                        k,
                        orig_coding_dim
                        - score_in_keypoints
                        + root_num
                        + knot_idx,
                    ] = 1
        else:
            raise NotImplementedError

        keypoints = copy_keypoints_to_dst(
            keypoints, keypoints_coding.astype(np.float32), score_in_keypoints
        )
        assert added_coding_dim > 0
        return keypoints, added_coding_dim

    def __call__(self, data):
        if "clip_keypoints" in data:
            added_finger_dim = 0
            finger_enc = None
            if self.use_finger_encoding:
                keypoints = data["clip_keypoints"]
                keypoints, added_finger_dim = self.add_finger_encoding(
                    keypoints,
                    finger_encoding_type=self.encoding_type,
                    finger_points_num=self.num_kps,
                    score_in_keypoints=self.use_kps_score,
                )
                if self.as_separate_input:
                    finger_enc = keypoints[
                        :, :, -1 - added_finger_dim : -1
                    ].copy()
                    keypoints = np.concatenate(
                        [
                            keypoints[:, :, : -1 - added_finger_dim],
                            keypoints[:, :, -1:],
                        ],
                        axis=-1,
                    )
                data["clip_keypoints"] = keypoints
            data["finger_enc"] = finger_enc
            data["added_finger_dim"] = added_finger_dim
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"use_finger_encoding={self.use_finger_encoding}"
        repr_str += f"encoding_type={self.encoding_type}"
        repr_str += f"as_separate_input={self.as_separate_input}"
        repr_str += f"num_kps={self.num_kps}"
        repr_str += f"use_kps_score={self.use_kps_score}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionKpsAddTemporalEncoding(object):
    """Add temporal encoding for gesture task.

    Args:
        use_temporal_encoding: whether to use temporal encoding.
            Defaults to False.
        encoding_type: method of temporal encoding.
            Support "naive", "multiratio1", "multiratio2","cos_sin".
            Defaults to "multiratio1".
        as_separate_input: whether to use temporal encoding as separate_input.
            Defaults to False.
    """

    def __init__(
        self,
        use_temporal_encoding: bool = False,
        encoding_type: str = "multiratio1",
        as_separate_input: bool = False,
    ):
        self.use_temporal_encoding = use_temporal_encoding
        self.encoding_type = encoding_type
        self.as_separate_input = as_separate_input

    def add_time_encoding(self, keypoints, encoding_type):
        assert encoding_type in [
            "naive",
            "cos_sin",
            "multiratio1",
            "multiratio2",
        ]
        encoding_dim = -1
        orig_coding_dim = keypoints.shape[2]
        seq_len = keypoints.shape[0]
        if encoding_type == "naive":
            encoding_dim = 1
            keypoints_coding = np.zeros(
                (keypoints.shape[0], keypoints.shape[1], orig_coding_dim + 1)
            )
            for k in range(seq_len):
                keypoints_coding[k, :, -1] = k * 1.0 / seq_len
        elif encoding_type == "multiratio1" or encoding_type == "multiratio2":
            encoding_dim = 4
            ratio_list = [1, 2, 4, 8]
            keypoints_coding = np.zeros(
                (
                    keypoints.shape[0],
                    keypoints.shape[1],
                    orig_coding_dim + encoding_dim,
                )
            )
            if encoding_type == "multiratio1":
                for k in range(seq_len):
                    for i in range(encoding_dim):
                        keypoints_coding[k, :, orig_coding_dim + i] = (
                            ((k // ratio_list[i]) * ratio_list[i])
                            * 1.0
                            / seq_len
                        )
            else:
                for k in range(seq_len):
                    for i in range(encoding_dim):
                        keypoints_coding[k, :, orig_coding_dim + i] = (
                            k * 1.0 / (seq_len * ratio_list[i])
                        )
        else:
            raise NotImplementedError
        keypoints = copy_keypoints_to_dst(keypoints, keypoints_coding)
        assert encoding_dim > 0
        return keypoints, encoding_dim

    def __call__(self, data):
        if "clip_keypoints" in data:
            added_time_dim = 0
            time_enc = None
            if self.use_temporal_encoding:
                keypoints = data["clip_keypoints"]
                keypoints, added_time_dim = self.add_time_encoding(
                    keypoints, encoding_type=self.encoding_type
                )
                if self.as_separate_input:
                    time_enc = keypoints[:, :, -1 - added_time_dim : -1].copy()
                    keypoints = np.concatenate(
                        [
                            keypoints[:, :, : -1 - added_time_dim],
                            keypoints[:, :, -1:],
                        ],
                        axis=-1,
                    )
                data["clip_keypoints"] = keypoints
            data["time_enc"] = time_enc
            data["added_time_dim"] = added_time_dim
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"use_temporal_encoding={self.use_temporal_encoding}"
        repr_str += f"encoding_type={self.encoding_type}"
        repr_str += f"as_separate_input={self.as_separate_input}"
        return repr_str
