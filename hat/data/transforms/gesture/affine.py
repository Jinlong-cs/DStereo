# Copyright (c) Horizon Robotics. All rights reserved.

import math
import random
from typing import Dict

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY

__all__ = ["ActionRotateKpsFrames", "ActionFlipKpsFrames"]


@OBJECT_REGISTRY.register
class ActionRotateKpsFrames(object):
    """Rotate clip data of kps and img for action task.

    Args:
        p_rotate: clip data rotate probability, range between [0, 1].
            Defaults to 1.
        rotate_range: angle range of rotate.
            Defaults to 15.
        feat_ch: channel of kps, such as 4 for (x,y,z,score),
            3 for (x,y,score). Defaults to 3.
        use_rgb_branch: whether to use rgb branch.
            Defaults to False.
    """

    def __init__(
        self,
        p_rotate: float = 1,
        rotate_range: int = 15,
        feat_ch: int = 3,
        use_rgb_branch: bool = False,
    ):
        self._rotate = p_rotate
        self._rotate_range = rotate_range
        self._use_rgb_branch = use_rgb_branch
        # 4:xyzs or 3:xys
        self.feat_ch = feat_ch

    def _rotate_points(
        self, src_points, angle, src_img_shape, dst_img_shape, do_clip=True
    ):
        if angle == 0:
            return src_points  # (num_points, 2)
        # img_shape: [h, w, c]
        src_img_center = [src_img_shape[1] / 2.0, src_img_shape[0] / 2.0]
        dst_img_center = [dst_img_shape[1] / 2.0, dst_img_shape[0] / 2.0]
        radian = angle / 180.0 * math.pi
        radian_sin = math.sin(radian)
        radian_cos = math.cos(radian)
        dst_points = np.zeros(src_points.shape, dtype=np.float64)
        src_x = src_points[:, 0] - src_img_center[0]
        src_y = src_points[:, 1] - src_img_center[1]
        dst_points[:, 0] = radian_cos * src_x + radian_sin * src_y
        dst_points[:, 1] = -radian_sin * src_x + radian_cos * src_y

        dst_points[:, 0] += dst_img_center[0]
        dst_points[:, 1] += dst_img_center[1]
        if do_clip:
            dst_points[:, 0] = np.clip(
                dst_points[:, 0], 0, dst_img_shape[1] - 1
            )
            dst_points[:, 1] = np.clip(
                dst_points[:, 1], 0, dst_img_shape[0] - 1
            )
        return dst_points

    def _rotate_boxes(self, boxes, angle, src_img_shape, dst_img_shape):
        # boxes: (num_boxes, 4)  [x1, y1, x2, y2]
        if angle == 0:
            return boxes
        num_boxes = boxes.shape[0]
        x1 = boxes[:, 0, np.newaxis]
        y1 = boxes[:, 1, np.newaxis]
        x2 = boxes[:, 2, np.newaxis]
        y2 = boxes[:, 3, np.newaxis]
        lt = np.hstack([x1, y1])
        rt = np.hstack([x2, y1])
        lb = np.hstack([x1, y2])
        rb = np.hstack([x2, y2])
        src_points = np.vstack([lt, rt, lb, rb])
        dst_points = self._rotate_points(
            src_points, angle, src_img_shape, dst_img_shape
        )
        dst_lt = dst_points[:num_boxes, :]
        dst_rt = dst_points[num_boxes : num_boxes * 2, :]
        dst_lb = dst_points[num_boxes * 2 : num_boxes * 3, :]
        dst_rb = dst_points[num_boxes * 3 :, :]
        dst_boxes = np.zeros(boxes.shape, dtype=boxes.dtype)
        dst_boxes[:, 0] = np.concatenate(
            (dst_lt[:, 0:1], dst_lb[:, 0:1], dst_rt[:, 0:1], dst_rb[:, 0:1]),
            axis=1,
        ).min(axis=1)
        dst_boxes[:, 1] = np.concatenate(
            (dst_lt[:, 1:2], dst_lb[:, 1:2], dst_rt[:, 1:2], dst_rb[:, 1:2]),
            axis=1,
        ).min(axis=1)
        dst_boxes[:, 2] = np.concatenate(
            (dst_lt[:, 0:1], dst_lb[:, 0:1], dst_rt[:, 0:1], dst_rb[:, 0:1]),
            axis=1,
        ).max(axis=1)
        dst_boxes[:, 3] = np.concatenate(
            (dst_lt[:, 1:2], dst_lb[:, 1:2], dst_rt[:, 1:2], dst_rb[:, 1:2]),
            axis=1,
        ).max(axis=1)
        return dst_boxes

    def _rotate_image(
        self, src, angle, flags=cv2.INTER_LINEAR, border_value=0
    ):
        if angle == 0:
            return src
        w = src.shape[1]
        h = src.shape[0]
        radian = angle / 180.0 * math.pi
        radian_sin = math.sin(radian)
        radian_cos = math.cos(radian)
        new_w = int(abs(radian_cos * w) + abs(radian_sin * h))
        new_h = int(abs(radian_sin * w) + abs(radian_cos * h))
        rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rot_mat[0, 2] += (new_w - w) / 2.0
        rot_mat[1, 2] += (new_h - h) / 2.0

        if (len(src.shape) == 3 and src.shape[2] == 1) or len(src.shape) == 2:
            border_value = (border_value,)
        else:
            border_value = (border_value, border_value, border_value)
        dst_img = cv2.warpAffine(
            src, rot_mat, (new_w, new_h), flags=flags, borderValue=border_value
        )
        return dst_img

    def _get_rotate_img_shape(self, angle, img_size):
        h, w = img_size
        radian = angle / 180.0 * math.pi
        radian_sin = math.sin(radian)
        radian_cos = math.cos(radian)
        new_w = int(abs(radian_cos * w) + abs(radian_sin * h))
        new_h = int(abs(radian_sin * w) + abs(radian_cos * h))
        return (new_h, new_w)

    def _rotate_rgb_branch(
        self, imgs, clip_boxes, img_shape, angle, new_img_shape
    ):

        for idx in range(len(imgs)):
            if imgs[idx] is not None:
                imgs[idx] = self._rotate_image(imgs[idx], angle)
        # todo: check this func
        clip_boxes = self._rotate_boxes(
            clip_boxes, angle, img_shape, new_img_shape
        )
        return imgs, clip_boxes

    def _rotate_kps_branch(
        self, src_kps, angle, src_img_shape, dst_img_shape, feat_ch=3
    ):
        # src_kps: (num_boxes, num_kps*(2/3/4))  [x1, y1, ...]
        if angle == 0:
            return src_kps
        num_boxes = src_kps.shape[0]
        assert src_kps.shape[1] % feat_ch == 0
        src_kps = src_kps.reshape((-1, feat_ch))
        dst_kps = src_kps.copy()
        dst_kps[:, :2] = self._rotate_points(
            src_kps[:, :2], angle, src_img_shape, dst_img_shape
        )
        return dst_kps.reshape((num_boxes, -1))

    def __call__(self, data):
        if np.random.choice([False, True], p=[1 - self._rotate, self._rotate]):
            angle = random.randint(-self._rotate_range, self._rotate_range)
            new_img_shape = self._get_rotate_img_shape(
                angle, data["img_shape"][:2]
            )
            # kps branch
            data["clip_keypoints"] = self._rotate_kps_branch(
                data["clip_keypoints"],
                angle,
                data["img_shape"],
                new_img_shape,
                feat_ch=self.feat_ch,
            )

            if self._use_rgb_branch:
                if data["frames"] is not None:
                    (
                        data["frames"],
                        data["clip_boxes"],
                    ) = self._rotate_rgb_branch(
                        data["frames"],
                        data["clip_boxes"],
                        data["img_shape"],
                        angle,
                        new_img_shape,
                    )
            data["img_shape"] = new_img_shape
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"rotate={self._rotate}"
        repr_str += f"rotate_range={self.__rotate_rangerotate}"
        repr_str += f"use_rgb_branch={self._use_rgb_branch}"
        repr_str += f"feat_ch={self.feat_ch}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionFlipKpsFrames(object):
    """Flip for img clip and convert label.

    Convert label according to whether or not to flip.
        example: wave left is mirrored and gt label
        become waving to the right.

    Args:
        p_flip: img flip probability. Defaults to 0.
        feat_ch: kps channel. Defaults to 3.
        flip_label_dict: label mapping before and after flipping.
            Defaults to None.
    """

    def __init__(
        self, p_flip: float = 0, feat_ch: int = 3, flip_label_dict: Dict = None
    ):
        self._flip = p_flip
        self.feat_ch = feat_ch
        self.flip_label_dict = flip_label_dict

    def _flip_boxes(self, boxes, img_width):
        # boxes: (num_boxes, 4)  [x1, y1, x2, y2]
        dst_boxes = boxes.copy()
        dst_boxes[:, 0] = img_width - boxes[:, 2] - 1.0
        dst_boxes[:, 2] = img_width - boxes[:, 0] - 1.0
        return dst_boxes

    def _flip_hand_kps(self, src_kps, img_width, feat_ch=3):
        # src_kps: (num_boxes, num_kps*feat_ch)
        num_kps = src_kps.shape[1] / feat_ch
        assert num_kps == 21, "just support 21 hand keypoints"
        dst_kps = src_kps.copy()
        dst_kps[:, ::feat_ch] = img_width - src_kps[:, ::feat_ch] - 1.0
        return dst_kps

    def __call__(self, data):
        if np.random.choice([False, True], p=[1 - self._flip, self._flip]):
            if "frames" in data:
                # rgb branch
                if data["frames"] is not None:
                    if "img_shape" not in data:
                        # todo: delete this assert
                        raise ValueError(
                            "img shape is taken by the class \
                            'ActionClipDataPreProcess'"
                        )

                    for idx in range(len(data["frames"])):
                        data["frames"][idx] = data["frames"][idx][:, ::-1, :]
                    data["clip_boxes"] = self._flip_boxes(
                        data["clip_boxes"], data["img_shape"][1]
                    )

            # kps branch
            data["clip_keypoints"] = self._flip_hand_kps(
                data["clip_keypoints"], data["img_shape"][1], self.feat_ch
            )
            data["act_label"] = self.flip_label_dict.get(
                data["act_label"], data["act_label"]
            )

            if "is_left_hand" not in data:
                raise ValueError(
                    "is_left_hand is taken by the class \
                    'ActionClipDataPreProcess'"
                )
            # change hand_attrs if kps flipped
            data["is_left_hand"] = 1 - data["is_left_hand"]
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"use_flip={self._flip}"
        repr_str += f"feat_ch={self.feat_ch}"
        repr_str += f"flip_label_dict={self.flip_label_dict}"
        return repr_str
