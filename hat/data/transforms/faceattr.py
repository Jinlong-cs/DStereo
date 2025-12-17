# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import random

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

logger = logging.getLogger(__name__)

__all__ = [
    "TransformAgeLabel",
    "RandomOcclusion",
]


@OBJECT_REGISTRY.register
class TransformAgeLabel(object):
    """Different presentation of age, used when calculating loss.

    Refer to https://arxiv.org/abs/1708.09687

    Args:
        age_classes: Predict age in [0, age_classes]
        sigma: Hyperparameter in age normal distribution. Defaults to 2.
    """

    def __init__(
        self,
        age_classes: int,
        sigma: int = 2,
    ):
        self.age_classes = age_classes
        self.sigma = sigma

    def get_ord_label_inst(self, label):
        label_ord = np.zeros(self.age_classes)
        if abs(label + 1) < 1e-6:
            label_ord[:] = -1
        else:
            label_ord[: int(label)] = 1
        return label_ord

    def _trans_dist(self, label):
        rng = np.arange(self.age_classes)
        diff = label - rng
        sigma = self.sigma
        a = -(diff ** 2) / (2 * sigma ** 2)
        b = sigma * np.sqrt(2 * np.pi)
        label_dist = np.exp(a) / b
        return label_dist

    def get_dist_label(self, label):
        label_dist = np.zeros(self.age_classes)
        if label == -1:
            label_dist[:] = 1
        else:
            label_dist = self._trans_dist(label)
        return label_dist

    def __call__(self, data):
        raw_age_label = data["age"]
        raw_age_label = min(raw_age_label, self.age_classes)
        # logical vertor, the length is age_classes
        # each bit means whether older than bit_idx or not
        ord_age_label = self.get_ord_label_inst(raw_age_label)
        # float error weight vertor
        # subject to normal distribution
        dist_age_label = self.get_dist_label(raw_age_label).reshape(
            -1,
        )
        data["ord_age"] = ord_age_label
        data["dist_age"] = dist_age_label
        return data


@OBJECT_REGISTRY.register
class RandomOcclusion(object):
    """Randomly occlusion the image with circles or rectangles.

    Args:
        occ_type : str, default "whole"
            Occlusion type, `whole` and `forehead` are available.
        prob : float, defualt 0.
        occ_num : int, defualt 1
            Number of occlusion regions.
        color_ratio : float, default 0.5
            Probability of using random color.
        shape_ratio : float, default 0.5
            Probability of circle block, only for `whole` type.
        size_ratio : float, default 0.3
            Occlusion size of height and width.
            In `whole` mode:
            it's the upper bound of occlusion on the whole image.
            In `forehead` mode:
            it's the lower bound of occlusion on the top 1/3.
    """

    def __init__(
        self,
        occ_type: str = "whole",
        prob: float = 0.0,
        occ_num: int = 1,
        color_ratio: float = 0.5,
        shape_ratio: float = 0.5,
        size_ratio: float = 0.3,
        occ_ratio: float = 0.3,
    ):
        self.occ_type = _as_list(occ_type)
        self.prob = prob
        self.occ_num = occ_num
        self.color_ratio = color_ratio
        self.shape_ratio = shape_ratio
        self.radius_ratio = size_ratio / 2.0
        self.occ_ratio = occ_ratio

    def __call__(self, data):
        if random.random() > self.prob:
            return data

        assert data["layout"] == "hwc"
        img = data["img"]
        mask = data.get("gt_mask", None)
        occ_type = random.choice(self.occ_type)
        assert isinstance(img, np.ndarray)

        img_h, img_w = img.shape[:2]
        for _ in range(self.occ_num):
            if img.ndim == 3 and img.shape[-1] == 3:
                if random.random() <= self.color_ratio:
                    color = (
                        np.mean(img[:, :, 0]),
                        np.mean(img[:, :, 1]),
                        np.mean(img[:, :, 2]),
                    )
                    color = tuple(map(int, color))
                else:
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
            else:
                color = np.random.randint(0, 255)
            if occ_type == "whole":
                center = (random.randint(0, img_w), random.randint(0, img_h))
                if random.random() <= self.shape_ratio:
                    radius = int(img_h * random.uniform(0, self.radius_ratio))
                    cv2.circle(img, center, radius, color, -1)
                    if mask is not None:
                        cv2.circle(mask, center, radius, 0, -1)
                else:
                    radius_x = int(
                        img_w * random.uniform(0, self.radius_ratio)
                    )
                    radius_y = int(
                        img_h * random.uniform(0, self.radius_ratio)
                    )
                    left_top = (center[0] - radius_x, center[1] - radius_y)
                    right_down = (center[0] + radius_x, center[1] + radius_y)
                    cv2.rectangle(img, left_top, right_down, color, -1)
                    if mask is not None:
                        cv2.rectangle(mask, left_top, right_down, 0, -1)
            elif occ_type == "forehead":
                max_h = img_h // 3
                center = (random.randint(0, img_w), random.randint(0, max_h))
                radius_x = int(img_w * random.uniform(self.radius_ratio, 1.0))
                radius_y = int(max_h * random.uniform(self.radius_ratio, 1.0))
                left_top = (center[0] - radius_x, center[1] - radius_y)
                right_down = (center[0] + radius_x, center[1] + radius_y)
                cv2.rectangle(img, left_top, right_down, color, -1)
                if mask is not None:
                    cv2.rectangle(mask, left_top, right_down, 0, -1)
            elif occ_type == "edge":
                ratio = random.uniform(1 / 4, 1 / 3)
                prob = random.uniform(0, 1)
                color = (0, 0, 0)
                if prob < 0.25:  # top
                    cv2.rectangle(
                        img, (0, 0), (img_w, int(img_h * ratio)), color, -1
                    )
                    if mask is not None:
                        cv2.rectangle(
                            mask, (0, 0), (img_w, int(img_h * ratio)), 0, -1
                        )

                elif prob < 0.5:  # down
                    cv2.rectangle(
                        img,
                        (0, int(img_h * (1 - ratio))),
                        (img_w, img_h),
                        color,
                        -1,
                    )
                    if mask is not None:
                        cv2.rectangle(
                            mask,
                            (0, int(img_h * (1 - ratio))),
                            (img_w, img_h),
                            0,
                            -1,
                        )
                elif prob < 0.75:  # left
                    cv2.rectangle(
                        img, (0, 0), (int(img_w * ratio), img_h), color, -1
                    )
                    if mask is not None:
                        cv2.rectangle(
                            mask, (0, 0), (int(img_w * ratio), img_h), 0, -1
                        )
                else:
                    cv2.rectangle(
                        img,
                        (int(img_w * (1 - ratio)), 0),
                        (img_w, img_h),
                        color,
                        -1,
                    )
                    if mask is not None:
                        cv2.rectangle(
                            mask,
                            (int(img_w * (1 - ratio)), 0),
                            (img_w, img_h),
                            0,
                            -1,
                        )
            elif occ_type == "jaw":
                bbox_local = data.get("gt_bboxes_local")
                x1, y1, x2, y2 = bbox_local.reshape(-1)
                start_r = 1 - self.occ_ratio
                y1_start = random.randint(
                    int(y1 + (y2 - y1) * start_r), int(y1 + (y2 - y1) * 0.9)
                )
                y1_start = min(y1_start, img_h)
                y2_start = random.randint(y1_start, img_h)
                color1 = random.randint(0, 50)
                color2 = random.randint(50, 128)
                left_top = (0, y1_start)
                right_down = (img_w, img_h)
                cv2.rectangle(img, left_top, (img_w, y2_start), color1, -1)
                cv2.rectangle(img, (0, y2_start), (img_w, img_h), color2, -1)
                if mask is not None:
                    cv2.rectangle(mask, left_top, right_down, 0, -1)
            else:
                raise ValueError("Not supported block type.")

        data["img"] = img
        if mask is not None:
            data["gt_mask"] = mask
        return data
