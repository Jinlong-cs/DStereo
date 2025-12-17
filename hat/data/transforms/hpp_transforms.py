import copy
import math
import random
from copy import deepcopy

# import matplotlib.cm as c
from typing import Tuple

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "GaussianNoiseWithChannel",
    "ChangeIntensity",
    "Shadow",
    "RandomHorizontalFilpWithPoints",
    "RandomTranslationWithPoints",
    "RandomRotateWithPoints",
    "HPPPoint2Map",
    "AlignOdometryLength",
]


@OBJECT_REGISTRY.register
class GaussianNoiseWithChannel(object):
    """Apply Gaussian noise to images.

    Args:
        prob: probility of applying the transform.
        mean: mean of noise.
        sigma: standard deviation of noise.
    """

    def __init__(
        self,
        prob: float,
        mean: float = 0,
        sigma: float = 20,
    ):
        self.prob = prob
        self.mean = mean
        self.sigma = sigma

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        img = data["img"]
        if random.random() <= self.prob:
            mean = (self.mean, self.mean, self.mean)
            sigma = (self.sigma, self.sigma, self.sigma)
            noise = np.zeros(img.shape, np.uint8)
            # channel different, noise different
            noise = cv2.randn(noise, mean, sigma)
            img = img + noise
            img = np.clip(img, 0, 255)
            data["img"] = img
            return data
        else:
            return data


@OBJECT_REGISTRY.register
class ChangeIntensity(object):
    """Change intensity of input images.

    Args:
        prob: probility of applying the transform.
    """

    def __init__(self, prob: float, rescale: float = 60):
        self.prob = prob
        self.rescale = rescale

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        img = data["img"]
        if random.random() <= self.prob:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            value = int(random.uniform(-1 * self.rescale, self.rescale))
            if value > 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = -1 * value
                v[v < lim] = 0
                v[v >= lim] -= lim
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            data["img"] = img
            return data
        else:
            return data


@OBJECT_REGISTRY.register
class Shadow(object):
    """Simulates shadows for input images."""

    def __init__(
        self, prob: float, min_alpha: float = 0.5, max_alpha: float = 0.75
    ):
        self.prob = prob
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        img = data["img"]
        if random.random() <= self.prob:
            top_x, bottom_x = np.random.randint(0, 512, 2)
            coin = 0
            rows, cols, _ = img.shape
            shadow_img = img.copy()
            if coin == 0:
                rand = np.random.randint(2)
                vertices = np.array(
                    [[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32
                )
                if rand == 0:
                    vertices = np.array(
                        [[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]],
                        dtype=np.int32,
                    )
                elif rand == 1:
                    vertices = np.array(
                        [
                            [top_x, 0],
                            [cols, 0],
                            [cols, rows],
                            [bottom_x, rows],
                        ],
                        dtype=np.int32,
                    )
                mask = img.copy()
                # i.e. 3 or 4 depending on your image
                channel_count = img.shape[2]
                ignore_mask_color = (0,) * channel_count
                cv2.fillPoly(mask, [vertices], ignore_mask_color)
                rand_alpha = np.random.uniform(self.min_alpha, self.max_alpha)
                cv2.addWeighted(
                    mask,
                    rand_alpha,
                    img,
                    1 - rand_alpha,
                    0.0,
                    shadow_img,
                )
            data["img"] = shadow_img
            return data
        else:
            return data


@OBJECT_REGISTRY.register
class RandomHorizontalFilpWithPoints(object):
    def __init__(self, flip_prob: float = 0.5):
        self.flip_prob = flip_prob

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        assert "points" in data
        if random.random() < self.flip_prob:
            img = data["img"]
            points = data["points"]
            _, width, _ = img.shape
            # flip img
            data["img"] = cv2.flip(img, 1)
            # flip x coordinate
            mid_x = width // 2
            points[:, 0] = 2 * mid_x - points[:, 0]
            data["points"] = points
            return data
        else:
            return data


@OBJECT_REGISTRY.register
class RandomTranslationWithPoints(object):
    """Randomly apply affine translation to input images and 2d points.

    Args:
        neg_filter: filter points that out of image. Default: True
        translate: Tuple (px, py). Two values will be uniformly sampled from
            the discrete interval [-px, px] and [-py, py], then translate
            image along x-axis and y-axis by sampled values.

    """

    def __init__(
        self,
        prob: float = 0.5,
        neg_filter: bool = True,
        translate: Tuple[float, float] = (50, 30),
    ):
        self.prob = prob
        self.neg_filter = neg_filter
        self.translate = translate

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        assert "points" in data
        img = data["img"]
        points = data["points"]
        if random.random() < self.prob:
            tx = np.random.randint(-1 * self.translate[0], self.translate[0])
            ty = np.random.randint(-1 * self.translate[1], self.translate[1])
            height, width = img.shape[:2]
            # translate img
            translate_img = cv2.warpAffine(
                img, np.float32([[1, 0, tx], [0, 1, ty]]), (width, height)
            )
            data["img"] = translate_img
            # translate point
            points += np.array([tx, ty], dtype=points.dtype)
            ignore_by_border = (
                (points[:, 0] < 0)
                + (points[:, 0] >= width)
                + (points[:, 1] < 0)
                + (points[:, 1] >= height)
            )
            if self.neg_filter:
                points = points[~ignore_by_border]
            data["points"] = points
            return data
        else:
            return data


@OBJECT_REGISTRY.register
class RandomRotateWithPoints(object):
    """Randomly apply affine rotate to input images and 2d points.

    Args:
        neg_filter: filter points that out of image. Default: True
        angle: a value will be uniformly sampled per image
            from the interval [-angle, angle] and used as the rotation value

    """

    def __init__(
        self,
        prob: float = 0.5,
        neg_filter: bool = False,
        angle: float = 10,
    ):
        self.prob = prob
        self.neg_filter = neg_filter
        self.angle = angle

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        assert "points" in data
        img = data["img"]
        points = data["points"]
        if random.random() < self.prob:
            if self.angle != 0:
                angle_ = np.random.randint(-self.angle, self.angle)
            else:
                angle_ = self.angle
            height, width = img.shape[:2]
            # rotate image
            M = cv2.getRotationMatrix2D((width // 2, height // 2), angle_, 1)
            data["img"] = cv2.warpAffine(img, M, (width, height))
            offset = np.array([width / 2, height / 2], dtype=np.float32)
            matrix = np.array(
                [
                    [
                        math.cos(angle_ / 180.0 * math.pi),
                        math.sin(-angle_ / 180.0 * math.pi),
                    ],
                    [
                        math.sin(angle_ / 180.0 * math.pi),
                        math.cos(angle_ / 180.0 * math.pi),
                    ],
                ],
                dtype=np.float32,
            )
            rotated_points = np.matmul((points - offset), matrix) + offset

            ignore_by_border = (
                (rotated_points[:, 0] < 0)
                + (rotated_points[:, 0] >= width)
                + (rotated_points[:, 1] < 0)
                + (rotated_points[:, 1] >= height)
            )
            # points[ignore_by_border] = self.ignore_id
            if self.neg_filter:
                rotated_points = rotated_points[~ignore_by_border]
            data["points"] = rotated_points
            return data
        else:
            return data


@OBJECT_REGISTRY.register
class HPPPoint2Map(object):
    """Convert point to GT Map.

    Args:
        output_stride: the output stride of feature
    """

    def __init__(self, output_stride: int = 8):
        self.output_stride = output_stride

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        assert "points" in data
        data["labels"] = (
            self.get_offset_map(data),
            self.get_instance_map(data),
        )

        return data

    def remove_invalid_points(self, points, width, height):
        ignore_by_border = (
            (points[:, 0] < 0)
            + (points[:, 0] >= width)
            + (points[:, 1] < 0)
            + (points[:, 1] >= height)
        )
        points = points[~ignore_by_border]
        return points

    def get_offset_map(self, data):
        height, width = data["img"].shape[:2]
        grid_h = height // self.output_stride
        grid_w = width // self.output_stride
        points = self.remove_invalid_points(
            data["points"].copy(), width, height
        )
        index_points = np.int_(points // self.output_stride)
        ground = np.zeros((3, grid_h, grid_w))
        # confidence map
        ground[0, index_points[:, 1], index_points[:, 0]] = 1.0
        # x and y offset map
        ground[1, index_points[:, 1], index_points[:, 0]] = (
            points[:, 0] / self.output_stride - index_points[:, 0]
        )
        ground[2, index_points[:, 1], index_points[:, 0]] = (
            points[:, 1] / self.output_stride - index_points[:, 1]
        )
        return ground

    def get_instance_map(self, data):
        height, width = data["img"].shape[:2]
        grid_h = height // self.output_stride
        grid_w = width // self.output_stride
        ground = np.zeros((1, grid_h * grid_w, grid_h * grid_w))
        points = self.remove_invalid_points(
            data["points"].copy(), width, height
        )
        index_points = np.int_(points // self.output_stride)
        # index_points = points.copy() // self.output_stride
        temp = np.zeros((1, grid_h, grid_w))
        temp[:, index_points[:, 1], index_points[:, 0]] = 1

        for i in range(grid_h * grid_w):  # make gt
            temp = temp[temp > -1]
            gt_one = deepcopy(temp)
            if temp[i] > 0:
                gt_one[temp == temp[i]] = 1  # same instance
                if temp[i] == 0:
                    # different instance, different class
                    gt_one[temp != temp[i]] = 3
                else:
                    # different instance, same class
                    gt_one[temp != temp[i]] = 2
                    # different instance, different class
                    gt_one[temp == 0] = 3
                ground[0][i] += gt_one
        return ground


@OBJECT_REGISTRY.register
class AlignOdometryLength(object):
    """keep consistency of the length of odometry.

    Args:
        length (int): target length. Default: 100.
        pad (int): pad value. Default: -2.
    """

    def __init__(
        self,
        length: int = 100,
        pad: int = -2,
    ):
        self.length = length
        self.pad = pad

    def __call__(self, data):
        assert "points" in data
        original_points = data["points"]
        length_diff = self.length - original_points.shape[0]
        pad_temp = -2 * np.ones((length_diff, 2))
        pad_points = np.concatenate([original_points, pad_temp], axis=0)
        data["points"] = pad_points

        return data


@OBJECT_REGISTRY.register
class HPPPerspectiveTransform(object):
    def __init__(
        self,
        prob: float = 0.5,
        w_scale_reduce: tuple = (0.2, 0.3),
        w_scale_dilate: tuple = (0.2, 0.4),
        h_scale: float = 0.2,
        pad_value: tuple = (114, 114, 114),
    ):
        """HPP perspective transform, used to simulate the image when car bumps.

        Args:
            w_scale_reduce (tuple, optional): scale factor along horizonal
                axis, reduce the length of top sides. Defaults to (0.2, 0.3).
            w_scale_dilate (tuple, optional): scale factor along horizonal
                axis, add the length of top sides, Defaults to (0.2, 0.4).
            h_scale (float, optional): scale factor along vectical axis.
                Defaults to 0.2.
            pad_value (tuple, optional): fill value.
                Defaults to (114, 114, 114).
        """
        self.prob = prob
        self.w_scale_reduce = w_scale_reduce
        self.w_scale_dilate = w_scale_dilate
        self.h_scale = h_scale
        self.pad_value = pad_value

    def get_delta_width(self, width):
        w_reduce_min = int(self.w_scale_reduce[0] * width)
        w_reduce_max = int(self.w_scale_reduce[1] * width)
        w_dilate_min = int(self.w_scale_dilate[0] * width)
        w_dilate_max = int(self.w_scale_dilate[1] * width)
        left_range_tmp1 = list(range(w_reduce_min, w_reduce_max))
        left_range_tmp2 = [-1 * i for i in range(w_dilate_min, w_dilate_max)]
        left_range = []
        left_range.extend(left_range_tmp1)
        left_range.extend(left_range_tmp2)
        right_range = -1 * np.array(left_range)
        right_range = right_range.tolist()
        return left_range, right_range

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        assert "points" in data

        image = data["img"]
        points = data["points"]
        h, w, _ = image.shape

        if random.random() < self.prob:
            left_range, right_range = self.get_delta_width(w)
            max_scale_h = h * self.h_scale
            rand_left = random.choice(left_range)
            rand_right = random.choice(right_range)
            # avoid generate right trapezoid area
            if rand_left * rand_right > 0:
                rand_right = rand_right * -1

            rand_h = random.randint(int(-1 * max_scale_h), int(max_scale_h))
            original_vertexs = np.array(
                [
                    [0, 0],  # left top
                    [w - 1, 0],  # right top
                    [0, h],  # left bottom
                    [w - 1, h - 1],  # right bottom
                ],
                dtype=np.float32,
            )
            transform_vertex = copy.deepcopy(original_vertexs)
            # only transform left-top and right-top vertexs
            transform_vertex[0, 0] += rand_left
            transform_vertex[1, 0] += rand_right
            transform_vertex[:2, 1] += rand_h
            points = np.array(data["points"], dtype=np.float32)
            ones = np.ones(points.shape[0]).reshape(points.shape[0], 1)
            homogeneous_points = np.concatenate([points, ones], axis=-1)
            pers_matrix = cv2.getPerspectiveTransform(
                original_vertexs, transform_vertex
            )
            pers_points = np.matmul(pers_matrix, homogeneous_points.T)
            # normalize points
            pers_points[0, :] /= pers_points[
                2,
            ]
            pers_points[1, :] /= pers_points[
                2,
            ]
            pers_points = pers_points[
                :2,
            ].T
            # print(pers_points.shape)
            ignore_by_border = (
                (pers_points[:, 0] < 0)
                + (pers_points[:, 0] >= w)
                + (pers_points[:, 1] < 0)
                + (pers_points[:, 1] >= h)
            )
            pers_points = pers_points[
                ~ignore_by_border,
            ]
            # transform image
            pers_image = cv2.warpPerspective(
                image,
                pers_matrix,
                dsize=(w, h),
                borderValue=self.pad_value,
            )

            data["img"] = pers_image
            data["points"] = np.int_(pers_points)

            return data
        else:
            return data


@OBJECT_REGISTRY.register
class RandomTranslationWithPointsV2(object):
    def __init__(
        self,
        prob: float = 0.5,
        neg_filter: bool = True,
        translate_x: list = None,  # [(-25, -15), (15, 25)]
        translate_y: list = None,  # [(0, 0)],
    ):
        """Randomly apply affine translation to input images and 2d points.

        Args:
            neg_filter (bool, optional): filter points that out of image.
                Defaults to True.
            translate_x (list, optional): translate range along x axix.
                Defaults to None.
            translate_y (list, optional): translate range along x axix.
                Defaults to None.
        """
        self.prob = prob
        self.neg_filter = neg_filter
        (
            self.translate_x_range,
            self.translate_y_range,
        ) = self.get_delta_translation(translate_x, translate_y)

    def get_delta_translation(self, translate_x, translate_y):
        translate_x_range = []
        for t_range in translate_x:
            assert t_range[0] <= t_range[1]
            translate_x_tmp = list(range(t_range[0], t_range[1]))
            translate_x_range.extend(translate_x_tmp)
        translate_y_range = []
        for t_range in translate_y:
            assert t_range[0] <= t_range[1]
            translate_y_tmp = list(range(t_range[0], t_range[1]))
            translate_y_range.extend(translate_y_tmp)
        return translate_x_range, translate_y_range

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(
                "Expect 'dict' input but get {}".format(type(data))
            )
        assert "img" in data
        assert "points" in data
        img = data["img"]
        points = data["points"]
        if random.random() < self.prob:
            if len(self.translate_x_range) != 0:
                tx = random.choice(self.translate_x_range)
            else:
                tx = 0
            if len(self.translate_y_range) != 0:
                ty = random.choice(self.translate_y_range)
            else:
                ty = 0
            height, width = img.shape[:2]
            # translate img
            translate_img = cv2.warpAffine(
                img, np.float32([[1, 0, tx], [0, 1, ty]]), (width, height)
            )
            data["img"] = translate_img
            # translate point
            points += np.array([tx, ty], dtype=points.dtype)
            ignore_by_border = (
                (points[:, 0] < 0)
                + (points[:, 0] >= width)
                + (points[:, 1] < 0)
                + (points[:, 1] >= height)
            )
            if self.neg_filter:
                points = points[~ignore_by_border]
            data["points"] = points
            return data
        else:
            return data
