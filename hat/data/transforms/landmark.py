# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import logging
import math
import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from .pupil_segmentation import Ellipse

logger = logging.getLogger(__name__)

__all__ = [
    "GaussianNoise",
    "RandomNoise",
    "SaltPepperNoise",
    "Lighting",
    "NormalizeLdmk",
    "CropRecROI",
    "GenerateGaussianHeatmap",
    "GenerateGaussianVector",
    "GenerateGRMITarget",
    "RandomShiftRotateScale",
    "ClipBoxes",
]


@OBJECT_REGISTRY.register
class Lighting(object):
    """Generate lighting noise on img.

    Randomly add PCA noise. Follow the AlexNet style.
    This class operate on tensor, please make sure input
    is tensor instead of numpy.

    Args:
        prob: Prob to generate lighting noise.
        alphastd: Level of the lighting noise.
    """

    def __init__(
        self,
        prob: float,
        alphastd: float,
    ):
        self.prob = prob
        self.alphastd = alphastd

    def __call__(self, data):
        assert "img" in data
        img = data["img"]
        img = img.to(torch.float32)
        if self.alphastd == 0:
            return data

        if random.random() <= self.prob:
            eigval = torch.Tensor([0.2175, 0.0188, 0.0045]).to(img.device)
            eigvec = torch.Tensor(
                [
                    [-0.5675, 0.7192, 0.4009],
                    [-0.5808, -0.0045, -0.8140],
                    [-0.5836, -0.6948, 0.4203],
                ]
            ).to(img.device)
            alpha = img.new().resize_(3).normal_(0, self.alphastd)
            rgb = (
                eigvec.type_as(img)
                .clone()
                .mul(alpha.view(1, 3).expand(3, 3))
                .mul(eigval.view(1, 3).expand(3, 3))
                .sum(1)
                .squeeze()
            )
            img = img.add(rgb.view(3, 1, 1).expand_as(img))
        data["img"] = img.to(torch.uint8)

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"prob={self.prob}"
        repr_str += f"alphastd={self.alphastd}"
        return repr_str


@OBJECT_REGISTRY.register
class GaussianNoise(object):
    """Generate gaussian noise on img.

    Args:
        prob: Prob to generate gaussian noise.
        mean: Mean of gaussian distribution. Defaults to 0.
        sigma: Sigma of gaussian distribution. Defaults to 2.
    """

    def __init__(
        self,
        prob: float,
        mean: float = 0,
        sigma: float = 2,
    ):
        self.prob = prob
        self.mean = mean
        self.sigma = sigma

    def __call__(self, data):
        assert "img" in data
        img = data["img"]
        if random.random() <= self.prob:
            noise = np.random.normal(
                self.mean, self.sigma, (img.shape[0], img.shape[1])
            )
            noise = noise[:, :, np.newaxis]
            img = img + noise
            img = np.clip(img, 0, 255)
        data["img"] = img.astype(np.uint8)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"prob={self.prob}"
        repr_str += f"mean={self.mean}"
        repr_str += f"sigma={self.sigma}"
        return repr_str


@OBJECT_REGISTRY.register
class RandomNoise(object):
    """Generate random noise on img.

    Args:
        prob: Prob to generate gaussian noise.
        min: Min value of uniform distribution. Defaults to -5.
        max: Max value of uniform distribution. Defaults to 5.
    """

    def __init__(
        self,
        prob: float,
        min: float = -5,
        max: float = 5,
    ):
        self.prob = prob
        self.min = min
        self.max = max

    def __call__(self, data):
        assert "img" in data
        img = data["img"]
        if random.random() <= self.prob:
            noise = np.random.uniform(
                self.min, self.max, (img.shape[0], img.shape[1])
            )
            noise = noise[:, :, np.newaxis]
            img = img + noise
            img = np.clip(img, 0, 255)
        data["img"] = img.astype(np.uint8)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"prob={self.prob}"
        repr_str += f"min={self.min}"
        repr_str += f"max={self.max}"
        return repr_str


@OBJECT_REGISTRY.register
class SaltPepperNoise(object):
    """Generate saltpepper noise on img.

    Args:
        prob: Prob to generate gaussian noise.
        s_ratio: Salt ratio. Defaults to 0.05.
        p_ratio: Pepper ratio. Defaults to 0.05.
    """

    def __init__(
        self,
        prob: float,
        s_ratio: float = 0.05,
        p_ratio: float = 0.05,
    ):
        self.prob = prob
        self.s_ratio = s_ratio
        self.p_ratio = p_ratio

    def __call__(self, data):
        assert "img" in data
        img = data["img"]
        if random.random() <= self.prob:
            s_num = int(self.s_ratio * img.shape[1] * img.shape[0])
            coords = [
                np.random.randint(0, i, int(s_num))
                for i in (img.shape[0], img.shape[1])
            ]
            img[tuple(coords)] = 255
            p_num = int(self.p_ratio * img.shape[1] * img.shape[0])
            coords = [
                np.random.randint(0, i, int(p_num))
                for i in (img.shape[0], img.shape[1])
            ]
            img[tuple(coords)] = 0
            img = np.clip(img, 0, 255)
        data["img"] = img.astype(np.uint8)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"prob={self.prob}"
        repr_str += f"s_ratio={self.s_ratio}"
        repr_str += f"p_ratio={self.p_ratio}"
        return repr_str


@OBJECT_REGISTRY.register
class NormalizeLdmk(object):
    """Normalize landmarks to -1~1.

    This is used for coordinate regression based landmark detection.

    Args:
        root_index: which landmark is set to be the root.
            For 3D hand keypoints, all coords minus the root to get relative
            coords. Defaults to None.
        norm_scale: normalization scale. Usually equal to input image size.
            Defaults to 128.0.
    """

    def __init__(
        self,
        root_index: Optional[int] = None,
        norm_scale: float = 128.0,
    ):
        self.root_index = root_index
        self.norm_scale = norm_scale

    def __call__(self, data):
        if "gt_ldmk" in data:
            data = self.normalize_ldmk(data)
        if "gt_lmkd_3d" in data:
            data = self.normalize_ldmk_3d(data)
        return data

    def normalize_ldmk(self, data):
        gt_ldmk = data["gt_ldmk"]
        gt_ldmk = gt_ldmk / self.norm_scale - 0.5
        data["gt_ldmk"] = gt_ldmk
        return data

    def normalize_ldmk_3d(self, data):
        raise NotImplementedError()

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"root_index={self.root_index}"
        repr_str += f"norm_scale={self.norm_scale}"
        return repr_str


@OBJECT_REGISTRY.register
class GenerateGaussianHeatmap(object):
    r"""Generate Gaussain heatmap ground truth for landmark detection.

    Standard encoding and unbiased-encoding are supported. And this method
    is modified based on GluonFace.
    In standard mode, coords are downsampled and round to generate heatmap.
    Noteworthily, plus 1 to the maximum element to enlarge the gap of the
    landmark location and its neighbourhood. The heatmap will boost the
    performance in most situations compared with the counterpart without
    this trick.
    In unbiased mode, coords are downsampled without quantization. It is
    common used for reducing quantization error of heatmap based methods.
    Such as DARK(https://arxiv.org/abs/1910.06278).

    Args:
        num_ldmk: number of landmarks.
        feat_stride: feature stride, which is the scale ratio of
            input / output. Defaults to 4.
        heatmap_shape: heatmap feature shape (H, W). Defaults to (32, 32).
        sigma: $\sigma$ of Gaussian formula. Defaults to 2.
        encoding_method: encoding method. "standard" or "unbiased".
            Defaults to "standard".
    """

    def __init__(
        self,
        num_ldmk: int,
        feat_stride: int = 4,
        heatmap_shape: Tuple[int, int] = (32, 32),
        sigma: float = 2,
        encoding_method: str = "standard",
    ):
        self.num_ldmk = num_ldmk
        self.feat_stride = feat_stride
        self.sigma = sigma
        self.encoding_method = encoding_method.lower()
        self.height, self.width = heatmap_shape
        self.radius = 3 * self.sigma
        x = np.arange(0, 6 * self.sigma + 1, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = self.radius
        self.kernel = np.exp(
            -((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
        )
        self.kernel[y0, x0] += 1

    def __call__(self, data):
        ldmk = data["gt_ldmk"]
        ldmk_attr = data["gt_ldmk_attr"]

        heatmap = self._transform_ldmk_to_heatmap(ldmk, ldmk_attr)
        heatmap_weight = np.ones_like(heatmap)
        # set different pixel weight
        heatmap_weight[heatmap == 0] = 0.1
        # ignore empty heatmap
        heatmap_weight[~np.any(heatmap, axis=(1, 2)), :, :] = 0
        # ignore invalid landmark
        heatmap_weight[np.where(ldmk_attr < 0), :, :] = 0
        data["gt_heatmap"] = heatmap
        data["gt_heatmap_weight"] = heatmap_weight
        return data

    def _transform_ldmk_to_heatmap(
        self, landmark: np.ndarray, ldmk_attr: np.ndarray
    ):
        heatmap = np.zeros((self.num_ldmk, self.height, self.width))
        for i in range(self.num_ldmk):
            if self.encoding_method in ["unbiased", "ellipse"]:
                heatmap[i] = self._unbiased_encode(landmark[i])
            elif self.encoding_method == "standard":
                heatmap[i] = self._standard_encode(landmark[i])
            else:
                raise ValueError(
                    f"Not supported heatmap encoding method: {self.encoding_method}."  # noqa E501
                )

        if self.encoding_method == "ellipse":
            # concatenate cigaret body map with landmark heatmap
            if np.any(ldmk_attr):
                heatmap_cigaret = self._ellipse_encode(landmark)
            else:
                heatmap_cigaret = np.zeros((self.height, self.width))

            heatmap = np.concatenate(
                [
                    heatmap,
                    heatmap_cigaret.reshape(
                        1,
                        self.height,
                        self.width,
                    ),
                ],
                axis=0,
            )

        return heatmap

    def _unbiased_encode(self, landmark: np.ndarray):
        """Use unbiased encode from DARK.

        More details about DARK please refer to
        https://arxiv.org/abs/1910.06278.
        """
        landmark = landmark / self.feat_stride
        target = np.zeros((self.height, self.width), dtype=np.float32)
        lmk_x, lmk_y = landmark[:2]
        x = np.arange(0, 2 * self.radius, 1, np.float32)
        y = x[:, np.newaxis]
        # get the up left and bottom right corners
        ul = [math.ceil(lmk_x - self.radius), math.ceil(lmk_y - self.radius)]
        br = [math.ceil(lmk_x + self.radius), math.ceil(lmk_y + self.radius)]
        # check heatmap
        if ul[0] >= self.width or ul[1] >= self.height:
            return target
        if br[0] <= 0 or br[1] <= 0:
            return target
        # generate kernel
        x0 = lmk_x - math.floor(lmk_x) + self.radius - 1  # float center_x
        y0 = lmk_y - math.floor(lmk_y) + self.radius - 1  # float center_y
        kernel = np.exp(
            -((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
        )
        # generate heatmap
        g_x = max(0, -ul[0]), min(br[0], self.width) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], self.height) - ul[1]
        img_x = max(0, ul[0]), min(br[0], self.width)
        img_y = max(0, ul[1]), min(br[1], self.height)
        target[img_y[0] : img_y[1], img_x[0] : img_x[1]] = kernel[
            g_y[0] : g_y[1], g_x[0] : g_x[1]
        ]

        return target

    def _standard_encode(self, landmark: np.ndarray):
        landmark = (landmark / self.feat_stride).round().astype(np.int32)
        target = np.zeros((self.height, self.width), dtype=np.float32)
        lmk_x, lmk_y = landmark[:2]

        ul = [int(lmk_x - self.radius), int(lmk_y - self.radius)]
        br = [int(lmk_x + self.radius + 1), int(lmk_y + self.radius + 1)]

        if ul[0] >= self.width or ul[1] >= self.height:
            return target
        if br[0] <= 0 or br[1] <= 0:
            return target

        g_x = max(0, -ul[0]), min(br[0], self.width) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], self.height) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], self.width)
        img_y = max(0, ul[1]), min(br[1], self.height)

        target[img_y[0] : img_y[1], img_x[0] : img_x[1]] = self.kernel[
            g_y[0] : g_y[1], g_x[0] : g_x[1]
        ]
        return target

    def _ellipse_encode(self, landmark: np.ndarray, eps: float = 0.0001):
        """Encode cigaret heatmap for smoke kps task.

        Reference: https://horizonrobotics.feishu.cn/wiki/wikcnLlF6L047P6Ph2II9L4mf9f.
        """  # noqa
        landmark = landmark / self.feat_stride
        self.X, self.Y = np.meshgrid(
            np.arange(self.height), np.arange(self.width)
        )

        # - calcualte vector of ldmk 0-1 and ldmk 2-3
        vec_01 = landmark[1, 0:2] - landmark[0, 0:2]  # vec 0-1
        vec_23 = landmark[3, 0:2] - landmark[2, 0:2]  # vec 2-3
        vec_01[np.abs(vec_01) < eps] = eps
        vec_23[np.abs(vec_23) < eps] = eps

        # - calcualte the projection of vector 2-3
        # on vertical direction of vector 0-1
        vec_01t = vec_01[::-1] * np.array([1, -1])  # vertical of vec 0-1
        length = np.linalg.norm(vec_01)  # length of vec 0-1
        width = np.abs(np.dot(vec_01t, vec_23) / length)  # length of width
        cent = (landmark[0, 0:2] + landmark[1, 0:2]) / 2  # Center of Ellipse

        # - calcualte the conv matrix
        R = (
            np.concatenate(
                [-vec_01t.reshape(1, -1), vec_01.reshape(1, -1)], axis=0
            )[:, ::-1]
            / length
        )
        A = np.diag([length, width]) * np.array([[0.5, 0], [0, 0.5]])
        invS = np.dot(np.dot(R, A), R.T)
        invS = np.linalg.inv(np.dot(invS, invS))

        # - calcualte heatmap
        heatmap = np.concatenate(
            [
                self.X[:, :, np.newaxis] - cent[0],
                self.Y[:, :, np.newaxis] - cent[1],
            ],
            axis=2,
        )
        heatmap = (np.dot(heatmap, invS) * heatmap).sum(axis=2)
        heatmap = (
            1 / (2 * np.pi) * np.linalg.norm(invS) * np.exp(-1 / 2 * heatmap)
        )  # noqa
        heatmap = heatmap / np.max(heatmap)

        return heatmap

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"num_ldmk={self.num_ldmk}"
        repr_str += f"feat_stride={self.feat_stride}"
        repr_str += f"heatmap_shape={(self.height, self.width)}"
        repr_str += f"sigma={self.sigma}"
        repr_str += f"encoding_method={self.encoding_method}"
        return repr_str


@OBJECT_REGISTRY.register
class CropRecROI(object):
    """Crop ROI from crop&resized rec.

    This is Modified based on GluonFace/CropImage.

    Args:
        crop_type: crop style, only "center", "random" and "scale" are
            supported. In "center" type, directly crop base_roi and resize
            the ROI to target_shape. In "random" type, height and width
            jitter in [1 - crop_jitter_range, 1 + crop_jitter_range]. ROI
            is cropped and resized with a variational aspect ratio. In "scale"
            type, height and width jitter synchronously with a fixed aspect
            ratio. That is to say, height and width share the same jitter
            ratio.
        target_shape: output image shape, (H, W, C).
        base_roi: basic ROI rects. Cropped ROI is calculated based on it.
        crop_jitter_range: maximum jitter ratio of the ROI.
        center_shift_range: maximum center shifting ratio. Defaults to 0.0.
        random_type: distribution to get random number. Defaults to "gaussian".
        crop_frames_prob_dist: probability distribution of different crop
            methods.There are three values in list, corresponding to the
            probability of "random", "scale" and "center".
            Defaults to [0.3, 0.7, 0].
        use_crop_limit: whether to use limit crop. Defaults to False.
    """

    def __init__(
        self,
        crop_type: str,
        target_shape: Tuple[int, int, int],
        base_roi: Tuple[float, float, float, float],
        crop_jitter_range: float,
        center_shift_range: float = 0.0,
        random_type: str = "gaussian",
        crop_frames_prob_dist: Tuple[float, float, float] = (0.3, 0.7, 0),
        use_crop_limit: bool = False,
    ):
        # TODO(yuhao.dou): Remove norm_ratio and make it easier.
        self.crop_type = crop_type.lower()
        self.target_shape = target_shape
        self.base_roi = base_roi
        self.crop_jitter_range = crop_jitter_range
        self.center_shift_range = center_shift_range
        self.random_type = random_type
        assert (
            sum(crop_frames_prob_dist) == 1
        ), "random_crop + scale_crop + center_crop should be 1.0"
        self.crop_frames_prob_dist = crop_frames_prob_dist
        self.use_crop_limit = use_crop_limit

    def __call__(self, data):
        if "img" in data:
            data = self.crop_img(data, self.crop_type)
            if "gt_ldmk" in data:
                self.crop_ldmk(data)
            if "gt_ldmk_3d" in data:
                self.crop_ldmk_3d(data)
            if "gt_pupil_ellipse_param" in data:
                self.crop_ellipse_param(data)
            data["img_shape"] = self.target_shape

        if "frames" in data:
            data = self.crop_frames(data)
        return data

    def crop_img(self, data, crop_type):
        data["img"], data["crop_rects"] = getattr(self, f"{crop_type}_crop")(
            data["img"]
        )
        return data

    def crop_frames(self, data):
        imgs = data["frames"]
        roi_imgs = []
        rec_shape = data["rec_shape"]
        for idx in range(len(imgs)):
            if imgs[idx] is None:
                roi_img = np.zeros(
                    (self.target_shape[0], self.target_shape[1], 3),
                    dtype=np.uint8,
                )
            else:
                if self.use_crop_limit:
                    limit_box = (
                        np.array(data["handboxes"][idx]).reshape((-1, 2))
                        - np.array(data["cropboxes"][idx]).reshape((-1, 2))[0]
                    )
                    limit_box[:, 0] *= rec_shape / data["raw_shapes"][idx][1]
                    limit_box[:, 1] *= rec_shape / data["raw_shapes"][idx][0]
                    limit_box = limit_box.flatten()
                else:
                    limit_box = None
                crop_type = np.random.choice(
                    ["random", "scale", "center"], p=self.crop_frames_prob_dist
                )
                roi_img, _ = getattr(self, f"{crop_type}_crop")(
                    imgs[idx], limit_box=limit_box
                )

            roi_imgs.append(roi_img)
        data["frames"] = roi_imgs
        return data

    def crop_ldmk(self, data):
        # TODO(yuhao.dou): gt_ldmk_attr should be changed during cropping.
        ldmk = data["gt_ldmk"].copy()
        rects = data["crop_rects"]
        ldmk[:, 0] -= rects[0]
        ldmk[:, 0] *= self.target_shape[1] / (rects[2] - rects[0])
        ldmk[:, 1] -= rects[1]
        ldmk[:, 1] *= self.target_shape[0] / (rects[3] - rects[1])
        data["gt_ldmk"] = ldmk
        return data

    def crop_ldmk_3d(self, data):
        raise NotImplementedError()

    def crop_ellipse_param(self, data):
        ellipse_param = data["gt_pupil_ellipse_param"].copy()
        rects = data["crop_rects"]
        ellipse_param[0] = ellipse_param[0] - rects[0]
        ellipse_param[1] = ellipse_param[1] - rects[1]
        scale_w = self.target_shape[1] / (rects[2] - rects[0])
        scale_h = self.target_shape[0] / (rects[3] - rects[1])
        if scale_h != 1 or scale_w != 1:
            scale_matrix = np.array(
                [[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]]
            )
            ellipse_param[-1] = np.deg2rad(ellipse_param[-1])
            ellipse_param = Ellipse(ellipse_param).transform(scale_matrix)[0][
                :-1
            ]
            ellipse_param[-1] = np.rad2deg(ellipse_param[-1])
        data["gt_pupil_ellipse_param"] = ellipse_param

    def _get_random_num(self):
        if self.random_type == "uniform":
            random_num = np.random.uniform(-1, 1)
        elif self.random_type == "gaussian":
            random_num = np.random.normal(0, 0.6, 1)[0] / 2
            random_num = np.minimum(random_num, 1.0)
            random_num = np.maximum(random_num, -1.0)
        else:
            raise ValueError(f"Not supported random type: {self.random_type}")
        return random_num

    def center_crop(self, img, limit_box=None):
        """Crop the center region of the image."""
        assert img.ndim == 3
        target_h, target_w, _ = self.target_shape
        rects = copy.deepcopy(self.base_roi)
        if limit_box is not None:
            rects[0] = limit_box[0] if rects[0] >= limit_box[0] else rects[0]
            rects[1] = limit_box[1] if rects[1] >= limit_box[1] else rects[1]
            rects[2] = limit_box[2] if rects[2] <= limit_box[2] else rects[2]
            rects[3] = limit_box[3] if rects[3] <= limit_box[3] else rects[3]
        rects = list(map(int, rects))
        img = img[rects[1] : rects[3], rects[0] : rects[2], :]
        img = cv2.resize(img, (target_w, target_h))
        return img, rects

    def random_crop(self, img, limit_box=None):
        assert img.ndim == 3
        img_h, img_w, _ = img.shape
        target_h, target_w, _ = self.target_shape
        rects = copy.deepcopy(self.base_roi)
        exp_h = rects[3] - rects[1]
        exp_w = rects[2] - rects[0]

        # 1. get random number
        random_num1 = self._get_random_num()
        random_num2 = self._get_random_num()
        random_num3 = self._get_random_num()
        random_num4 = self._get_random_num()

        # 2. random jitter rects
        rects[0] += random_num1 * (exp_w * self.crop_jitter_range)
        rects[1] += random_num2 * (exp_h * self.crop_jitter_range)
        rects[2] += random_num3 * (exp_w * self.crop_jitter_range)
        rects[3] += random_num4 * (exp_h * self.crop_jitter_range)
        if limit_box is not None:
            rects[0] = limit_box[0] if rects[0] >= limit_box[0] else rects[0]
            rects[1] = limit_box[1] if rects[1] >= limit_box[1] else rects[1]
            rects[2] = limit_box[2] if rects[2] <= limit_box[2] else rects[2]
            rects[3] = limit_box[3] if rects[3] <= limit_box[3] else rects[3]
        rects = list(map(int, rects))
        rects[0] = max(0, rects[0])
        rects[1] = max(0, rects[1])
        rects[2] = min(img_w, rects[2])
        rects[3] = min(img_h, rects[3])

        # 3. crop and resize image
        img = img[rects[1] : rects[3], rects[0] : rects[2]]
        img = cv2.resize(img, (target_w, target_h))

        return img, rects

    def scale_crop(self, img, limit_box=None):
        """Zoom the rects and crop the image."""
        assert img.ndim == 3
        img_h, img_w, _ = img.shape
        target_h, target_w, _ = self.target_shape
        rects = copy.deepcopy(self.base_roi)
        exp_h = rects[3] - rects[1]
        exp_w = rects[2] - rects[0]

        # 1. shift center
        shift_x = self._get_random_num() * self.center_shift_range * img_w
        shift_y = self._get_random_num() * self.center_shift_range * img_h
        rects[0] += int(shift_x)
        rects[1] += int(shift_y)
        rects[2] += int(shift_x)
        rects[3] += int(shift_y)

        # 2. get zoom ratio
        random_num = self._get_random_num()
        ratio = random_num * self.crop_jitter_range
        offset_h = int(0.5 * exp_h * ratio)
        offset_w = int(0.5 * exp_w * ratio)
        rects[0] -= offset_w
        rects[2] += offset_w
        rects[1] -= offset_h
        rects[3] += offset_h
        if limit_box is not None:
            rects[0] = limit_box[0] if rects[0] >= limit_box[0] else rects[0]
            rects[1] = limit_box[1] if rects[1] >= limit_box[1] else rects[1]
            rects[2] = limit_box[2] if rects[2] <= limit_box[2] else rects[2]
            rects[3] = limit_box[3] if rects[3] <= limit_box[3] else rects[3]
            rects = list(map(int, rects))

        # 3. crop and resize
        img = img[rects[1] : rects[3], rects[0] : rects[2]]
        img = cv2.resize(img, (target_w, target_h))
        return img, rects

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"crop_type={self.crop_type}"
        repr_str += f"target_shape={self.target_shape}"
        repr_str += f"base_roi={self.base_roi}"
        repr_str += f"crop_jitter_range={self.crop_jitter_range}"
        repr_str += f"center_shift_range={self.center_shift_range}"
        repr_str += f"random_type={self.random_type}"
        repr_str += f"crop_frames_prob_dist={self.crop_frames_prob_dist}"
        repr_str += f"use_crop_limit={self.use_crop_limit}"
        return repr_str


@OBJECT_REGISTRY.register
class GenerateGaussianVector(object):
    r"""Generate Gaussian Vector label.

    This is modified based on GluonFace. More details can be found in
    https://arxiv.org/pdf/2010.01318.pdf

    Args:
        num_ldmk: number of landmark.
        feat_stride: feature strides of input / output.
        vector_size: output vector size, (widht, height).
        sigma: $\sigma$ of Gaussian formula. Defaults to 2.
        encoding_method: "standard" or "unbiased" encoding. See
            `GenerateGaussianHeatmap` for more details. Default to "standard".
    """

    def __init__(
        self,
        num_ldmk: int,
        feat_stride: Union[float, Tuple[float, float]],
        vector_size: Tuple[int, int],
        sigma: float = 2,
        encoding_method: str = "standard",
    ):
        self.num_ldmk = num_ldmk
        self.feat_stride = feat_stride
        self.width, self.height = vector_size
        self.sigma = sigma
        self.encoding_method = encoding_method
        self.radius = 3 * sigma
        kernel_size = 2 * self.radius + 1
        center = kernel_size // 2
        self.clip = np.arange(0, kernel_size, 1, np.float32)
        self.clip = np.exp(
            -((self.clip - center) ** 2) / (2 * self.sigma ** 2)
        )
        self.clip[center] += 1

    def transform_ldmk_to_vector(self, ldmk: np.ndarray):
        vector_x = np.zeros((self.num_ldmk, self.width))
        vector_y = np.zeros((self.num_ldmk, self.height))
        for i in range(self.num_ldmk):
            if self.encoding_method == "unbiased":
                vector_x[i], vector_y[i] = self._unbiased_encode(ldmk[i])
            elif self.encoding_method == "standard":
                vector_x[i], vector_y[i] = self._standard_encode(ldmk[i])
            else:
                raise ValueError(
                    "Not suppored encoding method: {self.encoding_method}."
                )
        return vector_x, vector_y

    def _unbiased_encode(self, ldmk: np.ndarray):
        ldmk = ldmk / self.feat_stride
        x, y = ldmk[0], ldmk[1]
        vector_x_instance = np.zeros((self.width))
        vector_y_instance = np.zeros((self.height))
        ul = [math.ceil(x - self.radius), math.ceil(y - self.radius)]
        br = [math.ceil(x + self.radius), math.ceil(y + self.radius)]

        if ul[0] >= self.width or ul[1] >= self.height:
            return vector_x_instance, vector_y_instance
        if br[0] <= 0 or br[1] <= 0:
            return vector_x_instance, vector_y_instance

        # generate kernel
        kernel_x = np.arange(self.radius * 2)
        kernel_y = np.arange(self.radius * 2)
        x0 = x - math.floor(x) + self.radius - 1
        y0 = y - math.floor(y) + self.radius - 1
        kernel_x = np.exp(-((kernel_x - x0) ** 2) / (2 * self.sigma ** 2))
        kernel_y = np.exp(-((kernel_y - y0) ** 2) / (2 * self.sigma ** 2))

        g_x = max(0, -ul[0]), min(self.width, br[0]) - ul[0]
        g_y = max(0, -ul[1]), min(self.height, br[1]) - ul[1]
        img_x = max(0, ul[0]), min(self.width, br[0])
        img_y = max(0, ul[1]), min(self.height, br[1])

        vector_x_instance[img_x[0] : img_x[1]] = kernel_x[g_x[0] : g_x[1]]
        vector_y_instance[img_y[0] : img_y[1]] = kernel_y[g_y[0] : g_y[1]]

        return vector_x_instance, vector_y_instance

    def _standard_encode(self, ldmk: np.ndarray):
        ldmk = (ldmk / self.feat_stride).round().astype("int32")
        x, y = ldmk[0], ldmk[1]
        vector_x_instance = np.zeros((self.width))
        vector_y_instance = np.zeros((self.height))
        ul = [int(x - self.radius), int(y - self.radius)]
        br = [int(x + self.radius + 1), int(y + self.radius + 1)]

        if ul[0] >= self.width or ul[1] >= self.height:
            return vector_x_instance, vector_y_instance
        if br[0] <= 0 or br[1] <= 0:
            return vector_x_instance, vector_y_instance

        g_x = max(0, -ul[0]), min(self.width, br[0]) - ul[0]
        g_y = max(0, -ul[1]), min(self.height, br[1]) - ul[1]
        img_x = max(0, ul[0]), min(self.width, br[0])
        img_y = max(0, ul[1]), min(self.height, br[1])
        vector_x_instance[img_x[0] : img_x[1]] = self.clip[g_x[0] : g_x[1]]
        vector_y_instance[img_y[0] : img_y[1]] = self.clip[g_y[0] : g_y[1]]
        return vector_x_instance, vector_y_instance

    def __call__(self, data):
        ldmk = data["gt_ldmk"]
        # ldmk_attr = data["gt_ldmk_attr"]
        # TODO(yuhao.dou): Add landmark attribution processing.
        # TODO(yuhao.dou): Add 3D landmark.
        vector_x, vector_y = self.transform_ldmk_to_vector(ldmk.copy()[:, :2])
        data["gt_vector_x"] = vector_x
        data["gt_vector_y"] = vector_y
        vector_x_weight = np.ones_like(vector_x)
        vector_y_weight = np.ones_like(vector_y)
        vector_x_weight[~np.any(vector_x, axis=1)] = 0
        vector_y_weight[~np.any(vector_y, axis=1)] = 0
        vector_x_weight[vector_x < 1e-5] *= 0.1
        vector_y_weight[vector_y < 1e-5] *= 0.1
        data["gt_vector_weight_x"] = vector_x_weight
        data["gt_vector_weight_y"] = vector_y_weight
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"num_ldmk={self.num_ldmk}"
        repr_str += f"feat_stride={self.feat_stride}"
        repr_str += f"vector_size={(self.width, self.height)}"
        repr_str += f"sigma={self.sigma}"
        repr_str += f"encoding_method={self.encoding_method}"
        return repr_str


@OBJECT_REGISTRY.register
class RandomShiftRotateScale(object):
    """Random Shift, Rotate and Scale the image, landmark and box.

    This class shift,rotate and scale both image, landmark and box.
    Shift means shift the whole image. Scale means center crop
    the whole image without depending on the bounding box.
    It is generally not recommended to rotate the box
    because it will no longer be the minimum bounding box.
    Especially when the rotation angle is very large.
    Landmark points rotated out of the image will be ignored.
    {0:invisible,1:occlusion,2:fully visible,3:ignore}

    Args:
        rotate_prob: Probability of rotate. Defaults to 0.5.
        max_rotate_angle: The maximum of rotated angle. Defaults to 0.0.
        bounded: Rotate the image without cutoff or not. Defaults to True.
        shift_prob: Shift probability. Defaults to 0.0.
        max_shift_range: Range of x, y shift. Defaults to (0.0, 0.0).
            For example, for x-axis, it will shift
            (img_width*random.uniform(-shift[0],shift[0])) pixel.
        resize: Whether recover to raw shape after rotation. Defaults to False.
            If out_shape is not None, this operation will not take place.
        out_shape: Output shape of the image. Defaults to None.
        img_scale: Whether scale image. Defaults to False.
            It's important to note, image scale is not resize the image.
            It's actually a center cropping of the image.
        scale_range: Sample random number of img scale. Defaults to (1.0, 1.0).
        cv_flags: Method used to interpolation. Defaults to cv2.INTER_LINEAR.
        border_value: Value to fill the border. Defauls to 0.
    """

    def __init__(
        self,
        rotate_prob: float = 0.5,
        max_rotate_angle: float = 0.0,
        bounded: bool = True,
        shift_prob: float = 0.0,
        max_shift_range: Tuple[float, float] = (0.0, 0.0),
        resize: bool = False,
        out_shape: Optional[Tuple[int, int]] = None,
        img_scale: bool = False,
        scale_range: Tuple[float, float] = (1.0, 1.0),
        cv_flags: int = cv2.INTER_LINEAR,
        border_value: int = 0,
    ):
        self.rotate_prob = rotate_prob
        self.max_rotate_angle = max_rotate_angle
        self.bounded = bounded
        self.shift_prob = shift_prob
        self.max_shift_range = max_shift_range
        self.resize = resize
        self.scale_range = scale_range
        self.out_shape = out_shape
        self.img_scale = img_scale
        self.cv_flags = cv_flags
        self.border_value = border_value

    def _shift_rotate_scale_img(self, data):
        """Rotate and shift images by warpaffine with cutoff."""
        img = data["img"]
        img_h, img_w = img.shape[:2]
        new_h, new_w = img_h, img_w

        mat = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        offset_x, offset_y = 0, 0

        # get rotate matrix
        if random.random() < self.rotate_prob:
            angle = random.randint(
                -self.max_rotate_angle, self.max_rotate_angle
            )
            # the image center
            center = (img_w // 2, img_h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            mat[:2, :] = rot_mat
            if self.bounded:
                cos = np.abs(mat[0, 0])
                sin = np.abs(mat[0, 1])
                # compute the new bounding dimensions of the image
                new_w = int(img_h * sin + img_w * cos)
                new_h = int(img_h * cos + img_h * sin)

                # adjust the rotation matrix to take into account translation
                mat[0, 2] += new_w / 2 - center[0]
                mat[1, 2] += new_h / 2 - center[1]

        # get shift matrix
        if random.random() < self.shift_prob:
            max_x_shift = self.max_shift_range[0]
            max_y_shift = self.max_shift_range[1]

            # random translation
            offset_x = int(img_w * random.uniform(-max_x_shift, max_x_shift))
            offset_y = int(img_h * random.uniform(-max_y_shift, max_y_shift))
            offset_mat = np.float32(
                [[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]]
            )
            mat = np.dot(mat, offset_mat)

        # resize to raw image shape or specified out shape
        if self.resize and self.out_shape is None:
            scale_mat = np.float32(
                [[img_w / new_w, 0, 0], [0, img_h / new_h, 0], [0, 0, 1]]
            )
            mat = np.dot(scale_mat, mat)
            new_h, new_w = img_h, img_w
        elif self.out_shape is not None:
            scale_mat = np.float32(
                [
                    [self.out_shape[1] / new_w, 0, 0],
                    [0, self.out_shape[0] / new_h, 0],
                    [0, 0, 1],
                ]
            )
            mat = np.dot(scale_mat, mat)
            new_h, new_w = self.out_shape

        # crop image if needed
        if self.img_scale:
            self._cal_scale(new_h, new_w)
            crop_mat = np.float32(
                [[1, 0, -self.left], [0, 1, -self.top], [0, 0, 1]]
            )
            mat = np.dot(crop_mat, mat)
            new_h, new_w = (self.bottom - self.top), (self.right - self.left)

        # apply warp affine by using matrix above
        img = cv2.warpAffine(
            img, mat[:2, :], (new_w, new_h), borderValue=self.border_value
        )

        data["img"] = img
        return mat[:2, :]

    def _shift_rotate_scale_ldmk(self, data, mat):
        src_ldmk = data["gt_ldmk"]

        # src_ldmk: (num_points, 2 or 3)
        src_ldmk = src_ldmk.reshape((-1, src_ldmk.shape[-1]))
        dst_ldmk = src_ldmk.copy()
        dst_ldmk = self._shift_rotate_scale_points(src_ldmk, mat)
        # dst_ldmk: (num_points, 2 or 3)

        data["gt_ldmk"] = dst_ldmk

    def _shift_rotate_scale_ellipse_param(self, data, mat):
        ang_deg = np.rad2deg(np.arccos(mat[0, 0]))
        ellipse_param = data["gt_pupil_ellipse_param"]
        ellipse_param[-1] = ellipse_param[-1] - ang_deg
        ellipse_center = ellipse_param.copy()[:2]
        ellipse_center = np.dot(ellipse_center, mat[:, :2].T) + mat[:, 2]
        ellipse_param[:2] = ellipse_center
        data["gt_pupil_ellipse_param"] = ellipse_param

    def _shift_rotate_scale_points(self, points, mat):
        # src_points: (num_points, 2)
        src_points = points[:, :2]
        dst_points = np.dot(src_points, mat[:, :2].T) + mat[:, 2]

        points[:, :2] = dst_points
        return points

    def _shift_rotate_scale_box(self, data, mat):
        # boxes: (num_boxes, 4)  [x1, y1, x2, y2]
        boxes = data["gt_bboxes"]
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, 0)

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
        dst_points = self._shift_rotate_scale_points(src_points, mat)
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

        data["gt_bboxes"] = dst_boxes

    def _cal_scale(self, img_h, img_w):
        # cal scale area
        real_scale = random.uniform(self.scale_range[0], self.scale_range[1])
        real_h, real_w = int(real_scale * img_h), int(real_scale * img_w)
        real_l = int((img_w - real_w) * 0.5)
        real_t = int((img_h - real_h) * 0.5)
        real_r = real_l + real_w
        real_b = real_t + real_h
        self.left = np.maximum(real_l, 0)
        self.top = np.maximum(real_t, 0)
        self.right = np.minimum(real_r, img_w)
        self.bottom = np.minimum(real_b, img_h)

    def __call__(self, data):
        assert "img" in data

        # transform img
        mat = self._shift_rotate_scale_img(data)

        # transform label by using the same matrix as the image
        if "gt_ldmk" in data:
            self._shift_rotate_scale_ldmk(data, mat)
        if "gt_bboxes" in data:
            self._shift_rotate_scale_box(data, mat)
        if "gt_pupil_ellipse_param" in data:
            self._shift_rotate_scale_ellipse_param(data, mat)

        data["img"] = np.clip(data["img"], 0, 255)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"rotate_prob={self.rotate_prob}"
        repr_str += f"max_rotate_angle={self.max_rotate_angle}"
        repr_str += f"bounded={self.bounded}"
        repr_str += f"shift_prob={self.shift_prob}"
        repr_str += f"shift={self.max_shift_range}"
        repr_str += f"resize={self.resize}"
        repr_str += f"scale_range={self.scale_range}"
        repr_str += f"out_shape={self.out_shape}"
        repr_str += f"img_scale={self.img_scale}"
        repr_str += f"cv_flags={self.cv_flags}"
        repr_str += f"border_value={self.border_value}"
        return repr_str


@OBJECT_REGISTRY.register
class ClipBoxes(object):
    """Clip boxes outside the image."""

    def __call__(self, data):
        assert "gt_bboxes" in data
        assert "img" in data
        gt_bboxes = data["gt_bboxes"]
        img = data["img"]
        img_h, img_w, _ = img.shape

        gt_bboxes[:, 0] = np.maximum(np.minimum(gt_bboxes[:, 0], img_w - 1), 0)
        gt_bboxes[:, 1] = np.maximum(np.minimum(gt_bboxes[:, 1], img_h - 1), 0)
        gt_bboxes[:, 2] = np.maximum(np.minimum(gt_bboxes[:, 2], img_w - 1), 0)
        gt_bboxes[:, 3] = np.maximum(np.minimum(gt_bboxes[:, 3], img_h - 1), 0)

        data["gt_bboxes"] = gt_bboxes

        w = gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1
        assert w.all() and h.all()
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@OBJECT_REGISTRY.register
class GenerateGRMITarget(object):
    """Generate landmark heatmap and offset.

    If 'pixel' in ldmk_loss_type, generate heatmap and
    offsetmap as described in`"Towards Accurate Multi-person Pose Estimation in
    the Wild"  <https://arxiv.org/pdf/1701.01779.pdf>`_ paper.
    This paper's method is called G-RMI which is the class name means.

    If 'one-hot' in ldmk_loss_type, generate one-hot heatmap by
    caculate the ldmk locations in feature map by subtracting offset from
    keypoints then divided by stride. Heatmap is the differnce of ldmk
    locations in feature map and the floor of them.

    Args:
        num_ldmk: number of keypoints.
        feat_shape: The shape of output featuremap.
        ldmk_loss_type: keypoints loss.
        ldmk_use_gauss: Whether use gauss to encode. Defaults to False.
        ldmk_unbiased_encode: Whether use ubiased encode. Defaults to True.
        ldmk_keep_occlusion: Whether keep occlusion info. Defaults to True.
        ldmk_keep_outside_box: Whether keep ldmk outside. Defaults to False.
        ldmk_pos_distance: Keypoints distance ratio. Defaults to 0.1.
        default_value: Default value to fill ldmk label. Defaults to -1.
    """

    def __init__(
        self,
        num_ldmk,
        feat_shape,
        ldmk_loss_type,
        ldmk_use_gauss=False,
        ldmk_unbiased_encode=True,
        ldmk_keep_occlusion=True,
        ldmk_keep_outside_box=False,
        ldmk_pos_distance=0.1,
        default_value=-1,
    ):
        self.num_ldmk = num_ldmk
        self.feat_width = feat_shape[1]
        self.feat_height = feat_shape[0]
        self.ldmk_keep_occlusion = ldmk_keep_occlusion
        self.ldmk_keep_outside_box = ldmk_keep_outside_box
        self.ldmk_loss_type = ldmk_loss_type
        self.ldmk_unbiased_encode = ldmk_unbiased_encode
        self.ldmk_use_gauss = ldmk_use_gauss
        self.ldmk_pos_distance = ldmk_pos_distance
        self.default_value = default_value

    def __call__(self, data):
        person_boxes = []
        keypoints = []
        classes = data["gt_classes"]
        boxes = data["gt_bboxes"]
        keypoints = data["gt_ldmk"]
        assert len(boxes) == len(classes)
        for i in range(len(boxes)):
            # if this box is person
            if classes[i] == 1 or classes[i] == -1:
                person_boxes.append(boxes[i])
        person_boxes = np.array(person_boxes)
        num_boxes = person_boxes.shape[0]
        keypoints = keypoints.reshape((num_boxes * self.num_ldmk, 3))

        scales_xy = np.zeros((num_boxes, 2), dtype=np.float32)
        scales_xy[:, 0] = self.feat_width / (
            person_boxes[:, 2] - person_boxes[:, 0] + 1
        )
        scales_xy[:, 1] = self.feat_height / (
            person_boxes[:, 3] - person_boxes[:, 1] + 1
        )
        scales_xy = np.tile(
            scales_xy, (1, self.num_ldmk)
        )  # (num_boxes, num_ldmk*2)
        # (num_boxes * num_ldmk, 2)
        scales_xy = scales_xy.reshape((num_boxes * self.num_ldmk, 2))

        offsets_xy = person_boxes[:, :2]  # (num_boxes, 2)
        # (num_boxes, num_ldmk*2)
        offsets_xy = np.tile(offsets_xy, (1, self.num_ldmk))
        # (num_boxes * num_ldmk, 2)
        offsets_xy = offsets_xy.reshape((num_boxes * self.num_ldmk, 2))
        # (num_boxes * num_ldmk, 2)
        ldmk_xy = (keypoints[:, :2] - offsets_xy) * scales_xy
        ldmk_xy_int = np.floor(ldmk_xy)  # (num_boxes * num_ldmk, 2)

        if self.ldmk_keep_occlusion:
            vis = np.logical_and(keypoints[:, 2] > 0, keypoints[:, 2] < 3)
        else:
            vis = keypoints[:, 2] == 2

        if not self.ldmk_keep_outside_box:
            within_box = np.logical_and(
                np.logical_and(ldmk_xy_int[:, 0] >= 0, ldmk_xy_int[:, 1] >= 0),
                np.logical_and(
                    ldmk_xy_int[:, 0] < self.feat_width,  # noqa
                    ldmk_xy_int[:, 1] < self.feat_height,
                ),
            )  # noqa
            vis = np.logical_and(within_box, vis)  # (num_boxes * num_ldmk, )
        keep = np.where(vis == 1)[0]  # indices indicating valid keypoints

        if "one-hot" in self.ldmk_loss_type:
            ldmk_label = np.full(
                (num_boxes * self.num_ldmk,),
                fill_value=self.default_value,
                dtype=np.float32,
            )
        else:
            ldmk_label = np.full(
                (
                    num_boxes * self.num_ldmk,
                    self.feat_height * self.feat_width,
                ),
                fill_value=self.default_value,
                dtype=np.float32,
            )

        ldmk_label_weight = np.zeros(ldmk_label.shape, dtype=np.float32)
        ldmk_pos_offset = np.zeros(
            (num_boxes * self.num_ldmk, 2, self.feat_height * self.feat_width),
            dtype=np.float32,
        )
        ldmk_pos_offset_weight = np.zeros(
            ldmk_pos_offset.shape, dtype=np.float32
        )

        if len(keep) > 0:
            num_keep_per_box = float(len(keep)) / num_boxes
            if "one-hot" in self.ldmk_loss_type:
                assert not self.ldmk_keep_outside_box
                pos_offset_xy = (
                    ldmk_xy - ldmk_xy_int
                )  # (num_boxes * num_ldmk, 2)
                pos = (
                    ldmk_xy_int[:, 1] * self.feat_width + ldmk_xy_int[:, 0]
                )  # (num_boxes * num_ldmk,)
                keep_pos = pos[keep].astype(np.int32)
                ldmk_label[keep] = keep_pos
                ldmk_label_weight[keep] = 1.0 / len(keep)
                ldmk_pos_offset[keep, 0, keep_pos] = pos_offset_xy[keep, 0]
                ldmk_pos_offset[keep, 1, keep_pos] = pos_offset_xy[keep, 1]
                ldmk_pos_offset_weight[
                    keep, 0, keep_pos
                ] = num_keep_per_box / len(keep)
                ldmk_pos_offset_weight[
                    keep, 1, keep_pos
                ] = num_keep_per_box / len(keep)
                assert (
                    ldmk_pos_offset.min() >= 0 and ldmk_pos_offset.max() <= 1
                )
            elif "pixel" in self.ldmk_loss_type:
                feat_x_int = np.arange(0, self.feat_width)
                feat_y_int = np.arange(0, self.feat_height)
                feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
                feat_x_int = feat_x_int.reshape((-1,))
                feat_y_int = feat_y_int.reshape(
                    (-1,)
                )  # (feat_height * feat_width, )
                pos_distance = self.ldmk_pos_distance * self.feat_width
                num_keep_pos = 0
                for keep_i in keep:
                    if self.ldmk_use_gauss:
                        if self.ldmk_unbiased_encode:
                            pos_offset_x = ldmk_xy[keep_i, 0] - feat_x_int
                            pos_offset_y = ldmk_xy[keep_i, 1] - feat_y_int
                        else:
                            pos_offset_x = ldmk_xy_int[keep_i, 0] - feat_x_int
                            pos_offset_y = ldmk_xy_int[keep_i, 1] - feat_y_int
                        sigma = self.ldmk_gauss_sigma
                        thre = (3 * sigma) ** 2
                        dis = pos_offset_x ** 2 + pos_offset_y ** 2
                        keep_pos = np.where((dis <= thre) & (dis >= 0))[0]
                    else:
                        pos_offset_x = ldmk_xy[keep_i, 0] - feat_x_int
                        pos_offset_y = ldmk_xy[keep_i, 1] - feat_y_int
                        pos_offset_x /= pos_distance
                        pos_offset_y /= pos_distance
                        dis = pos_offset_x ** 2 + pos_offset_y ** 2
                        keep_pos = np.where((dis <= 1) & (dis >= 0))[0]
                    if len(keep_pos) > 0:
                        ldmk_label[keep_i] = 0
                        ldmk_label_weight[keep_i] = 1.0
                        if self.ldmk_use_gauss:
                            ldmk_label[keep_i, keep_pos] = np.exp(
                                dis[keep_pos] / (-2.0 * sigma * sigma)
                            )
                            if not self.ldmk_unbiased_encode:
                                ldmk_label[keep_i, np.where(dis == 0)[0]] += 1
                        else:
                            ldmk_label[keep_i, keep_pos] = 1
                        ldmk_pos_offset[keep_i, 0, keep_pos] = pos_offset_x[
                            keep_pos
                        ]
                        ldmk_pos_offset[keep_i, 1, keep_pos] = pos_offset_y[
                            keep_pos
                        ]
                        ldmk_pos_offset_weight[keep_i, 0, keep_pos] = 1.0
                        ldmk_pos_offset_weight[keep_i, 1, keep_pos] = 1.0
                        num_keep_pos += len(keep_pos)
                if num_keep_pos > 0:
                    if "cross_entropy" in self.ldmk_loss_type:
                        ldmk_label_weight[ldmk_label_weight > 0] = (
                            1.0 / num_keep_pos
                        )
                    elif "smooth_L1" in self.ldmk_loss_type:
                        ldmk_label_weight[ldmk_label_weight > 0] = (
                            num_keep_per_box / num_keep_pos
                        )
                    else:
                        raise ValueError(
                            "unknown ldmk loss type {}".format(
                                self.ldmk_loss_type
                            )
                        )
                    ldmk_pos_offset_weight[ldmk_pos_offset_weight > 0] = (
                        num_keep_per_box / num_keep_pos
                    )
            else:
                raise ValueError(
                    "unknown ldmk loss type {}".format(self.ldmk_loss_type)
                )

        if "one-hot" in self.ldmk_loss_type:
            ldmk_label = ldmk_label.reshape((num_boxes, self.num_ldmk, 1))
        else:
            ldmk_label = ldmk_label.reshape(
                (num_boxes, self.num_ldmk, self.feat_height, self.feat_width)
            )
        ldmk_label_weight = ldmk_label_weight.reshape(ldmk_label.shape)
        ldmk_pos_offset = ldmk_pos_offset.reshape(
            (num_boxes, self.num_ldmk * 2, self.feat_height, self.feat_width)
        )
        ldmk_pos_offset_weight = ldmk_pos_offset_weight.reshape(
            ldmk_pos_offset.shape
        )

        data["gt_heatmap"] = ldmk_label
        data["gt_heatmap_weight"] = ldmk_label_weight
        data["gt_offset"] = ldmk_pos_offset
        data["gt_offset_weight"] = ldmk_pos_offset_weight
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"num_ldmk={self.num_ldmk}"
        repr_str += f"feat_width={self.feat_width}"
        repr_str += f"feat_height={self.feat_height}"
        repr_str += f"ldmk_keep_occlusion={self.ldmk_keep_occlusion}"
        repr_str += f"ldmk_keep_outside_box={self.ldmk_keep_outside_box}"
        repr_str += f"ldmk_loss_type={self.ldmk_loss_type}"
        repr_str += f"ldmk_unbiased_encode={self.ldmk_unbiased_encode}"
        repr_str += f"ldmk_use_gauss={self.ldmk_use_gauss}"
        repr_str += f"ldmk_pos_distance={self.ldmk_pos_distance}"
        repr_str += f"default_value={self.default_value}"
        return repr_str
