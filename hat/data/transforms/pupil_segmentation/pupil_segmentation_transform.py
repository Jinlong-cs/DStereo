# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

from hat.registry import OBJECT_REGISTRY
from .utils import Ellipse

__all__ = [
    "GenerateEllipseMask",
    "GenerateEdgeWeightMap",
    "GenerateDistMap",
    "NormEllipseParam",
]


@OBJECT_REGISTRY.register
class GenerateEllipseMask(object):
    r"""Generate mask for pupil segmentation task.

    The mask is generated from the parameters of the ellipse,
    where the pupil region is 1 and the background is 0.
    """

    def _generate_mask(
        self,
        mask_shape: Sequence[float],
        ellipse_param: np.ndarray,
    ):
        ellipse_mask = np.zeros(mask_shape).astype(np.uint8)
        ellipse_mask = cv2.ellipse(
            ellipse_mask,
            (
                int(ellipse_param[0] + 0.5),
                int(ellipse_param[1] + 0.5),
            ),
            (
                int(ellipse_param[2] / 2 + 0.5),
                int(ellipse_param[3] / 2 + 0.5),
            ),
            int(ellipse_param[-1] + 0.5),
            0,
            360,
            1,
            -1,
        )
        return ellipse_mask

    def __call__(self, data):
        h, w, _ = data["img"].shape
        mask_shape = (h, w)
        gt_elllipse_mask = self._generate_mask(
            mask_shape, data["gt_pupil_ellipse_param"]
        )
        data["gt_pupil_mask"] = gt_elllipse_mask
        return data


@OBJECT_REGISTRY.register
class GenerateEdgeWeightMap(object):
    r"""Generate edge weights map for pupil segmentation task.

    The weights is assigned for cross entropy loss,
    which makes the loss function pay more attention to the boundary,
    so as to alleviate the problem of class imbalance.
    """

    def __init__(self, boundary_weight: int = 20):
        self.boundary_weight = boundary_weight

    def _transform_mask_to_weightmap(self, mask: np.ndarray):
        spatial_weights = cv2.Canny(mask.astype(np.uint8), 0, 1) / 255
        spatial_weights = (
            1
            + cv2.dilate(spatial_weights, (3, 3), iterations=1)
            * self.boundary_weight
        )
        return spatial_weights

    def __call__(self, data):
        mask = data["gt_pupil_mask"].astype(np.uint8)
        spatial_weights = self._transform_mask_to_weightmap(mask)
        data["spat_weights"] = spatial_weights
        return data


@OBJECT_REGISTRY.register
class GenerateDistMap(object):
    r"""Generate dist map for pupil segmentation task.

    The greater the distance from the boundary, the greater the value is.

    Reference:
        Boundary loss for highly unbalanced segmentation
        https://github.com/LIVIAETS/boundary-loss
    """

    def _transform_mask_to_distmap(self, posmask: np.ndarray):
        # Input: Mask. Will be converted to Bool.
        h, w = posmask.shape
        max_dist = np.sqrt((h - 1) ** 2 + (w - 1) ** 2)
        if np.any(posmask):
            assert len(posmask.shape) == 2
            res = np.zeros_like(posmask)
            posmask = posmask.astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                res = (
                    distance(negmask) * negmask
                    - (distance(posmask) - 1) * posmask
                )
            res = res / max_dist
        else:
            # No valid element exists for that category
            res = np.zeros_like(posmask)
        return res

    def __call__(self, data):
        mask = data["gt_pupil_mask"].astype(np.uint8)
        disp_map = self._transform_mask_to_distmap(mask)
        data["dist_map"] = disp_map
        return data


@OBJECT_REGISTRY.register
class NormEllipseParam(object):
    r"""Normalize the ellipse parameters.

    The ellipse parameters is normalizedto be in the range [-1, 1].
    """

    def _norm_ellipse_param(
        self, ellipse_param: np.ndarray, h_affine: np.ndarray
    ):
        # Drop the area
        norm_param = Ellipse(ellipse_param).transform(h_affine)[0][:-1]

        if norm_param[2] > norm_param[3]:
            # This rotates the ellipse by 90 degrees to ensure param 3 is
            # always greater than 2
            norm_param[[2, 3]] = norm_param[
                [3, 2]
            ]  # Exchange major and minor axis
            new_theta = norm_param[-1] + 0.5 * np.pi
            norm_param[-1] = new_theta
        if norm_param[-1] > np.pi:
            norm_param[-1] = norm_param[-1] - np.pi
        if norm_param[-1] < 0:
            norm_param[-1] = norm_param[-1] + np.pi
        return norm_param

    def __call__(self, data):
        h, w, _ = data["img"].shape
        h_affine = np.array([[2 / w, 0, -1], [0, 2 / h, -1], [0, 0, 1]])
        ellipse_param = data["gt_pupil_ellipse_param"]
        ellipse_param[-1] = np.deg2rad(ellipse_param[-1])
        norm_ellipse_param = self._norm_ellipse_param(ellipse_param, h_affine)
        data["gt_norm_pupil_ellipse_param"] = norm_ellipse_param
        return data
