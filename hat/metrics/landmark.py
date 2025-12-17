# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Optional, Union

import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["NormalizedMeanError"]

logger = logging.getLogger(__name__)

FACE_LDMK_NORM_TYPE = ["ION", "IPN", "SIZE", "DIAG"]


@OBJECT_REGISTRY.register
class NormalizedMeanError(EvalMetric):
    """A common metric for facial landmark detection.

    Noramlized Mean Error is the mean Euclidean distance of gt landmarks and
    predicted ones which is normlized by some kind of distance.

    Args:
        num_ldmk: number of landmarks.
        name: metric name that is shown in log.
        norm_type: normalization type. It must be chosen from
            FACE_LDMK_NORM_TYPE. Defaults to "ION".
            In case "ION", NME is noramlized by inter-ocular distance.
            In case "IPN", NME is normalized by inter-pupil distance.
            In case "SIZE", NME is normalized by sqrt of bbox area.
            In case "DIAG", NME is normalized by diagonal of bbox.
            Above-mentioned bbox is the outer minimum bounding rectangle
            of GT landmarks, not bbox annotated by human.
        mode: parse landmark from "coords" or "heatmap" or "vector".
        decoding_method: only accessible in "heatmap" or "vector" mode.
            "diff", "shift" and "taylor" are supported.
        feat_stride: feature stride of input / output.
    """

    def __init__(
        self,
        num_ldmk: int,
        name: str = "NME",
        norm_type: Optional[str] = None,
        mode: str = "coords",
        decoding_method: str = "",
        feat_stride: Optional[float] = None,
    ):
        super().__init__(name)
        self.num_ldmk = num_ldmk
        self.norm_type = norm_type
        self.mode = mode.lower()
        self.decode_heatmap = DecodeHeatmap(decoding_method, feat_stride)
        self.decode_vector = DecodeVector(decoding_method, feat_stride)

    def update(self, data):
        gt_ldmk = data["gt_ldmk"]  # N, num_ldmk, 2/3
        pr_ldmk = self._get_preds(data).reshape(gt_ldmk.shape)

        err = torch.mean(
            torch.sqrt(torch.square(gt_ldmk - pr_ldmk).sum(axis=2)), axis=1
        )
        norm_dist = self._get_norm_dist(gt_ldmk)
        err = (err / norm_dist).sum()
        if self.norm_type is not None:
            err = err * 100
        self.sum_metric += err
        self.num_inst += gt_ldmk.shape[0]

    def _get_preds(self, data):
        if self.mode == "coords":
            pr_ldmk = data["pr_ldmk"]
        elif self.mode == "heatmap":
            heatmap = data["pr_heatmap"]
            batch_size = heatmap.shape[0]
            pr_ldmk = torch.zeros(
                (batch_size, self.num_ldmk, 2), device=heatmap.device
            )
            for i in range(batch_size):
                for j in range(self.num_ldmk):
                    pr_ldmk[i, j] = self.decode_heatmap(heatmap[i, j])
        elif self.mode == "vector":
            vector_x = data["pr_vector_x"].squeeze(2)
            vector_y = data["pr_vector_y"].squeeze(3)
            vector = torch.stack([vector_x, vector_y], axis=1)  # N, 2, K, L
            batch_size = vector.shape[0]
            pr_ldmk = torch.zeros(
                (batch_size, self.num_ldmk, 2), device=vector.device
            )
            for i in range(batch_size):
                for j in range(self.num_ldmk):
                    pr_ldmk[i, j] = self.decode_vector(vector[i, :, j])
        else:
            raise ValueError(f"Not suppored decoding mode: {self.mode}.")
        return pr_ldmk

    def _get_norm_dist(self, gt_ldmk: torch.Tensor):
        # TODO: Finish all
        if self.norm_type is None:
            return 1
        elif self.norm_type == "ION":
            if self.num_ldmk == 68:
                # 300W
                dist = torch.sqrt(
                    torch.square(gt_ldmk[:, 36] - gt_ldmk[:, 45]).sum(axis=1)
                )
            elif self.num_ldmk == 98:
                # WFLW
                raise NotImplementedError()
            elif self.num_ldmk == 106:
                # JD106
                raise NotImplementedError()
            else:
                raise ValueError(
                    f"num_ldmk:{self.num_ldmk} is not supported in ION."
                )
        elif self.norm_type == "IPN":
            if self.num_ldmk == 5:
                dist = torch.sqrt(
                    torch.square(gt_ldmk[:, 0] - gt_ldmk[:, 1]).sum(axis=1)
                )
            else:
                raise NotImplementedError()
        elif self.norm_type == "SIZE":
            min_x = gt_ldmk[:, :, 0].min(axis=1)[0]
            max_x = gt_ldmk[:, :, 0].max(axis=1)[0]
            min_y = gt_ldmk[:, :, 1].min(axis=1)[0]
            max_y = gt_ldmk[:, :, 1].max(axis=1)[0]
            dist = torch.sqrt((max_x - min_x) * (max_y - min_y))
        elif self.norm_type == "DIAG":
            min_x = gt_ldmk[:, :, 0].min(axis=1)[0]
            max_x = gt_ldmk[:, :, 0].max(axis=1)[0]
            min_y = gt_ldmk[:, :, 1].min(axis=1)[0]
            max_y = gt_ldmk[:, :, 1].max(axis=1)[0]
            dist = torch.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        else:
            raise ValueError("Not supported Norm Type.")
        return dist


class DecodeHeatmap(object):
    def __init__(self, decoding_method: str, feat_stride: int = 4):
        self.decoding_method = decoding_method.lower()
        self.feat_stride = feat_stride

    def __call__(self, heatmap: torch.Tensor):
        def sign(x):
            if x > 0:
                return 1
            if x < 0:
                return -1
            return 0

        if torch.max(heatmap) <= 0:
            return np.array([-1, -1])
        height, width = heatmap.shape[0], heatmap.shape[1]
        max_index = torch.argmax(heatmap)
        y = max_index // width
        x = max_index - y * width
        esp = torch.tensor(1e-9, device=heatmap.device)

        diff_x, diff_y = 0, 0
        if x > 0 and x < width - 1:
            diff_x = heatmap[y][x + 1] - heatmap[y][x - 1]
        if y > 0 and y < height - 1:
            diff_y = heatmap[y + 1][x] - heatmap[y - 1][x]

        if self.decoding_method == "diff":
            # The simplest shifting strategy.
            x = x + diff_x * 1.0
            y = y + diff_y * 1.0
        elif self.decoding_method == "shift":
            # Standard deocde method
            x = x + 0.25 * sign(diff_x)
            y = y + 0.25 * sign(diff_y)
        elif self.decoding_method == "taylor":
            # This decoding method which is proposed in DARK Pose.
            # This method is only not recommended during training process
            # because of its complexity. The fowllowing code is only for
            # reference.More details about DARK please refer to
            # https://arxiv.org/abs/1910.06278.
            if 1 < x < width - 2 and 1 < y < height - 2:
                l1 = torch.log(torch.maximum(heatmap[y, x - 1], esp))
                l2 = torch.log(torch.maximum(heatmap[y, x - 2], esp))
                r1 = torch.log(torch.maximum(heatmap[y, x + 1], esp))
                r2 = torch.log(torch.maximum(heatmap[y, x + 2], esp))
                u1 = torch.log(torch.maximum(heatmap[y - 1, x], esp))
                u2 = torch.log(torch.maximum(heatmap[y - 2, x], esp))
                b1 = torch.log(torch.maximum(heatmap[y + 1, x], esp))
                b2 = torch.log(torch.maximum(heatmap[y + 2, x], esp))
                c = torch.log(torch.maximum(heatmap[y, x], esp))
                br = torch.log(torch.maximum(heatmap[y + 1, x + 1], esp))
                bl = torch.log(torch.maximum(heatmap[y + 1, x - 1], esp))
                ur = torch.log(torch.maximum(heatmap[y - 1, x + 1], esp))
                ul = torch.log(torch.maximum(heatmap[y - 1, x - 1], esp))
                dx = 0.5 * (r1 - l1)
                dy = 0.5 * (b1 - u1)
                dxx = 0.25 * (r2 - 2 * c + l2)
                dxy = 0.25 * (br + ul - bl - ur)
                dyy = 0.25 * (b2 - 2 * c + u2)
                derivative = torch.tensor([[dx], [dy]])
                hessian = torch.tensor([[dxx, dxy], [dxy, dyy]])
                if dxx * dyy - dxy ** 2 != 0:
                    hessian_inv = torch.inverse(hessian)
                    offset = -torch.matmul(hessian_inv, derivative)
                    offset = torch.squeeze(offset.T, axis=0)
                    x = x + offset[0]
                    y = y + offset[1]
            else:
                x = x + sign(diff_x)
                y = y + sign(diff_y)
        else:
            raise ValueError(
                f"Not supported decoding method {self.decoding_method}."
            )

        landmark = torch.stack([x, y], axis=0) * self.feat_stride
        return landmark


class DecodeVector(object):
    def __init__(
        self,
        decoding_method: str,
        feat_stride: float,
        use_bbs: bool = False,
        max_value: float = 1.4,
    ):
        self.decoding_method = decoding_method
        self.feat_stride = feat_stride
        self._factor = 1.0
        self._use_bbs = use_bbs
        self._max = max_value

    def __call__(self, vector):
        return self.post_process_instance(vector[0], vector[1])

    def _sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def _gauss_modulate(self, vector: torch.Tensor, sigma: float):
        """Gaussian modulation before decoding, which is proposed in DARK.

        Args:
            vector: Predicted vector.
            sigma: Gaussian sigma.

        Returns
            vector_gauss: Modulated vector.
        """
        origin_max = torch.max(vector)
        length = len(vector)
        vector_new = torch.zeros((length + 6 * sigma))
        vector_new[3 * sigma : -3 * sigma] = vector
        vector_gauss = torch.zeros_like(vector)
        kernel = torch.arange(6 * sigma + 1)
        kernel = torch.exp(-((kernel - 3 * sigma) ** 2) / (2 * sigma ** 2))
        for i in range(length):
            vector_gauss[i] = (
                kernel * vector_new[i : i + 6 * sigma + 1]
            ).sum()
        vector_gauss_max = torch.max(vector_gauss)
        vector_gauss = vector_gauss / vector_gauss_max * origin_max

        return vector_gauss

    def post_process_instance(
        self, vector_x: torch.Tensor, vector_y: torch.Tensor
    ):
        """Post-processing for one instance."""
        x = vector_x.argmax()
        y = vector_y.argmax()

        if 0 < x < len(vector_x) - 1:
            diff_x = self._get_offset(vector_x, x)
        elif self._use_bbs:
            diff_x = self._beyond_box(vector_x[x], x)
        else:
            diff_x = 0

        if 0 < y < len(vector_y) - 1:
            diff_y = self._get_offset(vector_y, y)
        elif self._use_bbs:
            diff_y = self._beyond_box(vector_y[y], y)
        else:
            diff_y = 0

        x = (x + diff_x) * self.feat_stride
        y = (y + diff_y) * self.feat_stride
        return torch.stack([x, y], axis=0)

    def _beyond_box(self, value: float, m: Union[int, torch.Tensor]):
        """Beyond Box Strategey(BBS).

        Args:
            value: Max value of the predicted vector.
            m: Maximum index of the predicted vector.

        Returns
            dist: The distance beyond the box. Negative one means the landmark
                locates at the left/upper side of the bbox. Postive one means
                right/down.
        """
        value = torch.clip(value, 0, self._max)
        dist = torch.sqrt(
            torch.log(value / self._max) * (-2 * self._sigma ** 2)
        )
        dist = -dist if m == 0 else dist
        return dist

    def _get_offset(self, vector: torch.Tensor, m: Union[int, torch.Tensor]):
        """Get the offset from the max index.

        Three decoding methods, `diff`, `shift` and `taylor`, are supported.
        """
        diff = vector[m + 1] - vector[m - 1]
        vector_size = len(vector)
        esp = torch.tensor(1e-9, device=vector.device)
        if self.decoding_method == "null":
            diff = 0
        elif self.decoding_method == "diff":
            diff = diff * self._factor
        elif self.decoding_method == "shift":  # standard method
            diff = self._sign(diff) * 0.25
        elif self.decoding_method == "taylor":  # 1-D tayloe decoding
            if 1 < m < vector_size - 2:
                vector = self._gauss_modulate(vector, 2)
                l2 = torch.log(max(vector[m - 2], esp))
                l1 = torch.log(max(vector[m - 1], esp))
                r1 = torch.log(max(vector[m + 1], esp))
                r2 = torch.log(max(vector[m + 2], esp))
                c0 = torch.log(max(vector[m], esp))
                dx = 0.5 * (r1 - l1)
                dxx = 0.25 * (r2 - 2 * c0 + l2)
                diff = -dx / dxx / vector.max()
            else:
                # choose one of the following way.
                diff = self._sign(diff) * 0.25
                # diff = diff * self._factor
        else:
            raise ValueError(
                f"Not supported decoding method: {self.decoding_method}."
            )

        return diff
