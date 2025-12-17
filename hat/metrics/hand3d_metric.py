# Copyright (c) Horizon Robotics. All rights reserved.
import logging

import torch

from hat.registry import OBJECT_REGISTRY
from .landmark import NormalizedMeanError

__all__ = ["Hand3dNME", "Hand3dJPE"]
logger = logging.getLogger(__name__)

HAND_LDMK_NORM_TYPE = ["MMCP", "SIZE", "DIAG"]


@OBJECT_REGISTRY.register
class Hand3dNME(NormalizedMeanError):
    """Calculate the Normalized Mean Error metric for hand3d reconstruction.

    Args:
        num_ldmk: Number of hand landmarks.
        name: Metric name that is shown in log.
        norm_type:
            Normalization type. It must be chosen from HAND_LDMK_NORM_TYPE.
            Defaults to "MMCP".
            In case "MMCP", NME is normalized by distance of \
                middle finger MCP to wrist.
            In case "SIZE", NME is normalized by sqrt of bbox area.
            In case "DIAG", NME is normalized by diagonal of bbox.
            Above-mentioned bbox is the outer minimum bounding rectangle of
            GT landmarks, not bbox annotated by human.
        mode:
            Control which space the nme error is calculated.
            In case "coords", NME is calculated by pred 2d landmarks.
            In case "reproj", NME is calculated by 2d reproj of \
                pred 3d landmarks.
    """

    def __init__(
        self,
        num_ldmk: int = 21,
        name: str = "NME",
        norm_type: str = "MMCP",
        mode: str = "coords",
    ):
        super().__init__(
            num_ldmk=num_ldmk,
            name=name,
            mode=mode.lower(),
        )
        self.norm_type = norm_type.upper()
        assert self.norm_type in HAND_LDMK_NORM_TYPE

    def update(self, label, data):
        gt_ldmk = label.get("gt_ldmk").clone()  # N, num_ldmk, 2/3
        pr_ldmk = self._get_preds(data).reshape(gt_ldmk.shape)
        assert pr_ldmk is not None, "Error: pr_ldmk in Hand3dNME is None."

        err = torch.mean(
            torch.sqrt(torch.square(gt_ldmk - pr_ldmk).sum(axis=2)), axis=1
        )
        norm_dist = self._get_norm_dist(gt_ldmk)
        err = err / norm_dist
        self.sum_metric += err.sum() * 100  # return nme in percentage
        self.num_inst += gt_ldmk.shape[0]
        del gt_ldmk

    def _get_preds(self, data):
        if self.mode == "coords":
            pr_ldmk = data.get("pred_ldmk").clone()
        elif self.mode == "reproj":
            pr_ldmk = data.get("pred_ldmk_proj").clone()

        return pr_ldmk

    def _get_norm_dist(self, gt_ldmk: torch.Tensor):

        if self.norm_type == "MMCP":
            if self.num_ldmk == 21:
                dist = torch.sqrt(
                    torch.square(gt_ldmk[:, 9] - gt_ldmk[:, 0]).sum(axis=1)
                )
            else:
                raise ValueError(
                    f"num_ldmk:{self.num_ldmk} is not supported in MMCP."
                )

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

        return dist


@OBJECT_REGISTRY.register
class Hand3dJPE(NormalizedMeanError):
    """Joint Position Error metric for hand3d reconstruction.

    Joint Position Error is the mean Euclidean distance of gt landmarks,
    and predicted ones in 3d space.

    Args:
        num_ldmk: Number of hand landmarks.
        name: Metric name that is shown in log.
        mode: Control which space the error is calculated.
            In case "root", JPE is calculated in hand coordinate system.
            In case "cam", JPE is calculated in camera coordinate system.
    """

    def __init__(
        self,
        num_ldmk: int = 21,
        name: str = "JPE",
        mode: str = "root",
    ):
        super().__init__(num_ldmk=num_ldmk, name=name, mode=mode.lower())

    def update(self, label, data):
        if self.mode == "root":
            gt_ldmk = label.get("ldmk3d_relat").clone()  # N, num_ldmk, 2/3
            gt_ldmk_vis = label.get("ldmk3d_vis").clone()  # N, num_ldmk, 2/3
            pr_ldmk = data.get("pred_ldmk3d_relat").clone()

        elif self.mode == "cam":
            gt_ldmk = label.get("gt_ldmk3d").clone()  # N, num_ldmk, 2/3
            gt_ldmk_vis = label.get("ldmk3d_vis").clone()  # N, num_ldmk, 2/3
            pr_ldmk = data.get("pred_ldmk3d").clone()

        else:
            print("Error Mode!")
        assert pr_ldmk is not None, "Error: pr_ldmk in Hand3dJPE is None."

        pr_ldmk = pr_ldmk.reshape(gt_ldmk.shape)
        gt_ldmk_vis = gt_ldmk_vis.reshape((-1, 1, 1))
        err = torch.sqrt(
            torch.sum((pr_ldmk * gt_ldmk_vis - gt_ldmk * gt_ldmk_vis) ** 2, 1)
        )

        self.sum_metric += err.sum()  # return jpe in millimeters
        self.num_inst += gt_ldmk_vis.sum()
        del gt_ldmk
