# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from hat.core.affine import get_vcs2bev_img_mat
from hat.core.bev_elevation_utils import decimal_div, decimal_minus
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import EvalResult
from hat.utils.apply_func import convert_numpy

__all__ = ["ANCBevVismaskMetric"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ANCBevVismaskMetric(EvalMetric):
    """Calculate error of bev&voxel vismask task.

    ANCBevVismaskMetric support call metrics and draw vis results.
    Visualization include 6v, gt/pred vismask.

    Args:
        bev_size: Bev size, in pixel.(order is (h,w)).
        vcs_range: Vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2)).
        eval_vcs_range: Roi vcs range which you care, in
            meter.(order is (bottom,right,top,left), e.g.(-12.8, -12.8, 25.6,
            12.8)).
        save_dir: dir to save pred or imgs.
        save_metric_path: the path to save metric results.
        vis: if True, save the vis results(.jpg).
        vis_interval: save frequency.
        vehicle_location: the location to draw self car.
        result_prefix: Prefix of aidi eval result.
    """

    def __init__(
        self,
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        task_name: Optional[bool] = "bev_vismask",
        gt_name: Optional[bool] = "gt_bev_elevation_vismask",
        pred_name: Optional[bool] = "bev_vismask",
        eval_vcs_range: Optional[Sequence[float]] = None,
        save_dir: str = None,
        save_metric_path: str = None,
        vis: bool = False,
        vis_interval: int = 1,
        result_prefix: str = "",
    ):
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.task_name = task_name
        self.gt_name = gt_name
        self.pred_name = pred_name
        self.eval_vcs_range = eval_vcs_range
        if self.eval_vcs_range is not None:
            assert (
                self.eval_vcs_range[0] >= self.vcs_range[0]
                and self.eval_vcs_range[2] <= self.vcs_range[2]
            )
            assert (
                self.eval_vcs_range[1] >= self.vcs_range[1]
                and self.eval_vcs_range[3] <= self.vcs_range[3]
            )
            self.ipm_range = self.init_eval_ipm_range()
        self.save_metric_path = save_metric_path
        self.save_dir_img = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if vis:
                save_dir_img = os.path.join(save_dir, "imgs")
                os.makedirs(save_dir_img, exist_ok=True)
                self.save_dir_img = save_dir_img
        self.save_dir = save_dir

        self.name = ["vismask_iou", "iou_agent"]
        self.vis = vis
        self.vis_interval = vis_interval
        self.bev_show_size = bev_size
        self.img_show_size = (
            int(self.bev_show_size[1] * 360 / 512),
            self.bev_show_size[1],
        )
        self.final_show_size = (
            self.bev_show_size[0] + self.img_show_size[0] * 2,
            self.img_show_size[1] * 3,
            3,
        )
        self._mat_vcs2bev = None
        self.result_prefix = result_prefix
        super(ANCBevVismaskMetric, self).__init__(self.name)

    @property
    def mat_vcs2bev(self):
        if self._mat_vcs2bev is None:
            self._mat_vcs2bev = get_vcs2bev_img_mat(
                self.vcs_range, self.bev_size
            )
        return self._mat_vcs2bev

    def _init_states(self):
        self.add_state(
            name="vismask_iou",
            default=torch.zeros(2),
            dist_reduce_fx="sum",
        )
        self.add_state(
            name="vismask_iou_agent",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state(
            name="num_inst",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state(
            name="num_inst_agent",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )

    def save_res(
        self,
        vis_mask_gt: torch.tensor,
        vis_mask_pred: torch.tensor,
        color_imgs: torch.tensor,
        timestamp: torch.tensor,
    ):
        timestamp = str(int(timestamp.cpu().numpy()[0] * 1000))
        vis_mask_gt = convert_numpy(vis_mask_gt)
        vis_mask_pred = convert_numpy(vis_mask_pred)

        vismask = np.zeros((*vis_mask_gt.shape, 3), dtype=np.uint8)
        vismask[..., 1] = vis_mask_gt * 255
        vismask[..., 2] = vis_mask_pred * 255
        vismask = _draw_ego(vismask, self.mat_vcs2bev)

        # resize 5v to front shape, then save concat 6v img.
        resized_imgs = []
        for _, color_img in enumerate(color_imgs):
            color_img = color_img.detach().cpu().numpy().transpose(1, 2, 0)
            color_img = cv2.cvtColor(color_img * 255, cv2.COLOR_RGB2BGR)
            color_img = cv2.resize(color_img, self.img_show_size[::-1])
            resized_imgs.append(color_img)

        front_views = np.concatenate(
            (
                resized_imgs[1],  # camera_front_left
                resized_imgs[0],  # camera_front
                resized_imgs[2],  # camera_front_right
            ),
            axis=1,
        )
        rear_views = np.concatenate(
            (
                resized_imgs[3],  # camera_rear_left
                resized_imgs[5],  # camera_rear
                resized_imgs[4],  # camera_rear_right
            ),
            axis=1,
        )

        img_res = np.zeros(self.final_show_size)
        img_res[: self.img_show_size[0], :, :] = front_views
        img_res[
            self.img_show_size[0] : -self.img_show_size[0],
            self.bev_show_size[1] : -self.bev_show_size[1],
            :,
        ] = vismask
        img_res[-self.img_show_size[0] :, :, :] = rear_views

        cv2.imwrite(
            os.path.join(self.save_dir_img, timestamp + ".jpg"), img_res
        )

    def save_res_json(self, res):
        if self.save_metric_path is None:
            return
        save_dir, _ = os.path.split(self.save_metric_path)
        os.makedirs(save_dir, exist_ok=True)
        with open(self.save_metric_path, "w") as f:
            f.write(json.dumps(res, ensure_ascii=False, indent=1))

    def init_eval_ipm_range(self):
        h, w = self.bev_size
        spatial_ratio = [
            decimal_div(
                abs(decimal_minus(self.vcs_range[2], self.vcs_range[0])), h
            ),
            decimal_div(
                abs(decimal_minus(self.vcs_range[3], self.vcs_range[1])), w
            ),
        ]

        top = int(
            decimal_div(
                decimal_minus(self.vcs_range[2], self.eval_vcs_range[2]),
                spatial_ratio[0],
            )
        )
        bottom = int(
            decimal_div(
                decimal_minus(self.vcs_range[2], self.eval_vcs_range[0]),
                spatial_ratio[0],
            )
        )
        left = int(
            decimal_div(
                decimal_minus(self.vcs_range[3], self.eval_vcs_range[3]),
                spatial_ratio[1],
            )
        )
        right = int(
            decimal_div(
                decimal_minus(self.vcs_range[3], self.eval_vcs_range[1]),
                spatial_ratio[1],
            )
        )

        top = np.clip(top, 0, h)
        bottom = np.clip(bottom, 0, h)
        left = np.clip(left, 0, w)
        right = np.clip(right, 0, w)

        return (bottom, right, top, left)

    def crop_roi(self, img):
        if self.eval_vcs_range is None:
            return img
        (bottom, right, top, left) = self.ipm_range
        return img[:, top:bottom, left:right]

    def call_vis_mask_metric_agent(
        self,
        vis_mask_gt: torch.tensor,
        vis_mask_pred: torch.tensor,
        agent_gt: torch.tensor,
    ):
        label = vis_mask_gt.squeeze().float()
        pred_label = vis_mask_pred.squeeze().float()
        agent_gt = agent_gt.squeeze().float()

        area_intersect = (
            (pred_label == 1) * (label == 1) * (agent_gt == 1)
        ).sum()
        area_pred_label = ((pred_label == 1) * (agent_gt == 1)).sum()
        area_label = ((label == 1) * (agent_gt == 1)).sum()
        area_union = area_pred_label + area_label - area_intersect

        iou = area_intersect / area_union
        if torch.isnan(iou):
            if area_label > 0:
                self.vismask_iou_agent += 1
            else:
                return
        else:
            self.vismask_iou_agent += iou
        self.num_inst_agent += 1

    def call_vis_mask_metric(
        self,
        vis_mask_gt: torch.tensor,
        vis_mask_pred: torch.tensor,
        bins: int = 2,
        maxn: int = 1,
    ):
        label = vis_mask_gt.squeeze().float()
        pred_label = vis_mask_pred.squeeze().float()

        intersect = pred_label[pred_label == label]

        area_intersect = torch.histc(intersect, bins=bins, max=maxn)
        area_pred_label = torch.histc(pred_label, bins=bins, max=maxn)
        area_label = torch.histc(label, bins=bins, max=maxn)
        area_union = area_pred_label + area_label - area_intersect
        iou = area_intersect / area_union

        if torch.isnan(iou).any():
            self.vismask_iou += 1
        else:
            self.vismask_iou += iou
        self.num_inst += 1

    def update(self, label: dict, pred: dict):
        vis_mask_gts_roi = self.crop_roi(
            img=label[self.task_name][self.gt_name]["vismask"]
        )
        agent_gts_roi = self.crop_roi(
            img=label[self.task_name][self.gt_name]["agent"]
        )
        vis_mask_preds_roi = self.crop_roi(img=pred[self.pred_name][0])
        color_imgs = label["color_imgs"]

        # vis_mask_gts_roi = self.crop_roi(vis_mask_gts["vismask"])
        for idx in range(vis_mask_gts_roi.shape[0]):
            timestamp = label["timestamp"][idx]
            agent_gt_ori = agent_gts_roi[idx]
            vis_mask_gt_roi = vis_mask_gts_roi[idx]
            vis_mask_pred_roi = vis_mask_preds_roi[idx]

            self.call_vis_mask_metric(vis_mask_gt_roi, vis_mask_pred_roi)
            self.call_vis_mask_metric_agent(
                vis_mask_gt_roi, vis_mask_pred_roi, agent_gt_ori
            )

            if (
                self.vis
                and self.save_dir_img
                and self.num_inst % self.vis_interval == 0
            ):
                batch_imgs = [img[idx] for img in color_imgs[0]]
                self.save_res(
                    vis_mask_gt_roi,
                    vis_mask_pred_roi,
                    batch_imgs,
                    timestamp,
                )

    def compute(self):
        all_inVis = float((self.vismask_iou[0] / self.num_inst).cpu().numpy())
        all_Vis = float((self.vismask_iou[1] / self.num_inst).cpu().numpy())
        agent_Vis = float(
            (self.vismask_iou_agent / self.num_inst_agent).cpu().numpy()
        )

        res = {
            "all_inVis": round(all_inVis, 4),
            "all_Vis": round(all_Vis, 4),
            "agent_Vis": round(agent_Vis, 4),
        }
        self.save_res_json(res)
        title = "\n Vismask metric: \n"
        name = (
            "all_inVis".ljust(10)
            + "all_Vis".ljust(10)
            + "agent_Vis".ljust(10)
            + "\n"
        )
        val = (
            str(round(all_inVis, 4)).ljust(10)
            + str(round(all_Vis, 4)).ljust(10)
            + str(round(agent_Vis, 4)).ljust(10)
            + "\n"
        )

        logger.info(title + name + val)
        summary = {}
        for k, v in zip(name.strip().split(), val.strip().split()):
            summary.update({f"{self.result_prefix} {k}": v})
        return EvalResult(summary=summary)


def _draw_ego(img_bev, mat_vcs2bev, dx=0, dy=0, thickness=1):
    """Draw ego car on bev img.

    Args:
        lane: lane pts
        mat_vcs2bev: transformation matrix
        dx, dy: the offset of ploted point
        thickness: line thickness
    """
    bev_ego_pts = (mat_vcs2bev @ np.array([[0], [0], [1]])).transpose()
    bev_ego_pts[:, 0] = bev_ego_pts[:, 0] / bev_ego_pts[:, 2]
    bev_ego_pts[:, 1] = bev_ego_pts[:, 1] / bev_ego_pts[:, 2]
    bev_ego_pts = bev_ego_pts[0, :2]
    cv2.rectangle(
        img_bev,
        (int(bev_ego_pts[0]) - 5 + dx, int(bev_ego_pts[1]) - 10 + dy),
        (int(bev_ego_pts[0]) + 5 + dx, int(bev_ego_pts[1]) + 10 + dy),
        color=(0, 255, 0),
        thickness=thickness,
    )
    return img_bev
