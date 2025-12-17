# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import json
import logging
import os
from typing import Dict, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from aidisdk.experiment import Table
except ImportError:
    Table = None
from scipy.optimize import linear_sum_assignment
from six import iteritems
from torch import distributed as dist

from hat.core.affine import get_vcs2bev_img_mat
from hat.metrics.metric import EvalMetric
from hat.models.task_modules.bev.crosspt_postprocess_utils import (
    cal_vcs_pts,
    gt_preprocess,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import EvalResult
from hat.visualize.bev_crosspoint import CrossPointVisualize

__all__ = ["ANCCrossPointMetric"]

logger = logging.getLogger(__name__)


def draw_dist_pr_curve(pts_stats_dict, curve_name, save_path):
    """Draw precision and recall within different distance.

    Args:
        pts_stats_dict: Stats(gt_num, tp_num, ...) with different
            distance thresh.
        curve_name: Combine with category and region as name.
        save_path: Curve save path.
    """
    dist_threshold_list = []
    precision_list = []
    recall_list = []
    for k, pt_dict in iteritems(pts_stats_dict):
        num_gt_pt = np.sum(pt_dict["num_gt_pt"])
        num_pred_pt = np.sum(pt_dict["num_pred_pt"])
        num_tp_pt = np.sum(pt_dict["num_tp_pt"])
        recall = np.sum(num_tp_pt) / (num_gt_pt + 1e-6)
        precision = np.sum(num_tp_pt) / (num_pred_pt + 1e-6)
        dist_threshold_list.append(float(k))
        precision_list.append(round(precision, 3))
        recall_list.append(round(recall, 3))
    plt.plot(dist_threshold_list, precision_list, label="precision", color="g")
    plt.plot(dist_threshold_list, recall_list, label="recall", color="b")
    for x, y in zip(dist_threshold_list, precision_list):
        plt.text(x, y, str(y), fontsize=10)
    for x, y in zip(dist_threshold_list, recall_list):
        plt.text(x, y, str(y), fontsize=10)
    plt.xlabel("dist threshold")
    plt.title(curve_name)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def cal_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def draw_pr_curve(recall, precision, category, save_path):
    plt.figure("P-R Curve")
    plt.title("Precision/Recall Curve")
    plt.xlim((0, 1.0))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall, precision, label=category)
    plt.legend()
    plt.savefig(save_path)


@OBJECT_REGISTRY.register
class ANCCrossPointMetric(EvalMetric):
    """Metric for bev-crosspoint task.

    Args:
        stride: Crosspoint output stride.
        target_categorys: Crosspoint output category.
        cls_group_map: Output categories of each group.
        subtype_weights: Subtype weights ues in cal ap.
        bev_size: Bev size, in pixel.(order is (h,w)).
        image_size: Image size, in pixel.(order is (h,w)).
        vcs_range: Vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2)).
        name: Metric name.
        eval_result_file: Eval result dir, if None not save eval result.
        metric_save_path: Path to save metric result.
        save_infer_output_dir: Infer result dir, if None not save infer result.
        visualize_output_dir: Infer result visualization dir,
            If None not visualize infer result.
        metric_cfg: Fine metric config, include region area, confidence thresh,
            distance list and so on.
        ignore_index: Ignored cls_id in train and val stage.
        multi_views_type: The type of restore, supported
            "bev_6v", "bev_7v", "bev_fisheye", "bev_10v", "bev_11v".
        lane_gt_dir: Laneline ground truth of val dataset image,
            need to generate offline.
        result_prefix: Prefix of aidi eval result.
    """

    def __init__(
        self,
        stride: int,
        target_categorys: Sequence[str],
        cls_group_map: Dict[str, Sequence[int]],
        subtype_weights: Sequence[float],
        bev_size: Sequence[int],
        image_size: Sequence[int],
        vcs_range: Sequence[float],
        name: str = "CrossPointMetric",
        eval_result_dir: Optional[str] = None,
        metric_save_path: Optional[str] = None,
        save_infer_output_dir: Optional[str] = None,
        visualize_output_dir: Optional[str] = None,
        metric_cfg: Optional[Dict] = None,
        ignore_index: int = 255,
        multi_views_type: str = "bev_6v",
        lane_gt_dir: Optional[str] = None,
        result_prefix: str = "",
    ):
        super(ANCCrossPointMetric, self).__init__(name=name)
        assert len(target_categorys) == len(subtype_weights)

        self.target_categorys = target_categorys
        self.cls_group_map = cls_group_map
        self.subtype_weights = subtype_weights
        self.attr_list = ["cls", "x", "y"]
        self.attr_num = len(self.attr_list)

        self.lane_gt_dir = lane_gt_dir
        self.eval_result_dir = eval_result_dir
        self.metric_save_path = metric_save_path
        self.save_infer_output_dir = save_infer_output_dir
        self.visualize_output_dir = visualize_output_dir

        self.ignore_index = ignore_index
        self.metric_cfg = metric_cfg
        self.vcs_range = vcs_range
        self.bev_h, self.bev_w = bev_size
        self.stride = stride
        self.out_h = self.bev_h // self.stride
        self.out_w = self.bev_w // self.stride
        _, _, self.top_offset, self.left_offset = vcs_range
        self.scope_h = vcs_range[2] - vcs_range[0]
        self.scope_w = vcs_range[3] - vcs_range[1]
        self.meter_per_out_pixel_h = self.scope_h / self.out_h
        self.meter_per_out_pixel_w = self.scope_w / self.out_w
        self.mat_vcs2bev = get_vcs2bev_img_mat(
            vcs_range=vcs_range,
            bev_size=bev_size,
        )
        self.mat_bev2vcs = np.linalg.inv(self.mat_vcs2bev)
        self.result_prefix = result_prefix
        assert multi_views_type in (
            "bev_6v",
            "bev_7v",
            "bev_10v",
            "bev_11v",
            "bev_fisheye",
        ), f"{multi_views_type} is not support now"
        self.multi_views_type = multi_views_type

        self.image_height, self.image_width = image_size
        if (
            self.multi_views_type == "bev_6v"
            or self.multi_views_type == "bev_7v"
        ):
            self.normal_img_heigt = self.image_height * 2
            self.fisheye_img_height = 0
        elif (
            self.multi_views_type == "bev_10v"
            or self.multi_views_type == "bev_11v"
        ):
            self.normal_img_heigt = self.image_height * 2
            self.fisheye_img_height = self.image_height
        elif self.multi_views_type == "bev_fisheye":
            self.normal_img_heigt = 0
            self.fisheye_img_height = self.image_height
        self.fisheye_img_height_origin = self.normal_img_heigt
        self.bev_res_height_origin = (
            self.normal_img_heigt + self.fisheye_img_height
        )
        self.bev_img_origin = {
            "camera_front_left": (0, 0),
            "camera_front": (self.image_width, 0),
            "camera_front_right": (self.image_width * 2, 0),
            "camera_front_30fov": (self.image_width * 3, 0),
            "camera_rear_left": (0, self.image_height),
            "camera_rear": (self.image_width, self.image_height),
            "camera_rear_right": (self.image_width * 2, self.image_height),
            "fisheye_left": (0, self.fisheye_img_height_origin),
            "fisheye_front": (
                self.image_width,
                self.fisheye_img_height_origin,
            ),
            "fisheye_rear": (
                self.image_width * 2,
                self.fisheye_img_height_origin,
            ),
            "fisheye_right": (
                self.image_width * 3,
                self.fisheye_img_height_origin,
            ),
        }

    def compute(self):
        if dist.get_rank() != 0:
            return None
        return self.cal_metric()

    def save_infer_file(self, pred_stats, label_stats, pack_name, image_name):
        """Save pred pts and gt pts as file."""
        os.makedirs(
            os.path.join(self.save_infer_output_dir, pack_name),
            exist_ok=True,
        )
        prediction_file = os.path.join(
            self.save_infer_output_dir,
            pack_name,
            image_name + ".json",
        )
        json.dump(
            {
                category: [np.array(pt).tolist() for pt in pts]
                for category, pts in pred_stats.items()
            },
            open(prediction_file, "w"),
        )
        gt_file = os.path.join(
            self.save_infer_output_dir,
            pack_name,
            image_name + "_gt.json",
        )
        json.dump(
            {
                category: [np.array(pt).tolist() for pt in pts]
                for category, pts in label_stats.items()
            },
            open(gt_file, "w"),
        )

    def visualize_crosspt(
        self,
        pred_pts,
        label_pts,
        image_files,
        origin_imgs,
        pack_name,
        image_name,
        ignore_mask,
    ):
        bev = np.zeros(
            (
                self.bev_res_height_origin + self.bev_h,
                self.image_width * 4,
                3,
            )
        )

        # Visualize crosspoints on image with lane gt for better
        # analization. If not exist, only show crosspoints result.
        lane_gt_img_path = (
            os.path.join(
                self.lane_gt_dir,
                pack_name,
                image_name + ".jpg",
            )
            if self.lane_gt_dir is not None
            else None
        )
        if lane_gt_img_path is not None and os.path.exists(lane_gt_img_path):
            lane_gt_img = cv2.imread(lane_gt_img_path)
        else:
            lane_gt_img = None

        img_bev = CrossPointVisualize.draw_crosspoint(
            label_pts,
            self.bev_h,
            self.bev_w,
            self.mat_vcs2bev,
            self.target_categorys,
            text="Label crosspoints",
            img_bev=lane_gt_img,
            is_gt=True,
            ignore_mask=ignore_mask,
        )
        bev[
            self.bev_res_height_origin : (
                self.bev_res_height_origin + self.bev_h
            ),
            0 : self.bev_w,
        ] = img_bev

        img_bev = CrossPointVisualize.draw_crosspoint(
            pred_pts,
            self.bev_h,
            self.bev_w,
            self.mat_vcs2bev,
            self.target_categorys,
            text="PRED crosspoints",
            img_bev=lane_gt_img,
            is_gt=False,
            ignore_mask=ignore_mask,
        )
        bev[
            self.bev_res_height_origin : (
                self.bev_res_height_origin + self.bev_h
            ),
            self.bev_w : self.bev_w * 2,
        ] = img_bev

        for idx, image_file in enumerate(image_files):
            if origin_imgs is not None:
                img = cv2.cvtColor(
                    np.array(origin_imgs[idx], np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
            else:
                continue
            img = cv2.resize(img, (self.image_width, self.image_height))
            h, w, _ = img.shape
            img_origin_x, img_origin_y = self.bev_img_origin[
                image_file.split("/")[-2]
            ]
            bev[
                img_origin_y : (img_origin_y + h),
                img_origin_x : (img_origin_x + w),
            ] = img
        cv2.putText(
            bev,
            f"{pack_name}/{image_name}.jpg",
            (50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
        )
        os.makedirs(
            os.path.join(self.visualize_output_dir, pack_name),
            exist_ok=True,
        )
        cv2.imwrite(
            os.path.join(
                self.visualize_output_dir,
                pack_name,
                image_name + ".jpg",
            ),
            bev,
        )

    def update(
        self,
        image_files_batch,
        origin_imgs_batch,
        gt_stats_batch,
        pred_pts_batch,
    ):
        batch_size = len(pred_pts_batch)
        for batch_id in range(batch_size):
            image_files = np.array(image_files_batch)[:, batch_id]
            origin_imgs = [imgs[batch_id] for imgs in origin_imgs_batch[0]]
            pred_pts = pred_pts_batch[batch_id]
            pred_stats = self.convert_pts(pred_pts)
            label_pts = []
            cls_id_offset = 0
            label_prob_crosspoint = None
            for group in self.cls_group_map:
                gt_stats_group = gt_preprocess(batch_id, group, gt_stats_batch)
                gt_stats_group["cls"][
                    gt_stats_group["prob"] == self.ignore_index
                ] = self.ignore_index
                gt_stats_group["cls"][
                    (gt_stats_group["prob"] == 1)
                    * (gt_stats_group["cls"] != self.ignore_index)
                ] += cls_id_offset
                if group == "crosspoints":
                    label_prob_crosspoint = gt_stats_group["prob"]
                label_pts_group = cal_vcs_pts(
                    gt_stats_group,
                    self.metric_cfg["ap_score_thresh"],
                    self.vcs_range,
                    self.meter_per_out_pixel_h,
                    self.meter_per_out_pixel_w,
                    self.ignore_index,
                    is_gt=True,
                )
                label_pts.extend(label_pts_group)
                cls_id_offset += len(self.cls_group_map[group])
            label_stats = self.convert_pts(label_pts)

            tmp = image_files[0].split("/")
            pack_name = tmp[-3]
            image_name = os.path.splitext(tmp[-1])[0]

            if self.save_infer_output_dir:
                self.save_infer_file(
                    pred_stats, label_stats, pack_name, image_name
                )
            if self.visualize_output_dir:
                label_prob_crosspoint[
                    label_prob_crosspoint != self.ignore_index
                ] = 0
                ignore_mask = np.transpose(label_prob_crosspoint, (1, 2, 0))
                ignore_mask = np.tile(ignore_mask, [1, 1, 3]).astype(np.uint8)
                ignore_mask = cv2.resize(
                    ignore_mask,
                    [self.bev_w, self.bev_h],
                    interpolation=cv2.INTER_NEAREST,
                )
                vis_pred_pts = [
                    pt
                    for pt in pred_pts
                    if pt[3] >= self.metric_cfg["score_thresh"]
                ]
                self.visualize_crosspt(
                    vis_pred_pts,
                    label_pts,
                    image_files,
                    origin_imgs,
                    pack_name,
                    image_name,
                    ignore_mask,
                )

    def convert_pts(self, pts):
        """Convert gt or pred pts to dict by cls_id.

        Args:
            pts: pts include all classes.
        """
        pts_stats = {}
        for pt in pts:
            cls_id = pt[2]
            cls_name = [
                k for k, v in self.target_categorys.items() if v == cls_id
            ][0]
            if cls_name not in pts_stats:
                pts_stats[cls_name] = []
            pts_stats[cls_name].append(pt)
        return pts_stats

    def bench_crosspoints(
        self,
        pred_pts,
        gt_pts,
        roi,
        dist_threshold=0.5,
        dist_percent_thresh=0.05,
        absolute_dist_range=25,
        score_thresh=0.5,
        ap_score_thresh=0.05,
    ):
        """Statistic metric of each instance.

        Args:
            pred_pts: Predict crosspt, each as [x, y, cls_id, cls_score].
            gt_pts: Ground truth crosspt, each as [x, y, cls_id, cls_score].
            roi: Metric bounding region.
            dist_threshold: P/R decisive distance threshold.
            dist_percent_thresh: P/R decisive distance percent threshold.
            absolute_dist_range: Use dist_threshold rather than
                dist_percent_thresh within this range.
            score_thresh: Positive instance confidence threshold.
            ap_score_thresh: Score thresh used to cal AP.

        Returns:
            Dict: Metric per instance, include dx, dy, dxy, gt_num,
                tp_num and so on.
        """
        while [] in gt_pts:
            gt_pts.remove([])
        while [] in pred_pts:
            pred_pts.remove([])
        num_gt_lane = len(gt_pts)
        num_pred_lane = len(pred_pts)
        gt_all = (
            np.vstack(gt_pts)[:, :2] if num_gt_lane > 0 else np.empty((0, 2))
        )
        gt_all = gt_all[
            np.logical_and(
                np.logical_and(gt_all[:, 0] >= roi[2], gt_all[:, 0] <= roi[3]),
                np.logical_and(gt_all[:, 1] >= roi[0], gt_all[:, 1] <= roi[1]),
            )
        ]
        filtered_pred_ap = [
            ins for ins in pred_pts if ins[3] > ap_score_thresh
        ]
        num_pred_lane = len(filtered_pred_ap)
        pred_all_ap = (
            np.vstack(filtered_pred_ap)
            if num_pred_lane > 0
            else np.empty((0, 4))
        )
        pred_all_ap = pred_all_ap[
            np.logical_and(
                np.logical_and(
                    pred_all_ap[:, 0] >= roi[2], pred_all_ap[:, 0] <= roi[3]
                ),
                np.logical_and(
                    pred_all_ap[:, 1] >= roi[0], pred_all_ap[:, 1] <= roi[1]
                ),
            )
        ]
        num_gt_pt = gt_all.shape[0]
        num_pred_pt_ap = pred_all_ap.shape[0]  # for ap calculate
        pred_all = pred_all_ap[pred_all_ap[:, 3] > score_thresh]
        num_pred_pt = pred_all.shape[0]  # for p/r calculate
        cal_ap_data = [
            {"is_tp": False, "score": pred[3]} for pred in pred_all_ap
        ]

        cost_mat = np.zeros((num_gt_pt, num_pred_pt))
        pred_all = pred_all[:, :2]
        for i in range(num_gt_pt):
            for j in range(num_pred_pt):
                gt_pt = gt_all[i]
                pred_pt = pred_all[j]
                cost_mat[i, j] = np.linalg.norm(gt_pt - pred_pt)
        gt_inds, pred_inds = linear_sum_assignment(cost_mat)

        cost_mat_ap = np.zeros((num_gt_pt, num_pred_pt_ap))
        pred_all_ap = pred_all_ap[:, :2]
        for i in range(num_gt_pt):
            for j in range(num_pred_pt_ap):
                gt_pt = gt_all[i]
                pred_pt = pred_all_ap[j]
                cost_mat_ap[i, j] = np.linalg.norm(gt_pt - pred_pt)
        gt_inds_ap, pred_inds_ap = linear_sum_assignment(cost_mat_ap)

        stats_with_distlist_dict = {}
        dxy_all = 0
        dx_all = 0
        dy_all = 0
        dxyp_all = 0
        num_tp_pt = 0
        cal_ap_data_copy = copy.deepcopy(cal_ap_data)

        for gt_ind, pred_ind in zip(gt_inds, pred_inds):
            dxy = cost_mat[gt_ind, pred_ind]
            dxyp = abs(dxy / np.linalg.norm(gt_all[gt_ind], 2))
            if dxy <= dist_threshold or dxyp <= dist_percent_thresh:
                dx = abs(pred_all_ap[pred_ind][0] - gt_all[gt_ind][0])
                dy = abs(pred_all_ap[pred_ind][1] - gt_all[gt_ind][1])
                num_tp_pt += 1  # for p/r calculate
                dxy_all += dxy
                dx_all += dx
                dy_all += dy
                dxyp = abs(dxy / np.linalg.norm(gt_all[gt_ind], 2))
                dxyp_all += (
                    dxyp
                    if dxy > absolute_dist_range
                    else np.minimum(dxyp, dist_percent_thresh)
                )

        for gt_ind_ap, pred_ind_ap in zip(gt_inds_ap, pred_inds_ap):
            dxy = cost_mat_ap[gt_ind_ap, pred_ind_ap]
            dxyp = abs(dxy / np.linalg.norm(gt_all[gt_ind_ap], 2))
            if dxy <= dist_threshold or dxyp <= dist_percent_thresh:
                is_tp = 1  # for ap calculate
            else:
                is_tp = 0
            cal_ap_data_copy[pred_ind_ap]["is_tp"] = is_tp

        stats_with_distlist_dict[str(dist_threshold)] = {
            "num_gt_pt": num_gt_pt,
            "num_pred_pt": num_pred_pt,
            "num_tp_pt": num_tp_pt,
            "dx": dx_all,
            "dy": dy_all,
            "dxy": dxy_all,
            "dxyp": dxyp_all,
            "cal_ap_data": cal_ap_data_copy,
        }

        result_dict = {}
        result_dict["num_gt_pt"] = num_gt_pt
        result_dict["num_pred_pt"] = num_pred_pt
        result_dict["stats_with_distlist_dict"] = stats_with_distlist_dict
        return result_dict

    def statsistic_cls_accuracy(
        self,
        pred_pts,
        gt_pts,
        roi,
        dist_threshold=0.5,
        dist_percent_thresh=0.05,
        score_thresh=0.5,
    ):
        """Statistic classify accuracy.

        Args:
            pred_pts: Predict crosspt.
            gt_pts: Ground truth crosspt.
            roi: Metric bounding region.
            dist_threshold: P/R decisive distance threshold.
            dist_percent_thresh: P/R decisive distance percent threshold.
            score_thresh: Positive instance confidence threshold.

        Returns:
            Dict: Metric per instance, include matched_num, true_num.
        """
        gt_all_cls = []
        pred_all_cls = []
        for key in gt_pts:
            gt_all_cls.extend(gt_pts[key])
        for key in pred_pts:
            pred_all_cls.extend(pred_pts[key])
        while [] in gt_all_cls:
            gt_all_cls.remove([])
        while [] in pred_all_cls:
            pred_all_cls.remove([])

        # changepoint branch only has one class,
        # classify accuracy is meaningless
        gt_all_cls = [
            ins
            for ins in gt_all_cls
            if int(ins[2]) != self.target_categorys["changepoint"]
        ]
        num_gt_pt = len(gt_all_cls)
        gt_all_cls = (
            np.vstack(gt_all_cls) if num_gt_pt > 0 else np.empty((0, 4))
        )
        gt_all_cls = gt_all_cls[
            np.logical_and(
                np.logical_and(
                    gt_all_cls[:, 0] >= roi[2], gt_all_cls[:, 0] <= roi[3]
                ),
                np.logical_and(
                    gt_all_cls[:, 1] >= roi[0], gt_all_cls[:, 1] <= roi[1]
                ),
            )
        ]
        pred_all_cls = [
            ins
            for ins in pred_all_cls
            if ins[3] > score_thresh
            and int(ins[2]) != self.target_categorys["changepoint"]
        ]
        num_pred_pt = len(pred_all_cls)
        pred_all_cls = (
            np.vstack(pred_all_cls) if num_pred_pt > 0 else np.empty((0, 4))
        )
        pred_all_cls = pred_all_cls[
            np.logical_and(
                np.logical_and(
                    pred_all_cls[:, 0] >= roi[2], pred_all_cls[:, 0] <= roi[3]
                ),
                np.logical_and(
                    pred_all_cls[:, 1] >= roi[0], pred_all_cls[:, 1] <= roi[1]
                ),
            )
        ]

        num_gt_pt = gt_all_cls.shape[0]
        num_pred_pt = pred_all_cls.shape[0]
        cost_mat = np.zeros((num_gt_pt, num_pred_pt))
        for i in range(num_gt_pt):
            for j in range(num_pred_pt):
                gt_pt = gt_all_cls[i]
                pred_pt = pred_all_cls[j]
                cost_mat[i, j] = np.linalg.norm(gt_pt[:2] - pred_pt[:2])
        gt_inds, pred_inds = linear_sum_assignment(cost_mat)

        matched_num = 0
        true_num = 0
        for gt_ind, pred_ind in zip(gt_inds, pred_inds):
            dxy = cost_mat[gt_ind, pred_ind]
            dxyp = abs(dxy / np.linalg.norm(gt_all_cls[gt_ind][:2], 2))
            if dxy <= dist_threshold or dxyp <= dist_percent_thresh:
                matched_num += 1
                if gt_all_cls[gt_ind][2] == pred_all_cls[pred_ind][2]:
                    true_num += 1

        acc_stats = {
            "matched_num": matched_num,
            "true_num": true_num,
        }
        return acc_stats

    def cal_crosspoint_ap(
        self, pred_stats, num_gt_pt_all, region_name, category
    ):
        while [] in pred_stats:
            pred_stats.remove([])
        pred_stats_all = []
        for i in pred_stats:
            pred_stats_all.extend(i)
        pred_stats_all.sort(key=lambda x: x["score"], reverse=True)
        tp = np.array([x["is_tp"] for x in pred_stats_all])
        fp = 1 - tp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / np.maximum(num_gt_pt_all, 1e-6)
        prec = tp / np.maximum(tp + fp, 1e-6)
        ap = round(cal_ap(rec, prec), 4)

        save_path = os.path.join(
            self.eval_result_dir, f"{region_name}_pr_curve.png"
        )
        draw_pr_curve(rec, prec, category, save_path)
        return ap

    def eval_pts(
        self,
        pts_stats_dict,
        dist_threshold,
        region_name,
        eval_result_file=None,
    ):
        """Statistic metric of whole dataset.

        Args:
            pts_stats_dict: Base statistics of different category and distance.
            dist_threshold: P/R decisive distance threshold.
            region_name: Metric region name, such as forward-20-50.
            eval_result_file: Metric result save path.
        """
        metric_title = (
            "{:>26} {:>8} {:>8} {:>8} {:>8} {:>8} "
            + "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n"
        )
        output_stats = {"all": {}}
        num_gt_pt_all = 0
        weight_num_gt_pt_all = 0
        num_pred_pt_all = 0
        num_tp_pt_all = 0
        dx_all = 0
        dy_all = 0
        dxy_all = 0
        dxyp_all = 0
        ap_all = 0
        weight_ap_all = 0  # mean AP cal by different cls weight

        for cls_id, category in enumerate(self.target_categorys):
            output_stats.update({category: {}})
            if len(pts_stats_dict[category]) == 0:
                return
            num_gt_pt = np.array(
                [
                    i["stats_with_distlist_dict"][str(dist_threshold)][
                        "num_gt_pt"
                    ]
                    for i in pts_stats_dict[category]
                ]
            )
            num_pred_pt = np.array(
                [
                    i["stats_with_distlist_dict"][str(dist_threshold)][
                        "num_pred_pt"
                    ]
                    for i in pts_stats_dict[category]
                ]
            )
            num_tp_pt = np.array(
                [
                    i["stats_with_distlist_dict"][str(dist_threshold)][
                        "num_tp_pt"
                    ]
                    for i in pts_stats_dict[category]
                ]
            )
            dx = np.array(
                [
                    i["stats_with_distlist_dict"][str(dist_threshold)]["dx"]
                    for i in pts_stats_dict[category]
                ]
            )
            dy = np.array(
                [
                    i["stats_with_distlist_dict"][str(dist_threshold)]["dy"]
                    for i in pts_stats_dict[category]
                ]
            )
            dxy = np.array(
                [
                    i["stats_with_distlist_dict"][str(dist_threshold)]["dxy"]
                    for i in pts_stats_dict[category]
                ]
            )
            dxyp = np.array(
                [
                    i["stats_with_distlist_dict"][str(dist_threshold)]["dxyp"]
                    for i in pts_stats_dict[category]
                ]
            )
            cal_ap_data = [
                i["stats_with_distlist_dict"][str(dist_threshold)][
                    "cal_ap_data"
                ]
                for i in pts_stats_dict[category]
            ]

            num_gt_pt_all += np.sum(num_gt_pt)
            weight_num_gt_pt_all += (
                np.sum(num_gt_pt) * self.subtype_weights[cls_id]
            )
            num_pred_pt_all += np.sum(num_pred_pt)
            num_tp_pt_all += np.sum(num_tp_pt)
            dx_all += np.sum(dx)
            dy_all += np.sum(dy)
            dxy_all += np.sum(dxy)
            dxyp_all += np.sum(dxyp)

            recall = np.sum(num_tp_pt) / (np.sum(num_gt_pt) + 1e-6)
            precision = np.sum(num_tp_pt) / (np.sum(num_pred_pt) + 1e-6)
            dx = np.sum(dx) / (np.sum(num_tp_pt) + 1e-6)
            dy = np.sum(dy) / (np.sum(num_tp_pt) + 1e-6)
            dxy = np.sum(dxy) / (np.sum(num_tp_pt) + 1e-6)
            dxyp = np.sum(dxyp) / (np.sum(num_tp_pt) + 1e-6)
            ap = self.cal_crosspoint_ap(
                cal_ap_data,
                np.sum(num_gt_pt),
                region_name,
                category,
            )
            ap_all += ap * np.sum(num_gt_pt)
            weight_ap_all += (
                ap * np.sum(num_gt_pt) * self.subtype_weights[cls_id]
            )

            output_stats[category]["ap"] = ap
            output_stats[category]["Precision"] = round(precision, 4)
            output_stats[category]["Recall"] = round(recall, 4)
            output_stats[category]["dx"] = round(dx, 4)
            output_stats[category]["dy"] = round(dy, 4)
            output_stats[category]["dxy"] = round(dxy, 4)
            output_stats[category]["dxyp"] = round(dxyp, 4)
            output_stats[category]["num_gt"] = int(np.sum(num_gt_pt))
            output_stats[category]["num_pred"] = int(np.sum(num_pred_pt))
            output_stats[category]["num_tp"] = int(np.sum(num_tp_pt))

        matched_num = np.sum(
            np.array([i["matched_num"] for i in pts_stats_dict["accuracy"]])
        )
        true_num = np.sum(
            np.array([i["true_num"] for i in pts_stats_dict["accuracy"]])
        )

        output_stats["all"]["ap"] = round(ap_all / (num_gt_pt_all + 1e-6), 4)
        output_stats["all"]["weight_ap"] = round(
            weight_ap_all / (weight_num_gt_pt_all + 1e-6), 4
        )
        output_stats["all"]["Precision"] = round(
            num_tp_pt_all / (num_pred_pt_all + 1e-6), 4
        )
        output_stats["all"]["Recall"] = round(
            num_tp_pt_all / (num_gt_pt_all + 1e-6), 4
        )
        output_stats["all"]["Acc"] = round(
            int(true_num) / (int(matched_num) + 1e-6), 4
        )
        output_stats["all"]["dx"] = round(dx_all / (num_tp_pt_all + 1e-6), 4)
        output_stats["all"]["dy"] = round(dy_all / (num_tp_pt_all + 1e-6), 4)
        output_stats["all"]["dxy"] = round(dxy_all / (num_tp_pt_all + 1e-6), 4)
        output_stats["all"]["dxyp"] = round(
            dxyp_all / (num_tp_pt_all + 1e-6), 4
        )
        output_stats["all"]["num_gt"] = int(num_gt_pt_all)
        output_stats["all"]["num_pred"] = int(num_pred_pt_all)
        output_stats["all"]["num_tp"] = int(num_tp_pt_all)

        metric_str = metric_title.format(
            region_name,
            "{:.4f}".format(output_stats["all"]["ap"]),
            "{:.4f}".format(output_stats["all"]["Precision"]),
            "{:.4f}".format(output_stats["all"]["Recall"]),
            "{:.4f}".format(output_stats["all"]["Acc"]),
            "{:.4f}".format(output_stats["all"]["dx"]),
            "{:.4f}".format(output_stats["all"]["dy"]),
            "{:.4f}".format(output_stats["all"]["dxy"]),
            "{:.4f}".format(output_stats["all"]["dxyp"]),
            "{:}".format(output_stats["all"]["num_gt"]),
            "{:}".format(output_stats["all"]["num_pred"]),
            "{:}".format(output_stats["all"]["num_tp"]),
        )
        metric_dict = {region_name: output_stats["all"]}
        del metric_dict[region_name]["weight_ap"]

        if eval_result_file:
            with open(eval_result_file, "w") as f:
                json.dump(output_stats, f, indent=2)
        return metric_str, metric_dict

    def cal_metric(self):
        summary_str = (
            "\n----------- Bev-crosspoint Summary Metrics -----------\n\n"
        )
        summary_dict = {}
        metric_title = (
            "{:>26} {:>8} {:>8} {:>8} {:>8} {:>8} "
            + "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n"
        )
        summary_str += metric_title.format(
            " ",
            "AP",
            "Prec",
            "Rec",
            "Acc",
            "dx",
            "dy",
            "dxy",
            "dxyp",
            "num_gt",
            "num_pred",
            "num_tp",
        )

        dist_threshold = self.metric_cfg["dist_threshold"]
        dist_percent_thresh = self.metric_cfg["dist_percent_thresh"]
        absolute_dist_range = dist_threshold / dist_percent_thresh
        score_thresh = self.metric_cfg["score_thresh"]
        ap_score_thresh = self.metric_cfg["ap_score_thresh"]
        region_dict = self.metric_cfg["region"]
        os.makedirs(
            os.path.join(self.eval_result_dir),
            exist_ok=True,
        )

        for region_name, roi in region_dict.items():
            crosspoint_stats_dict = {}
            crosspoint_stats_dict["accuracy"] = []
            for category in self.target_categorys:
                crosspoint_stats_dict.update({category: []})
            print(f"------ eval {region_name}------")
            for pack_name in os.listdir(self.save_infer_output_dir):
                infer_path = os.path.join(
                    self.save_infer_output_dir, pack_name
                )
                data_list = [
                    os.path.splitext(i)[0]
                    for i in os.listdir(infer_path)
                    if "_gt" not in i
                ]
                for image_name in data_list:
                    gt_file = os.path.join(infer_path, image_name + "_gt.json")
                    pred_file = os.path.join(infer_path, image_name + ".json")
                    gt = json.load(open(gt_file, "r"))
                    pred = json.load(open(pred_file, "r"))
                    for category in self.target_categorys:
                        gt_pts = [np.array(i) for i in gt.get(category, [])]
                        pred_pts = [
                            np.array(i) for i in pred.get(category, [])
                        ]
                        pts_stats = self.bench_crosspoints(
                            pred_pts,
                            gt_pts,
                            roi,
                            dist_threshold=dist_threshold,
                            dist_percent_thresh=dist_percent_thresh,
                            absolute_dist_range=absolute_dist_range,
                            score_thresh=score_thresh,
                            ap_score_thresh=ap_score_thresh,
                        )
                        if pts_stats is not None:
                            crosspoint_stats_dict[category].append(pts_stats)

                    acc_stats = self.statsistic_cls_accuracy(
                        pred,
                        gt,
                        roi,
                        dist_threshold=dist_threshold,
                        dist_percent_thresh=dist_percent_thresh,
                        score_thresh=score_thresh,
                    )
                    crosspoint_stats_dict["accuracy"].append(acc_stats)

            region_metric_str, region_metric_dict = self.eval_pts(
                crosspoint_stats_dict,
                dist_threshold,
                region_name,
                eval_result_file=os.path.join(
                    self.eval_result_dir, region_name + ".json"
                ),
            )

            summary_str += region_metric_str
            summary_dict.update(region_metric_dict)

        if self.metric_save_path:
            metric_save_dir, _ = os.path.split(self.metric_save_path)
            os.makedirs(metric_save_dir, exist_ok=True)
            with open(self.metric_save_path, "w") as f:
                json.dump(summary_dict, f, indent=2)
        logger.info(summary_str)
        res = summary_str.strip().split("\n")[2:]
        columns = ["items"] + res[0].split()
        data = []
        for sub_res in res[1:]:
            sub_data = {}
            for k, v in zip(columns, sub_res.strip().split()):
                if k != "items":
                    v = float(v)
                sub_data.update({k: v})
            data.append(sub_data)
        tables = [
            Table(
                name=self.result_prefix + "_" + self.name,
                columns=columns,
                data=data,
            )
        ]

        return EvalResult(tables=tables)
