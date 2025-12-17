# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from collections import defaultdict
from functools import reduce
from multiprocessing import Pool
from typing import Any, Dict, List, Optional

import numpy as np
import tqdm

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.core.box3d_utils import (
    compute_2d_box,
    compute_box_3d,
    compute_corners3d_lidar,
    compute_location_lidar,
    compute_yaw_lidar,
)
from hat.core.box_utils import bbox_overlaps
from hat.core.virtual_camera import CameraModelType
from hat.metrics.detection2d.utils import calap, calar
from hat.metrics.detection3d.compute_entry import cal_tp_error_auto
from hat.metrics.detection3d.generate_results import generate_auto_result
from hat.metrics.detection3d.utils import mioa_ignore
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import check_packages_available
from .basic_struct import ImgObj

logger = logging.getLogger(__name__)
__all__ = ["HorizonAutoMetric"]

AUTO_DEFAULT_METRICS = [
    "dx",
    "sdx",
    "dxp",
    "sdxp",
    "dy",
    "sdy",
    "dyp",
    "sdyp",
    "dxyp",
    "drot",
    "sdrot",
    "rot_cls_error_p",
    "dxy_10p_error",
    "dl",
    "sdl",
    "dlp",
    "sdlp",
    "dw",
    "sdw",
    "dwp",
    "sdwp",
    "dh",
    "sdh",
    "dhp",
    "sdhp",
]


@OBJECT_REGISTRY.register
class HorizonAutoMetric(EvalMetric):
    """Horizon Auto metric.

    Args:
        save_dir: Output dir to save file.
        eval_class: Name of class to evaluate.
        target_recalls: Target recalls to evaluate.
        metrics: Evaluation metrics.
        eval_cameras:Cameras to evaluate. Defaults to None.
        image_tag: images with tag to evaluate under given key-value.
        horizontal_roi: Horizontal roi size.Defaults to None.
        dep_thresh: Range of x. Defaults to [20, 50, 80, 120].
        y_thresh: Range of y. Defaults to None.
        iou_thresh: 2D iou thresh. Defaults to 0.2.
        score_thresh: Score thresh. Defaults to 0.0.
        gt_dist_thresh: Depth in front of each camera. Defaults to 300.
        clip_bbox_with_img_size: Image size, (width, height) format.
        eval_bbox_type: Bbox type, in ["all", "truncated", "non-truncated"].
        apply_dist_coeff: Whether to use dist_coeff. Defaults to False.
        enable_ignore: Whether enable ignore. Defaults to False.
        fisheye: whether to use fisheye. Defaults to False.
        lidar_error: Whether to use lidar error. Defaults to False.
        proj_z_thresh: Project thresh of z. Defaults to -1.5.
        mioa_thresh: Ignore mask thresh. Defaults to 0.6.
        num_worker: Num workers for calculating metric.
    """

    def __init__(
        self,
        save_dir: str,
        eval_class: str,
        target_recalls: List[float],
        metrics: List[str] = None,
        eval_cameras: List[str] = None,
        image_tag: Optional[Dict[str, Any]] = None,
        horizontal_roi: Optional[List[float]] = None,
        dep_thresh: List[float] = [20, 50, 80, 120],  # noqa B006
        y_thresh: List[float] = None,
        iou_thresh: float = 0.2,
        score_thresh: float = 0.0,
        gt_dist_thresh: float = 300,
        clip_bbox_with_img_size: Optional[List[int]] = None,
        eval_bbox_type: str = "all",
        apply_dist_coeff: bool = False,
        enable_ignore: bool = False,
        fisheye: bool = False,
        lidar_error: bool = False,
        proj_z_thresh: float = -1.5,
        mioa_thresh: float = 0.6,
        num_worker: int = 1,
        **kwargs,
    ):

        self.save_dir = save_dir
        assert eval_bbox_type in ["all", "truncated", "non-truncated"]

        self.eval_class = eval_class
        self.num_worker = num_worker

        if eval_cameras:
            assert isinstance(eval_cameras, (list, tuple))
            self.eval_cameras = eval_cameras
        else:
            if fisheye:
                self.eval_cameras = [
                    "__front_fe__",
                    "__right_fe__",
                    "__rear_fe__",
                    "__left_fe__",
                ]
            else:
                self.eval_cameras = [
                    "__front_left__",
                    "__front__",
                    "__front_right__",
                    "__right__",
                    "__rear_left__",
                    "__rear__",
                    "__rear_right__",
                ]

        self.image_tag = image_tag if image_tag is not None else {}
        assert isinstance(self.image_tag, dict)
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.gt_dist_thresh = gt_dist_thresh

        self.metrics = metrics if metrics is not None else AUTO_DEFAULT_METRICS
        self.dep_thresh = dep_thresh
        self.y_thresh = y_thresh
        self.clip_bbox_with_img_size = clip_bbox_with_img_size
        self.eval_bbox_type = eval_bbox_type
        self.apply_dist_coeff = apply_dist_coeff
        self.enable_ignore = enable_ignore
        self.horizontal_roi = horizontal_roi
        self.fisheye = fisheye

        self.flatten_camera_names = reduce(
            lambda x, y: x + "_" + y, self.eval_cameras
        )
        self.target_recalls = target_recalls
        # lidar_error means:
        # weather projection to lidar coordinate system and cal dx, dy, dxy...
        self.lidar_error = lidar_error
        self.proj_z_thresh = proj_z_thresh
        self.mioa_thresh = mioa_thresh

        self.all_dets = defaultdict(list)
        self.all_gts = defaultdict(list)
        self.img_info_list = []

    def update(self, gt_list: List[ImgObj], pred_list: List[ImgObj]):
        all_gts, img_info_list = self._parse_gts(gt_list)
        all_dets = self._parse_preds(pred_list)
        return all_gts, all_dets, img_info_list

    def _parse_preds(self, pred_list: List[ImgObj]):

        all_dets = defaultdict(list)

        for frame in pred_list:
            image_key = frame.image_key

            if not self._check_match(image_key):
                continue

            for pred_obj in frame.instances:
                if pred_obj.score < self.score_thresh:
                    continue
                pred_loc = pred_obj.location
                if (
                    np.linalg.norm([pred_loc[0], pred_loc[2]])
                    > self.gt_dist_thresh
                ):
                    continue
                det = {
                    "image_key": image_key,
                    "bbox_2d": pred_obj.bbox,
                    "score": pred_obj.score,
                    "depth": pred_obj.depth,
                    "dimensions": pred_obj.get_dim_by_format("hwl"),
                    "location": pred_obj.location,
                    "rotation_y": pred_obj.rotation_y,
                }
                all_dets[image_key].append(det)
        # To save mem usage when num_worker > 1.
        if self.num_worker <= 1:
            self.all_dets.update(all_dets)
        return all_dets

    def _parse_gts(self, gt_anno_list: List[ImgObj]):

        all_gts = defaultdict(list)
        img_info_list = []

        for frame in gt_anno_list:
            image_key = frame.image_key
            if not self._check_match(image_key):
                continue
            gt_image_tag = frame.attrs.get("tags", {})
            if not self._check_image_tag(gt_image_tag):
                continue
            gt_instances = frame.instances
            if len(gt_instances) == 0:
                continue

            for gt_obj in gt_instances:
                gt_loc = gt_obj.location
                if (
                    np.linalg.norm([gt_loc[0], gt_loc[2]])
                    >= self.gt_dist_thresh
                ):
                    gt_obj.ignore = True
                if self.horizontal_roi:
                    x = gt_obj.location[0]
                    if (
                        x > self.horizontal_roi[1]
                        or x < self.horizontal_roi[0]
                    ):  # noqa
                        continue
                if self.eval_bbox_type in ["truncated", "non-truncated"]:
                    img_size = np.array(self.clip_bbox_with_img_size)
                    assert img_size is not None
                    bbox = np.array(gt_obj.bbox)
                    bbox[2:] += bbox[:2]
                    if self.eval_bbox_type == "truncated":
                        if np.all(bbox[2:] < img_size) and np.all(
                            bbox[:2] >= [0, 0]
                        ):  # noqa
                            continue
                    elif self.eval_bbox_type == "non-truncated":
                        if np.any(bbox[2:] >= img_size) or np.any(
                            bbox[:2] < [0, 0]
                        ):  # noqa
                            continue

                gt_dict = {
                    "image_key": image_key,
                    "bbox_2d": gt_obj.bbox,
                    "ignore": gt_obj.ignore,
                    "depth": gt_obj.depth,
                    "dimensions": gt_obj.get_dim_by_format("hwl"),
                    "location": gt_obj.location,
                    "rotation_y": gt_obj.rotation_y,
                }
                all_gts[image_key].append(gt_dict)

            img_info = dict(  # noqa C408
                image_key=image_key,
                distCoeffs=frame.dist_coeffs,
                calib=frame.calib,
                img_width=frame.width,
                img_height=frame.height,
                lidar_to_camera=frame.lidar_to_camera,
                ignore_mask=frame.ignore_mask,
                camera_model=frame.camera_model,
            )
            img_info_list.append(img_info)

        # To save mem usage when num_worker > 1.
        if self.num_worker <= 1:
            self.all_gts.update(all_gts)
            self.img_info_list += img_info_list

        return all_gts, img_info_list

    def single_worker(self, img_info_list):
        metrics = {
            k: [[] for _ in range(len(self.dep_thresh) + 1)]
            for k in self.metrics
        }
        if self.y_thresh is not None:
            metrics_y = {
                k: [[] for _ in range(len(self.y_thresh) + 1)]
                for k in self.metrics
            }  # noqa
            y_gt_matched = np.zeros(len(self.y_thresh) + 1)
        else:
            metrics_y = None
            y_gt_matched = None

        matched_pairs = []
        img_fps = []

        num_gt, img_count, gt_missed, redundant_det = 0, 0, 0, 0
        all_scores, det_tp_mask = [], []
        gt_matched = np.zeros(len(self.dep_thresh) + 1)

        for img_info in img_info_list:

            image_key = img_info["image_key"]

            (
                gt_bboxes,
                gt_locs,
                gt_ignore_mask,
                _,
            ) = self._filter_gts_or_dets_per_image(img_info, is_gt=True)
            (
                det_bboxes,
                det_locs,
                _,
                det_scores,
            ) = self._filter_gts_or_dets_per_image(img_info, is_gt=False)

            if len(self.all_dets[image_key]) == 0:
                if self.enable_ignore:
                    num_gt += len(gt_bboxes) - sum(gt_ignore_mask)
                    gt_missed += len(gt_bboxes) - sum(gt_ignore_mask)
                else:
                    num_gt += len(gt_bboxes)
                    gt_missed += len(gt_bboxes)
                continue

            if len(gt_bboxes) == 0:
                redundant_det += len(self.all_dets[image_key])
                continue

            img_count += 1
            det_bboxes = np.array(det_bboxes)
            det_scores = np.array(det_scores)
            gt_bboxes = np.array(gt_bboxes)
            assert det_scores.shape[0] == det_bboxes.shape[0]

            if self.clip_bbox_with_img_size is not None:
                assert isinstance(self.clip_bbox_with_img_size, (list, tuple))
                gt_bboxes = np.clip(
                    gt_bboxes, [0, 0, 0, 0], self.clip_bbox_with_img_size * 2
                )
                det_bboxes = np.clip(
                    det_bboxes, [0, 0, 0, 0], self.clip_bbox_with_img_size * 2
                )

            if self.enable_ignore:
                gt_ignore_mask = np.array(gt_ignore_mask)
                matched_dict, redundant, det_ignored_mask = IoU_based_matching(
                    det_bboxes,
                    det_locs,
                    gt_bboxes,
                    gt_locs,
                    self.iou_thresh,
                    det_scores,
                    gt_ignore_mask,
                )  # noqa
            else:
                matched_dict, redundant = IoU_based_matching(
                    det_bboxes,
                    det_locs,
                    gt_bboxes,
                    gt_locs,
                    self.iou_thresh,
                    det_scores,
                )  # noqa

            redundant_det += len(redundant)

            if self.enable_ignore:
                num_gt += gt_ignore_mask.shape[0] - np.sum(gt_ignore_mask)
                all_scores += det_scores[np.invert(det_ignored_mask)].tolist()
                tp = np.ones(len(det_scores), dtype=bool)
                tp[redundant] = 0
                tp = tp[np.invert(det_ignored_mask)]
            else:
                num_gt += len(gt_bboxes)
                tp = np.ones(len(det_bboxes))
                tp[redundant] = 0
                all_scores += det_scores.tolist()

            det_tp_mask += tp.tolist()
            det_assigns = matched_dict["det_assign"]
            inds = np.array(list(range(len(det_assigns))))

            # remove gt with no assignment
            mask = det_assigns != -1
            det_assigns, inds = det_assigns[mask], inds[mask]
            if self.enable_ignore:
                gt_missed += np.sum(np.invert(mask)) - np.sum(gt_ignore_mask)
            else:
                gt_missed += np.sum(np.invert(mask))
            if np.sum(mask) == 0:
                continue
            # calculate error use tp
            res = cal_tp_error_auto(
                image_key,
                self.all_dets,
                self.all_gts,
                matched_dict,
                inds,
                det_assigns,
                mask,
                self.lidar_error,
                self.dep_thresh,
                self.y_thresh,
                gt_matched,
                y_gt_matched,
                metrics,
                metrics_y,
            )

            img_matched_pairs = {"meta": img_info, "matched": []}
            marked_gt_and_pred = {
                "meta": img_info,
                "gts": [],
                "dets": [],
                "fp": [],
            }
            for i, (gt_ind, det_ind) in enumerate(zip(inds, det_assigns)):
                gt = self.all_gts[image_key][gt_ind].copy()
                det = self.all_dets[image_key][det_ind].copy()
                m = {k: v[i] for k, v in res.items()}
                if "dxy_10p_error" in m.keys():
                    m["dxy_10p_error"] = 1 if m["dxy_10p_error"] else 0
                if "yaw_cls_error_p" in m.keys():
                    m["yaw_cls_error_p"] = 1 if m["yaw_cls_error_p"] else 0
                img_matched_pairs["matched"] += [
                    {"gt": gt, "det": det, "metrics": m}
                ]
            matched_pairs += [img_matched_pairs]
            # this img_fp is used to draw FP.
            marked_gt_and_pred["gts"] = self.all_gts[image_key].copy()
            marked_gt_and_pred["dets"] = self.all_dets[image_key].copy()
            marked_gt_and_pred["fp"] = np.zeros(len(det_scores))
            marked_gt_and_pred["fp"][redundant] += 1
            for i, (gt_ind, det_ind) in enumerate(  # noqa B007
                zip(inds, det_assigns)
            ):  # noqa B007
                m = {k: v[i] for k, v in res.items()}
                if "dxy_10p_error" in m.keys():
                    m["dxy_10p_error"] = 1 if m["dxy_10p_error"] else 0
                if "yaw_cls_error_p" in m.keys():
                    m["yaw_cls_error_p"] = 1 if m["yaw_cls_error_p"] else 0
                marked_gt_and_pred["dets"][det_ind].update(
                    {"metrics": m, "eval_type": "TP"}
                )
            img_fps += [marked_gt_and_pred]

        return (
            metrics,
            metrics_y,
            y_gt_matched,
            matched_pairs,
            img_fps,
            num_gt,
            img_count,
            gt_missed,
            redundant_det,
            all_scores,
            det_tp_mask,
            gt_matched,
        )

    def get(self):

        logger.info("load done! all_img: {}".format(len(self.all_gts.keys())))
        assert (
            len(self.img_info_list) > 0
        ), "Make sure image_key and eval_name is matched!"
        metrics = {
            k: [[] for _ in range(len(self.dep_thresh) + 1)]
            for k in self.metrics
        }
        if self.y_thresh is not None:
            metrics_y = {
                k: [[] for _ in range(len(self.y_thresh) + 1)]
                for k in self.metrics
            }  # noqa
            y_gt_matched = np.zeros(len(self.y_thresh) + 1)
        else:
            metrics_y = None
            y_gt_matched = None

        matched_pairs = []
        img_fps = []

        num_gt, img_count, gt_missed, redundant_det = 0, 0, 0, 0
        all_scores, det_tp_mask = [], []
        gt_matched = np.zeros(len(self.dep_thresh) + 1)

        if self.num_worker <= 1:
            (
                metrics,
                metrics_y,
                y_gt_matched,
                matched_pairs,
                img_fps,
                num_gt,
                img_count,
                gt_missed,
                redundant_det,
                all_scores,
                det_tp_mask,
                gt_matched,
            ) = self.single_worker(self.img_info_list)

        else:

            pool = Pool(self.num_worker)
            ret_list = []
            idx_interval = len(self.img_info_list) // self.num_worker
            for idx_ in range(0, len(self.img_info_list), idx_interval):
                ret = pool.apply_async(
                    self.single_worker,
                    (self.img_info_list[idx_ : idx_ + idx_interval],),
                )
                ret_list.append(ret)
            pool.close()
            for ret in tqdm.tqdm(ret_list, desc="Process: "):
                metrics_single = ret.get()[0]
                for key, val in metrics.items():
                    for dep_idx in range(len(self.dep_thresh) + 1):
                        val[dep_idx] += metrics_single[key][dep_idx]

                metrics_y_single = ret.get()[1]
                y_gt_matched_single = ret.get()[2]
                if self.y_thresh:
                    for key, val in metrics_y.items():
                        for dep_idx in range(len(self.y_thresh) + 1):
                            val[dep_idx] += metrics_y_single[key][dep_idx]

                    y_gt_matched += y_gt_matched_single
                matched_pairs += ret.get()[3]
                img_fps += ret.get()[4]
                num_gt += ret.get()[5]
                img_count += ret.get()[6]
                gt_missed += ret.get()[7]
                redundant_det += ret.get()[8]
                all_scores += ret.get()[9]
                det_tp_mask += ret.get()[10]
                gt_matched += ret.get()[11]

        metrics["img_fps"] = img_fps
        if self.y_thresh:
            metrics_y["counts"] = {
                "y_gt_matched": y_gt_matched.tolist(),
            }
            metrics["metrics_y"] = metrics_y
        metrics["counts"] = {
            "gt_matched": gt_matched.tolist(),
            "gt_missed": int(gt_missed),
            "redundant_det": int(redundant_det),
            "img_count": int(img_count),
        }
        metrics["meta"] = {
            "dep_thresh": self.dep_thresh,
            "horizontal_roi": self.horizontal_roi,
            "eval_camera": self.eval_cameras,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "gt_dist_thresh": self.gt_dist_thresh,
            "eval_class": self.eval_class,
        }

        det_tp_mask = np.array(det_tp_mask)
        all_scores = np.array(all_scores)
        arginds = np.argsort(-all_scores)
        all_scores = all_scores[arginds]
        det_tp_mask = det_tp_mask[arginds]
        det_fp_mask = 1 - det_tp_mask
        det_tp = np.cumsum(det_tp_mask)
        det_fp = np.cumsum(det_fp_mask)
        det_rate = (det_tp - det_fp) / (num_gt + 1e-6)
        max_ind = np.argmax(det_rate)
        max_det_rate = det_rate[max_ind]
        max_det_rate_score = all_scores[max_ind]

        metrics["max_det_rate"] = {
            "max_det_rate": max_det_rate,
            "score": max_det_rate_score,
        }
        metrics["matched_pairs"] = matched_pairs
        # cal ar ap
        recall = det_tp / (num_gt + 1e-6)
        precision = det_tp / (det_tp + det_fp + 1e-6)
        fppi = det_fp / float(len(self.img_info_list))
        fppi += 1e-6
        ar = calar(fppi, recall)
        ap, recall, precision = calap(recall, precision)
        recall = np.array(recall)
        precision = np.array(precision)
        results = {}
        results["aps"] = {
            "ap": ap,
            "rec": recall[-1] if len(recall) > 0 else np.inf,
            "detection_rate": det_rate[-1]
            if len(det_rate) > 0
            else np.inf,  # noqa
            "precision": precision[-1]
            if len(precision) > 0
            else np.inf,  # noqa
            "ar": ar,
            "num_image": len(self.img_info_list),
            "num_gt": num_gt,
            "num_tp": det_tp[-1] if len(det_tp) else 0,
            "num_fp": det_fp[-1] if len(det_fp) else 0,
            "num_fn": metrics["counts"]["gt_missed"],
        }
        results["recall"] = recall.tolist()
        results["precision"] = precision.tolist()
        results["fppi"] = fppi.tolist()
        results["conf"] = all_scores.tolist()
        results["detection_rate"] = det_rate.tolist()
        results["fp"] = det_fp.tolist()
        # results['all_scores'] = all_scores.tolist()
        thres_rec_pre_tp_fp_gt = []
        # cal drot dxyp.. by filter score. naive
        new_dxp = []
        new_dyp = []
        new_dxyp = []
        new_drot = []
        new_score = []
        for pairs in matched_pairs:
            pairs_matched = pairs["matched"]
            for pair in pairs_matched:
                new_dxp.append(pair["metrics"]["dxp"])
                new_dyp.append(pair["metrics"]["dyp"])
                new_dxyp.append(pair["metrics"]["dxyp"])
                new_drot.append(pair["metrics"]["drot"])
                new_score.append(pair["det"]["score"])
        new_dxp = np.array(new_dxp)
        new_dyp = np.array(new_dyp)
        new_dxyp = np.array(new_dxyp)
        new_drot = np.array(new_drot)
        new_score = np.array(new_score)

        results["scores_lut"] = {}
        thres_rec_pre_tp_fp_gt = []
        for thres in np.unique(all_scores):
            select = all_scores >= thres
            sub_tp = np.max(det_tp[select])
            sub_fp = np.max(det_fp[select])
            rec = sub_tp / (num_gt + 1e-6)
            pre = sub_tp / (sub_tp + sub_fp + 1e-6)
            new_select = new_score >= thres
            mean_dxp = np.mean(new_dxp[new_select])
            mean_dyp = np.mean(new_dyp[new_select])
            mean_dxyp = np.mean(new_dxyp[new_select])
            mean_drot = np.mean(new_drot[new_select])
            thres_rec_pre_tp_fp_gt.append(
                [
                    thres,
                    rec,
                    pre,
                    sub_tp,
                    sub_fp,
                    num_gt,
                    mean_dxp,
                    mean_dyp,
                    mean_dxyp,
                    mean_drot,
                ]
            )  # noqa
            results["scores_lut"][float(thres)] = {
                "threshold": thres,
                "recall": rec,
                "precision": pre,
                "num_tp": sub_tp,
                "num_fp": sub_fp,
                "num_gt": num_gt,
            }
        thres_rec_pre_tp_fp_gt = np.array(thres_rec_pre_tp_fp_gt)
        results["target_recalls"] = {}
        if len(thres_rec_pre_tp_fp_gt) == 0:
            pass
        else:
            thres_rec_pre_tp_fp_gt = np.array(thres_rec_pre_tp_fp_gt)
            for target_recall in self.target_recalls:
                valid_idx = np.where(
                    thres_rec_pre_tp_fp_gt[:, 1] > target_recall
                )[0]
                if valid_idx.shape[0] > 0:
                    idx = valid_idx[
                        thres_rec_pre_tp_fp_gt[valid_idx, 1].argmin()
                    ]
                else:
                    idx = np.argmin(
                        target_recall - thres_rec_pre_tp_fp_gt[:, 1]
                    )
                results["target_recalls"][target_recall] = {
                    "threshold": thres_rec_pre_tp_fp_gt[idx, 0],
                    "recall": thres_rec_pre_tp_fp_gt[idx, 1],
                    "precision": thres_rec_pre_tp_fp_gt[idx, 2],
                    "num_tp": thres_rec_pre_tp_fp_gt[idx, 3],
                    "num_fp": thres_rec_pre_tp_fp_gt[idx, 4],
                    "num_gt": thres_rec_pre_tp_fp_gt[idx, 5],
                    "dxp": thres_rec_pre_tp_fp_gt[idx, 6],
                    "dyp": thres_rec_pre_tp_fp_gt[idx, 7],
                    "dxyp": thres_rec_pre_tp_fp_gt[idx, 8],
                    "drot": thres_rec_pre_tp_fp_gt[idx, 9],
                }

        # dump to file
        generate_auto_result(
            results=results,
            metrics_res=metrics,
            metrics_type=self.metrics,
            dep_thresh=self.dep_thresh,
            lidar_error=self.lidar_error,
            y_thresh=self.y_thresh,
            output_dir=self.save_dir,
            result_json="result_auto.json",
            result_png="result_auto.png",
            table_json="tables.json",
            all_json="all.json",
        )

        ap_res = dict(  # noqa C408
            AP=round(results["aps"]["ap"], 4),
            AR=round(results["aps"]["ar"], 4),
        )
        logger.info(
            "AutoEval eval done! AR: {}, AR: {}".format(
                ap_res["AP"], ap_res["AR"]
            )
        )
        return ap_res

    def _filter_gts_or_dets_per_image(self, img_info, is_gt=True):

        image_key = img_info["image_key"]
        calib = img_info["calib"]
        dist_coeff = (
            img_info.get("distCoeffs", None) if self.apply_dist_coeff else None
        )
        lidar_to_camera = img_info["lidar_to_camera"]

        camera_model = img_info.get("camera_model", None)
        if camera_model is not None:
            camera_cls = CameraModelType[camera_model].value
            camera = camera_cls(
                camera_matrix=np.array(calib),
                distcoeffs=np.array(dist_coeff),
                image_size=[img_info["img_width"], img_info["img_height"]],
                is_virtual=False,
            )

        else:
            camera = None

        used_bboxes, used_locs, gt_ignore_masks, det_scores = [], [], [], []

        # fetch 2d box
        all_img_objs = self.all_gts if is_gt else self.all_dets

        valid_objs = [True] * len(all_img_objs[image_key])
        for idx, obj in enumerate(all_img_objs[image_key]):
            dimensions = obj["dimensions"]
            location = obj["location"]
            yaw = obj["rotation_y"]

            # https://gitlab.hobot.cc/auto/perception/ad/horizonadas_evalkit/-/blob/main/adas_eval/detection_3d/functional/auto_eval.py#L321-339  # noqa E501
            # TODO(mengyang.duan): refactor.
            if camera is not None:
                corners3d = compute_box_3d(
                    dim=dimensions,
                    location=location,
                    rotation_y=yaw,
                )
                projected_corner2d = camera.project_cam2pixel(corners3d)
                bbox2d = np.concatenate(
                    [
                        np.min(projected_corner2d, axis=0),
                        np.max(projected_corner2d, axis=0),
                    ]
                )
            else:
                bbox2d, corners3d = compute_2d_box(
                    dimensions,
                    location,
                    yaw,
                    calib,
                    dist_coeff=dist_coeff,
                    fisheye=self.fisheye,
                    img_wh=self.clip_bbox_with_img_size,
                    z_thresh=self.proj_z_thresh,
                )

            if is_gt:
                if bbox2d is None:
                    valid_objs[idx] = False
                    continue
            else:
                ignore_mask_nd = None
                if img_info["ignore_mask"] is not None:
                    if coco_mask is None:
                        check_packages_available("pycocotools")
                    ignore_mask_nd = coco_mask.decode(img_info["ignore_mask"])

                if bbox2d is None or (
                    ignore_mask_nd is not None
                    and mioa_ignore(
                        bbox2d,
                        ignore_mask_nd,
                        thresh=self.mioa_thresh,
                        decode_mask=False,
                    )
                ):  # noqa
                    valid_objs[idx] = False
                    continue

            obj.update({"bbox": bbox2d.tolist()})

            used_bboxes.append(obj["bbox"])
            used_locs.append(obj["location"])

            if not is_gt:
                assert obj["score"] >= 0
                det_scores.append(obj["score"])
            if self.enable_ignore and is_gt:
                gt_ignore_masks += [obj["ignore"]]

            if self.lidar_error:
                corners3d_lidar = compute_corners3d_lidar(
                    corners3d, lidar_to_camera
                )
                bev_bbox = corners3d_lidar[:4, :].reshape(-1, 3)
                bev_bbox = bev_bbox[:, :2]
                yaw_lidar = compute_yaw_lidar(corners3d_lidar)
                location_lidar = compute_location_lidar(
                    obj["location"], lidar_to_camera
                )
                obj.update(
                    {
                        "yaw_lidar": yaw_lidar,
                        "location_lidar": location_lidar[0].tolist(),
                        "bev_bbox": bev_bbox.tolist(),
                    }
                )

        used_objs = [
            obj
            for idx, obj in enumerate(all_img_objs[image_key])
            if valid_objs[idx]
        ]

        if is_gt:
            self.all_gts[image_key] = used_objs
        else:
            self.all_dets[image_key] = used_objs

        return used_bboxes, used_locs, gt_ignore_masks, det_scores

    def _check_match(self, img_name):
        matched = [int(camera in img_name) for camera in self.eval_cameras]
        return sum(matched) > 0

    def _check_image_tag(self, gt_image_tag):
        keep = True
        for key, value in self.image_tag.items():
            value = value if isinstance(value, list) else [value]
            if gt_image_tag.get(key, None) not in value:
                keep = False
                break
        return keep


def IoU_based_matching(
    det_boxes: np.ndarray,
    det_loc: List[List[float]],
    gt_boxes: np.ndarray,
    gt_loc: List[List[float]],
    iou_thresh: float,
    det_scores: float,
    gt_ignore_mask=None,
):

    det_loc = np.array(det_loc)
    gt_loc = np.array(gt_loc)
    assert len(det_boxes) == len(det_scores)

    overlaps = np.zeros(shape=gt_boxes.shape[0])
    det_assign = np.zeros(shape=gt_boxes.shape[0], dtype=np.int64) - 1
    matched_det = np.zeros(shape=det_boxes.shape[0], dtype=np.int64)

    valid_gt = np.ones(gt_boxes.shape[0], dtype=bool)
    pairwise_iou = bbox_overlaps(det_boxes, gt_boxes)

    det_sorted_idx = np.argsort(det_scores)[::-1]
    det_ignored_mask = np.zeros(shape=det_boxes.shape[0], dtype=bool)

    for dt_idx in det_sorted_idx:
        gt_mask = valid_gt
        det_iou = pairwise_iou[dt_idx] * gt_mask.astype(float)
        max_ind = np.argmax(det_iou)
        max_iou = det_iou[max_ind]
        if max_iou > iou_thresh:
            if gt_ignore_mask is not None and gt_ignore_mask[max_ind]:
                det_ignored_mask[dt_idx] = True
                matched_det[dt_idx] = -1
            else:
                det_assign[max_ind] = dt_idx
                matched_det[dt_idx] = 1
                overlaps[max_ind] = max_iou
                valid_gt[max_ind] = 0

    redundant = np.where(matched_det == 0)[0]

    if gt_ignore_mask is not None:
        return (
            {"overlaps": overlaps, "det_assign": det_assign},
            redundant,
            det_ignored_mask,
        )
    else:
        return {"overlaps": overlaps, "det_assign": det_assign}, redundant
