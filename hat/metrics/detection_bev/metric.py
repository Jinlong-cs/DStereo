# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from hat.core.box3d_utils import get_3dbox_corners
from hat.core.box_utils import bbox_overlaps
from hat.core.utils_3d import get_3dbox_dense_points
from hat.core.virtual_camera import CameraModelType
from hat.metrics.detection3d.utils import mioa_ignore
from hat.metrics.detection_bev.basic_struct import ImgObj, group_scence_by_clip
from hat.metrics.detection_bev.compute_entry import (
    cal_tp_error,
    gather_all_cls,
    get_clip_errors,
    get_tp_fp_info,
    match_by_key,
    pick_det_one_cls,
    statistic_error_by_depth,
    summarize_ap,
)
from hat.metrics.detection_bev.generate_result import (
    generate_bev_det_result,
    generate_temporal_bev_det_result,
)
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

logger = logging.getLogger(__name__)

__all__ = ["HorizonBEVDetMetric"]


@OBJECT_REGISTRY.register
class HorizonBEVDetMetric(EvalMetric):
    """Horizon BEV detection metric.

    Args:
        save_dir: Output dir to save file.
        eval_class: Name of class to evaluate.
        target_recalls: Target recalls to evaluate.
        target_precisions: Target precisions to evaluate.
        eval_cameras: Cameras to evaluate.
            Filter unused camera data in gt file.
        eval_bbox_type: Bbox type, in ["ALL", "truncated", "non-truncated"].
        eval_match_method: tp match method, Only support iou 3d NOW.
        eval_summarize_method: tp error cal methods.
        eval_multi_category: eval metric by multi category.
        dep_thresh: Range of x.
        max_depth: max eval depth, ignore if over.
        iou_thresh: 3D iou thresh. Defaults to 0.5.
        miou_thresh: Match prelabel 2d thresh. Defaults to 0.3.
        enable_ignore: Whether enable ignore. Defaults to False.
        horizon_truncate_offset: Truncate offset. Defaults to 0.
        eval_let_params: params for let_iou match, Optional[Dict[str, float]]
            for example:
            {
                "let_p_t": 0.2,
                "let_min_t": 4.0,
                "let_max_t": 10.0,
            }
        eval_scence_tag: scence_tag(s) to eval,
            Optional[Dict[str, Union[str, Sequence[str], Type(None)]]],
            for example:
            {
                "weather": ["sunny","cloudy"]
                "scene": "urban",
                "time": "day",
                "plate": None,
            }
        summary_table_file: name of the summary table file
            where all the eval metrics are saved, these metrics will
            be shown on the summary page on aidieval leaderboard.
    """

    def __init__(
        self,
        save_dir: str,
        eval_class: str,
        target_recalls: List[float],
        target_precisions: List[float],
        eval_cameras: List[str],
        eval_bbox_type: str,
        eval_match_method: str,
        eval_summarize_method: str,
        eval_segment_method: Tuple[str] = None,
        eval_multi_category: List[str] = None,
        dep_thresh: List[float] = None,
        max_depth: float = 300,
        iou_threshold: float = 0.5,
        dist_threshold: float = 0.5,
        relative_dist_threshold: float = 0.2,
        miou_thresh: float = 0.3,
        enable_ignore: bool = False,
        horizon_truncate_offset: int = 0,
        eval_let_params: Optional[Dict[str, float]] = None,
        eval_scence_tag: Optional[
            Dict[str, Union[str, Sequence[str], Type[None]]]
        ] = None,
        eval_vcs_range: Optional[Sequence[float]] = None,
        summary_table_file: str = "tables.json",
        **kwargs,
    ):
        assert eval_bbox_type in ["ALL", "truncated", "non-truncated"]
        assert eval_match_method.lower() in [
            "iou_3d",
            "center_dist",
            "let_iou",
            "iou_bev",
        ]
        self.save_dir = save_dir
        self.eval_class = eval_class
        self.target_recalls = target_recalls
        self.target_precisions = target_precisions
        self.eval_cameras = eval_cameras
        self.eval_bbox_type = eval_bbox_type
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.relative_dist_threshold = relative_dist_threshold
        self.max_depth = max_depth
        self.miou_thresh = miou_thresh
        self.enable_ignore = enable_ignore
        self.horizon_truncate_offset = horizon_truncate_offset
        self.eval_match_method = eval_match_method
        self.eval_summarize_method = eval_summarize_method
        self.eval_segment_method = eval_segment_method
        self.eval_multi_category = eval_multi_category
        self.dep_thresh = sorted(dep_thresh)
        self.eval_let_params = eval_let_params
        if self.eval_let_params is None:
            self.eval_let_params = {
                "let_p_t": 0.2,
                "let_min_t": 4.0,
                "let_max_t": 10.0,
            }
        self.enable_let_apl = (
            True if eval_match_method.lower() == "let_iou" else False
        )

        self.all_gts = []
        self.all_preds = []
        self.eval_scence_tag = eval_scence_tag
        self.eval_vcs_range = eval_vcs_range
        self.summary_table_file = summary_table_file

    def update(self, preds: List[dict], gts: List):
        all_gt, all_pred = self._parse_gts_and_preds(gts, preds)

        self.all_gts += all_gt
        self.all_preds += all_pred
        return all_gt, all_pred

    def _filter_by_camera(self, gt_scence):
        filter_img_objects = []
        for obj in gt_scence.image_objects.values():
            if obj.meta["camera"] not in self.eval_cameras:
                filter_img_objects.append(obj)
        [gt_scence.pop_image_object(obj) for obj in filter_img_objects]

        filter_obj3d = []
        for idx, obj in gt_scence.objects_3d.items():
            obj_filter = True
            ignore = True
            for img_obj in gt_scence.image_objects.values():
                if idx in img_obj.objects_3d:
                    obj_filter = False
                    if not img_obj.objects_3d[idx].ignore:
                        ignore = False
            if ignore:
                obj.ignore = True
            if obj_filter:
                filter_obj3d.append(obj)
        [gt_scence.pop_obj3d(obj) for obj in filter_obj3d]
        return gt_scence

    def _filter_by_scence_tag(self, gt_scence):
        gt_scence_tag = gt_scence.meta["scence_tag"]
        if gt_scence_tag is not None and self.eval_scence_tag is not None:
            for eval_tag, eval_sub_tag in self.eval_scence_tag.items():
                if eval_tag not in gt_scence_tag or eval_sub_tag is None:
                    continue
                eval_sub_tag = _as_list(eval_sub_tag)
                if gt_scence_tag[eval_tag] not in eval_sub_tag:
                    return None
        return gt_scence

    def _ignore_by_depth(self, gt_scence):
        for obj3d in gt_scence.objects_3d.values():
            depth = np.linalg.norm([obj3d.location[0], obj3d.location[1]])
            if depth >= self.max_depth:
                obj3d.ignore = True
        return gt_scence

    def _ignore_by_ignore_mask(self, gt_scence):
        for idx, obj3d in gt_scence.objects_3d.items():
            obj3d_ignore = None
            for img_obj in gt_scence.image_objects.values():
                if (
                    idx in img_obj.objects_3d
                    and not img_obj.objects_3d[idx].ignore
                ):
                    bbox_2d = img_obj.objects_3d_info[idx]["bbox"]
                    if mioa_ignore(
                        bbox_2d, img_obj.ignore_mask, self.mioa_thresh
                    ):
                        obj3d_ignore = (
                            (obj3d_ignore is True)
                            if obj3d_ignore is not None
                            else True
                        )
                        continue
                    obj3d_ignore = False

            if obj3d_ignore:
                obj3d.ignore = True

        return gt_scence

    def _ignore_by_bbox_type(self, gt_scence):
        if self.eval_bbox_type in ["truncated", "non-truncated"]:
            for idx, obj3d in gt_scence.objects_3d.items():
                is_truncated = True
                for img_obj in gt_scence.image_objects.values():
                    if idx in img_obj.objects_3d_info:
                        obj_img_info = img_obj.objects_3d_info[idx]
                        image_size = [img_obj.width, img_obj.height]

                        bbox = np.array(obj_img_info["bbox"])
                        if self.eval_bbox_type == "truncated":
                            if np.all(bbox[2:] < image_size) and np.all(
                                bbox[:2] >= [0, 0]
                            ):
                                is_truncated = False
                        elif self.eval_bbox_type == "non-truncated":
                            new_img_w = (
                                image_size[0] - self.horizon_truncate_offset
                            )
                            new_img_h = image_size[1]
                            if not (
                                np.any(bbox[2:] >= [new_img_w, new_img_h])
                                or np.any(
                                    bbox[:2]
                                    < [self.horizon_truncate_offset, 0]
                                )
                            ):
                                is_truncated = False
                if self.eval_bbox_type == "truncated" and not is_truncated:
                    obj3d.ignore = True
                elif self.eval_bbox_type == "non-truncated" and is_truncated:
                    obj3d.ignore = True
        return gt_scence

    def _project_vcs_to_img2d(
        self, bbox_3d, image_size, calib, camera_model="PinholeCamera"
    ):
        camera_tool = CameraModelType[camera_model].value()
        calib.update(
            {
                "image_width": image_size[0],
                "image_height": image_size[1],
            }
        )
        camera_tool = camera_tool.init_cam_param_by_dict(calib)
        dim = [bbox_3d.dim[2], bbox_3d.dim[1], bbox_3d.dim[0]]
        points_3d = get_3dbox_corners(bbox_3d.loc, dim, bbox_3d.yaw)
        points_img = camera_tool.project_vcs2cam(points_3d)
        is_valid = camera_tool.is_points_in_fov(points_img)
        if np.any(is_valid):
            points_dense_3d = get_3dbox_dense_points(
                bbox_3d.loc, dim, bbox_3d.yaw, pts_per_line=5
            )
            points_dense_img = camera_tool.project_vcs2cam(points_dense_3d)
            is_valid = camera_tool.is_points_in_fov(points_dense_img)
            if not np.any(is_valid):
                return None
            points_dense_2d = camera_tool.project_cam2pixel(
                points_dense_img[is_valid]
            )
            x1, y1 = (
                np.min(points_dense_2d, axis=0)[0],
                np.min(points_dense_2d, axis=0)[1],
            )
            x2, y2 = (
                np.max(points_dense_2d, axis=0)[0],
                np.max(points_dense_2d, axis=0)[1],
            )
            bbox_2d = [x1, y1, x2, y2]
            return bbox_2d
        else:
            return None

    def _ignore_fp_by_prelabel_2d(self, pred_scence, gt_scence):
        for k, obj3d in pred_scence.objects_3d.items():
            info = pred_scence.objects_3d_info[k]
            if info["is_tp"] == 0:
                ignore = None
                for img_key, img_obj in pred_scence.image_objects.items():
                    if img_obj.meta["prelabel_objects_2d"] is None:
                        continue
                    if k in img_obj.objects_3d:
                        gt_img_obj = gt_scence.image_objects[img_key]
                        pred_project_2d = img_obj.objects_3d_info[k]["bbox_2d"]
                        prelabel_2d = [
                            obj2d["bbox"]
                            for obj2d in gt_img_obj.meta["prelabel_objects_2d"]
                        ]
                        if len(prelabel_2d) == 0:
                            continue
                        overlaps = bbox_overlaps(
                            np.array([pred_project_2d]), np.array(prelabel_2d)
                        )
                        if overlaps.max() < self.miou_thresh:
                            continue
                        imax = np.argmax(overlaps)
                        max_prelabel = gt_img_obj.meta["prelabel_objects_2d"][
                            imax
                        ]
                        if max_prelabel["ass_id"] == -1:
                            ignore = (
                                ignore is True if ignore is not None else True
                            )
                        else:
                            ignore = False
                if ignore:
                    obj3d.ignore = True
                    info["is_tp"] = -1

        return pred_scence, gt_scence

    def _parse_gts_and_preds(self, gt_list, pred_list):
        # filter gt
        all_gt = []
        timestamp_gt = {}
        for gt in gt_list:
            gt.update_id()
            try:
                # filter scence by scence tag
                gt = self._filter_by_scence_tag(gt)
                if gt is None:
                    continue
                # filter obj3d by camera setting
                gt = self._filter_by_camera(gt)
                # ignore obj3d by depth
                gt = self._ignore_by_depth(gt)
                # filter obj3d by eval_bbox_type
                gt = self._ignore_by_bbox_type(gt)
                all_gt += [gt]
                timestamp_gt[gt.scence_key] = gt
            except BaseException as e:
                logger.error("load gt scence_key: %s error." % (gt.scence_key))
                logger.error(e)
                raise e

        # filter pred
        all_pred = []
        for pred in pred_list:
            pred.update_id()
            try:
                timestamp = pred.scence_key
                # filter by key
                if timestamp not in timestamp_gt:
                    continue
                pred_obj3d = pred.objects_3d
                infer_img = pred.meta["image_keys"]
                gt_scence = timestamp_gt[timestamp]
                # pre-define image object
                pred_image_objects = {}
                for camera in self.eval_cameras:
                    if camera in infer_img:
                        img_key = infer_img[camera]
                        gt_camera_meta = gt_scence.image_objects[img_key]
                        img_obj = ImgObj(
                            image_key=img_key,
                            objects_3d=None,
                            width=gt_camera_meta.width,
                            height=gt_camera_meta.height,
                            calib=gt_camera_meta.calib,
                            ignore_mask=gt_camera_meta.ignore_mask,
                            meta=gt_camera_meta.meta,
                        )
                        pred_image_objects[img_key] = img_obj

                filter_pred_obj3d = []
                for obj3d in pred_obj3d.values():
                    bbox_3d = obj3d.bbox_3d
                    # filter bbox3d by dim & depth
                    depth = np.linalg.norm(
                        [obj3d.location[0], obj3d.location[1]]
                    )
                    if not (
                        np.all(np.array(bbox_3d.dim) > 0)
                        and depth < self.max_depth
                    ):
                        filter_pred_obj3d.append(obj3d)
                        continue
                    # get each camera bbox_2d via calib projection
                    in_img2d = False
                    for key, img_obj in gt_scence.image_objects.items():
                        calib = img_obj.calib
                        image_size = [img_obj.width, img_obj.height]
                        bbox_2d = self._project_vcs_to_img2d(
                            bbox_3d,
                            image_size,
                            calib,
                            img_obj.meta["camera_model"],
                        )
                        if bbox_2d:
                            in_img2d = True
                            update_info = {"bbox_2d": bbox_2d}
                            pred_image_objects[key].add_obj3d(obj3d)
                            pred_image_objects[key].save_obj3d_in_img_info(
                                obj3d, update_info
                            )
                    # filter obj3d if not in any image
                    if not in_img2d:
                        filter_pred_obj3d.append(obj3d)
                [
                    pred.add_image_object(img_obj)
                    for img_obj in pred_image_objects.values()
                ]
                [pred.pop_obj3d(obj3d) for obj3d in filter_pred_obj3d]

                all_pred += [pred]
            except BaseException as e:
                logger.error(
                    "load det scence_key: %s error." % (pred.scence_key)
                )
                logger.error(e)
                raise e

        for pred_scence, gt_scence in match_by_key(
            all_pred, all_gt, "scence_key"
        ):
            if pred_scence is not None:
                pred_scence, gt_scence = get_tp_fp_info(
                    pred_scence,
                    gt_scence,
                    self.eval_match_method.lower(),
                    self.iou_threshold,
                    self.dist_threshold,
                    self.relative_dist_threshold,
                    self.enable_ignore,
                    self.eval_let_params,
                    self.eval_vcs_range,
                )
                # filter fp pred if lidar miss
                pred_scence, gt_scence = self._ignore_fp_by_prelabel_2d(
                    pred_scence,
                    gt_scence,
                )

        return all_gt, all_pred

    def get(self):
        gt_count = 0
        for gt_scence in self.all_gts:
            for idx, gt_obj3d in gt_scence.objects_3d.items():
                if gt_scence.objects_3d_info[idx]["is_fn"] != -1:
                    if not self.enable_ignore:
                        gt_count += 1
                    elif self.enable_ignore and not gt_obj3d.ignore:
                        gt_count += 1

        multi_class_eval = self.eval_multi_category is not None
        if multi_class_eval:
            cache_one_cls_data = {}
            for cls in self.eval_multi_category:
                (
                    all_cls_one_preds,
                    all_cls_one_gts,
                    gt_count_cls,
                ) = pick_det_one_cls(
                    self.all_preds, self.all_gts, cls, self.enable_ignore
                )
                cache_one_cls_data[cls] = {
                    "all_preds": all_cls_one_preds,
                    "all_gts": all_cls_one_gts,
                    "gt_count": gt_count_cls,
                }
            gather_preds, gather_gts = gather_all_cls(cache_one_cls_data)

        result_summary = defaultdict(lambda: {})
        # calculate segmentation metric like error of location
        if self.eval_segment_method is not None:
            for pred_scence, gt_scence in match_by_key(
                self.all_preds, self.all_gts, "scence_key"
            ):
                if pred_scence is not None:
                    # calculate tp error
                    pred_scence, gt_scence = cal_tp_error(
                        pred_scence,
                        gt_scence,
                        self.eval_segment_method,
                        self.eval_match_method.lower(),
                    )

            # statistic by depth range
            tp_errors_by_depth = statistic_error_by_depth(
                self.all_preds,
                self.all_gts,
                self.dep_thresh,
                self.eval_segment_method,
            )
            result_summary["tp_errors_by_depth"] = tp_errors_by_depth
            if multi_class_eval:
                # update tp error for vis
                for pred_scence, gt_scence in match_by_key(
                    gather_preds, gather_gts, "scence_key"
                ):
                    if pred_scence is not None:
                        # calculate tp error
                        pred_scence, gt_scence = cal_tp_error(
                            pred_scence,
                            gt_scence,
                            self.eval_segment_method,
                            self.eval_match_method.lower(),
                        )
                for cls in self.eval_multi_category:
                    all_cls_one_preds = cache_one_cls_data[cls]["all_preds"]
                    all_cls_one_gts = cache_one_cls_data[cls]["all_gts"]
                    for pred_scence, gt_scence in match_by_key(
                        all_cls_one_preds, all_cls_one_gts, "scence_key"
                    ):
                        if pred_scence is not None:
                            # calculate tp error
                            pred_scence, gt_scence = cal_tp_error(
                                pred_scence,
                                gt_scence,
                                self.eval_segment_method,
                                self.eval_match_method.lower(),
                            )
                    # statistic by depth range
                    tp_errors_by_depth = statistic_error_by_depth(
                        all_cls_one_preds,
                        all_cls_one_gts,
                        self.dep_thresh,
                        self.eval_segment_method,
                    )
                    result_summary[f"{cls}_results"][
                        "tp_errors_by_depth"
                    ] = tp_errors_by_depth

        # calculate overall metric like AP, NDS
        if self.eval_summarize_method == "AP":
            ap_3d, aph_3d, ar_3d, drot_90p, results = summarize_ap(
                self.all_preds,
                gt_count,
                self.target_recalls,
                self.target_precisions,
                self.enable_let_apl,
            )
            result_summary.update(results)
            leaderboard_results = dict(  # noqa
                AP_3D=round(ap_3d, 4),
                APH_3D=round(aph_3d, 4),
                AR_3D=round(ar_3d, 4),
                drot_90p=round(drot_90p, 3),
            )
            if multi_class_eval:
                # if assign eval_multi_category, calcuate mAP
                ap_list = []
                aph_list = []
                ar_list = []
                drot_90p_list = []
                for cls in self.eval_multi_category:
                    all_cls_one_preds = cache_one_cls_data[cls]["all_preds"]
                    all_cls_one_gts = cache_one_cls_data[cls]["all_gts"]
                    gt_count_cls = cache_one_cls_data[cls]["gt_count"]
                    ap_3d, aph_3d, ar_3d, drot_90p, results = summarize_ap(
                        all_cls_one_preds,
                        gt_count_cls,
                        self.target_recalls,
                        self.target_precisions,
                        self.enable_let_apl,
                    )
                    result_summary[f"{cls}_results"].update(results)
                    ap_list.append(ap_3d)
                    aph_list.append(aph_3d)
                    ar_list.append(ar_3d)
                    drot_90p_list.append(drot_90p)
                mAP = sum(ap_list) / len(ap_list)
                mAPH = sum(aph_list) / len(aph_list)
                mAR = sum(ar_list) / len(ar_list)
                mdrot_90p = sum(drot_90p_list) / len(drot_90p_list)
                leaderboard_results.update(
                    dict(  # noqa
                        mAP_3D=round(mAP, 4),
                        mAPH_3D=round(mAPH, 4),
                        mAR_3D=round(mAR, 4),
                        mdrot_90p=round(mdrot_90p, 3),
                    )
                )
        elif self.eval_summarize_method == "NDS":
            pass

        # convert gt/pred_scence struct data to dict, in order to dump json
        result_summary["gts_list"] = [gt.to_dict() for gt in self.all_gts]
        result_summary["preds_list"] = [
            pred.to_dict() for pred in self.all_preds
        ]
        if multi_class_eval:
            result_summary["gts_list"] = [gt.to_dict() for gt in gather_gts]
            result_summary["preds_list"] = [
                pred.to_dict() for pred in gather_preds
            ]

        # dump to file
        generate_bev_det_result(
            results=result_summary,
            output_dir=self.save_dir,
            gt_result_json="gts_result.json",
            pred_result_json="preds_result.json",
            eval_result_json="eval_result.json",
            result_png="result.png",
            table_json=self.summary_table_file,
            eval_multi_category=self.eval_multi_category,
            eval_match_method=self.eval_match_method.lower(),
        )

        logger.info("eval done")
        logger.info(leaderboard_results)
        return leaderboard_results

    def temporal_get(self):
        result_summary = defaultdict(lambda: {})
        all_gt_clips, gt_clip_group = group_scence_by_clip(
            self.all_gts, info_type="gt"
        )
        all_pred_clips, _ = group_scence_by_clip(
            self.all_preds, info_type="pred", refer_clip_group=gt_clip_group
        )
        total_clips_err, results = get_clip_errors(
            all_gt_clips, all_pred_clips
        )

        result_summary["gt_tracklets"] = results["gt_tracklets"]
        result_summary["pred_tracklets"] = results["pred_tracklets"]
        result_summary["eval_results"] = total_clips_err

        generate_temporal_bev_det_result(
            results=result_summary,
            output_dir=self.save_dir,
            gt_result_json="gts_temporal_result.json",
            pred_result_json="preds_temporal_result.json",
            table_json=self.summary_table_file,
            eval_match_method=self.eval_match_method.lower(),
        )

        leaderboard_temporal_err = {
            "temporal_dx": round(total_clips_err["filtered_locx_err"], 4),
            "temporal_dy": round(total_clips_err["filtered_locy_err"], 4),
        }
        logger.info("Temporal eval done")

        return leaderboard_temporal_err
