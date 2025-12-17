# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import logging
import os
from collections import defaultdict
from typing import List, Mapping, Optional, Sequence

import numpy as np

from hat.metrics.detection_bev_rotbox.compute_entry import gather_all_cls
from hat.metrics.detection_bev_rotbox.generate_result import gen_results
from hat.metrics.metric import EvalMetric
from hat.metrics.metric_3dv_utils import (
    ct_matching,
    ct_rot_matching,
    rotate_iou,
    rotate_iou_matching,
)
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["HorizonRotBoxDetMetric"]


@OBJECT_REGISTRY.register
class HorizonRotBoxDetMetric(EvalMetric):
    """Horizon BEV detection metric.

    Args:
        save_dir: Output dir to save file.
        id2label: class id mapping to class label.
        {
            0: ParkingLock_Open,
            1: ParkingLock_Close,
            2: CementColumn,
        }
        eval_category_ids: The categories to be evaluation.
        eval_class: eval class name, for example: bev_static_obstacle.
        depth_intervals: Depth range to validation, using (20, 50, 70)
            if not special.
        eval_vcs_range: The vcs range of you care, default is Dict().
            Order is [-x, -y, x, y].
        eval_multi_category: list of eval class label. For eaxmple:
            [ParkingLock_Open, ParkingLock_Close, CementColumn]
        ct_match_mode: when match mode is "ct", ct_match_mode have two:
            "percentage": according percent of gt distance
            "distance": according to center location
        iou_threshold: Threshold for IoU.
        gt_max_depth: Max depth for gts.
        metrics: Tuple of eval metrics, using
            ("dx", "dy", "dxy", "dw", "dh", "drot") if not special.
        match_mode: match mode, default is iou. stopline and speedbump
            use "ct_rot": acrroding to center location and rotation angle.
        dist_intervals: distance range to validation.
        yaw_amplitude: The amplitude of yaw, the yaw range of
            (stopline,crosswalk,...) is (-90,90), so the amplitude
            of yaw is 180, arrow is 360.
        ap_score_threshold: Threshold for score while calculate AP.
        ct_matching_thresholds: the mapping of gt distance(meter)
            and matching threshold(percent of gt distance). Only use
            when match_mode == "ct".
        compute_foreground_prec: Whether to get
            classification precision.
    """

    def __init__(
        self,
        save_dir: str,
        id2label: dict,
        eval_category_ids: Optional[Sequence[int]],
        eval_class: str,
        depth_intervals: dict,
        eval_vcs_range: dict,
        eval_multi_category: dict,
        ct_match_mode: Optional[str] = "percentage",
        iou_threshold: float = 0.2,
        gt_max_depth: int = 100,
        metrics: Optional[Sequence[str]] = (
            "dx",
            "dy",
            "dxy",
            "dw",
            "dh",
            "drot",
        ),
        match_mode: Optional[dict] = None,
        dist_intervals: Optional[Sequence[float]] = None,
        yaw_amplitude: int = 180,
        ap_score_threshold: float = 0.0,
        ct_matching_thresholds: Optional[Sequence[dict]] = None,
        compute_foreground_prec: bool = False,
        **kwargs,
    ):
        self.save_dir = save_dir
        self.eval_class = eval_class
        self.iou_threshold = iou_threshold
        self.id2label = id2label
        self.eval_multi_category = eval_multi_category
        self.gt_max_depth = gt_max_depth
        self.metrics = metrics
        self.match_mode = match_mode
        self.ct_match_mode = ct_match_mode
        self.yaw_amplitude = yaw_amplitude
        self.ap_score_threshold = ap_score_threshold
        self.ct_matching_thresholds = ct_matching_thresholds
        self.depth_intervals = depth_intervals
        self.eval_vcs_range = eval_vcs_range
        self.eval_category_ids = eval_category_ids
        self.dist_intervals = dist_intervals
        self.compute_foreground_prec = compute_foreground_prec
        self.all_res = defaultdict(list)
        self.eps = 1e-9
        self.score_threshold = 0.2
        self.metric_index = 0
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.collect_result = ["fn", "tp", "fp"]
        if self.compute_foreground_prec:
            self.collect_result.append("fp_miscls")

    def update(self, preds: List[dict], gts: List):
        res = self._parse_gts_and_preds(gts, preds)
        for k, v in res.items():
            if isinstance(v, list):
                self.all_res[k].extend(v)
            else:
                self.all_res[k].extend([v])
        return res

    def _is_in_range(self, loc, eval_vcs_range=None):
        if eval_vcs_range is None:
            return 1
        x, y = loc
        if (
            x < eval_vcs_range[0]
            or x > eval_vcs_range[2]
            or y < eval_vcs_range[1]
            or y > eval_vcs_range[3]
        ):
            return 0
        return 1

    def _get_vertices_from_vcs_box(self, wh, ct, yaw):
        """Get vertices of bounding box with format (w,h,cx,cy, yaw) in vcs.

        Args:
            wh: [w, h] in vcs along yaw direction and vertical to yaw.
            ct: [cx, cy], coordinate of center of bounding box in vcs.
            yaw: yaw of bounding box in vcs.

        Returns: coordinate (x, y) in vcs, 4 vertices of a bbox.

        """
        w, h = wh[:2]
        ctx, cty = ct

        p0 = [0.5 * w, 0.5 * h, 1]
        p1 = [0.5 * w, -0.5 * h, 1]
        p2 = [-0.5 * w, -0.5 * h, 1]
        p3 = [-0.5 * w, 0.5 * h, 1]
        points = np.array([p0, p1, p2, p3])
        R = np.array(
            [
                [np.cos(yaw), np.sin(yaw), ctx],
                [-np.sin(yaw), np.cos(yaw), cty],
                [0, 0, 1],
            ]
        )
        points = R.dot(points.T).T
        return points[:, :2]

    def get_misclassified_pred(
        self, misclassified_data, timestamps, all_result
    ):
        """Get all-category misclassified prediction."""
        all_miss_det_clsid = {}
        all_miss_gt_clsid = {}
        all_miss_det_box = {}
        all_miss_gt_box = {}
        # get all fps and all gts
        for timestamp in timestamps:
            all_miss_det_clsid[timestamp] = []
            all_miss_gt_clsid[timestamp] = []
            all_miss_det_box[timestamp] = np.zeros((0, 7), dtype=np.float32)
            all_miss_gt_box[timestamp] = np.zeros((0, 7), dtype=np.float32)
            for cid in self.eval_category_ids:
                val = misclassified_data[cid][timestamp]
                if len(val) == 0:
                    continue
                miss_det_idx = val[:, 0] == 0
                miss_gt_idx = val[:, 0] == 1
                # only preserve high score prediction.
                vaild_det_idx = val[miss_det_idx][:, 1] >= self.score_threshold
                all_miss_det_box[timestamp] = np.append(
                    all_miss_det_box[timestamp],
                    val[miss_det_idx][vaild_det_idx],
                    axis=0,
                )
                all_miss_det_clsid[timestamp].extend(
                    [cid] * vaild_det_idx.sum()
                )
                # preserve all gts.
                all_miss_gt_box[timestamp] = np.append(
                    all_miss_gt_box[timestamp], val[miss_gt_idx], axis=0
                )
                all_miss_gt_clsid[timestamp].extend([cid] * miss_gt_idx.sum())
        # dict for misclassification per category, score and depth are in it.
        miscls_fp = {}
        for cid in self.eval_category_ids:
            miscls_fp[cid] = np.zeros((0, 3), dtype=np.float32)
        # compute iou between fps and gts,
        # if iou>0 between different class pair<fp, gt>, then this fp can be
        # treated as misclassified pred.
        for timestamp in timestamps:
            miss_det_box = all_miss_det_box[timestamp]
            miss_gt_box = all_miss_gt_box[timestamp]
            miss_det_clsid = all_miss_det_clsid[timestamp]
            miss_gt_clsid = all_miss_gt_clsid[timestamp]
            if len(miss_det_box) and len(miss_gt_box):
                valid_gt = np.ones(miss_gt_box.shape[0], dtype=bool)
                pairwise_iou = rotate_iou(
                    miss_det_box[:, 2:], miss_gt_box[:, 2:]
                )
                for dt_idx in range(miss_det_box.shape[0]):
                    det_iou = pairwise_iou[dt_idx] * valid_gt.astype(float)
                    max_ind = np.argmax(det_iou)
                    max_iou = det_iou[max_ind]
                    dt_cls = miss_det_clsid[dt_idx]
                    gt_cls = miss_gt_clsid[max_ind]
                    # if iou=0 or fp and gt are the same class
                    # this fp does not belong to misclassification.
                    if max_iou == 0 or dt_cls == gt_cls:
                        continue
                    valid_gt[max_ind] = 0
                    miscls_fp[dt_cls] = np.append(
                        miscls_fp[dt_cls],
                        miss_det_box[dt_idx][1:4][None, :],
                        axis=0,
                    )

        for cid in self.eval_category_ids:
            all_result[f"{cid}_misfg_prec_data"] = []
            all_result[f"{cid}_misfg_prec_data"].extend([miscls_fp[cid]])

        return all_result

    def _eval_metric(
        self,
        det_res: Mapping,
        annotation: Mapping,
        iou_threshold: float,
        gt_max_depth: float,
        yaw_amplitude: int,
        match_mode: str = None,
        ct_match_mode: str = "percentage",
        eval_vcs_range: Optional[Sequence[float]] = None,
        ct_matching_thresholds: Optional[dict] = None,
        compute_foreground_prec: bool = False,
    ) -> Mapping:
        """Eval the metric between GT and pred boxes.

        Based on bev3d_bbox_eval(). For more details please refer to
        (hat/metrics/metric_3dv_utils.py)

        Args:
            det_res: The predict discrete obj boxes info.
            annotation: The ground truth discrete obj boxes info.
            iou_threshold: Threshold for IoU.
            gt_max_depth: Max depth for gts.
            yaw_amplitude: The amplitude of yaw, the
                yaw range of (stopline,crosswalk,...) is (-90,90),
                so the amplitude of yaw is 180, arrow is 360.
            match_mode: match mode, default is iou, or
                "ct_rot": acrroding to center location and rotation angle.
            eval_vcs_range: The vcs range of you care.
            ct_matching_thresholds: the mapping of gt distance(meter)
                and matching threshold(percent of gt distance).
                Only use when match_mode == "ct".
            compute_foreground_prec: Whether to compute classification precision,  # noqa
                if True, fp predictions and all gts will be returned.
        Returns:
            (Dict): Dict contains the results.
        """

        all_dets = defaultdict(list)
        all_gts = defaultdict(list)
        match_result = {}
        for det in det_res:
            if not self._is_in_range(det["pred_loc"], eval_vcs_range):
                continue
            all_dets[det["timestamp"]].append(det)

        for gt in annotation["annotations"]:
            gt_depth = abs(gt["vcs_discobj_loc"][0])  # vcs: abs(x) = depth
            if gt_depth > gt_max_depth:
                continue
            if not self._is_in_range(gt["vcs_discobj_loc"], eval_vcs_range):
                gt["vcs_discobj_ignore"] = 1
            all_gts[gt["timestamp"]].append(gt)  # 需过滤eval_vcs_range

        gt_missed = []
        det_redundant = []
        matched = []

        if compute_foreground_prec:
            all_valid_gt = {}
            miss_det_fp = {}

        for timestamp in annotation["timestamps"]:
            match_result.setdefault(timestamp, [])
            gts = all_gts[timestamp]
            dets = all_dets[timestamp]
            det_bbox3d, det_scores, det_locs, det_vcs_points = [], [], [], []
            gt_bbox3d, gt_locs, gt_ignores = [], [], []

            if compute_foreground_prec:
                all_valid_gt[timestamp] = []
                miss_det_fp[timestamp] = []

            if len(gts) == 0:
                for det in dets:
                    det_redundant.append([det["pred_score"]] + det["pred_loc"])
                    det["attr"] = "FP"
                    dim = det["pred_wh"]
                    yaw = det["pred_yaw"]
                    loc = det["pred_loc"]
                    vcs_points = self._get_vertices_from_vcs_box(dim, loc, yaw)
                    det["pred_vcs_points"] = vcs_points
                    tmp_det = copy.deepcopy(det)
                    for k, v in tmp_det.items():
                        if isinstance(v, np.ndarray):
                            tmp_det[k] = v.tolist()
                    match_result[timestamp].append(tmp_det)
                    if compute_foreground_prec:
                        score = det["pred_score"]
                        # the first value is a flag which defined 0
                        # for pred for differentiable with gt
                        miss_det_fp[timestamp].extend(
                            [[0, score, loc[0], loc[1], dim[0], dim[1], -yaw]]
                        )
                continue

            for gt in gts:
                dim = gt["vcs_discobj_wh"]
                yaw = gt["vcs_discobj_yaw"]
                loc = gt["vcs_discobj_loc"]
                # [x, y, w, h, -yaw], -yaw means change the yaw from \
                # counterclockwise -> clockwise
                bbox3d = [loc[0], loc[1], dim[0], dim[1], -yaw]
                gt_bbox3d.append(bbox3d)
                gt_locs.append(loc)
                gt_ignores.append(int(gt["vcs_discobj_ignore"]))

                if (
                    compute_foreground_prec
                    and int(gt["vcs_discobj_ignore"]) == 0
                ):
                    # the first value is a flag which defined 1
                    # for gt for differentiable with pred
                    all_valid_gt[timestamp].extend([[1, 1.0] + bbox3d])

            if len(dets) == 0:
                for gt, gt_ignore in zip(gts, gt_ignores):
                    if gt_ignore:
                        continue
                    loc = gt["vcs_discobj_loc"]
                    gt_missed.append(loc.tolist())

                for gt in gts:
                    if int(gt["vcs_discobj_ignore"]):
                        continue
                    gt["attr"] = "FN"
                    tmp_gt = copy.deepcopy(gt)
                    for k, v in tmp_gt.items():
                        if isinstance(v, np.ndarray):
                            tmp_gt[k] = v.tolist()
                    match_result[timestamp].append(tmp_gt)

                continue

            for det in dets:
                dim = det["pred_wh"]
                yaw = det["pred_yaw"]
                loc = det["pred_loc"]
                vcs_points = self._get_vertices_from_vcs_box(dim, loc, yaw)
                bbox3d = [loc[0], loc[1], dim[0], dim[1], -yaw]
                det_bbox3d.append(bbox3d)
                det_scores.append(det["pred_score"])
                det_locs.append(det["pred_loc"])
                det_vcs_points.append(vcs_points)

            det_bbox3d = np.array(det_bbox3d)
            det_scores = np.array(det_scores)
            gt_bbox3d = np.array(gt_bbox3d)
            det_locs = np.array(det_locs)
            gt_locs = np.array(gt_locs)

            assert det_bbox3d.shape[0] == det_scores.shape[0]

            if match_mode == "ct_rot":
                (matched_dict, redundant, _) = ct_rot_matching(
                    det_bbox3d,
                    det_locs,
                    gt_bbox3d,
                    gt_locs,
                    det_scores,
                    gt_ignores,
                )
            elif match_mode == "ct":
                (matched_dict, redundant, _) = ct_matching(
                    det_locs,
                    gt_locs,
                    det_scores,
                    ct_match_mode,
                    gt_ignores,
                    ct_matching_thresholds,
                )
            else:
                (matched_dict, redundant, _) = rotate_iou_matching(
                    det_bbox3d,
                    det_locs,
                    gt_bbox3d,
                    gt_locs,
                    det_scores,
                    iou_threshold,
                    gt_ignores,
                )

            det_redundant.extend(
                [[det_scores[i]] + det_locs[i].tolist() for i in redundant]
            )
            for i in redundant:
                dets[i]["attr"] = "FP"
                dets[i]["pred_vcs_points"] = det_vcs_points[i]
                tmp_det = copy.deepcopy(dets[i])
                for k, v in tmp_det.items():
                    if isinstance(v, np.ndarray):
                        tmp_det[k] = v.tolist()
                match_result[timestamp].append(tmp_det)
                if compute_foreground_prec:
                    dim = dets[i]["pred_wh"]
                    yaw = dets[i]["pred_yaw"]
                    loc = dets[i]["pred_loc"]
                    score = dets[i]["pred_score"]
                    miss_det_fp[timestamp].extend(
                        [[0, score, loc[0], loc[1], dim[0], dim[1], -yaw]]
                    )

            det_assigns = matched_dict["det_assign"]
            inds = np.array(list(range(len(det_assigns))))
            gt_miss_idx = det_assigns == -1
            gt_ignores = np.array(gt_ignores)
            gt_miss_idx[gt_ignores == 1] = False
            gt_missed.extend(
                [
                    gt_locs[i].tolist()
                    for i in range(len(inds))
                    if gt_miss_idx[i]
                ]
            )
            for i in range(len(inds)):
                if gt_miss_idx[i]:
                    gts[i]["attr"] = "FN"
                    tmp_gt = copy.deepcopy(gts[i])
                    for k, v in tmp_gt.items():
                        if isinstance(v, np.ndarray):
                            tmp_gt[k] = v.tolist()
                    match_result[timestamp].append(tmp_gt)

            mask = det_assigns != -1
            det_assigns, inds = det_assigns[mask], inds[mask]

            if len(det_assigns) == 0:
                continue
            assert len(det_assigns.tolist()) == len(inds.tolist())

            pred_socre = np.array([dets[i]["pred_score"] for i in det_assigns])
            pred_dim = np.array([dets[i]["pred_wh"] for i in det_assigns])
            pred_loc = np.array([dets[i]["pred_loc"] for i in det_assigns])
            pred_yaw_rad = np.array([dets[i]["pred_yaw"] for i in det_assigns])
            pred_yaw = np.rad2deg(pred_yaw_rad)

            gt_dim = np.array([gts[i]["vcs_discobj_wh"] for i in inds])
            gt_loc = np.array([gts[i]["vcs_discobj_loc"] for i in inds])
            gt_yaw_rad = np.array([gts[i]["vcs_discobj_yaw"] for i in inds])
            gt_yaw = np.rad2deg(gt_yaw_rad)

            dx = np.abs(pred_loc[:, 0] - gt_loc[:, 0])
            dy = np.abs(pred_loc[:, 1] - gt_loc[:, 1])
            dxy = (dx ** 2 + dy ** 2) ** 0.5
            dw = np.abs(pred_dim[:, 0] - gt_dim[:, 0])
            dh = np.abs(pred_dim[:, 1] - gt_dim[:, 1])
            abs_rot = np.abs(gt_yaw - pred_yaw) % yaw_amplitude
            drot = np.minimum(abs_rot, yaw_amplitude - abs_rot)

            det_idx = 0
            for det_i, gt_i in zip(det_assigns, inds):
                tmp_match_det = copy.deepcopy(dets[det_i])
                tmp_match_det["attr"] = "TP"
                tmp_match_det["pred_vcs_points"] = det_vcs_points[det_i]
                for k, v in tmp_match_det.items():
                    if isinstance(v, np.ndarray):
                        tmp_match_det[k] = v.tolist()
                tmp_match_det["dxy"] = dxy[det_idx]
                tmp_match_det["drot"] = drot[det_idx]

                tmp_match_gt = copy.deepcopy(gts[gt_i])
                tmp_match_gt["attr"] = "TP"
                for k, v in tmp_match_gt.items():
                    if isinstance(v, np.ndarray):
                        tmp_match_gt[k] = v.tolist()
                tmp_match_det["gt"] = tmp_match_gt
                match_result[timestamp].append(tmp_match_det)
                det_idx += 1

            matched.extend(
                [
                    [
                        pred_socre[i],
                        dx[i],
                        dy[i],
                        dxy[i],
                        dw[i],
                        dh[i],
                        drot[i],
                    ]
                    + gt_loc[i].tolist()
                    for i in range(len(dx))
                ]
            )
        Num_all = len(gt_missed) + len(det_redundant) + len(matched)
        Num_result_all = 0
        for i in match_result:
            Num_result_all += len(match_result[i])
        assert Num_all == Num_result_all
        result = np.zeros((Num_all, 9))
        i_num = 0
        for gt in gt_missed:
            result[i_num, :2] = gt
            i_num += 1
        for det in det_redundant:
            result[i_num, :3] = det
            i_num += 1
        for mat in matched:
            result[i_num] = mat
            i_num += 1

        if compute_foreground_prec:
            miss_result = {}
            for timestamp in miss_det_fp.keys():
                _miss_det_fp = np.array(miss_det_fp[timestamp])
                _all_valid_gt = np.array(all_valid_gt[timestamp])
                if len(_all_valid_gt) and len(_miss_det_fp):
                    miss_result[timestamp] = np.concatenate(
                        [_miss_det_fp, _all_valid_gt], axis=0
                    )
                elif len(_all_valid_gt):
                    miss_result[timestamp] = _all_valid_gt
                else:
                    miss_result[timestamp] = _miss_det_fp
        else:
            miss_result = None

        return result, match_result, miss_result

    def _parse_gts_and_preds(self, gt_list, pred_list):
        eval_timestamps = []
        det_objs_by_cid = {cid: [] for cid in self.eval_category_ids}
        gt_objs_by_cid = {cid: [] for cid in self.eval_category_ids}
        for output in pred_list:
            num_objs = len(output["pred_wh"])
            for obj_idx in range(num_objs):
                pred_items = {key: val[obj_idx] for key, val in output.items()}
                cid = pred_items["pred_bev_discobj_cls_id"]
                if cid in det_objs_by_cid:
                    det_objs_by_cid[cid] += [pred_items]

        for gt in gt_list:
            eval_timestamps.append(gt["timestamp"][0])
            meta = gt["meta"]
            annotations = {
                "vcs_discobj_loc": np.array(gt["vcs_discobj_loc"]),
                "vcs_discobj_wh": np.array(gt["vcs_discobj_wh"]),
                "vcs_discobj_ignore": np.array(gt["vcs_discobj_ignore"]),
                "vcs_discobj_rate_visible": np.array(
                    gt["vcs_discobj_rate_visible"]
                ),
                "vcs_discobj_cls": np.array(gt["vcs_discobj_cls"]),
                "vcs_discobj_yaw": np.array(gt["vcs_discobj_yaw"]),
                "vcs_points": np.array(gt["vcs_points"]),
                "uid": np.array(gt["uid"]),
            }
            for cid in self.eval_category_ids:
                cur_cid_idxs = (
                    annotations["vcs_discobj_cls"] == cid
                ).nonzero()[0]
                gt_cur_cid = {
                    key: val[cur_cid_idxs] for key, val in annotations.items()
                }
                num_objs_gt = gt_cur_cid[list(gt_cur_cid.keys())[0]].shape[0]
                for gt_obj_idx in range(num_objs_gt):
                    gt_items = {
                        key: val[gt_obj_idx] for key, val in gt_cur_cid.items()
                    }
                    gt_items["timestamp"] = gt["timestamp"][0]
                    gt_objs_by_cid[cid] += [gt_items]
        if self.compute_foreground_prec:
            # if need to compute classification precision,
            # we need to get all-category fps
            misclassified_data = {}
        all_result = {}
        all_result.setdefault("all_info_list", {})
        for cid in self.eval_category_ids:
            all_result.setdefault(f"{cid}", [])
            gt = {
                "timestamps": eval_timestamps,
                "annotations": gt_objs_by_cid[cid],
            }
            det = det_objs_by_cid[cid]
            res, match_result, miss_res = self._eval_metric(
                det,
                gt,
                self.iou_threshold,
                self.gt_max_depth,
                self.yaw_amplitude,
                self.match_mode.get(self.id2label[cid], "iou")
                if isinstance(self.match_mode, dict)
                else self.match_mode,
                self.ct_match_mode,
                self.eval_vcs_range.get(self.id2label[cid], None),
                ct_matching_thresholds=self.ct_matching_thresholds,
                compute_foreground_prec=self.compute_foreground_prec,
            )
            all_result["all_info_list"].setdefault(eval_timestamps[0], {})
            all_result["all_info_list"][eval_timestamps[0]].setdefault(
                "object_info", []
            )
            all_result["all_info_list"][eval_timestamps[0]][
                "object_info"
            ].extend(match_result[eval_timestamps[0]])

            all_result[str(cid)].extend([res])
            if self.compute_foreground_prec:
                misclassified_data[cid] = miss_res

        all_result["all_info_list"][eval_timestamps[0]]["meta"] = meta[0]
        if self.compute_foreground_prec:
            all_result = self.get_misclassified_pred(
                misclassified_data, eval_timestamps, all_result
            )
        return all_result

    def get(self):
        """Get evaluation metrics."""
        result_summary, leaderboard_results = gather_all_cls(
            all_res=self.all_res,
            eval_category_ids=self.eval_category_ids,
            collect_result_list=self.collect_result,
            eval_vcs_range=self.eval_vcs_range,
            id2label=self.id2label,
            depth_intervals=self.depth_intervals,
            dist_intervals=self.dist_intervals,
            gt_max_depth=self.gt_max_depth,
            metrics=self.metrics,
            ap_score_threshold=self.ap_score_threshold,
            score_threshold=self.score_threshold,
            eps=self.eps,
            compute_foreground_prec=self.compute_foreground_prec,
        )
        # write table
        # dump to file
        if result_summary:
            gen_results(
                results=result_summary,
                output_dir=self.save_dir,
                result_json="result.json",
                result_png="result.png",
                table_json="tables.json",
                eval_multi_category=self.eval_multi_category,
                eval_vcs_range=self.eval_vcs_range,
            )

            logger.info("eval done")
        logger.info(leaderboard_results)
        return leaderboard_results
