# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import glob
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import time
from collections import OrderedDict, defaultdict
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
from prettytable import PrettyTable

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.apply_func import convert_numpy as to_numpy
from hat.utils.bucket import url_to_local_path
from hat.utils.distributed import get_dist_info, rank_zero_only
from hat.visualize.bev_3d import Bev3DVisualize, get_3dboxcorner_in_vcs_numpy
from .metric import EvalMetric
from .metric_3dv_utils import (
    NpEncoder,
    bev3d_bbox_eval,
    bev3d_bbox_tag_eval,
    calap,
    collect_data,
    draw_curves,
)

__all__ = ["BEVDetEval", "BEVDetTagEval"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register_module
class BEVDetEval(EvalMetric):
    """The BEV 3D detection eval metrics.

    The BEV 3D detection metric calculation is based on the real3d eval metric,
    for more detail please refer to Read3dEval (hat/metrics/real3d.py)

    Args:
        eval_category_ids (tuple): The categories need to be evaluated.
            If "all" in it, a new category named all will be added into
            evaluation pipeline. Specifically, class more than 0 will be
            mapped to 0 for gt and pred, then, metrics for this new class 0,
            will be calculated independently.
        name(str): Name of this metric instance for display
        score_threshold (float, defualt: 0.1): Threshold for score.
        iou_threshold (float, defualt: 0.2): Threshold for IoU.
        gt_max_depth (float, default: 100): Max depth for gts.
        eval_vcs_range (tuple of float, default: None): Max vcs
            (bottom, right, top, left) for gts & preds,
            will override gt_max_depth if is not None.
        save_path (str, default: None): Path to save bev3d results,
            format: pred.pkl.
        save_score_thr (float, defualt: 0.0): The score thr for saved pred box.
        save_real3d_res (bool, default: False): Whether save the eval results
            of real3d data for aidi adas_eval. (TODO: Remove it when the real3d
            dataset not in training dataset, xiangyu.li)
        overrides_save_res (bool, default: False): Whether to override the
            pred.pkl in each metric eval interval, if True, only the latest
            pred.pkl will be save.
        save_vis_dir (str, default: None): Path to save bev3d visulize results.
        vis_setting (Dict, default: None): Settings of visualize.
        metrics (tuple of str, default: None): Tuple of eval metrics, using
            ("dxyp", "drot") if not special.
        depth_intervals (tuple of int, default: None): Depth range to
            validation, using (20, 50, 70) if not special.
        visibility_intervals (tuple of float, default: None): Piecewise
            interval, which used to evaluate in each visibility segment.
            When seting visibility_interval to (0,0.5,1), we will get
            evaluation results of different instance visibility intervals
            [0,0.5) and [0.5,1).
        enable_ignore (bool, default: True): Whether to use ignore_mask.
        ego_ignore_range (tuple of float, default: None): Ego range
            (bottom, right, top, left) to be ignored,(-0.6, -0.5, 2.0, 0.5)
            recommended based on the minimum tire diameter, wheelbase and track
        prcurv_save_path (str, default: None): Path to save pr results
            and curves, NOTE: since the ap calculate should contains all
            pred's results, so we should create a dict to save the ap res.
        verbose (bool, default: True): Whether print the recall/precision
            etc of sub-distance.
        id2label (dict, default: None): Mapping from category id to
            category's name.
        eval_occlusion (bool): Whether need to evaluate occlusion attribute
            prediction. Only evaluate occlusion attributes of TPs, and the
            evaluation scheme is:
            (num samples with correct occlusion classification in TPs) /
            (total num of TPs)
        occlusion_ignore_id (int): Default occlusion attributes ignore id.
        anno_name: key of anno for eval AP.
        eval_mode (str / list(str)): Matching mode between pred and gt
            used for evaluation. Only support bev_iou and let_iou. If you only
            want to select one of them, input a str. Else, input a list.
        let_iou_param (Dict): Enable when let_iou in eval_mode. Contains
            various parameters related to let iou. Including "p_t",
            "min_t", "max_t". For example:
            {"p_t": 0.35, "min_t": 4.0, "max_t": 8.0}
    """

    def __init__(
        self,
        eval_category_ids: Sequence[Union[int, str]],
        score_threshold: float,
        iou_threshold: float,
        gt_max_depth: float,
        name: str = "BEV3D",
        eval_vcs_range: Optional[Sequence[float]] = None,
        save_path: Optional[str] = None,
        save_score_thr: float = 0.0,
        save_real3d_res: bool = False,
        overrides_save_res: bool = False,
        save_vis_dir: Optional[str] = None,
        vis_setting: Optional[dict] = None,
        prcurv_save_path: Optional[dict] = None,
        metrics: Optional[Sequence[str]] = (
            "dx",
            "dxp",
            "dy",
            "dyp",
            "dxyp",
            "drot",
        ),
        depth_intervals: Optional[Sequence[str]] = (20, 50, 70),
        visibility_intervals: Optional[Sequence[float]] = None,
        enable_ignore: Optional[bool] = True,
        ego_ignore_range: Optional[Sequence[float]] = None,
        verbose: bool = True,
        id2label: Optional[dict] = None,
        eval_occlusion: bool = False,
        occlusion_ignore_id: int = -99,
        anno_name: str = "annos_bev_3d",
        eval_mode: Union[str, Sequence[str]] = "bev_iou",
        let_iou_param: Optional[Mapping[str, float]] = None,
        ct_rot_size_threshold: Optional[Mapping] = None,
    ):
        self.name = name
        self.eval_category_ids = eval_category_ids
        self.metrics = metrics
        self.gt_max_depth = gt_max_depth
        self.eval_vcs_range = eval_vcs_range
        self.ego_ignore_range = ego_ignore_range
        self.score_threshold = score_threshold
        self.depth_intervals = depth_intervals
        self.visib_intervals = visibility_intervals
        self.iou_threshold = iou_threshold
        self.save_path = save_path
        self.save_score_thr = save_score_thr
        self.overrides_save_res = overrides_save_res
        self.save_vis_dir = save_vis_dir
        self.verbose = verbose
        self.prcurv_save_path = prcurv_save_path
        if id2label is None:
            id2label = {}
        assert isinstance(id2label, dict), "id2label should be None or dict"
        self.id2label = id2label
        self.eval_occlusion = eval_occlusion
        self.occlusion_ignore_id = occlusion_ignore_id
        self.anno_name = anno_name
        self.eval_mode = _as_list(eval_mode)
        assert len(set(self.eval_mode)) == len(self.eval_mode)
        for mode in self.eval_mode:
            assert mode in ["bev_iou", "let_iou", "ct_rot_size"]
        if "let_iou" in self.eval_mode:
            assert isinstance(let_iou_param, Mapping)
            self.let_iou_param = let_iou_param
        if not ct_rot_size_threshold:
            ct_rot_size_threshold = {
                1000: {
                    "ct": 2,
                    "rot": 360,
                    "size": 0.2,
                }
            }
        self.ct_rot_size_threshold = ct_rot_size_threshold
        if save_vis_dir:
            self.bev3d_vis = Bev3DVisualize(
                save_path=save_vis_dir, **vis_setting
            )

        if save_real3d_res:
            assert (
                save_path
            ), "save_real3d_res cannot be True if save_path is None"
        self.save_real3d_res = save_real3d_res

        if self.visib_intervals:
            for start, end in zip(
                self.visib_intervals[:-1], self.visib_intervals[1:]
            ):
                assert (
                    start < end
                ), f"pls ensure {visibility_intervals} is in ascending order."

        if self.eval_vcs_range is not None:
            # gt_max_depth is overrided as it is contained in eval_vcs_range
            self.gt_max_depth = max(
                abs(self.eval_vcs_range[2]), abs(self.eval_vcs_range[0])
            )
        assert (
            self.gt_max_depth > depth_intervals[-1]
        ), "gt_max_depth must be greater than depth_intervals[-1]"
        self.eps = 1e-9

        self.metric_index = 0
        self.pkl_save_path = None
        self.rank_pkl_save_path = None
        # json_save_path used for save AP results
        self.json_save_path = None
        self.rank_json_save_path = None
        self.enable_ignore = enable_ignore
        depth_intervals = [0] + list(depth_intervals) + [self.gt_max_depth]
        self.all_depth_intervals = [
            "({},{})".format(start, end)
            for start, end in zip(depth_intervals[:-1], depth_intervals[1:])
        ]
        # Use overall evaluation by default, all intervals to be evaluated
        # are contained in all_vis_intervals
        self.all_vis_intervals = ["(0,1)"]
        if self.visib_intervals is not None:
            self.all_vis_intervals += [
                "({},{})".format(start, end)
                for start, end in zip(
                    self.visib_intervals[:-1], self.visib_intervals[1:]
                )
                if "({},{})".format(start, end) not in self.all_vis_intervals
            ]

        gt_cids = []
        det_cids = []
        for cid in eval_category_ids:
            gt_cids.append(cid)
            det_cids.append(cid)
        self.gt_cids = list(set(gt_cids))
        self.det_cids = list(set(det_cids))
        if "all" in self.gt_cids:
            self.gt_cids.pop(self.gt_cids.index("all"))
        if "all" in self.det_cids:
            self.det_cids.pop(self.det_cids.index("all"))

        if self.verbose:
            base_metric = ("TP", "FP", "FN", "Recall", "Precision")
            self.pr_metric = {}
            if "bev_iou" in self.eval_mode:
                self.pr_metric["bev_iou"] = base_metric
            if "ct_rot_size" in self.eval_mode:
                self.pr_metric["ct_rot_size"] = base_metric
            if "let_iou" in self.eval_mode:
                # Added AL and Precision_al evaluation metrics.
                # AL represents the sum of Longitudinal Affinity of each TP.
                # Precision_al represents AL / (TP + FP)
                self.pr_metric["let_iou"] = (
                    *base_metric,
                    "AL",
                    "Precision_al",
                )
            self.vis_metric = ("TP", "FN", "Recall")
        else:
            self.pr_metric = None
            self.vis_metric = None
        if self.prcurv_save_path:
            self.ap_metric = {}
            if "bev_iou" in self.eval_mode:
                self.ap_metric["bev_iou"] = ("AP_Dist",)
            if "ct_rot_size" in self.eval_mode:
                self.ap_metric["ct_rot_size"] = ("AP_Dist",)
            if "let_iou" in self.eval_mode:
                self.ap_metric["let_iou"] = ("AP_Dist", "APL_Dist")
        else:
            self.ap_metric = None
        super(BEVDetEval, self).__init__(name)

    def get_metric_names(self):
        metric_names = []
        for mode in self.eval_mode:
            for cate_id in self.eval_category_ids:
                for vis_interval in self.all_vis_intervals:
                    prefix = "category_{}_{}_{}_".format(
                        cate_id, vis_interval, mode
                    )
                    metric_names += [prefix + "seg_tp"]
                    metric_names += [prefix + "seg_fn"]
                    metric_names += [prefix + "seg_fp"]
                    metric_names += [prefix + "num_valid_image"]
                    metric_names += [prefix + "num_total_image"]
                    if mode == "let_iou":
                        metric_names += [prefix + "seg_al"]
                    for metric_name in self.metrics:
                        for dep_interval in self.all_depth_intervals:
                            metric_names += [
                                prefix + dep_interval + "_" + metric_name
                            ]
        return metric_names

    def _init_states(self):
        for name in self.get_metric_names():
            if "seg_" in name:
                self.add_state(
                    name,
                    default=torch.zeros((len(self.all_depth_intervals),)),
                    dist_reduce_fx="sum",
                )
            else:
                self.add_state(
                    name,
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.add_state(
                    name + "_num",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )

    def reset(self) -> None:
        super().reset()
        if self.prcurv_save_path:
            rank, word_size = get_dist_info()
            self.rank_json_save_path = os.path.join(
                self.prcurv_save_path,
                f"Eval_{str(self.metric_index)}",
                f"det_aps/rank_{rank}.json",
            )
            self.json_save_path = os.path.join(
                self.prcurv_save_path,
                f"Eval_{str(self.metric_index)}",
                f"pred_ap_{rank}.json",
            )
            json_save_root = os.path.dirname(self.rank_json_save_path)
            if not os.path.exists(json_save_root):
                os.makedirs(json_save_root, exist_ok=True)
        if self.save_path:
            # each rank's det results.
            self.rank_pkl_save_path = os.path.join(
                self.save_path,
                f"Eval_{str(self.metric_index)}",
                f"dets/rank_{rank}.pkl",
            )
            # all det's results.
            self.pkl_save_path = os.path.join(
                self.save_path,
                f"Eval_{str(self.metric_index)}",
                f"pred_rank_{rank}.pkl",
            )
            if self.overrides_save_res:
                self.rank_pkl_save_path = self.rank_pkl_save_path.replace(
                    f"Eval_{str(self.metric_index)}", "Eval_0"
                )
                self.pkl_save_path = self.pkl_save_path.replace(
                    f"Eval_{str(self.metric_index)}", "Eval_0"
                )
            pkl_save_root = os.path.dirname(self.rank_pkl_save_path)
            if not os.path.exists(pkl_save_root):
                os.makedirs(pkl_save_root, exist_ok=True)
        if self.save_vis_dir:
            os.makedirs(self.save_vis_dir, exist_ok=True)

    @rank_zero_only
    def reduce_rank_pkl(
        self,
    ):
        """Reduce all rank's pkl to total pred pkl."""

        pred_res = {}
        ordered_res = OrderedDict()
        rank_pkl_save_root = os.path.dirname(self.rank_pkl_save_path)
        pkl_file_list = glob.glob(rank_pkl_save_root + "/*.pkl")

        for pkl_file in pkl_file_list:
            rank_pred_res = self.load_pkl_file(pkl_file)
            pred_res.update(rank_pred_res)

        for key, value in sorted(pred_res.items(), key=lambda t: t[0]):
            ordered_res[key] = value

        logger.info("remove all rank's pkl...")
        os.system(f"rm {rank_pkl_save_root}/*.pkl")

        if ordered_res:
            with open(self.pkl_save_path, "wb") as f:
                logger.info(f"dump pred.pkl to {self.pkl_save_path}")
                pickle.dump(ordered_res, f)

    @rank_zero_only
    def reduce_ap_resluts(
        self,
        eval_mode,
    ):
        """Reduce all rank's ap results to total ap results.

        and cal ap metrics and draw p-r curves.

        """

        all_pred_aps = []
        rank_json_save_root = os.path.dirname(self.rank_json_save_path)
        json_file_list = glob.glob(
            rank_json_save_root + f"/*_{eval_mode}.json"
        )

        for json_file in json_file_list:
            with open(json_file, "r") as fp:
                rank_pred_aps = fp.read().splitlines()
            for _pred in rank_pred_aps:
                _aps = json.loads(_pred)
                all_pred_aps.append(_aps)

        logger.info("remove all rank's json...")
        os.system(f"rm {rank_json_save_root}/*_{eval_mode}.json")

        num_gt = defaultdict(int)
        det_tp_mask, all_scores, det_tp_pred_loc, det_gt_mask = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        if eval_mode == "let_iou":
            det_tp_let_al_mask = defaultdict(list)
        for res in all_pred_aps:
            for _cid, _cid_ap_res in res.items():
                if _cid != "all":
                    _cid = int(_cid)
                for _ap_res in _cid_ap_res:
                    det_tp_mask[_cid] += _ap_res["det_tp_mask"]
                    all_scores[_cid] += _ap_res["all_scores"]
                    det_tp_pred_loc[_cid] += _ap_res["det_tp_pred_loc"]
                    det_gt_mask[_cid] += _ap_res["det_gt_mask"]
                    num_gt[_cid] += _ap_res["num_gt"]
                    if eval_mode == "let_iou":
                        det_tp_let_al_mask[_cid] += _ap_res[
                            "det_tp_let_al_mask"
                        ]

        ap_results = {}
        max_det_rate_results = {}
        ap_depth_results = {
            cid: [0.0 for _ in range(len(self.depth_intervals) + 1)]
            for cid in self.eval_category_ids
        }
        ap_curves = defaultdict(dict)
        if eval_mode == "let_iou":
            apl_results = {}
            apl_depth_results = copy.deepcopy(ap_depth_results)
            apl_curves = defaultdict(dict)
        for cid in self.eval_category_ids:
            _det_tp_pred_loc = np.array(det_tp_pred_loc[cid])
            _det_tp_mask = np.array(det_tp_mask[cid])
            _all_scores = np.array(all_scores[cid])
            _det_gt_mask = np.array(det_gt_mask[cid])
            if eval_mode == "let_iou":
                _det_tp_let_al_mask = np.array(det_tp_let_al_mask[cid])

            if len(_det_tp_pred_loc) == 0 or len(_det_gt_mask) == 0:
                for dep_ind in range(len(self.depth_intervals) + 1):
                    ap_depth_results[cid][dep_ind] = 0.0
            else:
                det_dep_thresh_inds = np.sum(
                    abs(_det_tp_pred_loc[:, 0:1]) > self.depth_intervals,
                    axis=-1,
                )
                dep_inds, cnts = np.unique(
                    det_dep_thresh_inds, return_counts=True
                )

                gt_count = np.zeros(len(self.depth_intervals) + 1)
                gt_dep_thresh_inds = np.sum(
                    abs(_det_gt_mask[:, 0:1]) > self.depth_intervals, axis=-1
                )
                gt_dep_id, gt_cnts = np.unique(
                    gt_dep_thresh_inds, return_counts=True
                )
                gt_count[gt_dep_id] += gt_cnts

                # cal ap in different detth threshold.
                for dep_ind in dep_inds:
                    mask = det_dep_thresh_inds == dep_ind
                    dep_tp_depth = _det_tp_mask[mask]
                    all_score_depth = _all_scores[mask]
                    arginds_dep = np.argsort(-all_score_depth)
                    all_score_depth = all_score_depth[arginds_dep]
                    dep_tp_depth = dep_tp_depth[arginds_dep]
                    det_fp_mask_depth = 1 - dep_tp_depth
                    det_tp = np.cumsum(dep_tp_depth)
                    det_fp = np.cumsum(det_fp_mask_depth)
                    cus_recall = det_tp / (gt_count[dep_ind] + 1e-6)
                    cus_precision = det_tp / (det_tp + det_fp + 1e-6)
                    ap_depth = calap(cus_recall, cus_precision)
                    # NOTE: Since the TP match in all depth range first and
                    # then split according to depth range, the TP matching
                    # will cross different depth, e.g. TP: vcs_x=19.9 match
                    # GT: vcs_x=20.1, which will cause num_tp > num_gt in
                    # range(0-20), so we clip ap > 1 in such case to 1.
                    if ap_depth > 1.0:
                        ap_depth = 1.0
                    ap_depth_results[cid][dep_ind] = ap_depth
                    if eval_mode == "let_iou":
                        det_tp_let_al_depth = _det_tp_let_al_mask[mask]
                        det_tp_let_al_depth = det_tp_let_al_depth[arginds_dep]
                        det_tp_let_al = np.cumsum(det_tp_let_al_depth)
                        cus_precision_al = det_tp_let_al / (
                            det_tp + det_fp + 1e-6
                        )
                        apl_depth = calap(cus_recall, cus_precision_al)
                        if apl_depth > 1.0:
                            apl_depth = 1.0
                        apl_depth_results[cid][dep_ind] = apl_depth

            # cal all result's ap and save results.
            arginds = np.argsort(-_all_scores)
            _all_scores = _all_scores[arginds]
            _det_tp_mask = _det_tp_mask[arginds]
            det_fp_mask = 1 - _det_tp_mask
            det_tp = np.cumsum(_det_tp_mask)
            det_fp = np.cumsum(det_fp_mask)
            cus_recall = det_tp / (num_gt[cid] + 1e-6)
            cus_precision = det_tp / (det_tp + det_fp + 1e-6)
            ap = calap(cus_recall, cus_precision)
            ap_results[cid] = ap
            # calculate detection rate
            detection_rate = (det_tp - det_fp) / (num_gt[cid] + 1e-6)
            if not len(detection_rate):
                score_at_mdt = 0.0
                max_det_rate = 0.0
                tp_at_mdt = 0.0
                fp_at_mdt = 0.0
                recall_at_mdt = 0.0
                precision_at_mdt = 0.0
                f1_at_mdt = 0.0
            else:
                max_det_rate_ind = np.argmax(detection_rate)
                max_det_rate = detection_rate[max_det_rate_ind]
                score_at_mdt = _all_scores[max_det_rate_ind]
                tp_at_mdt = det_tp[max_det_rate_ind]
                fp_at_mdt = det_fp[max_det_rate_ind]
                recall_at_mdt = cus_recall[max_det_rate_ind]
                precision_at_mdt = cus_precision[max_det_rate_ind]
                f1_at_mdt = (
                    2
                    * recall_at_mdt
                    * precision_at_mdt
                    / (recall_at_mdt + precision_at_mdt + 1e-6)
                )
            max_det_rate_results[cid] = {
                "score": score_at_mdt,
                "max_det_rate": max_det_rate,
                "tp": tp_at_mdt,
                "fp": fp_at_mdt,
                "fn": num_gt[cid] - tp_at_mdt,
                "recall": recall_at_mdt,
                "precision": precision_at_mdt,
                "f1": f1_at_mdt,
            }
            # save pr results for pr-curves
            ap_curves[cid]["recall"] = cus_recall
            ap_curves[cid]["precision"] = cus_precision
            ap_curves[cid]["conf"] = _all_scores
            ap_curves[cid]["num_gt"] = num_gt[cid] + 1e-6
            ap_curves[cid]["detection_rate"] = detection_rate

            save_path_name, save_path_suffix = os.path.splitext(
                self.json_save_path
            )
            save_path = save_path_name + f"_{eval_mode}" + save_path_suffix
            with open(save_path, "w") as f:
                json.dump(ap_curves, f, cls=NpEncoder)

            # draw_pr_curves
            draw_curves(
                [save_path],
                ["all_ap"],
                os.path.join(
                    os.path.dirname(save_path),
                    f"pr_curv_eval_{self.metric_index-1}_{eval_mode}_AP.png",
                ),
                draw_detection_rate=True,
            )

            if eval_mode == "let_iou":
                _det_tp_let_al_mask = _det_tp_let_al_mask[arginds]
                det_tp_let_al = np.cumsum(_det_tp_let_al_mask)
                cus_precision_al = det_tp_let_al / (det_tp + det_fp + 1e-6)
                apl = calap(cus_recall, cus_precision_al)
                # calculate detection rate
                detection_rate = (det_tp - det_fp) / (num_gt[cid] + 1e-6)
                apl_results[cid] = apl

                apl_curves[cid]["recall"] = cus_recall
                apl_curves[cid]["precision"] = cus_precision_al
                apl_curves[cid]["conf"] = _all_scores
                apl_curves[cid]["num_gt"] = num_gt[cid] + 1e-6
                apl_curves[cid]["detection_rate"] = detection_rate

                save_path = save_path_name + "_apl_let_iou." + save_path_suffix
                with open(save_path, "w") as f:
                    json.dump(apl_curves, f, cls=NpEncoder)
                draw_curves(
                    [save_path],
                    ["all_apl"],
                    os.path.join(
                        os.path.dirname(save_path),
                        f"pr_curv_eval_{self.metric_index-1}_{eval_mode}_APL.png",  # noqa E501
                    ),
                    draw_detection_rate=True,
                )
        if eval_mode in ["bev_iou", "ct_rot_size"]:
            return ap_results, ap_depth_results, max_det_rate_results
        elif eval_mode == "let_iou":
            return (
                ap_results,
                ap_depth_results,
                max_det_rate_results,
                apl_results,
                apl_depth_results,
            )
        else:
            raise NotImplementedError(
                "eval_mode must in bev_iou, let_iou, ct_rot_size"
            )

    def get(self):
        eval_result = self.compute()
        if self.save_path:
            self.reduce_rank_pkl()
        return self.name, eval_result

    @staticmethod
    def flat_batch(batch):
        if len(batch["timestamp"].reshape(-1)) != len(batch["timestamp"]):
            batch_size = batch["timestamp"].shape[0]
            clip_size = batch["timestamp"].shape[1]
            batch["timestamp"] = torch.flatten(
                batch["timestamp"].flip(1), 0, 1
            )
            for anno_key in batch["annos_bev_3d"]:
                batch["annos_bev_3d"][anno_key] = torch.flatten(
                    batch["annos_bev_3d"][anno_key], 0, 1
                )
            if "tag_info" in batch:
                tag_info = defaultdict(list)
                tag_names = batch["tag_info"][0].keys()
                for tag_name in tag_names:
                    for j in range(batch_size):
                        for i in reversed(range(clip_size)):
                            tag_info[tag_name].append(
                                batch["tag_info"][i][tag_name][j]
                            )
                batch["tag_info"] = tag_info
            pack_dirs = []
            for j in range(batch_size):
                for _ in reversed(range(clip_size)):
                    pack_dirs.append(batch["pack_dir"][j])
            batch["pack_dir"] = pack_dirs

            if "meta_info" in batch:
                calib_path = []
                for j in range(batch_size):
                    for _ in reversed(range(clip_size)):
                        calib_path.append(batch["meta_info"]["calib_path"][j])
                batch["meta_info"]["calib_path"] = calib_path
            if "origin_imgs" in batch:
                origin_imgs = [[]]
                view_num = len(batch["origin_imgs"][0])
                origin_imgs = []
                for idx in range(view_num):
                    view_imgs = []
                    for j in range(batch_size):
                        for i in reversed(range(clip_size)):
                            view_imgs.append(batch["origin_imgs"][i][idx][j])
                    origin_imgs.append(tuple(view_imgs))
                batch["origin_imgs"] = [
                    origin_imgs,
                ]
            if "ego_pose" in batch:
                ego_pose = []
                for row in range(4):
                    row_data = []
                    for column in range(4):
                        column_data = []
                        for j in range(batch_size):
                            for i in reversed(range(clip_size)):
                                column_data.append(
                                    batch["ego_pose"][i][row][column][j]
                                )
                        row_data.append(tuple(column_data))
                    ego_pose.append(row_data)
                batch["ego_pose"] = ego_pose

    def update(self, batch, output):
        self.flat_batch(batch)
        _gt_group_by_cid, _det_group_by_cid, _timestamps = collect_data(
            batch,
            output,
            self.gt_cids,
            self.det_cids,
            self.save_real3d_res,
            self.eval_occlusion,
            self.anno_name,
        )
        if "all" in self.eval_category_ids:
            batch_all = copy.deepcopy(batch)
            output_all = copy.deepcopy(output)
            annos_bev_3d_cls = batch_all[self.anno_name]["vcs_cls_"]
            annos_bev_3d_cls[annos_bev_3d_cls > 0] = 0
            batch_all[self.anno_name]["vcs_cls_"] = annos_bev_3d_cls
            output_all["bev3d_cls_id"] = torch.zeros_like(
                output_all["bev3d_cls_id"]
            )
            _gt_group_by_cid_all, _det_group_by_cid_all, _ = collect_data(
                batch_all,
                output_all,
                [0],
                [0],
                self.save_real3d_res,
                self.eval_occlusion,
                self.anno_name,
            )
            _gt_group_by_cid["all"] = _gt_group_by_cid_all[0]
            _det_group_by_cid["all"] = _det_group_by_cid_all[0]

        if "bev_iou" in self.eval_mode:
            res_aps_bev_iou = defaultdict(list)
        if "let_iou" in self.eval_mode:
            res_aps_let_iou = defaultdict(list)
        if "ct_rot_size" in self.eval_mode:
            res_aps_ct_rot_size = defaultdict(list)
        for cid in self.eval_category_ids:
            gt = {
                "timestamps": _timestamps,
                "annotations": _gt_group_by_cid[cid],
            }
            det = _det_group_by_cid[cid]
            eval_input = [
                det,
                gt,
                self.depth_intervals,
                self.score_threshold,
                self.iou_threshold,
                self.gt_max_depth,
                self.eval_vcs_range,
                self.enable_ignore,
                self.all_vis_intervals,
                self.ego_ignore_range,
                self.eval_occlusion,
                self.occlusion_ignore_id,
            ]
            if "bev_iou" in self.eval_mode:
                bev_res = bev3d_bbox_eval(*eval_input, "bev_iou")
                self.eval_result_process(
                    bev_res, "bev_iou", cid, res_aps_bev_iou
                )
            if "let_iou" in self.eval_mode:
                let_res = bev3d_bbox_eval(
                    *eval_input, "let_iou", self.let_iou_param
                )
                self.eval_result_process(
                    let_res, "let_iou", cid, res_aps_let_iou
                )
            if "ct_rot_size" in self.eval_mode:
                bev_res = bev3d_bbox_eval(
                    *eval_input,
                    "ct_rot_size",
                    ct_rot_size_threshold=self.ct_rot_size_threshold,
                )
                self.eval_result_process(
                    bev_res, "ct_rot_size", cid, res_aps_ct_rot_size
                )
        if self.save_path:
            self.save_results(batch, output)

        if self.prcurv_save_path:
            if "bev_iou" in self.eval_mode:
                (
                    save_path_name,
                    save_path_suffix,
                ) = os.path.splitext(self.rank_json_save_path)
                save_path = save_path_name + "_bev_iou" + save_path_suffix
                with open(save_path, "a") as f:
                    write_item = json.dumps(res_aps_bev_iou)
                    f.write(write_item + "\n")
            if "let_iou" in self.eval_mode:
                (
                    save_path_name,
                    save_path_suffix,
                ) = os.path.splitext(self.rank_json_save_path)
                save_path = save_path_name + "_let_iou" + save_path_suffix
                with open(save_path, "a") as f:
                    write_item = json.dumps(res_aps_let_iou)
                    f.write(write_item + "\n")
            if "ct_rot_size" in self.eval_mode:
                (
                    save_path_name,
                    save_path_suffix,
                ) = os.path.splitext(self.rank_json_save_path)
                save_path = save_path_name + "_ct_rot_size" + save_path_suffix
                with open(save_path, "a") as f:
                    write_item = json.dumps(res_aps_ct_rot_size)
                    f.write(write_item + "\n")
        # save BEV3D visualize results
        if self.save_vis_dir:
            self.bev3d_vis.save_imgs(output, batch)

    def save_results(self, batch, output):
        pkl_res = self.convert_to_save_format(batch, output)
        with open(self.rank_pkl_save_path, "ab") as f:
            pickle.dump(pkl_res, f)

    def convert_to_save_format(self, batch, output):
        save_res = defaultdict()
        assert "timestamp" in batch
        batch_timestamps = np.array(batch["timestamp"].cpu())
        if self.save_real3d_res:
            batch_timestamps = [
                str(int(_bs_time)) for _bs_time in batch_timestamps
            ]
            rec_date = batch["pack_dir"]
            batch_timestamps = [
                date + "__" + time
                for date, time in zip(rec_date, batch_timestamps)
            ]
        else:
            batch_timestamps = [
                str(int(_bs_time * 1000)) for _bs_time in batch_timestamps
            ]
        batch_size, _ = output["bev3d_ct"].shape[:2]

        for i in range(batch_size):
            front_img_timestamp = batch_timestamps[i]
            if self.save_real3d_res:
                key = batch_timestamps[i]
            else:
                pack_dir = batch["pack_dir"][i]
                # the default timestamp in auto3dv is camera_front
                key = os.path.join(pack_dir, front_img_timestamp)

            save_res[key] = {}
            save_res[key]["gt"] = {}
            save_res[key]["pred"] = {}

            save_res[key]["pred"] = self.get_valid_pred_instance(
                output, self.save_score_thr, i
            )

            save_res[key]["gt"] = self.get_valid_gt_instance(
                batch, i, self.anno_name
            )
            save_res[key]["timestamp"] = str(front_img_timestamp)

        return save_res

    @staticmethod
    def get_valid_gt_instance(batch, idx, anno_name="annos_bev_3d"):
        annos = batch[anno_name]
        gt_cls = annos["vcs_cls_"][idx]
        gt_cls_valid = gt_cls > -99
        keep_annos = {}
        for key in annos.keys():
            keep_annos[key] = to_numpy(annos[key][idx][None, gt_cls_valid])
        return keep_annos

    @staticmethod
    def get_valid_pred_instance(det, score_thresh, idx):
        det_scores = det["bev3d_score"][idx]
        score_filer_idx = det_scores > score_thresh
        keep_dets = {}
        for key in det.keys():
            keep_dets[key] = to_numpy(
                det[key][idx][score_filer_idx][None, ...]
            )
        return keep_dets

    def load_pkl_file(self, pkl_path):
        result = {}
        f = open(pkl_path, "rb")
        while True:
            try:
                data = pickle.load(f)
                result.update(data)
            except EOFError:
                break
        f.close()
        return result

    def compute(self):
        self.metric_index += 1
        # reduce all rank's ap results.
        if self.prcurv_save_path and dist.get_rank() == 0:
            all_ap_results, ap_depth_results = {}, {}
            if "bev_iou" in self.eval_mode:
                (
                    all_ap_results["bev_iou"],
                    ap_depth_results["bev_iou"],
                    _,
                ) = self.reduce_ap_resluts("bev_iou")
            if "let_iou" in self.eval_mode:
                (
                    all_ap_results["let_iou"],
                    ap_depth_results["let_iou"],
                    _,
                    all_apl_results_let,
                    apl_depth_results_let,
                ) = self.reduce_ap_resluts("let_iou")
            if "ct_rot_size" in self.eval_mode:
                (
                    all_ap_results["ct_rot_size"],
                    ap_depth_results["ct_rot_size"],
                    _,
                ) = self.reduce_ap_resluts("ct_rot_size")

        metirc_dict_all = {}
        for eval_mode in self.eval_mode:
            for cid in self.eval_category_ids:
                for interval in self.all_vis_intervals:
                    names, values = [], []
                    prefix = "category_{}_{}_{}_".format(
                        cid, interval, eval_mode
                    )
                    num_valid_image = int(
                        getattr(self, prefix + "num_valid_image", None)
                        .cpu()
                        .numpy()
                    )
                    num_total_image = int(
                        getattr(self, prefix + "num_total_image", None)
                        .cpu()
                        .numpy()
                    )
                    # porcess seg precision-recall
                    pr_results = {}
                    ap_results = {}
                    seg_recall = []
                    seg_precision = []
                    seg_tp = (
                        getattr(self, prefix + "seg_tp", None).cpu().numpy()
                    )
                    seg_fn = (
                        getattr(self, prefix + "seg_fn", None).cpu().numpy()
                    )
                    seg_fp = (
                        getattr(self, prefix + "seg_fp", None).cpu().numpy()
                    )
                    if eval_mode == "let_iou":
                        seg_precision_al = []
                        seg_al = (
                            getattr(self, prefix + "seg_al", None)
                            .cpu()
                            .numpy()
                        )

                    for i in range(len(seg_tp)):
                        tmp_precision = seg_tp[i] / (
                            seg_tp[i] + seg_fp[i] + self.eps
                        )
                        tmp_recall = seg_tp[i] / (
                            seg_tp[i] + seg_fn[i] + self.eps
                        )
                        seg_recall.append(tmp_recall)
                        seg_precision.append(tmp_precision)
                        if eval_mode == "let_iou":
                            tmp_precision_al = seg_al[i] / (
                                seg_tp[i] + seg_fp[i] + self.eps
                            )
                            seg_precision_al.append(tmp_precision_al)

                    pr_results["TP"] = seg_tp
                    pr_results["FP"] = seg_fp
                    pr_results["FN"] = seg_fn
                    pr_results["Recall"] = seg_recall
                    pr_results["Precision"] = seg_precision
                    if self.prcurv_save_path and dist.get_rank() == 0:
                        ap_results["AP_Dist"] = ap_depth_results[eval_mode][
                            cid
                        ]
                    tp = seg_tp.sum()
                    fn = seg_fn.sum()
                    fp = seg_fp.sum()
                    precision = tp / (tp + fp + self.eps)
                    recall = tp / (tp + fn + self.eps)
                    if eval_mode == "let_iou":
                        pr_results["AL"] = seg_al
                        pr_results["Precision_al"] = seg_precision_al
                        if self.prcurv_save_path and dist.get_rank() == 0:
                            ap_results["APL_Dist"] = apl_depth_results_let[cid]
                        al = seg_al.sum()
                        precision_al = al / (tp + fp + self.eps)
                    logger_show_metric = self.metrics

                    if self.verbose:
                        for metric_name in self.pr_metric[eval_mode]:
                            for dep_interval, metric in zip(
                                self.all_depth_intervals,
                                pr_results[metric_name],
                            ):
                                names += [
                                    prefix + dep_interval + "_" + metric_name
                                ]
                                values += [metric]
                        if self.prcurv_save_path and dist.get_rank() == 0:
                            for ap_metric_name in self.ap_metric[eval_mode]:
                                for dep_interval, metric in zip(
                                    self.all_depth_intervals,
                                    ap_results[ap_metric_name],
                                ):
                                    names += [
                                        prefix
                                        + dep_interval
                                        + "_"
                                        + ap_metric_name
                                    ]
                                    values += [metric]

                    for metric in self.metrics:
                        for dep_interval in self.all_depth_intervals:
                            metric_name = prefix + dep_interval + "_" + metric
                            val = (
                                getattr(self, metric_name, None).cpu().numpy()
                            )
                            num = int(
                                getattr(self, metric_name + "_num", None)
                                .cpu()
                                .numpy()
                            )
                            names += [metric_name]
                            values += [val / (num + self.eps)]
                    label = self.id2label[cid] if cid in self.id2label else ""
                    summary_str = (
                        "~~~~ %s class %s [ %s ] %s Summary metrics ~~~~\n"
                        % (
                            self.name,
                            str(cid),
                            label,
                            eval_mode,
                        )
                    )
                    summary_str += "Summary:\n"
                    summary_str += "BEV_3D Overview: \n"
                    nl = "\n"
                    metirc_dict = {}
                    metirc_dict[
                        "all_depth_intervals"
                    ] = self.all_depth_intervals
                    # Only evaluation in whole interval supports PR metric.
                    # Otherwise we can only settle for using TP metric.
                    if interval == "(0,1)":
                        summary_str += f" total_images : {num_total_image} \
                                    {nl} valid_images : {num_valid_image} \
                                    {nl} TP: {tp} {nl} FP: {fp} {nl} FN: {fn} \
                                    {nl} Recall: {recall: .5f} \
                                    {nl} Precision: {precision: .5f} \
                                    {nl}"
                        if eval_mode == "let_iou":
                            summary_str += f" AL: {al: .5f} \
                                    {nl} Precision_al: {precision_al: .5f} \
                                    {nl}"
                            metirc_dict["AL"] = al
                            metirc_dict["Precision_al"] = precision_al
                        if self.prcurv_save_path and dist.get_rank() == 0:
                            summary_str += f" AP: {all_ap_results[eval_mode][cid]: .4f} \
                                            {nl}"
                            if eval_mode == "let_iou":
                                summary_str += f" APL: {all_apl_results_let[cid]: .4f} \
                                            {nl}"
                        logger_show_metric += self.pr_metric[eval_mode]
                        metirc_dict["total_images"] = num_total_image
                        metirc_dict["valid_images"] = num_valid_image
                        metirc_dict["TP"] = tp
                        metirc_dict["FP"] = fp
                        metirc_dict["FN"] = fn
                        metirc_dict["Recall"] = recall
                        metirc_dict["Precision"] = precision
                        if self.prcurv_save_path and dist.get_rank() == 0:
                            metirc_dict["AP"] = all_ap_results[eval_mode][cid]
                            if eval_mode == "let_iou":
                                metirc_dict["APL"] = all_apl_results_let[cid]
                    else:
                        summary_str += f" visibility : {interval} \
                                    {nl} total_images : {num_total_image} \
                                    {nl} valid_images : {num_valid_image} \
                                    {nl} TP: {tp} {nl} FN: {fn} \
                                    {nl} Recall: {recall: .5f} \
                                    {nl}"
                        logger_show_metric += self.vis_metric
                        metirc_dict["visibility"] = interval
                        metirc_dict["total_images"] = num_total_image
                        metirc_dict["valid_images"] = num_valid_image
                        metirc_dict["TP"] = tp
                        metirc_dict["FN"] = fn
                        metirc_dict["Recall"] = recall
                    metirc_dict["depthwise_metric"] = {}
                    for _metric in logger_show_metric:
                        json_interval_dict = {}
                        summary_str += _metric.ljust(10)
                        value_str = "".ljust(10)
                        for _dep_interval in self.all_depth_intervals:
                            _metric_name = (
                                prefix + _dep_interval + "_" + _metric
                            )
                            if _metric_name in names:
                                name_idx = names.index(_metric_name)
                                summary_str += _dep_interval.ljust(15)
                                format = (
                                    ".3f"
                                    if _metric not in ["TP", "FP", "FN"]
                                    else ".1f"
                                )
                                value_str += (
                                    f"{values[name_idx]: {format}}".ljust(15)
                                )
                                json_interval_dict[_dep_interval] = values[
                                    name_idx
                                ]
                        metirc_dict["depthwise_metric"][
                            _metric
                        ] = json_interval_dict
                        summary_str += "\n"
                        summary_str += value_str
                        summary_str += "\n"

                    if (
                        interval == "(0,1)"
                        and self.prcurv_save_path
                        and dist.get_rank() == 0
                    ):
                        for _metric in self.ap_metric[eval_mode]:
                            json_ap_metric_dict = {}
                            summary_str += _metric.ljust(10)
                            value_str = "".ljust(10)
                            for _dep_interval in self.all_depth_intervals:
                                _metric_name = (
                                    prefix + _dep_interval + "_" + _metric
                                )
                                if _metric_name in names:
                                    name_idx = names.index(_metric_name)
                                    summary_str += _dep_interval.ljust(15)
                                    format = (
                                        ".3f"
                                        if _metric not in ["TP", "FP", "FN"]
                                        else ".1f"
                                    )
                                    value_str += (
                                        f"{values[name_idx]: {format}}".ljust(
                                            15
                                        )
                                    )
                                    json_ap_metric_dict[
                                        _dep_interval
                                    ] = values[name_idx]
                            metirc_dict["depthwise_metric"][
                                _metric
                            ] = json_ap_metric_dict
                            summary_str += "\n"
                            summary_str += value_str
                            summary_str += "\n"
                    metirc_dict_all[cid] = metirc_dict
                    logger.info(summary_str)
        return ["recall", "precision"], [recall, precision], metirc_dict_all

    def eval_result_process(self, res, eval_mode, cid, res_aps, metrics=None):
        for vidx, interval in enumerate(self.all_vis_intervals):
            prefix = "category_{}_{}_{}_".format(cid, interval, eval_mode)
            device = getattr(self, prefix + "seg_tp", None).device
            seg_tp = torch.tensor(
                res["counts"]["gt_matched"][vidx], device=device
            )
            seg_fn = torch.tensor(
                res["counts"]["gt_missed"][vidx], device=device
            )
            seg_fp = torch.tensor(
                res["counts"]["redundant_det"], device=device
            )
            num_valid_image = res["counts"]["timestamp_count"]
            num_total_image = res["counts"]["total_timestamp_count"]
            setattr(
                self,
                prefix + "seg_tp",
                getattr(self, prefix + "seg_tp", None) + seg_tp,
            )
            setattr(
                self,
                prefix + "seg_fn",
                getattr(self, prefix + "seg_fn", None) + seg_fn,
            )
            setattr(
                self,
                prefix + "seg_fp",
                getattr(self, prefix + "seg_fp", None) + seg_fp,
            )
            setattr(
                self,
                prefix + "num_valid_image",
                getattr(self, prefix + "num_valid_image", None)
                + num_valid_image,
            )
            setattr(
                self,
                prefix + "num_total_image",
                getattr(self, prefix + "num_total_image", None)
                + num_total_image,
            )
            if eval_mode == "let_iou":
                seg_al = torch.tensor(
                    res["counts"]["gt_al"][vidx], device=device
                )
                setattr(
                    self,
                    prefix + "seg_al",
                    getattr(self, prefix + "seg_al", None) + seg_al,
                )
            if metrics is None:
                metrics = self.metrics
            for metric_name in metrics:
                if metric_name not in res:
                    continue
                for dep_interval, metric in zip(
                    self.all_depth_intervals, res[metric_name][interval]
                ):
                    name = prefix + dep_interval + "_" + metric_name
                    setattr(
                        self,
                        name,
                        getattr(self, name, None) + np.sum(metric),
                    )
                    setattr(
                        self,
                        name + "_num",
                        getattr(self, name + "_num", None) + len(metric),
                    )
            # ap results
            cid_aps = res["counts"]["result_aps"]
            res_aps[cid].append(cid_aps)


@OBJECT_REGISTRY.register_module
class BEVDetTagEval(BEVDetEval):
    """The BEV 3D detection eval metrics with tag.

    The BEV 3D detection metric calculation is based on the real3d eval metric,
    for more detail please refer to Read3dEval (hat/metrics/real3d.py)

    Args:
        metric_save_dir (str): dir to save metric.
        group_pred_by_cls (bool): Group prediction by category.
            If True, cls precision will be 100% while category-wise eval.
        cls_ignore_id (int): class ind of ignore category, default -99.
        show_al_metrics (bool): Show AL metrics while using let iou matcher.
        auto_threshold (bool): Show the metric at max detection rate.
        category_wise_threshold (bool): Using category-wise threshold
            while auto thresholding.
        ct_rot_size_threshold (Dict): The threshold of center/rot/size matcher.
        base_taggers (Dcit): The base tagger for combination.
        taggers (Dict): The infos of taggers.
        **kwargs :
            Please see :py:class:`BEVDetEval`.
    """

    def __init__(
        self,
        eval_category_ids: Sequence[Union[int, str]],
        score_threshold: float,
        iou_threshold: float,
        gt_max_depth: float,
        metric_save_dir: str,
        name: str = "BEV3D",
        eval_vcs_range: Optional[Sequence[float]] = None,
        save_path: Optional[str] = None,
        save_score_thr: float = 0.0,
        save_real3d_res: bool = False,
        overrides_save_res: bool = False,
        save_vis_dir: Optional[str] = None,
        vis_setting: Optional[dict] = None,
        prcurv_save_path: Optional[dict] = None,
        metrics: Optional[Sequence[str]] = (
            "dx",
            "dxp",
            "dy",
            "dyp",
            "dxyp",
            "drot",
        ),
        depth_intervals: Optional[Sequence[str]] = (20, 50, 70),
        visibility_intervals: Optional[Sequence[float]] = None,
        enable_ignore: Optional[bool] = True,
        ego_ignore_range: Optional[Sequence[float]] = None,
        verbose: bool = True,
        id2label: Optional[dict] = None,
        eval_occlusion: bool = False,
        occlusion_ignore_id: int = -99,
        anno_name: str = "annos_bev_3d",
        eval_mode: Union[str, Sequence[str]] = "bev_iou",
        let_iou_param: Optional[Mapping[str, float]] = None,
        group_pred_by_cls: Optional[bool] = False,
        eval_category_cls: Optional[bool] = True,
        cls_ignore_id: Optional[int] = -99,
        show_al_metrics: Optional[bool] = False,
        auto_threshold: Optional[bool] = False,
        category_wise_threshold: Optional[bool] = True,
        ct_rot_size_threshold: Optional[Mapping] = None,
        base_taggers: Optional[Mapping] = None,
        taggers: Optional[Mapping] = None,
        distance_wise_mode: Optional[str] = "depth",
        eval_velo: Optional[bool] = False,
        eval_stability: Optional[bool] = False,
        gt_select_keys: Optional[Sequence[str]] = None,
        pred_select_keys: Optional[Sequence[str]] = None,
    ):
        super(BEVDetTagEval, self).__init__(
            eval_category_ids,
            score_threshold,
            iou_threshold,
            gt_max_depth,
            name,
            eval_vcs_range,
            save_path,
            save_score_thr,
            save_real3d_res,
            overrides_save_res,
            save_vis_dir,
            vis_setting,
            prcurv_save_path,
            metrics,
            depth_intervals,
            visibility_intervals,
            enable_ignore,
            ego_ignore_range,
            verbose,
            id2label,
            eval_occlusion,
            occlusion_ignore_id,
            anno_name,
            eval_mode,
            let_iou_param,
            ct_rot_size_threshold,
        )
        self.metrics = []
        base_metric = []
        for metric_i in metrics:
            if metric_i in (
                "TP",
                "FP",
                "FN",
                "Recall",
                "Precision",
                "Det_Rate",
            ):
                base_metric += [metric_i]
            else:
                if metric_i == "occlusion":
                    metric_i = "occl"
                self.metrics += [metric_i]
        self.metrics = tuple(self.metrics)
        base_metric = tuple(base_metric)
        self.show_al_metrics = show_al_metrics
        self.distance_wise_mode = distance_wise_mode
        self.eval_stability = eval_stability
        self.category_wise_threshold = category_wise_threshold
        self.cls_ignore_id = cls_ignore_id
        self.group_pred_by_cls = group_pred_by_cls
        self.auto_threshold = auto_threshold
        self.metric_save_dir = metric_save_dir

        if self.verbose:
            self.pr_metric = {}
            if "bev_iou" in self.eval_mode:
                self.pr_metric["bev_iou"] = base_metric
            if "ct_rot_size" in self.eval_mode:
                self.pr_metric["ct_rot_size"] = base_metric
            if "let_iou" in self.eval_mode:
                # Added AL and Precision_al evaluation metrics.
                # AL represents the sum of Longitudinal Affinity of each TP.
                # Precision_al represents AL / (TP + FP)
                if self.show_al_metrics:
                    self.pr_metric["let_iou"] = (
                        *base_metric,
                        "AL",
                        "Precision_al",
                    )
                else:
                    self.pr_metric["let_iou"] = base_metric

        if self.prcurv_save_path:
            if "let_iou" in self.eval_mode:
                if not self.show_al_metrics:
                    self.ap_metric["let_iou"] = ("AP_Dist",)
        if not base_taggers:
            base_taggers = {}
        assert isinstance(base_taggers, dict)
        if not taggers:
            taggers = {k_i: {"base_tags": [k_i]} for k_i in base_taggers}
        assert isinstance(taggers, dict)

        self.base_taggers = {}
        self.taggers = {}
        # init tagger param
        for cid in eval_category_ids:
            self.base_taggers[cid] = self.get_distance_taggers()
            self.base_taggers[cid].update(copy.deepcopy(base_taggers))
            if cid == "all":
                for _cid in eval_category_ids:
                    if _cid != "all":
                        self.base_taggers[cid].update(
                            {
                                self.id2label[_cid]: {
                                    "gt_key": "vcs_cls_",
                                    "pred_key": "bev3d_cls_id",
                                    "mapper": {"ids": [_cid]},
                                    "tagger_fn": "tag_by_category",
                                }
                            }
                        )

        for cid in eval_category_ids:
            dist_keys = self.get_distance_taggers().keys()
            self.taggers[cid] = {
                k_i: {"base_tags": [k_i]} for k_i in dist_keys
            }
            self.taggers[cid].update(copy.deepcopy(taggers))
            if cid == "all":
                for _cid in eval_category_ids:
                    if _cid != "all":
                        self.taggers[cid].update(
                            {
                                self.id2label[_cid]: {
                                    "base_tags": [self.id2label[_cid]]
                                }
                            }
                        )

        self.eval_cls = eval_category_cls
        self.eval_velo = eval_velo
        # the key mapper
        if not gt_select_keys:
            gt_select_keys = {
                "vcs_dim_": "dim",
                "vcs_rot_z_": "yaw",
                "vcs_loc_": "loc",
                "vcs_cls_": "cls",
            }
        if not pred_select_keys:
            pred_select_keys = {
                "bev3d_dim": "dim",
                "bev3d_rot": "yaw",
                "bev3d_ct": "loc_xy",
                "bev3d_loc_z": "loc_z",
                "bev3d_cls_id": "cls",
                "bev3d_score": "score",
            }
        self.gt_select_keys = gt_select_keys
        self.pred_select_keys = pred_select_keys
        if eval_stability:
            self.gt_select_keys.update(
                {
                    "obj_idxes": "obj_idxes",
                }
            )
            self.metrics = self.metrics + ("f_dv", "f_dr")
        if eval_occlusion:
            self.gt_select_keys.update(
                {
                    "vcs_occlusion_": "occlusion",
                }
            )
            self.pred_select_keys.update(
                {
                    "bev3d_occlusion_id": "occlusion",
                }
            )
        if eval_velo:
            self.gt_select_keys.update(
                {
                    "vcs_velocities": "velocities",
                }
            )
            self.pred_select_keys.update(
                {
                    "bev3d_velocities": "velocities",
                }
            )

    def get_distance_taggers(self):
        # distance tagger
        taggers = {}
        if self.distance_wise_mode == "xy":
            for distance in self.all_depth_intervals:
                min_max = distance[1:-1].split(",")
                taggers.update(
                    {
                        f"xy{distance}": {
                            "gt_key": "vcs_loc_",
                            "pred_key": "bev3d_ct",
                            "mapper": {
                                "range": [
                                    [float(min_max[0]), float(min_max[1])],
                                ]
                            },
                            "tagger_fn": "tag_by_range",
                            "transforms": lambda loc: np.linalg.norm(loc[:2]),
                        },
                    }
                )
        elif self.distance_wise_mode == "depth":
            taggers = {}
        else:
            raise NotImplementedError

        return taggers

    def reset(self) -> None:
        super().reset()
        self.all_instances = {
            mode: {
                interval: {
                    cid: {"gt": [], "pred": [], "meta": []}
                    for cid in self.eval_category_ids
                }
                for interval in self.all_vis_intervals
            }
            for mode in self.eval_mode
        }
        npy_dir = os.path.join(self.metric_save_dir, "results")
        rank, _ = get_dist_info()
        bev3d_evalflag = os.path.join(
            npy_dir, "bev3d_evalflag_{}.txt".format(rank)
        )
        if os.path.exists(bev3d_evalflag):
            os.remove(bev3d_evalflag)

    def get_cids(self, cls_data):
        batch_size, num_objs = cls_data.shape[:2]
        cids = []
        # process model's pred
        for bs in range(batch_size):
            for obj_idx in range(num_objs):
                cids += [cls_data[bs][obj_idx].cpu().numpy().item()]
        return list(set(cids) - {self.cls_ignore_id})

    def collect_data(self, batch, output):
        _gt_group_by_cid, _det_group_by_cid, _timestamps = collect_data(
            batch=batch,
            output=output,
            gt_cids=self.gt_cids,
            det_cids=self.det_cids,
            save_real3d_res=self.save_real3d_res,
            eval_occlusion=self.eval_occlusion,
            ann_name=self.anno_name,
        )

        batch_all = copy.deepcopy(batch)
        output_all = copy.deepcopy(output)
        all_pred_cids = self.get_cids(output_all["bev3d_cls_id"])
        all_gt_cids = self.get_cids(batch_all[self.anno_name]["vcs_cls_"])
        all_cids = list(set(all_pred_cids + all_gt_cids))

        _gt_group_by_cid_all, _det_group_by_cid_all, _ = collect_data(
            batch=batch_all,
            output=output_all,
            gt_cids=all_cids,
            det_cids=all_cids,
            save_real3d_res=self.save_real3d_res,
            eval_occlusion=self.eval_occlusion,
            ann_name=self.anno_name,
        )
        _all_pred_det, _all_gt_det = [], []
        for cid in _det_group_by_cid_all:
            _all_pred_det += _det_group_by_cid_all[cid]
        for cid in _gt_group_by_cid_all:
            _all_gt_det += _gt_group_by_cid_all[cid]
        _gt_group_by_cid["all"] = _all_gt_det
        _det_group_by_cid["all"] = _all_pred_det
        if not self.group_pred_by_cls:
            for cid in self.eval_category_ids:
                _det_group_by_cid[cid] = copy.deepcopy(
                    _det_group_by_cid["all"]
                )
        return _gt_group_by_cid, _det_group_by_cid, _timestamps

    def update_instances(
        self,
        img_num,
        batch_matched_dict,
        timestamps,
        pack_dirs,
        event_id=None,
        ego_pose=None,
    ):
        # for compatible with e2e eval
        for eval_mode in self.eval_mode:
            for cid in self.eval_category_ids:
                detail_info = batch_matched_dict[eval_mode][cid]["detail"]
                for interval in batch_matched_dict[eval_mode][cid]["detail"]:
                    for idx in range(img_num):
                        gt_instance = {}
                        gt_values = detail_info[interval]["gt"]["value"][idx]
                        gt_depth_wise = detail_info[interval]["gt"][
                            "depth_wise"
                        ][idx]
                        if gt_values:
                            for key, value in self.gt_select_keys.items():
                                assert (
                                    value in gt_values
                                ), f"{value}, {gt_values.keys()}"
                                assert len(gt_values[value]) == len(
                                    gt_values["dim"]
                                )
                                gt_instance[key] = gt_values[value]
                            loc = gt_values["loc"]
                            dim = gt_values["dim"]
                            gt_instance["boxes"] = np.concatenate(
                                [
                                    loc[:, :2],
                                    dim[:, [2, 1]],
                                ],
                                -1,
                            )
                            gt_instance["bev_loc_z"] = loc[:, 2]
                            gt_instance["heights"] = dim[:, 0]
                            gt_instance["labels"] = gt_values["cls"]
                            if "velo" in gt_values:
                                gt_instance["velocities"] = gt_values["velo"]
                            gt_instance["scores"] = np.ones(len(loc))
                            gt_instance["depth_wise"] = {}
                            for dep_ind, dep_interval in enumerate(
                                self.all_depth_intervals
                            ):
                                gt_instance["depth_wise"][
                                    dep_interval
                                ] = gt_depth_wise[dep_ind]
                            gt_instance["taggers"] = {}
                            for tag_name in self.taggers[cid]:
                                gt_instance["taggers"][tag_name] = detail_info[
                                    interval
                                ]["gt"][tag_name][idx]

                        pred_instance = {}
                        pred_values = detail_info[interval]["pred"]["value"][
                            idx
                        ]
                        pred_depth_wise = detail_info[interval]["pred"][
                            "depth_wise"
                        ][idx]
                        if pred_values:
                            for key, value in self.pred_select_keys.items():
                                assert value in pred_values
                                assert len(pred_values[value]) == len(
                                    pred_values["dim"]
                                )
                                pred_instance[key] = pred_values[value]
                            loc_xy = pred_values["loc_xy"]
                            loc_z = pred_values["loc_z"]
                            dim = pred_values["dim"]
                            pred_instance["boxes"] = np.concatenate(
                                [
                                    loc_xy,
                                    dim[:, [2, 1]],
                                ],
                                -1,
                            )
                            pred_instance["bev_loc_z"] = loc_z
                            pred_instance["heights"] = dim[:, 0]
                            pred_instance["labels"] = pred_values["cls"]
                            if "velo" in pred_values:
                                pred_instance["velocities"] = pred_values[
                                    "velo"
                                ]
                            pred_instance["scores"] = pred_values["score"]
                            pred_instance["depth_wise"] = {}
                            for dep_ind, dep_interval in enumerate(
                                self.all_depth_intervals
                            ):
                                pred_instance["depth_wise"][
                                    dep_interval
                                ] = pred_depth_wise[dep_ind]
                            pred_instance["taggers"] = {}
                            for tag_name in self.taggers[cid]:
                                pred_instance["taggers"][
                                    tag_name
                                ] = detail_info[interval]["pred"][tag_name][
                                    idx
                                ]

                        self.all_instances[eval_mode][interval][cid][
                            "gt"
                        ].append(gt_instance)
                        self.all_instances[eval_mode][interval][cid][
                            "pred"
                        ].append(pred_instance)

                        self.all_instances[eval_mode][interval][cid][
                            "meta"
                        ].append(
                            {
                                "timestamp": np.array([timestamps[idx]])
                                if isinstance(timestamps[idx], int)
                                else np.array([timestamps[idx].item()]),
                                "pack_dir": pack_dirs[idx],
                            }
                        )
                        if event_id:
                            self.all_instances[eval_mode][interval][cid][
                                "meta"
                            ][-1].update({"event_id": event_id[idx]})
                        if ego_pose:
                            pose = [
                                [pose_ii[idx] for pose_ii in pose_i]
                                for pose_i in ego_pose
                            ]
                            self.all_instances[eval_mode][interval][cid][
                                "meta"
                            ][-1].update({"ego_pose": pose})

    def update_fn(self, _gt_group_by_cid, _det_group_by_cid, _timestamps):
        # match pred and gt for a batch
        all_res = {eval_mode: {} for eval_mode in self.eval_mode}
        if "bev_iou" in self.eval_mode:
            res_aps_bev_iou = defaultdict(list)
        if "let_iou" in self.eval_mode:
            res_aps_let_iou = defaultdict(list)
        if "ct_rot_size" in self.eval_mode:
            res_aps_ct_rot_size = defaultdict(list)
        for cid in self.eval_category_ids:
            gt = {
                "timestamps": _timestamps,
                "annotations": _gt_group_by_cid[cid],
            }
            det = _det_group_by_cid[cid]
            eval_input = [
                det,
                gt,
                self.depth_intervals,
                self.score_threshold,
                self.iou_threshold,
                self.gt_max_depth,
                self.eval_vcs_range,
                self.enable_ignore,
                self.all_vis_intervals,
                self.ego_ignore_range,
                self.eval_occlusion,
                self.occlusion_ignore_id,
            ]
            if "bev_iou" in self.eval_mode:
                bev_res = bev3d_bbox_tag_eval(
                    *eval_input,
                    "bev_iou",
                    base_taggers=self.base_taggers[cid],
                    taggers=self.taggers[cid],
                    target_cid=cid,
                    eval_velo=self.eval_velo,
                    gt_select_keys=self.gt_select_keys,
                    pred_select_keys=self.pred_select_keys,
                )

                self.eval_result_process(
                    bev_res, "bev_iou", cid, res_aps_bev_iou
                )
                all_res["bev_iou"][cid] = bev_res

            if "let_iou" in self.eval_mode:
                let_res = bev3d_bbox_tag_eval(
                    *eval_input,
                    "let_iou",
                    self.let_iou_param,
                    base_taggers=self.base_taggers[cid],
                    taggers=self.taggers[cid],
                    target_cid=cid,
                    eval_velo=self.eval_velo,
                    gt_select_keys=self.gt_select_keys,
                    pred_select_keys=self.pred_select_keys,
                )
                self.eval_result_process(
                    let_res, "let_iou", cid, res_aps_let_iou
                )
                all_res["let_iou"][cid] = let_res
            if "ct_rot_size" in self.eval_mode:
                ct_res = bev3d_bbox_tag_eval(
                    *eval_input,
                    "ct_rot_size",
                    base_taggers=self.base_taggers[cid],
                    taggers=self.taggers[cid],
                    ct_rot_size_threshold=self.ct_rot_size_threshold,
                    target_cid=cid,
                    eval_velo=self.eval_velo,
                    gt_select_keys=self.gt_select_keys,
                    pred_select_keys=self.pred_select_keys,
                )
                self.eval_result_process(
                    ct_res, "ct_rot_size", cid, res_aps_ct_rot_size
                )
                all_res["ct_rot_size"][cid] = ct_res

        if self.prcurv_save_path:
            if "bev_iou" in self.eval_mode:
                (
                    save_path_name,
                    save_path_suffix,
                ) = os.path.splitext(self.rank_json_save_path)
                save_path = save_path_name + "_bev_iou" + save_path_suffix
                with open(save_path, "a") as f:
                    write_item = json.dumps(res_aps_bev_iou)
                    f.write(write_item + "\n")
            if "let_iou" in self.eval_mode:
                (
                    save_path_name,
                    save_path_suffix,
                ) = os.path.splitext(self.rank_json_save_path)
                save_path = save_path_name + "_let_iou" + save_path_suffix
                with open(save_path, "a") as f:
                    write_item = json.dumps(res_aps_let_iou)
                    f.write(write_item + "\n")
            if "ct_rot_size" in self.eval_mode:
                (
                    save_path_name,
                    save_path_suffix,
                ) = os.path.splitext(self.rank_json_save_path)
                save_path = save_path_name + "_ct_rot_size" + save_path_suffix
                with open(save_path, "a") as f:
                    write_item = json.dumps(res_aps_ct_rot_size)
                    f.write(write_item + "\n")

        return all_res

    def update(self, batch, output):
        self.flat_batch(batch)
        _gt_group_by_cid, _det_group_by_cid, _timestamps = self.collect_data(
            batch, output
        )
        all_res = self.update_fn(
            _gt_group_by_cid, _det_group_by_cid, _timestamps
        )
        batch_timestamps = (np.array(batch["timestamp"].cpu()) * 1000).astype(
            int
        )
        event_id = batch.get("tag_info", {}).get("event_id", None)
        pack_dir = batch["pack_dir"]
        ego_pose = batch.get("ego_pose", [])
        self.update_instances(
            len(batch["timestamp"]),
            all_res,
            batch_timestamps,
            pack_dir,
            event_id,
            ego_pose,
        )
        if self.save_path:
            self.save_results(batch, output)
        # save BEV3D visualize results
        if self.save_vis_dir:
            self.bev3d_vis.save_imgs(output, batch)

    def save_instance(self):
        rank, world_size = get_dist_info()
        npy_dir = os.path.join(self.metric_save_dir, "results")
        os.makedirs(npy_dir, exist_ok=True)
        for mode in self.eval_mode:
            for interval in self.all_vis_intervals:
                interval_name = "-".join(
                    [
                        _.translate({ord(i): None for i in "()"})
                        for _ in interval.split(",")
                    ]
                )
                for cid in self.eval_category_ids:
                    self.save_npy(
                        os.path.join(
                            npy_dir,
                            f"matched_dict_{mode}_interval"
                            + f"{interval_name}_cid{cid}_rank{rank}.npy",
                        ),
                        self.all_instances[mode][interval][cid],
                    )
        del self.all_instances
        if rank != 0:
            finish_file = open(
                os.path.join(npy_dir, "bev3d_evalflag_{}.txt".format(rank)),
                "w",
            )
            finish_file.close()
        else:
            for i in range(1, world_size):
                while not os.path.exists(
                    os.path.join(npy_dir, "bev3d_evalflag_{}.txt".format(i))
                ):
                    time.sleep(10)
                os.remove(
                    os.path.join(npy_dir, "bev3d_evalflag_{}.txt".format(i))
                )
            time.sleep(10)
            process_list = []
            manager = mp.Manager()
            return_dict = manager.dict()
            for mode in self.eval_mode:
                for interval in self.all_vis_intervals:
                    interval_name = "-".join(
                        [
                            _.translate({ord(i): None for i in "()"})
                            for _ in interval.split(",")
                        ]
                    )
                    for cid in self.eval_category_ids:
                        npy_path_pattern = [
                            os.path.join(
                                npy_dir,
                                f"matched_dict_{mode}_"
                                + f"interval{interval_name}_cid{cid}_rank{rank_i}.npy",  # noqa
                            )
                            for rank_i in range(world_size)
                        ]
                        process_list += [
                            mp.Process(
                                target=BEVDetTagEval.cat_and_save_instance,
                                args=(
                                    cid,
                                    interval,
                                    mode,
                                    npy_path_pattern,
                                    npy_dir,
                                    return_dict,
                                ),
                            )
                        ]
                        process_list[-1].start()
            for sub_proc in process_list:
                sub_proc.join()
            all_npy_path = []
            for path in return_dict:
                all_npy_path.append(path + "\n")
            finish_file = open(
                os.path.join(npy_dir, "bev3d_evalflag_0.txt"), "w"  # noqa
            )
            finish_file.writelines(all_npy_path)
            finish_file.close()

    @staticmethod
    def save_npy(path, data):
        # save npy.
        with open(path, "wb") as fn:
            np.save(fn, data)

    @staticmethod
    def cat_and_save_instance(
        cid, interval, mode, npy_path_pattern, npy_dir, return_dict
    ):
        # concat all the instances from different ranks.
        all_match_instances = BEVDetTagEval.load_instances(
            cid, interval, mode, npy_path_pattern
        )
        interval_name = "-".join(
            [
                _.translate({ord(i): None for i in "()"})
                for _ in interval.split(",")
            ]
        )
        for npy_file in npy_path_pattern:
            os.remove(npy_file)
        npy_path = os.path.join(
            npy_dir,
            f"matched_dict_{mode}_"
            + f"interval{interval_name}_cid{cid}_rankall.npy",  # noqa
        )
        BEVDetTagEval.save_npy(
            npy_path,
            all_match_instances,
        )
        return_dict[npy_path] = 1
        return return_dict

    def compute(self):
        # compute the total metric after all inference finished.
        self.metric_index += 1
        metirc_dict_all = {}
        self.save_instance()
        rank, _ = get_dist_info()
        for eval_mode in self.eval_mode:
            # reduce all rank's ap results.
            if self.prcurv_save_path and rank == 0:
                all_ap_results, ap_depth_results, max_det_rate_results = (
                    {},
                    {},
                    {},
                )
                if eval_mode == "bev_iou":
                    (
                        all_ap_results["bev_iou"],
                        ap_depth_results["bev_iou"],
                        max_det_rate_results["bev_iou"],
                    ) = self.reduce_ap_resluts("bev_iou")
                elif eval_mode == "let_iou":
                    (
                        all_ap_results["let_iou"],
                        ap_depth_results["let_iou"],
                        max_det_rate_results["let_iou"],
                        all_apl_results_let,
                        _,
                    ) = self.reduce_ap_resluts("let_iou")
                elif eval_mode == "ct_rot_size":
                    (
                        all_ap_results["ct_rot_size"],
                        ap_depth_results["ct_rot_size"],
                        max_det_rate_results["ct_rot_size"],
                    ) = self.reduce_ap_resluts("ct_rot_size")
            for cid in self.eval_category_ids:
                for interval in self.all_vis_intervals:
                    metirc_dict = {}
                    prefix = "category_{}_{}_{}_".format(
                        cid, interval, eval_mode
                    )
                    num_valid_image = int(
                        getattr(self, prefix + "num_valid_image", None)
                        .cpu()
                        .numpy()
                    )
                    num_total_image = int(
                        getattr(self, prefix + "num_total_image", None)
                        .cpu()
                        .numpy()
                    )

                    if self.auto_threshold and rank == 0:
                        if self.category_wise_threshold:
                            threshold = max_det_rate_results[eval_mode][cid][
                                "score"
                            ]
                            mertic_results = self.eval_metrics(
                                cid, interval, eval_mode, threshold
                            )
                        else:
                            threshold = max_det_rate_results[eval_mode]["all"][
                                "score"
                            ]
                            mertic_results = self.eval_metrics(
                                cid, interval, eval_mode, threshold
                            )
                    elif rank == 0:
                        mertic_results = self.eval_metrics(
                            cid, interval, eval_mode, self.score_threshold
                        )
                    else:
                        mertic_results = {
                            "overall": {
                                "TP": 0,
                                "FP": 0,
                                "FN": 0,
                                "Recall": 0,
                                "Precision": 0,
                            }
                        }

                    label = self.id2label[cid] if cid in self.id2label else ""
                    summary_str = f"~~~~ {self.name} class {cid} [ {label} ] {eval_mode} metrics ~~~~"  # noqa
                    metirc_dict["all_tags"] = list(mertic_results.keys())
                    # Only evaluation in whole interval supports PR metric.
                    # Otherwise we can only settle for using TP metric.
                    if interval == "(0,1)":
                        metirc_dict["total_images"] = num_total_image
                        metirc_dict["valid_images"] = num_valid_image
                        metirc_dict["TP"] = mertic_results["overall"]["TP"]
                        metirc_dict["FP"] = mertic_results["overall"]["FP"]
                        metirc_dict["FN"] = mertic_results["overall"]["FN"]
                        metirc_dict["Recall"] = mertic_results["overall"][
                            "Recall"
                        ]
                        metirc_dict["Precision"] = mertic_results["overall"][
                            "Precision"
                        ]
                        if self.prcurv_save_path and rank == 0:
                            metirc_dict["AP"] = all_ap_results[eval_mode][cid]
                            metirc_dict["mAP"] = all_ap_results[eval_mode][cid]
                            if eval_mode == "let_iou":
                                metirc_dict["APL"] = all_apl_results_let[cid]
                                metirc_dict["mAPL"] = all_apl_results_let[cid]
                    else:
                        metirc_dict["visibility"] = interval
                        metirc_dict["total_images"] = num_total_image
                        metirc_dict["valid_images"] = num_valid_image
                        metirc_dict["TP"] = mertic_results["overall"]["TP"]
                        metirc_dict["FN"] = mertic_results["overall"]["FN"]
                        metirc_dict["Recall"] = mertic_results["overall"][
                            "Recall"
                        ]
                        metirc_dict["Precision"] = mertic_results["overall"][
                            "Precision"
                        ]

                    metirc_dict["tagwise_metric"] = {}
                    if self.auto_threshold and rank == 0:
                        title = f"Detail Metrics at MaxDetRate, [Total_Images={num_total_image}, Valid_Images={num_valid_image}, Threshold={threshold}], cc=cutin_corner"  # noqa
                    else:
                        title = f"Detail Metrics at Threshold={self.score_threshold}, [total_images: {num_total_image}, valid_images: {num_valid_image}], cc=cutin_corner"  # noqa
                    # build table
                    tb, metric_dict = self.build_table(
                        eval_mode,
                        interval,
                        mertic_results,
                        title,
                    )
                    metirc_dict["tagwise_metric"].update(metric_dict)
                    metirc_dict_all[cid] = metirc_dict
                    summary_str += "\n" + tb.get_string()
                    logger.info(summary_str)
                    if rank == 0:
                        html_str = (
                            "<html>"
                            + tb.get_html_string(format=True)
                            + "</html>"
                        )
                        save_dir = os.path.join(self.metric_save_dir, "tables")
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        file_path = os.path.join(
                            save_dir,
                            f"{self.name}-class{cid}[{label}]-{eval_mode}.html",  # noqa
                        )
                        with open(file_path, mode="w", encoding="utf-8") as f:
                            f.write(html_str)
                        logger.info(f"Html table is saved in {save_dir}")
        return (
            ["recall", "precision"],
            [metirc_dict["Recall"], metirc_dict["Precision"]],
            metirc_dict_all,
        )

    def eval_metrics(
        self, cid, interval, eval_mode, score_threshold, npy_path_pattern=None
    ):
        # calculate all bev3d metrics
        error_metric_names = self.metrics
        results = {"overall": {}}
        for dep_interval in self.all_depth_intervals:
            results[dep_interval] = {}
        for tagger_name in self.taggers[cid]:
            results[tagger_name] = {}
        npy_dir = os.path.join(self.metric_save_dir, "results")
        all_match_instances = self.load_instances(
            cid, interval, eval_mode, npy_path_pattern, npy_dir
        )
        (
            gt_values,
            pred_values,
            gt_tag_masks,
            pred_tag_masks,
        ) = self.concatenate_instances(cid, all_match_instances)
        score_mask = pred_values["pred_score"] >= score_threshold
        overall_error = {metric_name: [] for metric_name in error_metric_names}
        # depth-wise errors
        for dep_interval in self.all_depth_intervals:
            gt_matched_mask = gt_tag_masks[dep_interval] == 1
            pred_matched_mask = pred_tag_masks[dep_interval] == 1

            error = self.cal_error(
                pred_matched_mask,
                pred_values,
                gt_matched_mask,
                gt_values,
            )

            for metric_name in error_metric_names:
                if metric_name not in ["f_dv", "f_dr"]:
                    if re.match(r".+%\d+", metric_name):
                        _metric_name, percentage = metric_name.split("%")
                        metric_i = error[_metric_name][
                            score_mask[pred_matched_mask]
                        ]
                        if score_mask[pred_matched_mask].any():
                            results[dep_interval][metric_name] = np.percentile(
                                metric_i, int(percentage)
                            )
                            overall_error[metric_name] += metric_i.tolist()
                        else:
                            results[dep_interval][metric_name] = float(0)
                    else:
                        metric_i = error[metric_name][
                            score_mask[pred_matched_mask]
                        ]
                        if score_mask[pred_matched_mask].any():
                            results[dep_interval][
                                metric_name
                            ] = metric_i.mean().item()
                            overall_error[metric_name] += metric_i.tolist()
                        else:
                            results[dep_interval][metric_name] = float(0)
            if self.eval_stability:
                results[dep_interval].update(
                    self.cal_stability(
                        pred_matched_mask,
                        pred_values,
                        gt_matched_mask,
                        gt_values,
                        score_threshold,
                    )
                )
        # errors for overall
        for metric_name in error_metric_names:
            if metric_name not in ["f_dv", "f_dr"]:
                if re.match(r".+%\d+", metric_name):
                    _metric_name, percentage = metric_name.split("%")
                    if overall_error[metric_name]:
                        results["overall"][metric_name] = np.percentile(
                            np.array(overall_error[metric_name]),
                            int(percentage),
                        )
                    else:
                        results["overall"][metric_name] = float(0)
                else:
                    if overall_error[metric_name]:
                        results["overall"][metric_name] = (
                            np.array(overall_error[metric_name]).mean().item()
                        )
                    else:
                        results["overall"][metric_name] = float(0)

        if self.eval_stability:
            all_gt_mask, all_pred_mask = self.get_overall_mask(
                self.all_depth_intervals, gt_tag_masks, pred_tag_masks
            )
            gt_matched_mask = all_gt_mask == 1
            pred_matched_mask = all_pred_mask == 1
            del all_gt_mask, all_pred_mask
            results["overall"].update(
                self.cal_stability(
                    pred_matched_mask,
                    pred_values,
                    gt_matched_mask,
                    gt_values,
                    score_threshold,
                )
            )

        # tagger-wise errors
        for tagger_name in self.taggers[cid]:
            gt_matched_mask = gt_tag_masks[tagger_name] == 1
            pred_matched_mask = pred_tag_masks[tagger_name] == 1
            error = self.cal_error(
                pred_matched_mask,
                pred_values,
                gt_matched_mask,
                gt_values,
            )

            for metric_name in error_metric_names:
                if metric_name not in ["f_dv", "f_dr"]:
                    if re.match(r".+%\d+", metric_name):
                        _metric_name, percentage = metric_name.split("%")
                        metric_i = error[_metric_name][
                            score_mask[pred_matched_mask]
                        ]
                        if score_mask[pred_matched_mask].any():
                            results[tagger_name][metric_name] = np.percentile(
                                metric_i, int(percentage)
                            )
                        else:
                            results[tagger_name][metric_name] = float(0)
                    else:
                        metric_i = error[metric_name][
                            score_mask[pred_matched_mask]
                        ]
                        if score_mask[pred_matched_mask].any():
                            results[tagger_name][
                                metric_name
                            ] = metric_i.mean().item()
                        else:
                            results[tagger_name][metric_name] = float(0)
            if self.eval_stability:
                results[tagger_name].update(
                    self.cal_stability(
                        pred_matched_mask,
                        pred_values,
                        gt_matched_mask,
                        gt_values,
                        score_threshold,
                    )
                )
        overall_tp, overall_fp, overall_fn = 0.0, 0.0, 0.0
        overall_scores, overall_tp_masks, overall_fp_masks = [], [], []
        # depth-wise pr
        for dep_interval in self.all_depth_intervals:
            gt_mask = gt_tag_masks[dep_interval]
            pred_mask = pred_tag_masks[dep_interval]

            tp = ((pred_mask == 1) * score_mask).sum().astype(np.float32)
            fp = ((pred_mask == 0) * score_mask).sum().astype(np.float32)
            fn = (gt_mask == 0).sum() + (
                (pred_mask == 1) * (1 - score_mask)
            ).sum().astype(np.float32)
            gt_count = tp + fn

            recall = tp / (tp + fn + 1e-6)
            precision = tp / (tp + fp + 1e-6)

            all_scores = pred_values["pred_score"][
                (pred_mask >= 0) * score_mask
            ]
            all_tp_masks = pred_mask[(pred_mask >= 0) * score_mask]
            all_fp_masks = 1 - all_tp_masks
            detection_rate = (tp - fp) / (gt_count + 1e-6)
            results[dep_interval].update(
                {
                    "Pred": tp + fp,
                    "GT": gt_count,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": 2 * recall * precision / (recall + precision + 1e-6),
                    "Det_Rate": detection_rate,
                    "AP": self.cal_ap(
                        gt_count, all_scores, all_tp_masks, all_fp_masks
                    ),
                }
            )
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            overall_scores.append(all_scores)
            overall_tp_masks.append(all_tp_masks)
            overall_fp_masks.append(all_fp_masks)
        # overall pr
        overall_precision = overall_tp / (overall_tp + overall_fp + 1e-6)
        overall_recall = overall_tp / (overall_tp + overall_fn + 1e-6)
        overall_f1 = (
            2
            * overall_recall
            * overall_precision
            / (overall_recall + overall_precision + 1e-6)
        )
        overall_scores = np.concatenate(overall_scores)
        overall_tp_masks = np.concatenate(overall_tp_masks)
        overall_fp_masks = np.concatenate(overall_fp_masks)
        detection_rate = (overall_tp - overall_fp) / (
            overall_tp + overall_fn + 1e-6
        )
        results["overall"].update(
            {
                "Pred": overall_tp + overall_fp,
                "GT": overall_tp + overall_fn,
                "TP": overall_tp,
                "FP": overall_fp,
                "FN": overall_fn,
                "Precision": overall_precision,
                "Recall": overall_recall,
                "F1": overall_f1,
                "Det_Rate": detection_rate,
                "AP": self.cal_ap(
                    overall_tp + overall_fn,
                    overall_scores,
                    overall_tp_masks,
                    overall_fp_masks,
                ),
            }
        )

        # tagger-wise pr
        for tagger_name in self.taggers[cid]:
            gt_mask = gt_tag_masks[tagger_name]
            pred_mask = pred_tag_masks[tagger_name]

            tp = ((pred_mask == 1) * score_mask).sum().astype(np.float32)
            fp = ((pred_mask == 0) * score_mask).sum().astype(np.float32)
            fn = (gt_mask == 0).sum() + (
                (pred_mask == 1) * (1 - score_mask)
            ).sum().astype(np.float32)
            gt_count = tp + fn

            recall = tp / (tp + fn + 1e-6)
            precision = tp / (tp + fp + 1e-6)

            all_scores = pred_values["pred_score"][
                (pred_mask >= 0) * score_mask
            ]
            all_tp_masks = pred_mask[(pred_mask >= 0) * score_mask]
            all_fp_masks = 1 - all_tp_masks
            detection_rate = (tp - fp) / (gt_count + 1e-6)
            results[tagger_name].update(
                {
                    "Pred": tp + fp,
                    "GT": gt_count,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": 2 * recall * precision / (recall + precision + 1e-6),
                    "Det_Rate": detection_rate,
                    "AP": self.cal_ap(
                        gt_count, all_scores, all_tp_masks, all_fp_masks
                    ),
                }
            )

        if self.distance_wise_mode != "depth":
            for dep_interval in self.all_depth_intervals:
                results.pop(dep_interval)
        del gt_values, pred_values
        return results

    @staticmethod
    def load_instances(
        cid, interval, eval_mode, npy_path_pattern=None, npy_dir=None
    ):
        # load the instancs' matched result and value.
        all_match_instances = defaultdict(list)
        if not npy_path_pattern:
            while not os.path.exists(
                os.path.join(npy_dir, "bev3d_evalflag_0.txt")
            ):
                time.sleep(10)
            interval_name = "-".join(
                [
                    _.translate({ord(i): None for i in "()"})
                    for _ in interval.split(",")
                ]
            )
            npy_files = [
                os.path.join(
                    npy_dir,
                    f"matched_dict_{eval_mode}_"
                    + f"interval{interval_name}_cid{cid}_rankall.npy",
                ),
            ]
        else:
            npy_files = []
            if isinstance(npy_path_pattern, list):
                for pattern_i in npy_path_pattern:
                    npy_files += glob.glob(url_to_local_path(pattern_i))
            else:
                npy_files = glob.glob(url_to_local_path(npy_path_pattern))
        assert npy_files
        for file in npy_files:
            rank_match_instances = np.load(
                file,
                allow_pickle=True,
            ).item()
            exist_frames = set()
            keep_ids = []
            meta = rank_match_instances["meta"]
            # remove the duplicate data caused by multi GPUs
            for index, meta_i in enumerate(meta):
                if (
                    str(meta_i["timestamp"]) + meta_i["pack_dir"]
                    not in exist_frames
                ):
                    exist_frames.add(
                        str(meta_i["timestamp"]) + meta_i["pack_dir"]
                    )
                    keep_ids.append(index)
            for k, v in rank_match_instances.items():
                v = [v[i] for i in keep_ids]
                all_match_instances[k].extend(v)
        return all_match_instances

    @staticmethod
    def global_loc(loc, ego_pose):
        # covert instance loc to global vcs
        ego_pose = ego_pose.reshape((1, 4, 4))
        padded_loc = np.concatenate([loc, np.ones_like(loc[:, [0]])], -1)
        global_loc = ego_pose @ padded_loc[:, :, None]
        return global_loc[:, :3, 0]

    @staticmethod
    def global_yaw(yaw, ego_pose):
        # covert instance yaw to global vcs
        ego_pose = ego_pose.reshape((1, 4, 4))
        vcs_yaw_vec = np.stack(
            [
                np.cos(yaw),
                np.sin(yaw),
                np.zeros_like(yaw),
            ],
            -1,
        )[:, :, None]
        global_yaw_vec = ego_pose[:, :3, :3] @ vcs_yaw_vec
        global_yaw = np.arctan2(
            global_yaw_vec[:, 0, 0], global_yaw_vec[:, 1, 0]
        )
        return global_yaw

    def concatenate_instances(self, cid, all_match_instances):
        # concatenate all value into a numpy array
        img_num = len(all_match_instances["meta"])
        gt_tag_masks = {}
        pred_tag_masks = {}
        gt_loc = [np.zeros([1, 3])]
        gt_dim = [np.zeros([1, 3])]
        gt_yaw = [np.zeros(1)]
        pred_loc = [np.zeros([1, 3])]
        pred_dim = [np.zeros([1, 3])]
        pred_yaw = [np.zeros(1)]
        pred_score = [np.zeros(1)]
        gt_cls = [np.zeros(1)]
        pred_cls = [np.zeros(1)]
        gt_occlusion = [np.zeros(1)]
        pred_occlusion = [np.zeros(1)]
        gt_velo = [np.zeros([1, 3])]
        pred_velo = [np.zeros([1, 3])]
        global_gt_loc = [np.zeros([1, 3])]
        global_pred_loc = [np.zeros([1, 3])]
        global_gt_yaw = [np.zeros([1])]
        global_pred_yaw = [np.zeros([1])]
        gt_meta = [{"timestamp": np.array([-1]), "pack_dir": ""}]
        pred_meta = [{"timestamp": np.array([-1]), "pack_dir": ""}]
        gt_img2inds = defaultdict(list)
        pred_img2inds = defaultdict(list)
        gt_event2inds = defaultdict(list)
        pred_event2inds = defaultdict(list)
        timeline_ids = [np.array([-1])]
        event_track_ids = [np.array([-1])]
        timestamps = [np.array([0])]
        event_int_ids = {}
        for img_ind in range(img_num):
            gt_info = all_match_instances["gt"][img_ind]
            pred_info = all_match_instances["pred"][img_ind]
            meta = all_match_instances["meta"][img_ind]
            if (meta["pack_dir"], meta["timestamp"].item()) not in gt_img2inds:
                gt_img2inds[
                    (meta["pack_dir"], meta["timestamp"].item())
                ].append(0)
            if (
                meta["pack_dir"],
                meta["timestamp"].item(),
            ) not in pred_img2inds:
                pred_img2inds[
                    (meta["pack_dir"], meta["timestamp"].item())
                ].append(0)
            if meta.get("event_id", ""):
                if (meta["event_id"]) not in gt_event2inds:
                    gt_event2inds[meta["event_id"]].append(0)
                if (meta["event_id"]) not in pred_event2inds:
                    pred_event2inds[meta["event_id"]].append(0)
                if meta["event_id"] not in event_int_ids:
                    event_int_ids[meta["event_id"]] = len(event_int_ids)
            ego_pose = np.array(meta.get("ego_pose", []))
            if gt_info:
                gt_img2inds[
                    (meta["pack_dir"], meta["timestamp"].item())
                ].extend(
                    list(
                        range(
                            len(gt_meta),
                            len(gt_meta) + len(gt_info["vcs_loc_"]),
                        )
                    )
                )
                if "event_id" in meta:
                    gt_event2inds[meta["event_id"]].extend(
                        list(
                            range(
                                len(gt_meta),
                                len(gt_meta) + len(gt_info["vcs_loc_"]),
                            )
                        )
                    )
                gt_loc += [gt_info["vcs_loc_"]]
                gt_dim += [gt_info["vcs_dim_"]]
                gt_yaw += [gt_info["vcs_rot_z_"]]
                gt_cls += [gt_info["vcs_cls_"]]
                gt_meta += [meta] * len(gt_info["vcs_loc_"])
                timestamps += [meta["timestamp"]] * len(gt_info["vcs_loc_"])
                if self.eval_stability:
                    if "event_id" in meta and "ego_pose" in meta:
                        global_loc = self.global_loc(gt_loc[-1], ego_pose)
                        global_yaw = self.global_yaw(gt_yaw[-1], ego_pose)
                        global_gt_loc += [global_loc]
                        global_gt_yaw += [global_yaw]
                        for obj_ind in range(len(gt_info["vcs_loc_"])):
                            timeline_ids += [
                                np.array(
                                    [
                                        int(
                                            f'{event_int_ids[meta["event_id"]]}{gt_info["obj_idxes"][obj_ind].item()}{meta["timestamp"].item()}'  # noqa
                                        )
                                    ]
                                )
                            ]
                            event_track_ids += [
                                np.array(
                                    [
                                        int(
                                            f'{event_int_ids[meta["event_id"]]}{gt_info["obj_idxes"][obj_ind].item()}'  # noqa
                                        )
                                    ]
                                )
                            ]
                    else:
                        timeline_ids += [np.array([-1])] * len(
                            gt_info["vcs_loc_"]
                        )
                        event_track_ids += [np.array([-1])] * len(
                            gt_info["vcs_loc_"]
                        )
                        global_gt_loc += [gt_loc[-1]]
                        global_gt_yaw += [gt_yaw[-1]]
                if self.eval_occlusion:
                    gt_occlusion += [gt_info["vcs_occlusion_"]]
                if self.eval_velo:
                    gt_velo += [gt_info["vcs_velocities"]]
                depth_wise_mask = gt_info["depth_wise"]
                for dep_interval in self.all_depth_intervals:
                    if dep_interval not in gt_tag_masks:
                        gt_tag_masks[dep_interval] = [np.zeros(1) - 99]
                    gt_tag_masks[dep_interval] += [
                        depth_wise_mask[dep_interval]
                    ]
                tag_mask = gt_info["taggers"]
                for tagger_name in tag_mask:
                    if tagger_name not in gt_tag_masks:
                        gt_tag_masks[tagger_name] = [np.zeros(1) - 99]
                    gt_tag_masks[tagger_name] += [tag_mask[tagger_name]]

            if pred_info:
                pred_img2inds[
                    (meta["pack_dir"], meta["timestamp"].item())
                ].extend(
                    list(
                        range(
                            len(pred_meta),
                            len(pred_meta) + len(pred_info["bev3d_dim"]),
                        )
                    )
                )
                if "event_id" in meta:
                    pred_event2inds[meta["event_id"]].extend(
                        list(
                            range(
                                len(pred_meta),
                                len(pred_meta) + len(pred_info["bev3d_dim"]),
                            )
                        )
                    )
                pred_loc += [
                    np.concatenate(
                        [
                            pred_info["bev3d_ct"],
                            pred_info["bev3d_loc_z"][:, None],
                        ],
                        axis=1,
                    )
                ]
                pred_dim += [pred_info["bev3d_dim"]]
                pred_yaw += [pred_info["bev3d_rot"]]
                pred_cls += [pred_info["bev3d_cls_id"]]
                pred_score += [pred_info["bev3d_score"]]
                pred_meta += [meta] * len(pred_info["bev3d_dim"])
                if self.eval_stability:
                    if "event_id" in meta and "ego_pose" in meta:
                        global_loc = self.global_loc(pred_loc[-1], ego_pose)
                        global_yaw = self.global_yaw(pred_yaw[-1], ego_pose)
                        global_pred_loc += [global_loc]
                        global_pred_yaw += [global_yaw]
                    else:
                        global_pred_loc += [pred_loc[-1]]
                        global_pred_yaw += [pred_yaw[-1]]
                if self.eval_occlusion:
                    pred_occlusion += [pred_info["bev3d_occlusion_id"]]
                if self.eval_velo:
                    pred_velo += [pred_info["bev3d_velocities"]]

                depth_wise_mask = pred_info["depth_wise"]
                for dep_interval in self.all_depth_intervals:
                    if dep_interval not in pred_tag_masks:
                        pred_tag_masks[dep_interval] = [np.zeros(1) - 99]
                    pred_tag_masks[dep_interval] += [
                        depth_wise_mask[dep_interval]
                    ]
                tag_mask = pred_info["taggers"]
                for tagger_name in tag_mask:
                    if tagger_name not in pred_tag_masks:
                        pred_tag_masks[tagger_name] = [np.zeros(1) - 99]
                    pred_tag_masks[tagger_name] += [tag_mask[tagger_name]]
        if not gt_tag_masks and not pred_tag_masks:
            for dep_interval in self.all_depth_intervals:
                gt_tag_masks[dep_interval] = [np.zeros(1) - 99]
                pred_tag_masks[dep_interval] = [np.zeros(1) - 99]
            for tagger_name in self.taggers[cid]:
                gt_tag_masks[tagger_name] = [np.zeros(1) - 99]
                pred_tag_masks[tagger_name] = [np.zeros(1) - 99]

        gt_loc = np.concatenate(gt_loc)
        gt_dim = np.concatenate(gt_dim)
        gt_yaw = np.concatenate(gt_yaw)
        gt_cls = np.concatenate(gt_cls)
        pred_loc = np.concatenate(pred_loc)
        pred_dim = np.concatenate(pred_dim)
        pred_yaw = np.concatenate(pred_yaw)
        pred_cls = np.concatenate(pred_cls)
        pred_score = np.concatenate(pred_score)
        if self.eval_stability:
            timeline_ids = np.concatenate(timeline_ids)
            event_track_ids = np.concatenate(event_track_ids)
            timestamps = np.concatenate(timestamps)
            global_pred_loc = np.concatenate(global_pred_loc)
            global_pred_yaw = np.concatenate(global_pred_yaw)
            global_gt_loc = np.concatenate(global_gt_loc)
            global_gt_yaw = np.concatenate(global_gt_yaw)
        assert len(gt_meta) == len(gt_loc)
        assert len(pred_meta) == len(pred_loc)
        gt_values = {
            "gt_loc": gt_loc,
            "gt_dim": gt_dim,
            "gt_yaw": gt_yaw,
            "gt_cls": gt_cls,
            "meta": np.array(gt_meta),
            "dict_contents": {
                "img2inds": gt_img2inds,
                "event2inds": gt_event2inds,
            },
        }
        pred_values = {
            "pred_loc": pred_loc,
            "pred_dim": pred_dim,
            "pred_yaw": pred_yaw,
            "pred_cls": pred_cls,
            "pred_score": pred_score,
            "meta": np.array(pred_meta),
            "dict_contents": {
                "img2inds": pred_img2inds,
                "event2inds": pred_event2inds,
            },
        }
        for tag in gt_tag_masks:
            gt_tag_masks[tag] = np.concatenate(gt_tag_masks[tag])
            if tag not in pred_tag_masks:
                pred_tag_masks[tag] = [np.zeros(1) - 99]
        for tag in pred_tag_masks:
            pred_tag_masks[tag] = np.concatenate(pred_tag_masks[tag])
            if tag not in gt_tag_masks:
                gt_tag_masks[tag] = np.zeros(1) - 99
        if self.eval_stability:
            gt_values["timeline_ids"] = timeline_ids
            gt_values["event_track_ids"] = event_track_ids
            gt_values["timestamps"] = timestamps
            gt_values["global_gt_loc"] = global_gt_loc
            gt_values["global_gt_yaw"] = global_gt_yaw
            pred_values["global_pred_loc"] = global_pred_loc
            pred_values["global_pred_yaw"] = global_pred_yaw

        if self.eval_occlusion:
            gt_occlusion = np.concatenate(gt_occlusion)
            pred_occlusion = np.concatenate(pred_occlusion)
            gt_values["gt_occlusion"] = gt_occlusion
            pred_values["pred_occlusion"] = pred_occlusion
        if self.eval_velo:
            gt_velo = np.concatenate(gt_velo)
            pred_velo = np.concatenate(pred_velo)
            gt_values["gt_velo"] = gt_velo
            pred_values["pred_velo"] = pred_velo

        return gt_values, pred_values, gt_tag_masks, pred_tag_masks

    @staticmethod
    def cal_ap(gt_count, scores, tp_masks, fp_masks):
        # cal ap
        arginds = np.argsort(-scores)
        scores = scores[arginds]
        tp_masks = tp_masks[arginds]
        fp_masks = fp_masks[arginds]
        det_tp = np.cumsum(tp_masks)
        det_fp = np.cumsum(fp_masks)
        cus_recall = det_tp / (gt_count + 1e-6)
        cus_precision = det_tp / (det_tp + det_fp + 1e-6)
        ap = calap(cus_recall, cus_precision)
        if ap > 1.0:
            ap = 1.0
        return float(ap)

    def cal_error(
        self,
        pred_matched_mask,
        pred_values,
        gt_matched_mask,
        gt_values,
    ):
        # cal the common error metrics
        pred_loc = pred_values["pred_loc"][pred_matched_mask]
        pred_dim = pred_values["pred_dim"][pred_matched_mask]
        pred_yaw = pred_values["pred_yaw"][pred_matched_mask]
        pred_cls = pred_values["pred_cls"][pred_matched_mask]
        pred_box = np.concatenate([pred_loc, pred_dim, pred_yaw[:, None]], -1)
        pred_yaw = np.rad2deg(pred_yaw) % 360.0

        gt_loc = gt_values["gt_loc"][gt_matched_mask]
        gt_dim = gt_values["gt_dim"][gt_matched_mask]
        gt_yaw = gt_values["gt_yaw"][gt_matched_mask]
        gt_cls = gt_values["gt_cls"][gt_matched_mask]
        gt_box = np.concatenate([gt_loc, gt_dim, gt_yaw[:, None]], -1)
        gt_yaw = np.rad2deg(gt_yaw) % 360.0

        if "pred_occlusion" in pred_values and "gt_occlusion" in gt_values:
            pred_occlusion = pred_values["pred_occlusion"][pred_matched_mask]
            gt_occlusion = gt_values["gt_occlusion"][gt_matched_mask]
        if "pred_velo" in pred_values and "gt_velo" in gt_values:
            pred_velo = pred_values["pred_velo"][pred_matched_mask]
            gt_velo = gt_values["gt_velo"][gt_matched_mask]

        dx = np.abs(pred_loc[:, 0] - gt_loc[:, 0])
        dy = np.abs(pred_loc[:, 1] - gt_loc[:, 1])
        dxy = (dx ** 2 + dy ** 2) ** 0.5
        dw = np.abs(pred_dim[:, 1] - gt_dim[:, 1])
        dl = np.abs(pred_dim[:, 2] - gt_dim[:, 2])
        dh = np.abs(pred_dim[:, 0] - gt_dim[:, 0])
        dxp = dx / (np.abs(gt_loc[:, 0]) + 1.0)
        dyp = dy / (np.abs(gt_loc[:, 1]) + 1.0)
        dxyp = dxy / (
            np.abs(gt_loc[:, 0] ** 2 + gt_loc[:, 1] ** 2) ** 0.5 + 1.0
        )
        dxy_10p_error = dxyp <= 0.1
        dz = np.abs(pred_loc[:, 2] - gt_loc[:, 2])
        dist = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        rdist = dist / (np.linalg.norm(gt_loc, axis=1) + 1.0)
        dwp = dw / (np.abs(gt_dim[:, 1]) + 1e-6)
        dlp = dl / (np.abs(gt_dim[:, 2]) + 1e-6)
        dhp = dh / (np.abs(gt_dim[:, 0]) + 1e-6)
        rsize = (dwp + dlp + dhp) / 3.0
        abs_rot = np.abs(gt_yaw - pred_yaw)
        drot = np.minimum(abs_rot, 360.0 - abs_rot)

        # find min(abs(y)) GT corner point
        gt_corners = get_3dboxcorner_in_vcs_numpy(gt_box)[:, :, :2]
        gt_corners_y = np.abs(gt_corners[:, :, 1])
        gt_corners_index = gt_corners_y.argmin(axis=1)
        gt_cipv_corners_index = np.abs(gt_corners[:, :, 0]).argmin(axis=1)
        gt_cutin_corner = gt_corners[
            np.arange(len(gt_corners)), gt_corners_index
        ]
        gt_cipv_corner = gt_corners[
            np.arange(len(gt_corners)), gt_cipv_corners_index
        ]
        # find a pred_corner nearest to the gt_cutin_corner
        pred_corners = get_3dboxcorner_in_vcs_numpy(pred_box)[:, :, :2]
        xy_dist = np.linalg.norm(
            gt_cutin_corner[:, None, :] - pred_corners, axis=2
        )
        xy_cipv_dist = np.linalg.norm(
            gt_cipv_corner[:, None, :] - pred_corners, axis=2
        )
        pred_corners_index = xy_dist.argmin(axis=1)
        pred_cipv_corners_index = xy_cipv_dist.argmin(axis=1)
        pred_cutin_corner = pred_corners[
            np.arange(len(pred_corners)), pred_corners_index, :2
        ]
        pred_cipv_corner = pred_corners[
            np.arange(len(pred_corners)), pred_cipv_corners_index, :2
        ]
        # cal the distance between the GT and pred cutin corner points
        cutin_corner_dxy = np.linalg.norm(
            gt_cutin_corner - pred_cutin_corner, axis=1
        )
        cutin_corner_dx = np.abs(
            gt_cutin_corner[:, 0] - pred_cutin_corner[:, 0]
        )
        cutin_corner_dy = np.abs(
            gt_cutin_corner[:, 1] - pred_cutin_corner[:, 1]
        )
        cipv_corner_dxy = np.linalg.norm(
            gt_cipv_corner - pred_cipv_corner, axis=1
        )
        cipv_corner_dx = np.abs(gt_cipv_corner[:, 0] - pred_cipv_corner[:, 0])
        cipv_corner_dy = np.abs(gt_cipv_corner[:, 1] - pred_cipv_corner[:, 1])

        error = {
            "dx": dx,
            "dy": dy,
            "dxy": dxy,
            "dw": dw,
            "dl": dl,
            "dh": dh,
            "dxp": dxp,
            "dyp": dyp,
            "dxyp": dxyp,
            "dwp": dwp,
            "dlp": dlp,
            "dhp": dhp,
            "drot": drot,
            "dxy_10p_error": dxy_10p_error,
            "rdist": rdist,
            "rsize": rsize,
            "cc_dxy": cutin_corner_dxy,
            "cc_dx": cutin_corner_dx,
            "cc_dy": cutin_corner_dy,
            "cipv_dxy": cipv_corner_dxy,
            "cipv_dx": cipv_corner_dx,
            "cipv_dy": cipv_corner_dy,
        }
        if self.eval_cls:
            if (gt_cls == self.cls_ignore_id).all():
                cls = np.zeros_like(gt_cls)
            else:
                cls = (pred_cls == gt_cls).astype(np.float32)
            error.update(cls=cls)
        if self.eval_occlusion:
            if (gt_occlusion == self.occlusion_ignore_id).all():
                occlusion = np.zeros_like(gt_occlusion)
            else:
                occlusion = (pred_occlusion == gt_occlusion).astype(np.float32)
            error.update(occl=occlusion)
        if "pred_velo" in pred_values and "gt_velo" in gt_values:
            error.update(
                {
                    "dvx": np.abs(pred_velo[:, 0] - gt_velo[:, 0]),
                    "dvy": np.abs(pred_velo[:, 1] - gt_velo[:, 1]),
                    "drot_evs": np.minimum(drot, 180 - drot),
                }
            )
        return error

    @staticmethod
    def cal_stability(
        pred_matched_mask,
        pred_values,
        gt_matched_mask,
        gt_values,
        score_threshold=0.0,
    ):
        # eval stability by fake velocities in global vcs.
        pred_loc = pred_values["global_pred_loc"][pred_matched_mask]
        pred_yaw = pred_values["global_pred_yaw"][pred_matched_mask]
        pred_score = pred_values["pred_score"][pred_matched_mask]
        score_mask = pred_score >= score_threshold
        pred_yaw = np.rad2deg(pred_yaw) % 360.0

        gt_loc = gt_values["global_gt_loc"][gt_matched_mask]
        gt_yaw = gt_values["global_gt_yaw"][gt_matched_mask]
        gt_yaw = np.rad2deg(gt_yaw) % 360.0

        timeline_ids = gt_values["timeline_ids"][gt_matched_mask]
        event_track_ids = gt_values["event_track_ids"][gt_matched_mask]
        timestamps = gt_values["timestamps"][gt_matched_mask]
        vaild_timeline_id = (timeline_ids != -1) * score_mask
        timeline_ids = timeline_ids[vaild_timeline_id]
        f_dv = float("inf")
        f_dr = float("inf")
        if len(timeline_ids) > 1:
            time_sort_inds = np.argsort(timeline_ids)

            event_track_ids = event_track_ids[vaild_timeline_id][
                time_sort_inds
            ]
            time_pred_loc = pred_loc[vaild_timeline_id][time_sort_inds]
            time_gt_loc = gt_loc[vaild_timeline_id][time_sort_inds]
            time_pred_yaw = pred_yaw[vaild_timeline_id][time_sort_inds]
            time_gt_yaw = gt_yaw[vaild_timeline_id][time_sort_inds]
            timestamps = timestamps[vaild_timeline_id][time_sort_inds]

            diff_times = (timestamps[1:] - timestamps[:-1]) / 1000.0 + 1e-4
            time_pred_velo = (
                time_pred_loc[1:] - time_pred_loc[:-1]
            ) / diff_times[:, None]
            pred_yaw_diff = np.abs(time_pred_yaw[1:] - time_pred_yaw[:-1])
            time_pred_yaw_rate = (
                np.minimum(pred_yaw_diff, 360.0 - pred_yaw_diff)
            ) / diff_times
            time_gt_velo = (time_gt_loc[1:] - time_gt_loc[:-1]) / diff_times[
                :, None
            ]
            gt_yaw_diff = np.abs(time_gt_yaw[1:] - time_gt_yaw[:-1])
            time_gt_yaw_rate = (
                np.minimum(gt_yaw_diff, 360.0 - gt_yaw_diff)
            ) / diff_times
            vaild_mask = event_track_ids[1:] == event_track_ids[:-1]
            vaild_mask = vaild_mask * (event_track_ids[:-1] != -1)

            time_pred_velo = time_pred_velo[vaild_mask]
            time_pred_yaw_rate = time_pred_yaw_rate[vaild_mask]
            time_gt_velo = time_gt_velo[vaild_mask]
            time_gt_yaw_rate = time_gt_yaw_rate[vaild_mask]
            if vaild_mask.any():
                f_dv = (
                    np.linalg.norm(
                        time_pred_velo[:, :2] - time_gt_velo[:, :2], -1
                    )
                    ** 0.5
                ).mean()  # noqa
                f_dr = np.abs(time_pred_yaw_rate - time_gt_yaw_rate).mean()

        error = {}
        error["f_dv"] = f_dv
        error["f_dr"] = f_dr
        return error

    @staticmethod
    def get_overall_mask(all_depth_intervals, gt_tag_masks, pred_tag_masks):
        # concat the mask from all distance range.
        gt_mask = None
        pred_mask = None
        total = 0
        for dep_interval in all_depth_intervals:
            if gt_mask is None:
                gt_mask = gt_tag_masks[dep_interval].copy()
            else:
                tag_mask = gt_tag_masks[dep_interval]
                mask = (tag_mask >= 0) * (gt_mask == -1)
                gt_mask[mask] = tag_mask[mask]
            if pred_mask is None:
                pred_mask = pred_tag_masks[dep_interval].copy()
                total += (pred_mask >= 0).sum()
            else:
                tag_mask = pred_tag_masks[dep_interval]
                total += (tag_mask >= 0).sum()
                mask = (tag_mask >= 0) * (pred_mask == -1)
                pred_mask[mask] = tag_mask[mask]
        assert (
            total == (pred_mask >= 0).sum()
        ), f"{total}, {(pred_mask>=0).sum()}"
        return gt_mask, pred_mask

    def tabel_header(self, eval_mode, interval):
        # init header
        row_header = [""]
        if "F1" not in self.pr_metric[eval_mode]:
            new_pr_metric = list(self.pr_metric[eval_mode])
            new_pr_metric.insert(3, "F1")
            self.pr_metric[eval_mode] = tuple(new_pr_metric)
        if "Pred" not in self.pr_metric[eval_mode]:
            new_pr_metric = list(self.pr_metric[eval_mode])
            new_pr_metric.insert(0, "Pred")
            self.pr_metric[eval_mode] = tuple(new_pr_metric)
        if "GT" not in self.pr_metric[eval_mode]:
            new_pr_metric = list(self.pr_metric[eval_mode])
            new_pr_metric.insert(0, "GT")
            self.pr_metric[eval_mode] = tuple(new_pr_metric)
        if self.verbose:
            if interval == "(0,1)":
                for metric_name in self.pr_metric[eval_mode]:
                    row_header += [metric_name]
            else:
                for metric_name in self.vis_metric:
                    row_header += [metric_name]
        if self.ap_metric is not None:
            for ap_metric_name in self.ap_metric[eval_mode]:
                row_header += [ap_metric_name]
                if row_header[-1] in ["AP_Dist", "APL_Dist"]:
                    row_header[-1] = row_header[-1][:-5]

        for metric in self.metrics:
            row_header += [metric]
        return row_header

    def build_table(self, eval_mode, interval, mertic_results, title):
        row_header = self.tabel_header(eval_mode, interval)
        # build rows
        rows = [[0 for _ in row_header] for __ in mertic_results]
        metric_dict = {}
        for row_ind, row_name in enumerate(mertic_results):
            rows[row_ind][0] = row_name.ljust(15)
            for metric_ind, metric_name in enumerate(row_header):
                if (
                    row_name in mertic_results
                    and metric_name in mertic_results[row_name]
                ):
                    rows[row_ind][metric_ind] = mertic_results[row_name][
                        metric_name
                    ]
                    if metric_name not in metric_dict:
                        metric_dict[metric_name] = {}
                    metric_dict[metric_name][row_name] = rows[row_ind][
                        metric_ind
                    ]
        float_format = self.tabel_format(row_header, rows)
        tb = PrettyTable()
        tb.field_names = row_header
        tb.title = title
        tb._float_format = float_format
        tb.add_rows(rows)
        return tb, metric_dict

    def tabel_format(self, row_header, rows):
        # format
        float_format = {}
        for metric_ind, metric_name in enumerate(row_header):
            if metric_ind == 0:
                continue
            if metric_name in [
                "TP",
                "FP",
                "FN",
                "GT",
                "Pred",
            ]:
                row_header[metric_ind] = "#" + metric_name
                float_format[row_header[metric_ind]] = ".0"
            elif metric_name in [
                "F1",
                "Recall",
                "Precision",
                "Det_Rate",
                "AP",
                "dxp",
                "dyp",
                "dxyp",
                "occlusion",
                "occl",
                "rdist",
                "rsize",
                "cls",
            ]:
                for row_value in rows:
                    row_value[metric_ind] *= 100
                if metric_name == "Recall":
                    metric_name = "Rec"
                if metric_name == "Precision":
                    metric_name = "Prec"
                row_header[metric_ind] = metric_name + "%"
                float_format[row_header[metric_ind]] = ".1"
            elif metric_name in ["drot"]:
                row_header[metric_ind] = metric_name + "°"
                float_format[row_header[metric_ind]] = ".1"
            elif metric_name.split("%")[0] in [
                "dx",
                "dy",
                "dl",
                "dw",
                "cc_dx",
                "cc_dy",
                "cc_dxy",
                "cipv_dx",
                "cipv_dy",
                "cipv_dxy",
            ]:
                for row_value in rows:
                    row_value[metric_ind] *= 100
                row_header[metric_ind] = metric_name + "(cm)"
                float_format[row_header[metric_ind]] = ".1"
            else:
                float_format[metric_name] = ".1"
        return float_format
