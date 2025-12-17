from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from tqdm import tqdm

from hat.metrics.metric_3dv_utils import let_iou_matching, rotate_iou_matching

EPSILON = 1e-8
IGNORE_VALUE = 999
__all__ = [
    "dict_select",
    "MetricTrajpred",
    "e2e_dynamic_bbox_eval",
    "decode_psc_rot_numpy",
]


def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))


def decode_psc_rot_numpy(
    bev3d_rot, coef_sin, coef_cos, N_steps_PSC_rot=3, use_sigmoid=False
) -> np.ndarray:
    """Numpy implement of `decode_psc_rot` in BEV3Decoder."""
    if use_sigmoid:
        bev3d_rot = sigmoid_numpy(bev3d_rot) * 2 - 1
    phase_sin = np.sum(
        bev3d_rot[..., :N_steps_PSC_rot] * coef_sin,
        axis=-1,
        keepdims=False,
    )
    phase_cos = np.sum(
        bev3d_rot[..., :N_steps_PSC_rot] * coef_cos,
        axis=-1,
        keepdims=False,
    )
    phase_mod = phase_cos ** 2 + phase_sin ** 2
    phase = -np.arctan2(phase_sin, phase_cos)

    phase[phase_mod < 0.0001] *= 0
    return phase


def cat_all_keys(input_list):
    if len(input_list) == 0:
        return []
    elif len(input_list) == 1:
        return input_list[0]
    else:
        outdata = {}
        for key in input_list[0].keys():
            # if isinstance(input_list[0][key], torch.Tensor):
            uu = [tmp[key] for tmp in input_list]
            outdata[key] = np.concatenate(uu, 0)
        return outdata


def dict_select(d: dict, index):
    new_instances = {}
    for k, v in d.items():
        new_instances[k] = v[index]
    return new_instances


def e2e_dynamic_bbox_eval(
    det_res: Mapping,
    annotation: Mapping,
    dep_thresh: Optional[Sequence[str]],
    score_threshold: float,
    iou_threshold: float,
    gt_max_depth: float,
    eval_vcs_range: Optional[Sequence[float]],
    enable_ignore: bool,
    vis_intervals: Optional[Sequence[str]],
    ego_ignore_range: Optional[Sequence[float]],
    time_delta: float,
    eval_mode: str = "bev_iou",
    let_iou_param: Optional[Mapping[str, float]] = None,
) -> Mapping:
    """Eval the metric between GT and pred boxes of bev3d.

    Args:
        det_res: the predict 3d boxes info.
        annotation: the ground truth 3d boxes info.
        dep_thresh: Depth range to
            validation.
        score_threshold: Threshold for score.
        iou_threshold: Threshold for IoU.
        gt_max_depth: Max depth for gts.
        eval_vcs_range: Max vcs
            (bottom, right, top, left) for gts & preds.
        enable_ignore: Whether to use ignore_mask.
        ego_ignore_range: Ego range
            (bottom, right, top, left) to be ignored,(-0.6, -0.5, 2.0, 0.5)
            recommended based on the minimum tire diameter, wheelbase and track
        vis_intervals: Piecewise interval of visibility.
        time_delta: Time interval between two frames.
        eval_mode: the way used to count the metric, in ["bev_iou",
            "let_iou"]
        let_iou_param: the parameters used to calculate the let_iou metric.
    Returns:
        Dict contains the results.
    """
    assert eval_mode in ["bev_iou", "let_iou"]

    if (
        len(annotation["annotations"]) > 0
        and (len(det_res) > 0)
        and ("vcs_velocities" in annotation["annotations"][0])
        and ("bev3d_velocities" in det_res[0])
    ):
        eval_velo = True
    else:
        eval_velo = False

    timestamp_count = 0
    total_timestamp_count = len(annotation["timestamps"])

    all_dets = defaultdict(list)
    all_gts = defaultdict(list)
    for det in det_res:
        if det["bev3d_score"] < score_threshold:
            continue
        if eval_vcs_range is not None:
            if not (
                eval_vcs_range[0] < det["bev3d_ct"][0] < eval_vcs_range[2]
                and eval_vcs_range[1] < det["bev3d_ct"][1] < eval_vcs_range[-1]
            ):
                continue
        if ego_ignore_range is not None:
            if (
                ego_ignore_range[0]
                <= det["bev3d_ct"][0]
                <= ego_ignore_range[2]
                and ego_ignore_range[1]
                <= det["bev3d_ct"][1]
                <= ego_ignore_range[-1]
            ):
                continue
        all_dets[det["timestamp"]].append(det)

    for gt in annotation["annotations"]:
        gt_depth = abs(gt["vcs_loc_"][0])  # vcs: abs(x)=depth

        if eval_vcs_range is not None:
            if not (
                eval_vcs_range[0] < gt["vcs_loc_"][0] < eval_vcs_range[2]
                and eval_vcs_range[1] < gt["vcs_loc_"][1] < eval_vcs_range[-1]
            ):
                continue
        else:
            if gt_depth > gt_max_depth:
                continue
        if ego_ignore_range is not None:
            if (
                ego_ignore_range[0] <= gt["vcs_loc_"][0] <= ego_ignore_range[2]
                and ego_ignore_range[1]
                <= gt["vcs_loc_"][1]
                <= ego_ignore_range[-1]
            ):
                continue
        all_gts[gt["timestamp"]].append(gt)

    all_metric = [
        "dx",
        "dxp",
        "dy",
        "dyp",
        "dxy",
        "dxyp",
        "dw",
        "dwp",
        "dl",
        "dlp",
        "dh",
        "dhp",
        "drot",
        "drot_evs",
        "dvx",
        "dvy",
        "dvx_0s_evs",
        "dvx_3s_evs",
        "dvy_0s_evs",
        "dvy_3s_evs",
    ]
    metrics = {}
    for k in all_metric:
        metrics[k] = {}
        for vi in vis_intervals:
            metrics[k][vi] = [[] for _ in range(len(dep_thresh) + 1)]

    gt_matched = np.zeros((len(vis_intervals), len(dep_thresh) + 1))
    gt_missed = np.zeros((len(vis_intervals), len(dep_thresh) + 1))
    redundant_det = np.zeros(len(dep_thresh) + 1)
    if eval_mode == "let_iou":
        gt_al = np.zeros((len(vis_intervals), len(dep_thresh) + 1))

    num_gt = 0
    det_tp_mask = []
    det_gt_mask = []
    det_tp_pred_loc = []
    all_scores = []
    matched_dict_in_batch = []
    if eval_mode == "let_iou":
        det_tp_let_al_mask = []

    for timestamp in annotation["timestamps"]:
        det_bbox3d, det_scores, det_locs, det_yaw = [], [], [], []
        gt_bbox3d, gt_locs, gt_yaw, gt_ignore, gt_visible = [], [], [], [], []

        # fetch 3d ground truth box
        for gt in all_gts[timestamp]:
            dim = gt["vcs_dim_"]
            yaw = gt["vcs_rot_z_"]
            loc = gt["vcs_loc_"]
            bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
            gt_bbox3d.append(bbox3d)
            gt_locs.append(gt["vcs_loc_"])
            gt_yaw.append(gt["vcs_rot_z_"])
            gt_visible.append(gt["vcs_visible_"])
            if enable_ignore:
                gt_ignore.append(gt["vcs_ignore_"])

        for det in all_dets[timestamp]:
            dim = det["bev3d_dim"]
            yaw = det["bev3d_rot"]
            loc = det["bev3d_ct"]
            # [x, y, l, w, -yaw], -yaw means change the yaw from \
            # counterclockwise -> clockwise
            bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
            det_bbox3d.append(bbox3d)
            assert det["bev3d_score"] >= 0
            det_scores.append(det["bev3d_score"])
            det_locs.append(det["bev3d_ct"])
            det_yaw.append(det["bev3d_rot"])

        det_locs = np.array(det_locs)
        gt_locs = np.array(gt_locs)
        gt_visible = np.array(gt_visible)
        if len(all_gts[timestamp]) == 0:
            if det_locs.any():
                pred_dep_thresh_inds = np.sum(
                    abs(det_locs[:, 0:1]) > dep_thresh, axis=-1
                )
                pred_dep_inds, pred_cnts = np.unique(
                    pred_dep_thresh_inds, return_counts=True
                )
                redundant_det[pred_dep_inds] += pred_cnts
            continue
        else:
            timestamp_count += 1

        if len(all_dets[timestamp]) == 0:
            for vis_idx, interval in enumerate(vis_intervals):
                vis_lthr, vis_rthr = [
                    float(_.translate({ord(i): None for i in "()"}))
                    for _ in interval.split(",")
                ]
                visib_gt_ind = (gt_visible >= vis_lthr) * (
                    gt_visible < vis_rthr
                )

                if enable_ignore:
                    valid_gt_ind = np.array(gt_ignore) == 0
                    gt_dep_thresh_inds = np.sum(
                        abs(gt_locs[visib_gt_ind * valid_gt_ind][:, 0:1])
                        > dep_thresh,
                        axis=-1,
                    )
                else:
                    gt_dep_thresh_inds = np.sum(
                        abs(gt_locs[visib_gt_ind][:, 0:1]) > dep_thresh,
                        axis=-1,
                    )

                gt_dep_inds, gt_cnts = np.unique(
                    gt_dep_thresh_inds, return_counts=True
                )
                gt_missed[vis_idx][gt_dep_inds] += gt_cnts
            # bev3d ap metric bug fix
            gt_det_loc = gt_locs.copy()
            if enable_ignore:
                gt_det_loc = gt_det_loc[np.invert(gt_ignore)]
            num_gt += len(gt_bbox3d) - sum(gt_ignore)
            det_gt_mask += gt_det_loc.tolist()
            continue

        det_bbox3d = np.array(det_bbox3d)
        det_scores = np.array(det_scores)
        gt_bbox3d = np.array(gt_bbox3d)

        assert det_bbox3d.shape[0] == det_scores.shape[0]

        if eval_mode == "bev_iou":
            (matched_dict, redundant, det_ignored_mask) = rotate_iou_matching(
                det_bbox3d,
                det_locs,
                gt_bbox3d,
                gt_locs,
                det_scores,
                iou_threshold,
                gt_ignore,
            )
        elif eval_mode == "let_iou":
            (matched_dict, redundant, det_ignored_mask) = let_iou_matching(
                det_bbox3d,
                gt_bbox3d,
                det_scores,
                iou_threshold,
                gt_ignore,
                let_iou_param,
            )
        else:
            raise NotImplementedError
        matched_dict_in_batch.append(matched_dict)
        redundant_det_mask = np.zeros(det_locs.shape[0], dtype=bool)
        for ind in redundant:
            redundant_det_mask[ind] = 1
        redundant_det_dep = np.sum(
            abs(det_locs[redundant_det_mask, 0:1]) > dep_thresh, axis=-1
        )
        redundant_det_dep_inds, redundant_det_dep_cnts = np.unique(
            redundant_det_dep, return_counts=True
        )
        redundant_det[redundant_det_dep_inds] += redundant_det_dep_cnts
        all_scores += det_scores[np.invert(det_ignored_mask)].tolist()
        tp = np.ones(len(det_scores), dtype=bool)
        tp[redundant] = 0
        tp = tp[np.invert(det_ignored_mask)]
        tp_det_loc = det_locs.copy()
        tp_det_loc = tp_det_loc[np.invert(det_ignored_mask)]
        gt_det_loc = gt_locs.copy()
        if enable_ignore:
            gt_det_loc = gt_det_loc[np.invert(gt_ignore)]
        num_gt += len(gt_bbox3d) - sum(gt_ignore)
        if eval_mode == "let_iou":
            al = matched_dict["let_al_det"]
            al = al[np.invert(det_ignored_mask)]
            det_tp_let_al_mask += al.tolist()

        det_tp_mask += tp.tolist()
        det_gt_mask += gt_det_loc.tolist()
        det_tp_pred_loc += tp_det_loc.tolist()
        det_assigns = matched_dict["det_assign"]
        inds = np.array(list(range(len(det_assigns))))

        mask = det_assigns != -1
        det_assigns, inds = det_assigns[mask], inds[mask]
        for vis_idx, interval in enumerate(vis_intervals):
            vis_lthr, vis_rthr = [
                float(_.translate({ord(i): None for i in "()"}))
                for _ in interval.split(",")
            ]
            gt_visible_mask = (gt_visible >= vis_lthr) * (
                gt_visible < vis_rthr
            )

            if enable_ignore:
                gt_missed_mask = np.invert(mask) * np.invert(
                    np.array(gt_ignore) == 1
                )
            else:
                gt_missed_mask = np.invert(mask)
            gt_missed_dep_thresh_inds = np.sum(
                abs(gt_locs[gt_visible_mask * gt_missed_mask, 0:1])
                > dep_thresh,
                axis=-1,
            )
            gt_missed_dep_inds, gt_missed_cnts = np.unique(
                gt_missed_dep_thresh_inds, return_counts=True
            )
            gt_missed[vis_idx][gt_missed_dep_inds] += gt_missed_cnts

            if np.sum(mask) == 0:
                continue

            pred_dim = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_dim"]
                    for assign in det_assigns
                ]
            )
            pred_loc = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_ct"]
                    for assign in det_assigns
                ]
            )
            pred_yaw_rad = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_rot"]
                    for assign in det_assigns
                ]
            )
            pred_yaw = np.rad2deg(pred_yaw_rad) % 360.0
            if eval_velo:
                pred_velo = np.array(
                    [
                        all_dets[timestamp][assign]["bev3d_velocities"]
                        for assign in det_assigns
                    ]
                )
                pred_velo_appear_times = np.array(
                    [
                        all_dets[timestamp][assign]["appear_time"]
                        for assign in det_assigns
                    ]
                )
                pred_velo_0s_mask = pred_velo_appear_times == 1
                pred_velo_3s_mask = pred_velo_appear_times <= (3 / time_delta)

            gt_dim = np.array(
                [all_gts[timestamp][ind]["vcs_dim_"] for ind in inds]
            )
            gt_loc = np.array(
                [all_gts[timestamp][ind]["vcs_loc_"] for ind in inds]
            )
            gt_depth = np.array(
                [all_gts[timestamp][ind]["vcs_loc_"][0] for ind in inds]
            )
            gt_yaw_rad = np.array(
                [all_gts[timestamp][ind]["vcs_rot_z_"] for ind in inds]
            )
            gt_vis = np.array(
                [all_gts[timestamp][ind]["vcs_visible_"] for ind in inds]
            )
            gt_yaw = np.rad2deg(gt_yaw_rad) % 360.0
            gt_vis_mask = (gt_vis >= vis_lthr) * (gt_vis < vis_rthr)
            if eval_velo:
                gt_velo = np.array(
                    [all_gts[timestamp][ind]["vcs_velocities"] for ind in inds]
                )

            dep_thresh_inds = np.sum(
                abs(gt_depth[gt_vis_mask][:, np.newaxis]) > dep_thresh, axis=-1
            )
            dep_inds, cnts = np.unique(dep_thresh_inds, return_counts=True)
            gt_matched[vis_idx][dep_inds.tolist()] += cnts

            if eval_mode == "let_iou":
                let_al_gt = matched_dict["let_al_gt"][mask]
                for dep_ind in dep_inds.tolist():
                    gt_al[vis_idx][dep_ind] += let_al_gt[
                        dep_thresh_inds == dep_ind
                    ].sum()

            dx = np.abs(pred_loc[:, 0] - gt_loc[:, 0])
            dy = np.abs(pred_loc[:, 1] - gt_loc[:, 1])
            dxy = (dx ** 2 + dy ** 2) ** 0.5
            dw = np.abs(pred_dim[:, 1] - gt_dim[:, 1])
            dl = np.abs(pred_dim[:, 2] - gt_dim[:, 2])
            dh = np.abs(pred_dim[:, 0] - gt_dim[:, 0])
            if eval_velo:
                assert gt_velo.shape[0] == pred_velo_0s_mask.shape[0]
                assert gt_velo.shape[0] == pred_velo_3s_mask.shape[0]
                dvx = np.abs(pred_velo[:, 0] - gt_velo[:, 0])
                dvy = np.abs(pred_velo[:, 1] - gt_velo[:, 1])
                dvx_0s = dvx[pred_velo_0s_mask]
                dvy_0s = dvy[pred_velo_0s_mask]
                dvx_3s = dvx[pred_velo_3s_mask]
                dvy_3s = dvy[pred_velo_3s_mask]
            else:
                dvx = np.ones(pred_dim.shape[0]) * IGNORE_VALUE
                dvy = np.ones(pred_dim.shape[0]) * IGNORE_VALUE
                dvx_0s = np.ones(pred_dim.shape[0]) * IGNORE_VALUE
                dvx_3s = np.ones(pred_dim.shape[0]) * IGNORE_VALUE
                dvy_0s = np.ones(pred_dim.shape[0]) * IGNORE_VALUE
                dvy_3s = np.ones(pred_dim.shape[0]) * IGNORE_VALUE
                pred_velo_0s_mask = [False] * pred_dim.shape[0]
                pred_velo_3s_mask = [False] * pred_dim.shape[0]

            dxp = dx / np.abs(gt_loc[:, 0])
            dyp = dy / np.abs(gt_loc[:, 1])
            dxyp = dxy / np.abs(gt_loc[:, 0] ** 2 + gt_loc[:, 1] ** 2) ** 0.5
            dxy_10p_error = dxyp <= 0.1
            dwp = dw / np.abs(gt_dim[:, 1])
            dlp = dl / np.abs(gt_dim[:, 2])
            dhp = dh / np.abs(gt_dim[:, 0])
            abs_rot = np.abs(gt_yaw - pred_yaw)
            drot = np.minimum(abs_rot, 360.0 - abs_rot)
            drot_evs = drot.copy()

            res = {
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
                "drot_evs": drot_evs,
                "dxy_10p_error": dxy_10p_error,
                "dvx": dvx,
                "dvy": dvy,
                "dvx_0s_evs": dvx_0s,
                "dvx_3s_evs": dvx_3s,
                "dvy_0s_evs": dvy_0s,
                "dvy_3s_evs": dvy_3s,
            }
            dep_thresh_inds = np.sum(
                abs(gt_depth[:, np.newaxis]) > dep_thresh, axis=-1
            )
            for key, val in metrics.items():
                for dep_ind in dep_inds:
                    dep_mask = dep_thresh_inds == dep_ind
                    if key == "dvx_0s_evs" or key == "dvy_0s_evs":
                        dep_mask_key = dep_mask[pred_velo_0s_mask]
                        gt_vis_mask_key = gt_vis_mask[pred_velo_0s_mask]
                    elif key == "dvx_3s_evs" or key == "dvy_3s_evs":
                        dep_mask_key = dep_mask[pred_velo_3s_mask]
                        gt_vis_mask_key = gt_vis_mask[pred_velo_3s_mask]
                    else:
                        dep_mask_key = dep_mask
                        gt_vis_mask_key = gt_vis_mask
                    val[interval][dep_ind] += res[key][
                        dep_mask_key * gt_vis_mask_key
                    ].tolist()

    result_aps = {
        "det_tp_mask": det_tp_mask,
        "det_gt_mask": det_gt_mask,
        "det_tp_pred_loc": det_tp_pred_loc,
        "all_scores": all_scores,
        "num_gt": float(num_gt),
        "matched_dict_in_batch": matched_dict_in_batch,
    }
    metrics["counts"] = {
        "gt_matched": gt_matched,
        "gt_missed": gt_missed,
        "redundant_det": redundant_det,
        "timestamp_count": timestamp_count,
        "total_timestamp_count": total_timestamp_count,
        "result_aps": result_aps,
    }
    if eval_mode == "let_iou":
        metrics["counts"]["result_aps"][
            "det_tp_let_al_mask"
        ] = det_tp_let_al_mask
        metrics["counts"]["gt_al"] = gt_al
    return metrics


class MetricTrajpred:
    """Metric for trajectory prediction.

    Calculate the ADE and FDE for the predicted trajectories.

    Args:
        k_values: The k values for ADE and FDE.
        metric: The metric to calculate. Default: ("ADE", "FDE").
        savepath: The path to save the results. Default: "./".
        is_filter_by_yaw: Whether to filter the predictions by yaw.
        mode: The mode to calculate the metric.
            Default: ("all", "veh", "cyc", "ped").
    """

    def __init__(
        self,
        k_values: tuple = (1, 5),
        metric: tuple = ("ADE", "FDE"),
        savepath: str = "./",
        is_filter_by_yaw: bool = False,
        mode: tuple = ("all", "veh", "cyc", "ped"),
    ):
        self.k_values = k_values
        self.metric = metric
        self.savepath = savepath
        self.is_filter_by_yaw = is_filter_by_yaw

        self.mode = mode

        self.performance = {}
        for mode in self.mode:
            self.initial_key_dicts(mode)

    def initial_key_dicts(self, mode):
        if "ADE" in self.metric:
            for kkk in self.k_values:
                self.performance["{}-ADE{}".format(mode, kkk)] = 0.0
        if "FDE" in self.metric:
            for kkk in self.k_values:
                self.performance["{}-FDE{}".format(mode, kkk)] = 0.0
        self.performance["{}-TP".format(mode)] = 0
        self.performance["{}-FP".format(mode)] = 0
        self.performance["{}-FN".format(mode)] = 0
        self.performance["{}-validTP".format(mode)] = 0

    def clear_key_dicts(self):
        for mode in self.mode:
            self.initial_key_dicts(mode)

    def post_avg_over_obj(self):
        for cur_mode in self.mode:
            num = self.performance["{}-validTP".format(cur_mode)]
            num = 1 if num < 1 else num
            if "ADE" in self.metric:
                for kkk in self.k_values:
                    self.performance["{}-ADE{}".format(cur_mode, kkk)] /= num
            if "FDE" in self.metric:
                for kkk in self.k_values:
                    self.performance["{}-FDE{}".format(cur_mode, kkk)] /= num

    def updata_FDE_ADE(self, gt_trajs, gt_masks, dt_trajs, cur_mode):
        """
        Obtain l2-norm of each point, then mean or minimun it.

        Args:
            gt_trajs:[num_obj, traj_len, 2]
            gt_masks:[num_obj, traj_len, 2]
            dt_trajs:[num_obj, num_modal, traj_len, 2],
                sorted by prob decreasely.

        """
        gt_masks = gt_masks[..., 0]  # [num_obj, traj_len]
        valid_TP = (
            np.sum(gt_masks, axis=-1) > 0
        )  # some masks are all 0,will not calc
        self.performance["{}-validTP".format(cur_mode)] += np.sum(valid_TP)
        gt_trajs = gt_trajs[valid_TP]
        gt_masks = gt_masks[valid_TP]
        dt_trajs = dt_trajs[valid_TP]
        final_index = np.array(
            [np.nonzero(tmp)[0][-1] for tmp in gt_masks], dtype=np.int32
        )
        if len(dt_trajs) > 0:
            for kkk in self.k_values:
                dist = np.linalg.norm(
                    gt_trajs[:, np.newaxis] - dt_trajs[:, :kkk], axis=-1
                )
                dist[np.isnan(dist)] = 0  # [num_obj, num_modal, traj_len]
                if "ADE" in self.metric:
                    tmp = np.sum(dist * gt_masks[:, np.newaxis], axis=-1) / (
                        1e-10 + np.sum(gt_masks[:, np.newaxis], axis=-1)
                    )  # [num_obj, num_modal]
                    self.performance[
                        "{}-ADE{}".format(cur_mode, kkk)
                    ] += np.sum(np.min(tmp, axis=-1))
                if "FDE" in self.metric:
                    tmp = dist[
                        np.arange(len(final_index)), :, final_index
                    ]  # [num_obj, num_modal]
                    self.performance[
                        "{}-FDE{}".format(cur_mode, kkk)
                    ] += np.sum(np.min(tmp, axis=-1))

    def update_metrics(
        self,
        gt_valid_ids: List,
        gt_trajs: Dict,
        gt_masks: Dict,
        dt_valid_ids: List,
        dt_trajs: Dict,
        dt_probs: Dict,
        mode: List,
        miss_dt_ids: bool = None,
    ):
        """
        Update the trajectory metrics.

        Args:
            gt_valid_ids: Valid id list of GT.[N]
            gt_trajs: [N, traj_len, 2], GT trajectories corresponding to
                gt_valid_ids.
            gt_masks: [N, traj_len, 2], GT masks correspongding to
                gt_valid_ids.
            dt_valid_ids: [K], Valid ids of tracking(last history and
                current frame are both exists).
            dt_trajs: K:[modal, traj_len, 2], Network predict trajectories
                corresponding to dt_valid_ids.
            dt_probs: K:[modal], Predict distribution corresponding to
                dt_valid_ids.
            miss_dt_ids: miss-matched detect ids.

        """
        if miss_dt_ids is not None:
            self.performance["{}-FP".format(mode)] += len(miss_dt_ids)
        if len(dt_valid_ids) <= 0:
            self.performance["{}-FN".format(mode)] += len(gt_valid_ids)
            return
        if len(gt_valid_ids) <= 0:
            return

        num_modal = dt_probs[dt_valid_ids[0]].shape[-1]
        traj_len = gt_trajs[gt_valid_ids[0]].shape[-2]

        gt_valid_ids = set(gt_valid_ids)
        dt_valid_ids = set(dt_valid_ids)

        inner_ids = gt_valid_ids & dt_valid_ids
        self.performance["{}-TP".format(mode)] += len(inner_ids)
        self.performance["{}-FN".format(mode)] += len(
            gt_valid_ids - dt_valid_ids
        )

        if len(inner_ids) > 0:
            # inner-set calculate FDE,ADE
            inner_gt_trajs = [
                gt_trajs[key].reshape([1, traj_len, 2])[:, 0:]
                for key in inner_ids
            ]
            inner_gt_masks = [
                gt_masks[key].reshape([1, traj_len, 2])[:, 0:]
                for key in inner_ids
            ]

            inner_dt_trajs = []
            for key in inner_ids:
                cur_prob = dt_probs[key]
                cur_dt_traj = dt_trajs[key].reshape(
                    [1, num_modal, traj_len, 2]
                )
                sort_indics = np.argsort(cur_prob)[::-1]  # num_modal
                cur_dt_traj = cur_dt_traj[:, sort_indics]
                inner_dt_trajs.append(cur_dt_traj)

            inner_gt_trajs = np.concatenate(
                inner_gt_trajs, 0
            )  # [num_obj, traj_len, 2]
            inner_gt_masks = np.concatenate(
                inner_gt_masks, 0
            )  # [num_obj, traj_len, 2]
            inner_dt_trajs = np.concatenate(
                inner_dt_trajs, 0
            )  # [num_obj, num_modal, traj_len, 2]

            self.updata_FDE_ADE(
                inner_gt_trajs.copy(),
                inner_gt_masks.copy(),
                inner_dt_trajs.copy(),
                mode,
            )

    def get_track_ids(self, gt_indices_dict, gt_instances, timestamp):
        gt_indices = gt_indices_dict[timestamp]
        obj_idxes = gt_instances[timestamp].obj_idxes
        return [obj_idxes[iii] for iii in gt_indices]

    def normalize_yaw_array(self, yaw_array: np.array):
        """Convert the yaw_array to [-pi, pi].

        Args:
            yaw_array: the yaw [rad].
        """

        yaw_array -= 2 * np.pi * np.floor((yaw_array + np.pi) / (2 * np.pi))

        return yaw_array

    def filter_by_yaw(
        self,
        list_instance,
        list_gt_instance,
        yaw_diff_bounce_thr=3.14 / 3,
        use_calc_yaw=True,
    ):
        for ins_index in range(len(list_instance)):
            dt_instances = list_instance[ins_index]
            gt_instances = list_gt_instance[ins_index]

            if "mem_coord" not in dt_instances.keys():
                continue

            cache_position = dt_instances[
                "mem_coord"
            ].copy()  # [valid_q_num, mem_len]

            valid_q_num, num_mem, dims = cache_position.shape[:3]
            assert dims == 4

            x_list = cache_position[..., 0].copy()  # [valid_q_num, mem_len]
            y_list = cache_position[..., 1].copy()
            cache_cos = cache_position[..., 2].copy()
            cache_sin = cache_position[..., 3].copy()
            yaw_list = np.arctan2(cache_sin, cache_cos)
            lcf_classid = dt_instances["labels"]

            state_flag = np.zeros_like(lcf_classid)
            for idx, _ in enumerate(lcf_classid):
                his_x, his_y, all_yaw = x_list[idx], y_list[idx], yaw_list[idx]
                his_x = his_x[~np.isnan(his_x)]
                his_y = his_y[~np.isnan(his_y)]
                all_yaw = all_yaw[~np.isnan(all_yaw)]

                if use_calc_yaw:
                    selected_yaw = np.arctan2(np.diff(his_y), np.diff(his_x))
                else:
                    selected_yaw = all_yaw

                if selected_yaw is not None:
                    selected_yaw = np.unwrap(selected_yaw)
                if selected_yaw is None:
                    state_flag[idx] = 1
                    continue
                if len(selected_yaw) > 1:
                    yaw_array_diff = np.diff(selected_yaw)
                    yaw_array_diff = np.abs(
                        self.normalize_yaw_array(yaw_array_diff)
                    )
                    max_yaw_diff = np.max(yaw_array_diff)
                    if max_yaw_diff > yaw_diff_bounce_thr:
                        state_flag[idx] = 2
                        continue

            keep = 0 == state_flag
            list_instance[ins_index] = dict_select(dt_instances, keep)
            list_gt_instance[ins_index] = dict_select(gt_instances, keep)
        return list_instance, list_gt_instance

    def filter_by_lateral_range(
        self, list_instance, list_gt_instance, lateral_range=(-3, 3)
    ):
        for ins_index in range(len(list_instance)):
            dt_instances = list_instance[ins_index]
            gt_instances = list_gt_instance[ins_index]

            cache_position = gt_instances["boxes"].copy()

            y_list = cache_position[..., 1].copy()
            lcf_classid = dt_instances["labels"]

            state_flag = np.zeros_like(lcf_classid)
            state_flag[y_list > lateral_range[1]] = 1
            state_flag[y_list < lateral_range[0]] = 1
            keep = 0 == state_flag
            list_instance[ins_index] = dict_select(dt_instances, keep)
            list_gt_instance[ins_index] = dict_select(gt_instances, keep)
        return list_instance, list_gt_instance

    def __call__(
        self,
        gt_trajpreds_list,
        input_match_instance,
        FilterByLateralrange=False,
        lateral_range=(-3, 3),
    ):
        for timestamp in tqdm(range(1, len(gt_trajpreds_list))):
            # gt trajs
            trajectory_pred = gt_trajpreds_list[timestamp].copy()
            gt_trajs_dict = trajectory_pred["gt_traj_list"][0]
            gt_classes_dict = trajectory_pred["valid_classes"][0]
            gt_masks_dict = trajectory_pred["gt_mask_list"][0]

            past_ids_list = gt_trajpreds_list[timestamp - 1]["ids_list"][
                0
            ].copy()
            gt_trajs_dict = {
                key: gt_trajs_dict[key]
                for key in gt_trajs_dict.keys()
                if key in past_ids_list
            }
            gt_masks_dict = {
                key: gt_masks_dict[key]
                for key in gt_masks_dict.keys()
                if key in past_ids_list
            }

            gt_valid_ids = list(gt_trajs_dict.keys())
            gt_trajs_Arr = {
                key: gt_trajs_dict[key] for key in gt_trajs_dict.keys()
            }
            gt_masks_Arr = {
                key: gt_masks_dict[key] for key in gt_trajs_dict.keys()
            }

            gt_valid_classes = {
                key: gt_classes_dict[key]
                for key in gt_classes_dict.keys()
                if key in gt_valid_ids
            }
            assert len(gt_valid_classes) == len(gt_valid_ids)

            # network output trajs.
            cur_match_instance = input_match_instance["gt_instances"][
                timestamp
            ].copy()
            past_match_instance = input_match_instance["gt_instances"][
                timestamp - 1
            ].copy()

            mismatch_dt_inst = input_match_instance["unmatch_dt"][
                timestamp
            ].copy()
            if len(mismatch_dt_inst) > 0:
                miss_dt_ids = np.concatenate(
                    [tmp["obj_idxes"] for tmp in mismatch_dt_inst], 0
                )
                miss_dt_labels = np.concatenate(
                    [tmp["labels"] for tmp in mismatch_dt_inst], 0
                )
            else:
                miss_dt_ids = np.array([])
                miss_dt_labels = np.array([])

            # match indexes
            if len(cur_match_instance) > 0 and len(past_match_instance) > 0:
                # filter by detect yaws.
                cur_dt_instances = input_match_instance["dt_instances"][
                    timestamp
                ].copy()
                if self.is_filter_by_yaw:
                    cur_dt_instances, cur_match_instance = self.filter_by_yaw(
                        cur_dt_instances, cur_match_instance
                    )

                if FilterByLateralrange:
                    (
                        cur_dt_instances,
                        cur_match_instance,
                    ) = self.filter_by_lateral_range(
                        cur_dt_instances,
                        cur_match_instance,
                        lateral_range=lateral_range,
                    )

                cur_match_gt_track_id = np.concatenate(
                    [tmp["obj_idxes"] for tmp in cur_match_instance], 0
                )
                past_match_gt_track_id = np.concatenate(
                    [tmp["obj_idxes"] for tmp in past_match_instance], 0
                )

                cur_dt_TrajRegs = np.concatenate(
                    [tmp["TrajRegs"] for tmp in cur_dt_instances], 0
                )  # (n_obj, n_modal, traj_len, 2)
                cur_dt_TrajScores = np.concatenate(
                    [tmp["TrajScores"] for tmp in cur_dt_instances], 0
                )  # (n_obj, n_modal)
                cur_dt_ids = np.concatenate(
                    [tmp["obj_idxes"] for tmp in cur_dt_instances], 0
                )  # (n_obj, 1)
                cur_dt_labels = np.concatenate(
                    [tmp["labels"] for tmp in cur_dt_instances], 0
                )  # (n_obj, 1)

                dt_trajs_Arr = {}
                dt_probs_Arr = {}
                dt_rand_assign_ids = []
                dt_labels = {}
                for match_dt_idx, tmp_id in enumerate(cur_match_gt_track_id):
                    if tmp_id in past_match_gt_track_id:
                        dt_rand_assign_ids.append(cur_dt_ids[match_dt_idx])
                        dt_trajs_Arr[tmp_id] = cur_dt_TrajRegs[match_dt_idx]
                        dt_probs_Arr[tmp_id] = cur_dt_TrajScores[match_dt_idx]
                        dt_labels[tmp_id] = cur_dt_labels[match_dt_idx]
            else:
                dt_trajs_Arr = {}
                dt_probs_Arr = {}
                dt_labels = {}

            for mode in self.mode:
                if "all" != mode:
                    cur_class = self.mode.index(mode) - 1
                    cur_miss_dt_ids = miss_dt_ids[miss_dt_labels == cur_class]
                    cur_gt_valid_ids = [
                        key
                        for key in gt_valid_classes.keys()
                        if gt_valid_classes[key] == cur_class
                    ]
                    cur_gt_trajs = {
                        key: gt_trajs_Arr[key]
                        for key in gt_trajs_Arr.keys()
                        if key in cur_gt_valid_ids
                    }
                    cur_gt_masks = {
                        key: gt_masks_Arr[key]
                        for key in gt_masks_Arr.keys()
                        if key in cur_gt_valid_ids
                    }

                    cur_dt_valid_ids = [
                        key
                        for key in dt_labels.keys()
                        if dt_labels[key] == cur_class
                    ]
                    cur_dt_trajs_Arr = {
                        key: dt_trajs_Arr[key]
                        for key in dt_trajs_Arr.keys()
                        if key in cur_dt_valid_ids
                    }
                    cur_dt_probs_Arr = {
                        key: dt_probs_Arr[key]
                        for key in dt_probs_Arr.keys()
                        if key in cur_dt_valid_ids
                    }
                else:
                    cur_miss_dt_ids = miss_dt_ids
                    cur_gt_valid_ids = list(gt_valid_classes.keys())
                    cur_gt_trajs = gt_trajs_Arr
                    cur_gt_masks = gt_masks_Arr
                    cur_dt_valid_ids = list(dt_labels.keys())
                    cur_dt_trajs_Arr = dt_trajs_Arr
                    cur_dt_probs_Arr = dt_probs_Arr
                self.update_metrics(
                    cur_gt_valid_ids,
                    cur_gt_trajs,
                    cur_gt_masks,
                    cur_dt_valid_ids,
                    cur_dt_trajs_Arr,
                    cur_dt_probs_Arr,
                    mode,
                    miss_dt_ids=cur_miss_dt_ids,
                )
        self.post_avg_over_obj()  # avg

        # print
        if not FilterByLateralrange:
            outstr = "traj-pred-metrics: \n"
        else:
            outstr = "traj-pred-metrics-lateral-range: \n"

        outkeys = []
        outlines = []
        traj_metric = {}
        for cur_mode in self.mode:
            traj_metric[cur_mode] = {}
            cur_str = cur_mode + "\t"
            if "ADE" in self.metric:
                for kkk in self.k_values:
                    cur_key = "ADE{}".format(kkk)
                    if cur_key not in outkeys:
                        outkeys.append(cur_key)
                    tmp = self.performance["{}-ADE{}".format(cur_mode, kkk)]
                    cur_str = cur_str + "{:.3f}\t".format(tmp)
                    traj_metric[cur_mode][cur_key] = tmp

            if "FDE" in self.metric:
                for kkk in self.k_values:
                    cur_key = "FDE{}".format(kkk)
                    if cur_key not in outkeys:
                        outkeys.append(cur_key)
                    tmp = self.performance["{}-FDE{}".format(cur_mode, kkk)]
                    cur_str = cur_str + "{:.3f}\t".format(tmp)
                    traj_metric[cur_mode][cur_key] = tmp

            cur_str = cur_str + "{}\t".format(
                self.performance["{}-TP".format(cur_mode)]
            )
            cur_str = cur_str + "{}\t".format(
                self.performance["{}-FN".format(cur_mode)]
            )
            cur_str = cur_str + "{}\t".format(
                self.performance["{}-FP".format(cur_mode)]
            )
            cur_str = cur_str + "{}\t".format(
                self.performance["{}-validTP".format(cur_mode)]
            )
            traj_metric[cur_mode]["TP"] = self.performance[
                "{}-TP".format(cur_mode)
            ]
            traj_metric[cur_mode]["FN"] = self.performance[
                "{}-FN".format(cur_mode)
            ]
            traj_metric[cur_mode]["FP"] = self.performance[
                "{}-FP".format(cur_mode)
            ]
            traj_metric[cur_mode]["ValidTP"] = self.performance[
                "{}-validTP".format(cur_mode)
            ]
            outlines.append(cur_str + "\n")

        output_str = outstr
        outkeys = outkeys + [
            "TP",
            "FN",
            "FP",
            "ValidTP",
        ]  # ["TP", "FP", "FN", "ValidTP"]

        cur_key_str = "\t"
        for tmp in outkeys:
            cur_key_str = cur_key_str + "{}\t".format(tmp)
        output_str = output_str + cur_key_str + "\n"

        for cur_line in outlines:
            output_str = output_str + cur_line

        traj_metric["outkeys"] = outkeys
        traj_metric["mode"] = self.mode
        return output_str, traj_metric
