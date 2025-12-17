from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from hatbc.message.structure import BBox3D
except ImportError:
    BBox3D = None

from hat.core.box3d_utils import boxes_iou3d_vcs
from hat.metrics.detection2d.utils import calap, calar
from hat.metrics.detection3d.compute_entry import AutoEvalCalculator
from hat.metrics.detection3d.generate_results import cal_percentile_vaule
from hat.utils.package_helper import require_packages


def match_by_key(preds, gts, key):
    for gt in gts:
        id = getattr(gt, key)
        for pred in preds:
            if getattr(pred, key) == id:
                yield pred, gt
                break
        else:
            yield None, gt


@require_packages("hatbc")
def center_distance(pred: "BBox3D", gt: "BBox3D"):
    return np.linalg.norm(np.array([pred.x, pred.y]) - np.array([gt.x, gt.y]))


@require_packages("hatbc")
def let_iou(
    pred: "BBox3D",
    gt: "BBox3D",
    p_t: float = 0.2,
    min_t: float = 4.0,
    max_t: float = 10.0,
) -> float:

    pred_bbox_center = np.array([pred.x, pred.y])
    gt_bbox_center = np.array([gt.x, gt.y])
    e_loc = pred_bbox_center - gt_bbox_center
    g_norm = np.linalg.norm(gt_bbox_center)
    p_norm = np.linalg.norm(pred_bbox_center)
    u_g = gt_bbox_center / g_norm
    u_p = pred_bbox_center / p_norm
    e_lon = (e_loc.dot(u_g)) * u_g
    e_lon_norm = np.linalg.norm(e_lon)
    tl = max(p_t * g_norm, min_t)
    tl = min(tl, max_t)
    al = 1 - min(e_lon_norm / tl, 1.0)
    if al > 0.0:
        pred_aligned_center = (gt_bbox_center.dot(u_p)) * u_p
        pred = BBox3D(
            dim=pred.dim,
            loc=[pred_aligned_center[0], pred_aligned_center[1], pred.loc[2]],
            yaw=pred.yaw,
        )
    pred = pred.loc.data + [pred.dim[2], pred.dim[0], pred.dim[1]] + [pred.yaw]
    gt = gt.loc.data + [gt.dim[2], gt.dim[0], gt.dim[1]] + [gt.yaw]
    _, iou3d = boxes_iou3d_vcs(pred, gt)
    return iou3d, al


def let_iou_matrix(
    preds: list,
    gts: list,
    p_t: float,
    min_t: float,
    max_t: float,
):

    num_pred = len(preds)
    num_gt = len(gts)
    rotate_ious = np.zeros((num_pred, num_gt), dtype=np.float64)
    pairwise_al = np.zeros(((num_pred, num_gt)), dtype=np.float64)
    for idx_pred in range(num_pred):
        for idx_gt in range(num_gt):
            pred = preds[idx_pred]
            gt = gts[idx_gt]
            tmp_iou, al = let_iou(pred, gt, p_t, min_t, max_t)
            pairwise_al[idx_pred][idx_gt] = al
            rotate_ious[idx_pred][idx_gt] = tmp_iou

    return rotate_ious, pairwise_al


def get_heading_accuracy(pd_heading, gt_heading):
    # Caculate heading diff between pred & gt
    diff_heading = abs(
        normalize_angle(pd_heading) - normalize_angle(gt_heading)
    )
    # Normalize heading error to [0, PI] (+PI and -PI are the same).
    if diff_heading > np.pi:
        diff_heading = 2.0 * np.pi - diff_heading
    # Clamp the range to avoid numerical errors.
    return min(1.0, max(0.0, 1.0 - diff_heading / np.pi))


def normalize_angle(theta):
    return theta - 2 * np.pi * np.floor((theta + np.pi) / (2 * np.pi))


def get_tp_fp_info(
    pred_scence: List[dict],
    gt_scence: List[dict],
    eval_match_method: str,
    iou_thr: float = None,
    dist_thr: float = None,
    relative_dist_thr: float = None,
    enable_ignore: bool = False,
    eval_let_params: Optional[dict] = None,
    eval_vcs_range: Optional[Sequence[float]] = None,
) -> Dict:

    preds = [(id, pred.bbox_3d) for id, pred in pred_scence.objects_3d.items()]
    preds = sorted(preds, key=lambda x: x[1].score, reverse=True)
    preds_id = [pred[0] for pred in preds]
    preds_ignore = [pred_scence.objects_3d[id].ignore for id in preds_id]
    preds = [pred[1] for pred in preds]
    gts = [
        (id, gt.bbox_3d, gt.meta["uid"])
        for id, gt in gt_scence.objects_3d.items()
    ]
    gts_id = [gt[0] for gt in gts]
    gts_uid = [gt[2] for gt in gts]
    gts_ignore = [gt_scence.objects_3d[id].ignore for id in gts_id]
    gts = [gt[1] for gt in gts]

    num_pred = len(preds)
    num_gt = len(gts)
    tp = np.zeros(num_pred)
    tp_h = np.zeros(num_pred)
    match_idx = np.zeros(num_pred) - 1
    match_overlap = np.zeros(num_pred) - 1
    match_dist = np.zeros(num_pred) - 1
    fp = np.zeros(num_pred)
    fn = np.zeros(num_gt)
    conf = np.zeros(num_pred)
    gt_checked = np.zeros(len(gts))
    ignored_mask = np.array(preds_ignore)

    if "let_iou" in eval_match_method:
        assert (
            "let_p_t" in eval_let_params
            and "let_min_t" in eval_let_params
            and "let_max_t" in eval_let_params
        )
        let_p_t = eval_let_params["let_p_t"]
        let_min_t = eval_let_params["let_min_t"]
        let_max_t = eval_let_params["let_max_t"]

        lon_affinity = np.zeros(len(preds))
        gt_matchd_pred_idx = np.zeros(len(gts)) - 1
        pairwise_iou, pairwise_al = let_iou_matrix(
            preds=preds,
            gts=gts,
            p_t=let_p_t,
            min_t=let_min_t,
            max_t=let_max_t,
        )
        tmp_max_confidence = np.zeros(num_gt)

        for pred_idx, pred in enumerate(preds):
            pred_iou = pairwise_iou[pred_idx]
            pred_iou[pred_iou < iou_thr] = 0.0
            pred_al = pairwise_al[pred_idx]
            pred_score = pred.score
            max_conf = 0
            if len(gts) > 0:
                pred_joint_confidence = pred_iou * pred_al * pred_score
                max_gt_ind = np.argmax(pred_joint_confidence)
                max_conf = pred_joint_confidence[max_gt_ind]
                max_iou = pred_iou[max_gt_ind]
                # max_pred_ind = pred_idx
            if max_conf > 0 and tmp_max_confidence[max_gt_ind] < max_conf:
                if enable_ignore and gts_ignore[max_gt_ind]:
                    ignored_mask[pred_idx] = 1
                    continue
                else:
                    matched_pred_idx = int(gt_matchd_pred_idx[max_gt_ind])
                    gt_matchd_pred_idx[max_gt_ind] = pred_idx
                    match_idx[pred_idx] = max_gt_ind
                    match_overlap[pred_idx] = max_iou
                    if matched_pred_idx != -1:
                        match_idx[matched_pred_idx] = -1
                        tp[matched_pred_idx] = 0
                        fp[matched_pred_idx] = 1.0
                    lon_affinity[pred_idx] = pred_al[max_gt_ind]
                    tmp_max_confidence[max_gt_ind] = max_conf
                    tp[pred_idx] = 1.0
                    # tp_h
                    heading_accuracy = get_heading_accuracy(
                        pred.yaw, gts[max_gt_ind].yaw
                    )
                    tp_h[pred_idx] = heading_accuracy
                    fp[pred_idx] = 0
            else:
                fp[pred_idx] = 1.0

        fn = np.invert(np.array(gt_matchd_pred_idx >= 0, dtype=np.bool8))

    elif "iou" in eval_match_method:
        assert iou_thr is not None
        for pidx, det in enumerate(preds):
            conf[pidx] = det.score
            max_overlap = -np.inf
            jmax = -1
            if len(gts) > 0:
                overlaps = []
                det_tmp = (
                    det.loc.data
                    + [det.dim[2], det.dim[0], det.dim[1]]
                    + [det.yaw]
                )
                for gt in gts:
                    gt_tmp = (
                        gt.loc.data
                        + [gt.dim[2], gt.dim[0], gt.dim[1]]
                        + [gt.yaw]
                    )
                    if eval_match_method == "iou_3d":
                        _, iou3d = boxes_iou3d_vcs(det_tmp, gt_tmp)
                        overlaps.append(iou3d)
                    elif eval_match_method == "iou_bev":
                        iou_bev, _ = boxes_iou3d_vcs(det_tmp, gt_tmp)
                        overlaps.append(iou_bev)
                    else:
                        assert TypeError(
                            f"Get unsupported match type: {eval_match_method}!"
                        )

                max_overlap = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if max_overlap > iou_thr:
                if enable_ignore and gts_ignore[jmax]:
                    ignored_mask[pidx] = 1
                    continue
                if gt_checked[jmax] == 0:
                    tp[pidx] = 1.0
                    match_idx[pidx] = jmax
                    match_overlap[pidx] = max_overlap
                    gt_checked[jmax] = 1
                    # tp_h
                    heading_accuracy = get_heading_accuracy(
                        det.yaw, gts[jmax].yaw
                    )
                    tp_h[pidx] = heading_accuracy
                else:
                    fp[pidx] = 1.0
            else:
                fp[pidx] = 1.0
        fn = np.invert(np.array(gt_checked, dtype=np.bool8))
    else:
        assert dist_thr is not None
        assert relative_dist_thr is not None
        for pidx, det in enumerate(preds):
            conf[pidx] = det.score
            min_dist = np.inf
            relative_min_dist = np.inf
            jmin = -1

            if len(gts) > 0:
                dists = [center_distance(det, gt) for gt in gts]
                min_dist = np.min(dists)
                jmin = np.argmin(dists)
                match_gt = gts[jmin]
                relative_min_dist = min_dist / np.linalg.norm(
                    np.array([match_gt.x, match_gt.y])
                )

            if min_dist <= dist_thr and relative_min_dist <= relative_dist_thr:
                if enable_ignore and gts_ignore[jmin]:
                    ignored_mask[pidx] = 1
                    continue
                if gt_checked[jmin] == 0:
                    tp[pidx] = 1.0
                    match_idx[pidx] = jmin
                    match_dist[pidx] = min_dist
                    gt_checked[jmin] = 1
                    # tp_h
                    heading_accuracy = get_heading_accuracy(
                        det.yaw, gts[jmin].yaw
                    )
                    tp_h[pidx] = heading_accuracy
                else:
                    fp[pidx] = 1.0
            else:
                fp[pidx] = 1.0
        fn = np.invert(np.array(gt_checked, dtype=np.bool8))

    fn = fn.astype(int)
    if enable_ignore:
        ignored_mask = ignored_mask.astype(bool)
        tp[ignored_mask] = -1
        fp[ignored_mask] = -1
        if len(gts_ignore) > 0:
            fn[np.array(gts_ignore)] = -1

    # update preds metric (tp, tph, fp, match_gt_idx, match_overlap) info
    matched_ignore_by_eval_vcs_range = {}
    for idx in range(len(preds)):
        pred_obj3d = pred_scence.objects_3d[preds_id[idx]]
        match_gt_obj3d_id = (
            gts_uid[int(match_idx[idx])] if int(match_idx[idx]) != -1 else -1
        )
        pred_ignore_by_eval_vcs_range = False
        if eval_vcs_range is not None:
            (
                gt_ignore_by_eval_vcs_range,
                pred_ignore_by_eval_vcs_range,
            ) = update_metric_by_eval_vcs_range(
                pred_obj3d=pred_obj3d,
                matched_gt_obj3d=None
                if match_gt_obj3d_id == -1
                else gt_scence.objects_3d[gts_id[int(match_idx[idx])]],
                eval_vcs_range=eval_vcs_range,
            )
            if match_gt_obj3d_id != -1:
                matched_ignore_by_eval_vcs_range.update(
                    {
                        int(match_idx[idx]): (
                            gt_ignore_by_eval_vcs_range,
                            pred_ignore_by_eval_vcs_range,
                        )
                    }
                )
        pred_metric_info = dict(  # noqa
            is_tp=tp[idx] if not pred_ignore_by_eval_vcs_range else -1,
            tp_h=tp_h[idx] if not pred_ignore_by_eval_vcs_range else -1,
            match_gt_obj3d_id=match_gt_obj3d_id,
        )
        if "let_iou" in eval_match_method:
            pred_metric_info.update({"match_lon_affinity": lon_affinity[idx]})
        if "iou" in eval_match_method:
            pred_metric_info.update({"match_overlap": match_overlap[idx]})
        else:
            pred_metric_info.update({"match_dist": match_dist[idx]})
        pred_scence.save_obj3d_in_scence_info(pred_obj3d, pred_metric_info)
    # update gts metric (fn) info
    gt_metric_info = {}
    for idx in range(len(gts)):
        gt_obj3d = gt_scence.objects_3d[gts_id[idx]]
        gt_ignore_by_eval_vcs_range = False
        if eval_vcs_range is not None:
            if matched_ignore_by_eval_vcs_range.get(idx, None) is not None:
                (
                    gt_ignore_by_eval_vcs_range,
                    _,
                ) = matched_ignore_by_eval_vcs_range.get(idx)
            else:
                (
                    gt_ignore_by_eval_vcs_range,
                    _,
                ) = update_metric_by_eval_vcs_range(
                    pred_obj3d=None,
                    matched_gt_obj3d=gt_obj3d,
                    eval_vcs_range=eval_vcs_range,
                )
        gt_metric_info = {
            "is_fn": fn[idx] if not gt_ignore_by_eval_vcs_range else -1
        }
        gt_scence.save_obj3d_in_scence_info(gt_obj3d, gt_metric_info)

    return pred_scence, gt_scence


def summarize_ap(
    preds_scence: List,
    gt_count: int,
    target_recalls: List[float],
    target_precisions: List[float],
    enable_let_apl: bool = False,
) -> Dict:
    """Get ap results.

    Args:
        preds_scence: List of pred_scence obj.
        gt_count: Number of ground truth box.
        target_recalls: Target recalls.
        target_precisions: Target precisions.

    Returns:
        Dict: summary.
    """
    res_all = defaultdict(lambda: [])
    num_scence = len(preds_scence)
    for pred_scence in preds_scence:
        for id, info in pred_scence.objects_3d_info.items():
            obj3d = pred_scence.objects_3d[id]
            if info["is_tp"] != -1:
                res_all["tp"] += [info["is_tp"]]
                res_all["tp_h"] += [info["tp_h"]]
                res_all["fp"] += [1 - info["is_tp"]]
                res_all["conf"] += [obj3d.score]
                res_all["dx"] += [info["dx"]] if "dx" in info else [0]
                res_all["dxp"] += [info["dxp"]] if "dxp" in info else [0]
                res_all["dy"] += [info["dy"]] if "dy" in info else [0]
                res_all["dyp"] += [info["dyp"]] if "dyp" in info else [0]
                res_all["dxyp"] += [info["dxyp"]] if "dxyp" in info else [0]
                res_all["drot"] += [info["drot"]] if "drot" in info else [0]
                if enable_let_apl and info["is_tp"] == 1:
                    res_all["det_affinity"] += [info["match_lon_affinity"]]

    tp = np.array(res_all["tp"])
    tp_h = np.array(res_all["tp_h"])
    fp = np.array(res_all["fp"])
    conf = np.array(res_all["conf"])

    dx = np.array(res_all["dx"])
    dxp = np.array(res_all["dxp"])
    dy = np.array(res_all["dy"])
    dyp = np.array(res_all["dyp"])
    dxyp = np.array(res_all["dxyp"])
    drot = np.array(res_all["drot"])
    if enable_let_apl:
        det_affinity = np.array(res_all["det_affinity"])

    argsort = np.argsort(-conf)
    conf = conf[argsort]
    tp = tp[argsort]
    tp_h = tp_h[argsort]
    is_tp = tp.astype(np.bool_)
    fp = fp[argsort]
    dx = dx[argsort]
    dxp = dxp[argsort]
    dy = dy[argsort]
    dyp = dyp[argsort]
    dxyp = dxyp[argsort]
    drot = drot[argsort]
    drot_90p, drot_95p = cal_percentile_vaule(drot, [0.9, 0.95])
    dxyp_90p, dxyp_95p = cal_percentile_vaule(dxyp, [0.9, 0.95])
    drot_mean = np.mean(drot)
    dxyp_mean = np.mean(dxyp)

    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    tp_h = np.cumsum(tp_h, axis=0)

    if gt_count == 0:
        recalls = np.zeros(tp.shape)
    else:
        recalls = tp / float(gt_count)
    assert np.all(0 <= recalls) & np.all(recalls <= 1)

    precisions = tp / (tp + fp + 1e-6)
    precisions_h = tp_h / (tp + fp + 1e-6)
    assert np.all(0 <= precisions) & np.all(precisions <= 1)
    ap, recall, precision = calap(recalls, precisions)
    aph, _, _ = calap(recalls, precisions_h)
    apl = None
    if enable_let_apl:
        det_tp_num = tp.max() if tp.any() else 0
        mean_al = det_affinity.sum() / (det_tp_num + 1e-6)
        apl = mean_al * ap
    recall = np.array(recall)
    precision = np.array(precision)

    fppi = fp / float(num_scence)
    fppi += 1e-6
    detection_rate = (tp - fp) / (gt_count + 1e-6)
    max_detection_rate = (
        max(detection_rate.tolist()) if len(detection_rate) > 0 else np.inf
    )
    score_at_max_dr = (
        conf[detection_rate.tolist().index(max_detection_rate)]
        if len(detection_rate) > 0
        else np.inf
    )
    drot_at_max_dr = (
        np.mean(drot[conf >= score_at_max_dr])
        if len(detection_rate) > 0
        else np.inf
    )
    drot_at_max_dr_90p, drot_at_max_dr_95p = (
        cal_percentile_vaule(drot[conf >= score_at_max_dr], [0.9, 0.95])
        if len(detection_rate) > 0
        else [np.inf, np.inf]
    )
    max_dr_dxyp_90p, max_dr_dxyp_95p = (
        cal_percentile_vaule(dxyp[conf >= score_at_max_dr], [0.9, 0.95])
        if len(detection_rate) > 0
        else [np.inf, np.inf]
    )
    max_dr_dx = (
        np.mean(dx[conf >= score_at_max_dr])
        if len(detection_rate) > 0
        else np.inf
    )
    max_dr_dy = (
        np.mean(dy[conf >= score_at_max_dr])
        if len(detection_rate) > 0
        else np.inf
    )
    max_dr_dxyp = (
        np.mean(dxyp[conf >= score_at_max_dr])
        if len(detection_rate) > 0
        else np.inf
    )

    ar = calar(fppi, recalls)

    results = {}
    results["aps"] = {
        "ap": ap,
        "aph": aph,
        "rec": recall[-1] if len(recall) > 0 else np.inf,
        "detection_rate": detection_rate[-1]
        if len(detection_rate) > 0
        else np.inf,
        "precision": precision[-1] if len(precision) > 0 else np.inf,
        "ar": ar,
        "num_scence": num_scence,
        "num_gt": gt_count,
        "num_tp": tp[-1] if len(tp) else 0,
        "num_fp": fp[-1] if len(fp) else 0,
        "drot": drot_mean,
        "drot@90%": drot_90p,
        "drot@95%": drot_95p,
        "dxyp": dxyp_mean,
        "dxyp@90%": dxyp_90p,
        "dxyp@95%": dxyp_95p,
    }
    if apl is not None:
        results["aps"].update({"apl": apl})
    results["recall"] = recall.tolist()
    results["precision"] = precision.tolist()
    results["fppi"] = fppi.tolist()
    results["conf"] = conf.tolist()
    results["detection_rate"] = detection_rate.tolist()
    results["fp"] = fp.tolist()

    results["scores_lut"] = {}
    thres_rec_pre_tp_fp_gt_error = []
    for thres in np.unique(conf):
        select = conf >= thres
        sub_tp = np.max(tp[select])
        sub_fp = np.max(fp[select])
        sub_dxp = np.mean(dxp[select & is_tp])
        sub_dyp = np.mean(dyp[select & is_tp])
        sub_dxyp = np.mean(dxyp[select & is_tp])
        sub_drot = np.mean(drot[select & is_tp])
        sub_drot_90p, sub_drot_95p = cal_percentile_vaule(
            drot[select & is_tp], [0.9, 0.95]
        )

        rec = sub_tp / (gt_count + 1e-6)
        pre = sub_tp / (sub_tp + sub_fp + 1e-6)
        thres_rec_pre_tp_fp_gt_error.append(
            [
                thres,
                rec,
                pre,
                sub_tp,
                sub_fp,
                gt_count,
                sub_dxp,
                sub_dyp,
                sub_dxyp,
                sub_drot,
                sub_drot_90p,
                sub_drot_95p,
            ]
        )
        results["scores_lut"][float(thres)] = {
            "threshold": thres,
            "recall": rec,
            "precision": pre,
            "num_tp": sub_tp,
            "num_fp": sub_fp,
            "num_gt": gt_count,
            "dxp": sub_dxp,
            "dyp": sub_dyp,
            "dxyp": sub_dxyp,
            "drot": sub_drot,
            "drot@90%": sub_drot_90p,
            "drot@95%": sub_drot_95p,
        }
    thres_rec_pre_tp_fp_gt_error = np.array(thres_rec_pre_tp_fp_gt_error)

    results["max_detection_rate"] = {}
    if len(thres_rec_pre_tp_fp_gt_error) == 0:
        pass
    else:
        valid_idx = np.where(
            thres_rec_pre_tp_fp_gt_error[:, 0] > score_at_max_dr
        )[0]
        if valid_idx.shape[0] > 0:
            idx = valid_idx[
                thres_rec_pre_tp_fp_gt_error[valid_idx, 0].argmin()
            ]
        else:
            idx = np.argmin(
                score_at_max_dr - thres_rec_pre_tp_fp_gt_error[:, 0]
            )
        results["max_detection_rate"].update(
            {
                "max_dr": max_detection_rate,
                "score@max_dr": score_at_max_dr,
                "recall@max_dr": thres_rec_pre_tp_fp_gt_error[idx, 1],
                "precision@max_dr": thres_rec_pre_tp_fp_gt_error[idx, 2],
                "num_tp@max_dr": thres_rec_pre_tp_fp_gt_error[idx, 3],
                "num_fp@max_dr": thres_rec_pre_tp_fp_gt_error[idx, 4],
                "num_gt@max_dr": thres_rec_pre_tp_fp_gt_error[idx, 5],
                "drot@max_dr": drot_at_max_dr,
                "drot@90%@max_dr": drot_at_max_dr_90p,
                "drot@95%@max_dr": drot_at_max_dr_95p,
                "dx@max_dr": max_dr_dx,
                "dy@max_dr": max_dr_dy,
                "dxyp@max_dr": max_dr_dxyp,
                "dxyp@90%@max_dr": max_dr_dxyp_90p,
                "dxyp@95%@max_dr": max_dr_dxyp_95p,
            }
        )

    results["target_recalls"] = {}
    if len(thres_rec_pre_tp_fp_gt_error) == 0:
        pass
    else:
        for target_recall in target_recalls:
            valid_idx = np.where(
                thres_rec_pre_tp_fp_gt_error[:, 1] > target_recall
            )[0]
            if valid_idx.shape[0] > 0:
                idx = valid_idx[
                    thres_rec_pre_tp_fp_gt_error[valid_idx, 1].argmin()
                ]
            else:
                idx = np.argmin(
                    target_recall - thres_rec_pre_tp_fp_gt_error[:, 1]
                )
            results["target_recalls"][target_recall] = {
                "threshold": thres_rec_pre_tp_fp_gt_error[idx, 0],
                "recall": thres_rec_pre_tp_fp_gt_error[idx, 1],
                "precision": thres_rec_pre_tp_fp_gt_error[idx, 2],
                "num_tp": thres_rec_pre_tp_fp_gt_error[idx, 3],
                "num_fp": thres_rec_pre_tp_fp_gt_error[idx, 4],
                "num_gt": thres_rec_pre_tp_fp_gt_error[idx, 5],
                "dxp": thres_rec_pre_tp_fp_gt_error[idx, 6],
                "dyp": thres_rec_pre_tp_fp_gt_error[idx, 7],
                "dxyp": thres_rec_pre_tp_fp_gt_error[idx, 8],
                "drot": thres_rec_pre_tp_fp_gt_error[idx, 9],
                "drot@90%": thres_rec_pre_tp_fp_gt_error[idx, 10],
                "drot@95%": thres_rec_pre_tp_fp_gt_error[idx, 11],
            }
        results["target_precisions"] = {}
        for target_precision in target_precisions:
            valid_idx = np.where(
                thres_rec_pre_tp_fp_gt_error[:, 2] > target_precision
            )[0]
            if valid_idx.shape[0] > 0:
                idx = valid_idx[
                    thres_rec_pre_tp_fp_gt_error[valid_idx, 2].argmin()
                ]
            else:
                idx = np.argmin(
                    target_precision - thres_rec_pre_tp_fp_gt_error[:, 2]
                )
            results["target_precisions"][target_precision] = {
                "threshold": thres_rec_pre_tp_fp_gt_error[idx, 0],
                "recall": thres_rec_pre_tp_fp_gt_error[idx, 1],
                "precision": thres_rec_pre_tp_fp_gt_error[idx, 2],
                "num_tp": thres_rec_pre_tp_fp_gt_error[idx, 3],
                "num_fp": thres_rec_pre_tp_fp_gt_error[idx, 4],
                "num_gt": thres_rec_pre_tp_fp_gt_error[idx, 5],
                "dxp": thres_rec_pre_tp_fp_gt_error[idx, 6],
                "dyp": thres_rec_pre_tp_fp_gt_error[idx, 7],
                "dxyp": thres_rec_pre_tp_fp_gt_error[idx, 8],
                "drot": thres_rec_pre_tp_fp_gt_error[idx, 9],
                "drot@90%": thres_rec_pre_tp_fp_gt_error[idx, 10],
                "drot@95%": thres_rec_pre_tp_fp_gt_error[idx, 11],
            }

    return ap, aph, ar, drot_90p, results


def cal_tp_error(
    pred_scence,
    gt_scence,
    eval_metric,
    eval_match_method,
) -> Dict:
    """Get tp error.

    Args:
        pred_scence: pred scence object.
        gt_scence: gt scence object.
        eval_metric: eval segmentation metric.

    Returns:
        Dict: summary.
    """
    tp_id_list = [
        oid
        for oid, info in pred_scence.objects_3d_info.items()
        if info["is_tp"] == 1
    ]
    tp_match_gt_id_list = [
        info["match_gt_obj3d_id"]
        for info in pred_scence.objects_3d_info.values()
        if info["is_tp"] == 1
    ]
    tp_match_gt = [
        gt_scence.get_obj3d_by_uid(id) for id in tp_match_gt_id_list
    ]

    if len(tp_match_gt) > 0:
        tp_pred_dim = np.array(
            [pred_scence.objects_3d[id].dimensions for id in tp_id_list]
        )
        tp_pred_loc = np.array(
            [pred_scence.objects_3d[id].location for id in tp_id_list]
        )
        tp_pred_yaw = np.array(
            [pred_scence.objects_3d[id].rotation_y for id in tp_id_list]
        )
        tp_pred_score = np.array(
            [pred_scence.objects_3d[id].score for id in tp_id_list]
        )
        if "iou" in eval_match_method:
            tp_match_overlap = np.array(
                [
                    pred_scence.objects_3d_info[id]["match_overlap"]
                    for id in tp_id_list
                ]
            )
            diou_error = 1.0 - tp_match_overlap
        else:
            tp_match_dist = np.array(
                [
                    pred_scence.objects_3d_info[id]["match_dist"]
                    for id in tp_id_list
                ]
            )
            dist_error = tp_match_dist

        tp_match_gt_dim = np.array([obj3d.dimensions for obj3d in tp_match_gt])
        tp_match_gt_loc = np.array([obj3d.location for obj3d in tp_match_gt])
        tp_match_gt_yaw = np.array([obj3d.rotation_y for obj3d in tp_match_gt])
        tp_match_gt_depth = np.array(
            [
                np.linalg.norm([obj3d.location[0], obj3d.location[1]])
                for obj3d in tp_match_gt
            ]
        )

        dx = AutoEvalCalculator.dx(tp_match_gt_loc[:, 0], tp_pred_loc[:, 0])
        sdx = AutoEvalCalculator.sdx(tp_match_gt_loc[:, 0], tp_pred_loc[:, 0])
        dy = AutoEvalCalculator.dy(tp_match_gt_loc[:, 1], tp_pred_loc[:, 1])
        sdy = AutoEvalCalculator.sdy(tp_match_gt_loc[:, 1], tp_pred_loc[:, 1])
        dxy = AutoEvalCalculator.dxy(dx, dy)
        dxp = AutoEvalCalculator.dxp(dx, tp_match_gt_loc[:, 0])
        sdxp = AutoEvalCalculator.sdxp(sdx, tp_match_gt_loc[:, 0])
        dyp = AutoEvalCalculator.dyp(dy, tp_match_gt_loc[:, 1])
        sdyp = AutoEvalCalculator.sdyp(sdy, tp_match_gt_loc[:, 1])
        dxyp = AutoEvalCalculator.dxyp(
            dxy, tp_match_gt_loc[:, 0], tp_match_gt_loc[:, 1]
        )
        dxy_10p_error = AutoEvalCalculator.dxy_10p_error(dxyp)
        dh = AutoEvalCalculator.dh(tp_match_gt_dim[:, 2], tp_pred_dim[:, 2])
        sdh = AutoEvalCalculator.sdh(tp_match_gt_dim[:, 2], tp_pred_dim[:, 2])
        dw = AutoEvalCalculator.dw(tp_match_gt_dim[:, 1], tp_pred_dim[:, 1])
        sdw = AutoEvalCalculator.sdw(tp_match_gt_dim[:, 1], tp_pred_dim[:, 1])
        dl = AutoEvalCalculator.dl(tp_match_gt_dim[:, 0], tp_pred_dim[:, 0])
        sdl = AutoEvalCalculator.sdl(tp_match_gt_dim[:, 0], tp_pred_dim[:, 0])
        dhp = AutoEvalCalculator.dhp(dh, tp_match_gt_dim[:, 2])
        sdhp = AutoEvalCalculator.sdhp(sdh, tp_match_gt_dim[:, 2])
        dwp = AutoEvalCalculator.dwp(dw, tp_match_gt_dim[:, 1])
        sdwp = AutoEvalCalculator.sdwp(sdw, tp_match_gt_dim[:, 1])
        dlp = AutoEvalCalculator.dlp(dl, tp_match_gt_dim[:, 0])
        sdlp = AutoEvalCalculator.sdlp(sdl, tp_match_gt_dim[:, 0])
        abs_rot = AutoEvalCalculator.abs_rot(tp_match_gt_yaw, tp_pred_yaw)
        dyaw = AutoEvalCalculator.dyaw(tp_match_gt_yaw, tp_pred_yaw)
        drot = AutoEvalCalculator.drot(abs_rot)
        sdrot = AutoEvalCalculator.sdrot(dyaw)
        rot_cls_error_p = AutoEvalCalculator.rot_cls_error_p(dyaw)

        res_metric = {
            "dx": dx,
            "sdx": sdx,
            "dxp": dxp,
            "sdxp": sdxp,
            "dy": dy,
            "sdy": sdy,
            "dyp": dyp,
            "sdyp": sdyp,
            "dxy": dxy,
            "dxyp": dxyp,
            "dh": dh,
            "sdh": sdh,
            "dhp": dhp,
            "sdhp": sdhp,
            "dw": dw,
            "sdw": sdw,
            "dwp": dwp,
            "sdwp": sdwp,
            "dl": dl,
            "sdl": sdl,
            "dlp": dlp,
            "sdlp": sdlp,
            "drot": drot,
            "sdrot": sdrot,
            "dxy_10p_error": dxy_10p_error,
            "rot_cls_error_p": rot_cls_error_p,
            "score": tp_pred_score,
            "tp_match_gt_depth": tp_match_gt_depth,
        }
        if "iou" in eval_match_method:
            res_metric.update({"diou_error": diou_error})
            save_info_key = eval_metric + [
                "diou_error",
                "score",
                "tp_match_gt_depth",
            ]
        else:
            res_metric.update({"dist_error": dist_error})
            save_info_key = eval_metric + [
                "dist_error",
                "score",
                "tp_match_gt_depth",
            ]

    if "iou" in eval_match_method:
        save_info_key = eval_metric + [
            "diou_error",
            "score",
            "tp_match_gt_depth",
        ]
    else:
        save_info_key = eval_metric + [
            "dist_error",
            "score",
            "tp_match_gt_depth",
        ]
    for key in save_info_key:
        for idx, id in enumerate(tp_id_list):
            res = res_metric[key]
            tp_obj3d = pred_scence.objects_3d[id]
            metric = res[idx]
            update = {key: metric}
            pred_scence.save_obj3d_in_scence_info(tp_obj3d, update)

    return pred_scence, gt_scence


def statistic_error_by_depth(
    preds_scence,
    gts_scence,
    dep_thresh,
    eval_metrics,
):
    dep_thresh = sorted(dep_thresh)
    depth_range = {}
    for idx in range(len(dep_thresh) + 1):
        if idx == 0:
            dep = dep_thresh[idx]
            depth_range[idx] = "~" + str(dep)
        elif idx == len(dep_thresh):
            dep = dep_thresh[idx - 1]
            depth_range[idx] = str(dep) + "~"
        else:
            depth_range[idx] = (
                str(dep_thresh[idx - 1]) + "~" + str(dep_thresh[idx])
            )

    # build error table
    metric_table = defaultdict(lambda: defaultdict(lambda: []))
    tp_drot = defaultdict()
    tp_dxyp = defaultdict()
    for dep in depth_range.values():
        # for line in metric_table[dep]:
        metric_table[dep]["gt_matched"] = 0
        metric_table[dep]["gt_missed"] = 0
        metric_table[dep]["dep_recall"] = 0
        tp_drot[dep] = []
        tp_dxyp[dep] = []
        for key in eval_metrics:
            metric_table[dep][key] = []

    # statistic error by depth
    for pred_scence in preds_scence:
        for info in pred_scence.objects_3d_info.values():
            if info["is_tp"] == 1.0:
                tp_match_gt_depth = info["tp_match_gt_depth"]
                range_idx = -1
                for idx, dep in enumerate(dep_thresh):
                    if tp_match_gt_depth >= dep:
                        range_idx = idx
                    else:
                        break
                metric_line = metric_table[depth_range[range_idx + 1]]
                metric_line["gt_matched"] += 1
                for metric in eval_metrics:
                    metric_line[metric].append(info[metric])
                tp_drot[depth_range[range_idx + 1]].append(info["drot"])
                tp_dxyp[depth_range[range_idx + 1]].append(info["dxyp"])

    for gt_scence in gts_scence:
        for uid, obj in gt_scence.objects_3d.items():
            if gt_scence.objects_3d_info[uid]["is_fn"] == 1.0:
                gt_depth = np.linalg.norm([obj.location[0], obj.location[1]])
                range_idx = -1
                for idx, dep in enumerate(dep_thresh):
                    if gt_depth >= dep:
                        range_idx = idx
                    else:
                        break
                metric_line = metric_table[depth_range[range_idx + 1]]
                metric_line["gt_missed"] += 1

    # cal mean error
    for dep in depth_range.values():
        for key in eval_metrics:
            elem = metric_table[dep][key]
            if len(elem) == 0:
                metric_table[dep][key] = 0
            else:
                metric_table[dep][key] = sum(elem) / len(elem)
        if (
            metric_table[dep]["gt_missed"] + metric_table[dep]["gt_matched"]
            > 0
        ):
            metric_table[dep]["dep_recall"] = metric_table[dep][
                "gt_matched"
            ] / (
                metric_table[dep]["gt_missed"]
                + metric_table[dep]["gt_matched"]
            )
        (
            metric_table[dep]["drot@90%"],
            metric_table[dep]["drot@95%"],
        ) = cal_percentile_vaule(tp_drot[dep], [0.9, 0.95])
        (
            metric_table[dep]["dxyp@90%"],
            metric_table[dep]["dxyp@95%"],
        ) = cal_percentile_vaule(tp_dxyp[dep], [0.9, 0.95])

    return metric_table


def pick_det_one_cls(
    all_preds,
    all_gts,
    eval_cls,
    enable_ignore=True,
):
    all_preds = deepcopy(all_preds)
    all_gts = deepcopy(all_gts)
    for pred, gt in zip(all_preds, all_gts):
        pred.update_id()
        gt.update_id()
    gt_count = 0
    for pred, gt in match_by_key(all_preds, all_gts, "scence_key"):
        if pred is None:
            continue
        # reset pred sample eval type
        for oid, obj3d in pred.objects_3d.items():
            pred_cls = obj3d.category.lower()
            is_tp = pred.objects_3d_info[oid]["is_tp"]
            if is_tp == 0:
                if eval_cls.lower() != pred_cls:
                    pred.objects_3d_info[oid]["is_tp"] = -1
            elif is_tp == 1:
                match_gt_uid = pred.objects_3d_info[oid]["match_gt_obj3d_id"]
                match_gt, match_gt_id = gt.get_obj3d_by_uid(
                    match_gt_uid, return_id=True
                )
                if pred_cls in ["no_cls_output", "ignore"]:
                    pred.objects_3d_info[oid]["is_tp"] = -1
                    gt.objects_3d_info[match_gt_id]["is_fn"] = -1

                elif eval_cls.lower() not in match_gt.category.lower().split(
                    "|"
                ):
                    pred.objects_3d_info[oid]["is_tp"] = -1
                    gt.objects_3d_info[match_gt_id]["is_fn"] = -1

                elif pred_cls not in match_gt.category.lower().split("|"):
                    pred.objects_3d_info[oid]["is_tp"] = 0

        # reset gt sample eval type
        for oid, obj3d in gt.objects_3d.items():
            gt_cls = obj3d.category.lower()
            if (
                eval_cls.lower() in gt_cls.split("|")
                and gt.objects_3d_info[oid]["is_fn"] != -1
            ):
                if not enable_ignore:
                    gt_count += 1
                elif enable_ignore and not obj3d.ignore:
                    gt_count += 1
            else:
                gt.objects_3d_info[oid]["is_fn"] = -1

    return all_preds, all_gts, gt_count


def gather_all_cls(cls_data):
    preds_list = [data["all_preds"] for data in cls_data.values()]
    gts_list = [data["all_gts"] for data in cls_data.values()]
    # gather gt eval result by all cls
    gts_ret = []
    for cls_gts in zip(*gts_list):
        new_gt_scence = deepcopy(cls_gts[0])
        for idx, gt_objs in enumerate(
            zip(*[gt.objects_3d.values() for gt in cls_gts])
        ):
            ignore = np.all([gt.ignore for gt in gt_objs])
            list(new_gt_scence.objects_3d.values())[idx].ignore = ignore
        for idx, gt_objs_info in enumerate(
            zip(*[gt.objects_3d_info.values() for gt in cls_gts])
        ):
            is_fn = 1 in [gt.get("is_fn", 1) for gt in gt_objs_info]
            if np.all(
                np.array([gt.get("is_fn", 1) for gt in gt_objs_info])[:] == -1
            ):
                is_fn = -1
            list(new_gt_scence.objects_3d_info.values())[idx]["is_fn"] = int(
                is_fn
            )
        new_gt_scence.update_id()
        gts_ret.append(new_gt_scence)
    # gather pred eval result by all cls
    preds_ret = []
    for cls_preds in zip(*preds_list):
        new_pred_scence = deepcopy(cls_preds[0])
        for idx, pred_objs in enumerate(
            zip(*[pred.objects_3d.values() for pred in cls_preds])
        ):
            ignore = np.all([pred.ignore for pred in pred_objs])
            list(new_pred_scence.objects_3d.values())[idx].ignore = ignore
        for idx, pred_objs_info in enumerate(
            zip(*[pred.objects_3d_info.values() for pred in cls_preds])
        ):
            for pred_cls in pred_objs_info:
                if pred_cls["is_tp"] in [0, 1]:
                    for k, v in pred_cls.items():
                        list(new_pred_scence.objects_3d_info.values())[idx][
                            k
                        ] = v
        new_pred_scence.update_id()
        preds_ret.append(new_pred_scence)
    return preds_ret, gts_ret


def is_within_eval_vcs_range(obj3d, eval_vcs_range):
    assert len(eval_vcs_range) == 4  # (bottom, right, top, left)
    if not (
        eval_vcs_range[0] < obj3d.bbox_3d.x < eval_vcs_range[2]
        and eval_vcs_range[1] < obj3d.bbox_3d.y < eval_vcs_range[-1]
    ):
        return False
    return True


def update_metric_by_eval_vcs_range(
    pred_obj3d, matched_gt_obj3d, eval_vcs_range
):
    gt_ignore_by_eval_vcs_range = False
    pred_ignore_by_eval_vcs_range = False
    if eval_vcs_range is None:
        return gt_ignore_by_eval_vcs_range, pred_ignore_by_eval_vcs_range

    if pred_obj3d is not None and matched_gt_obj3d is None:
        if not is_within_eval_vcs_range(pred_obj3d, eval_vcs_range):
            pred_ignore_by_eval_vcs_range = True

    elif pred_obj3d is not None and matched_gt_obj3d is not None:
        if not is_within_eval_vcs_range(matched_gt_obj3d, eval_vcs_range):
            pred_ignore_by_eval_vcs_range = True
            gt_ignore_by_eval_vcs_range = True

    elif pred_obj3d is None and matched_gt_obj3d is not None:
        if not is_within_eval_vcs_range(matched_gt_obj3d, eval_vcs_range):
            gt_ignore_by_eval_vcs_range = True

    return gt_ignore_by_eval_vcs_range, pred_ignore_by_eval_vcs_range


def cal_filtered_diff(raw_input, sigma=10):
    raw_input = np.array(raw_input, dtype=float)
    filtered_input = gaussian_filter1d(raw_input, sigma, mode="nearest")

    filtered_diff = (abs(filtered_input - raw_input) / filtered_input).mean()
    return filtered_diff


def find_track_obj_by_scence_key(tracklet_list, scence_key):
    rt_track_obj = None
    for track_obj in tracklet_list:
        if track_obj.scence_key == scence_key:
            rt_track_obj = track_obj
            break
    return rt_track_obj


def get_tracklet_temporal_info(tracklet_list):
    for _, tracklet_obj in enumerate(tracklet_list):
        pre_tracklet_obj = find_track_obj_by_scence_key(
            tracklet_list, tracklet_obj.pre_scence_key
        )
        if not pre_tracklet_obj:
            continue
        if (
            "global_loc" not in tracklet_obj.object_3d_info
            or tracklet_obj.object_3d_info["global_loc"] is None
        ):
            continue
        if (
            "global_loc" not in pre_tracklet_obj.object_3d_info
            or pre_tracklet_obj.object_3d_info["global_loc"] is None
        ):
            continue
        assert (
            tracklet_obj.delta_time >= 0
        ), "non-first tracklet_obj should have delta_time attribute."
        if (
            not hasattr(tracklet_obj.object_3d, "velocity")
            or tracklet_obj.object_3d.velocity is None
        ):
            cal_velocity = list(
                map(
                    lambda x: (x[0] - x[1]) / (tracklet_obj.delta_time + 1e-6),
                    zip(
                        tracklet_obj.object_3d_info["global_loc"],
                        pre_tracklet_obj.object_3d_info["global_loc"],
                    ),
                )
            )
        else:
            cal_velocity = tracklet_obj.object_3d.velocity
        tracklet_obj.object_3d_info.update({"velocity": cal_velocity})

        if (
            not hasattr(tracklet_obj.object_3d, "acceleration")
            or tracklet_obj.object_3d.acceleration is None
        ):
            if "velocity" in pre_tracklet_obj.object_3d_info:
                cal_acceleration = list(
                    map(
                        lambda x: (x[0] - x[1])
                        / (tracklet_obj.delta_time + 1e-6),
                        zip(
                            tracklet_obj.object_3d_info["velocity"],
                            pre_tracklet_obj.object_3d_info["velocity"],
                        ),
                    )
                )
            else:
                cal_acceleration = None
        else:
            cal_acceleration = tracklet_obj.object_3d.acceleration
        tracklet_obj.object_3d_info.update({"acceleration": cal_acceleration})

        if (
            not hasattr(tracklet_obj.object_3d, "yaw_rate")
            or tracklet_obj.object_3d.yaw_rate is None
        ):
            cal_yaw_rate = np.rad2deg(
                tracklet_obj.object_3d_info["global_yaw"]
                - pre_tracklet_obj.object_3d_info["global_yaw"]
            ) / (tracklet_obj.delta_time + 1e-6)
        else:
            cal_yaw_rate = tracklet_obj.object_3d.yaw_rate
        tracklet_obj.object_3d_info.update({"yaw_rate": cal_yaw_rate})

    return tracklet_list


def cal_tracklet_error(gt_tracklet, pred_tracklet):
    assert len(gt_tracklet) >= len(
        pred_tracklet
    ), "len(pred_tracklet) longer than len(gt_tracklet)!"
    metric_list = [
        "velocity",
        "acceleration",
        "yaw_rate",
        "global_yaw",
        "global_loc",
    ]
    per_tracklet_err = {
        "global_yaw_err": 0,
        "global_loc_err": np.array([0, 0, 0]),
        "velocity_err": np.array([0, 0, 0]),
        "acceleration_err": np.array([0, 0, 0]),
        "yaw_rate_err": 0,
        "filtered_locx_err": 0,
        "filtered_locy_err": 0,
    }
    raw_loc_x = []
    raw_loc_y = []
    for pred_instance, gt_instance in match_by_key(
        pred_tracklet, gt_tracklet, "scence_key"
    ):
        ego_pose = gt_instance.object_3d_info["ego_pose"]
        pred_instance.update_global_loc(ego_pose)

    gt_tracklet = get_tracklet_temporal_info(gt_tracklet)
    pred_tracklet = get_tracklet_temporal_info(pred_tracklet)

    for pred_instance, gt_instance in match_by_key(
        pred_tracklet, gt_tracklet, "scence_key"
    ):
        # calucate error vs gt
        for metric in metric_list:
            gt_metric = gt_instance.object_3d_info.get(metric, None)
            pred_metric = pred_instance.object_3d_info.get(metric, None)
            if gt_metric and pred_metric:
                metric_err = (
                    list(
                        map(
                            lambda x: abs(x[0] - x[1]),
                            zip(pred_metric, gt_metric),
                        )
                    )
                    if isinstance(gt_metric, list)
                    else abs(pred_metric - gt_metric)
                )
                per_tracklet_err[f"{metric}_err"] = (
                    per_tracklet_err[f"{metric}_err"] + np.array(metric_err)
                    if isinstance(metric_err, list)
                    else per_tracklet_err[f"{metric}_err"] + metric_err
                )
                pred_instance.object_3d_info.update(
                    {f"{metric}_err": metric_err}
                )

        raw_loc_x.append(pred_instance.object_3d.location[0])
        raw_loc_y.append(pred_instance.object_3d.location[1])
    # calucate error vs filtered_tracklet
    per_tracklet_err["filtered_locx_err"] = cal_filtered_diff(raw_loc_x)
    per_tracklet_err["filtered_locy_err"] = cal_filtered_diff(raw_loc_y)

    return per_tracklet_err


def get_clip_errors(all_gt_clips, all_pred_clips):
    total_clips_err = {
        "tracklet_count": 0,
        "clip_count": 0,
        "global_yaw_err": 0,
        "global_loc_err": np.array([0, 0, 0]),
        "velocity_err": np.array([0, 0, 0]),
        "acceleration_err": np.array([0, 0, 0]),
        "yaw_rate_err": 0,
        "filtered_locx_err": 0,
        "filtered_locy_err": 0,
    }
    results = {
        "gt_tracklets": defaultdict(lambda: {}),
        "pred_tracklets": defaultdict(lambda: {}),
    }
    clip_count = 0
    total_tracklet_count = 0
    for pred_clip, gt_clip in match_by_key(
        all_pred_clips, all_gt_clips, "clip_timestamp"
    ):
        per_clip_err = {
            "global_yaw_err": 0,
            "global_loc_err": np.array([0, 0, 0]),
            "velocity_err": np.array([0, 0, 0]),
            "acceleration_err": np.array([0, 0, 0]),
            "yaw_rate_err": 0,
            "filtered_locx_err": 0,
            "filtered_locy_err": 0,
        }
        tracklet_count = 0
        if pred_clip:
            gt_tracklets = gt_clip.get_tracklets()
            pred_tracklets = pred_clip.get_tracklets()
            for track_id, gt_tracklet in gt_tracklets.items():
                if track_id not in pred_tracklets:  # TODO： tracklet长度
                    continue
                pred_tracklet = pred_tracklets[track_id]

                # get tracklet-wise metric
                per_tracklet_err = cal_tracklet_error(
                    gt_tracklet, pred_tracklet
                )
                per_clip_err["velocity_err"] = (
                    per_clip_err["velocity_err"]
                    + per_tracklet_err["velocity_err"]
                )
                per_clip_err["acceleration_err"] = (
                    per_clip_err["acceleration_err"]
                    + per_tracklet_err["acceleration_err"]
                )
                per_clip_err["yaw_rate_err"] = (
                    per_clip_err["yaw_rate_err"]
                    + per_tracklet_err["yaw_rate_err"]
                )
                per_clip_err["filtered_locx_err"] = (
                    per_clip_err["filtered_locx_err"]
                    + per_tracklet_err["filtered_locx_err"]
                )
                per_clip_err["filtered_locy_err"] = (
                    per_clip_err["filtered_locy_err"]
                    + per_tracklet_err["filtered_locy_err"]
                )
                tracklet_count += 1

            for key in per_clip_err.keys():
                per_clip_err[key] = per_clip_err[key] / (tracklet_count + 1e-3)

            total_clips_err["velocity_err"] = (
                total_clips_err["velocity_err"] + per_clip_err["velocity_err"]
            )
            total_clips_err["acceleration_err"] = (
                total_clips_err["acceleration_err"]
                + per_clip_err["acceleration_err"]
            )
            total_clips_err["yaw_rate_err"] = (
                total_clips_err["yaw_rate_err"] + per_clip_err["yaw_rate_err"]
            )
            total_clips_err["filtered_locx_err"] = (
                total_clips_err["filtered_locx_err"]
                + per_clip_err["filtered_locx_err"]
            )
            total_clips_err["filtered_locy_err"] = (
                total_clips_err["filtered_locy_err"]
                + per_clip_err["filtered_locy_err"]
            )
            clip_count += 1

        total_tracklet_count += tracklet_count
        results["gt_tracklets"][gt_clip.clip_timestamp] = [
            {id: [tracklet.convert_to_dict() for tracklet in gt_tracklet]}
            for id, gt_tracklet in gt_tracklets.items()
        ]
        results["pred_tracklets"][pred_clip.clip_timestamp] = [
            {id: [tracklet.convert_to_dict() for tracklet in pred_tracklet]}
            for id, pred_tracklet in pred_tracklets.items()
        ]

    for key in total_clips_err.keys():
        total_clips_err[key] = total_clips_err[key] / (clip_count + 1e-3)

    total_clips_err["tracklet_count"] = total_tracklet_count
    total_clips_err["clip_count"] = clip_count

    return total_clips_err, results
