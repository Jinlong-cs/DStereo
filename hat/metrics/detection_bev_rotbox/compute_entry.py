import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

from hat.metrics.detection2d.utils import calap


def match_by_key(preds, gts, key):
    for gt in gts:
        id = gt.get(key)[0]
        for pred in preds:
            if pred.get(key)[0] == id:
                yield pred, gt
                break
        else:
            yield None, gt


def collect_result_by_dep(results, dep_thresh):
    results_dep = [{} for _ in range(len(dep_thresh) + 1)]
    for name, res in results.items():
        res = np.array(res)
        res_np = res[:, -2] if res.size > 1 else res[:, 0]
        if dep_thresh[0] < 0:
            dep_inds = np.sum(res_np[:, np.newaxis] > dep_thresh, axis=-1)
        else:
            dep_inds = np.sum(abs(res_np[:, np.newaxis]) > dep_thresh, axis=-1)
        for i in range(len(dep_thresh) + 1):
            results_dep[i][name] = res[dep_inds == i]

    return results_dep


def collect_result_by_dist(results, dist_thresh):
    """Split results by distance.

    Args:
        results(dict): metric results.
            example:
                dict(
                    'tp': np.array(shape=(tp_ins_num, 9)),
                    'fp': np.array(shape=(fp_ins_num, 3)),
                    'fn': np.array(shape=(fn_ins_num, 2)),
                ).
            The dim 1 elements of the tp value includes of
                (det_score, dx, dy, dxy, dw, dh, drot, gt_x, gt_y).
            The dim 1 elements of the fp value includes of
                (det_score, det_x, det_y).
            The dim 1 elements of the fn value includes of
                (gt_x, gt_y).
        dist_thresh(list): distance range to validation.
    Returns:
        (list): squence of results(dict) in each distance range.

    """
    results_dist = [{} for _ in range(len(dist_thresh) + 1)]
    for name, res in results.items():
        res = np.array(res)
        # res[:, -2:] indicate vcs location of each instance.
        res_np = (
            np.linalg.norm(res[:, -2:], axis=1) if len(res.shape) > 1 else res
        )
        dist_inds = np.sum(abs(res_np[:, np.newaxis]) > dist_thresh, axis=-1)
        for i in range(len(dist_thresh) + 1):
            results_dist[i][name] = res[dist_inds == i]

    return results_dist


def voc_ap(res, score_threshold=0):
    if len(res["tp"]) == 0:
        return 0
    scores_matched = np.array(res["tp"])[:, 0]
    redundant = np.array(res["fp"])[:, 0]

    redundant = sorted(redundant, reverse=True)
    scores_matched = sorted(scores_matched, reverse=True)
    gts_total = len(res["fn"]) + len(scores_matched)

    recalls = np.array([])
    precisions = np.array([])

    tp, fp = 0, 0
    Num_matched = len(scores_matched)
    Num_redundant = len(redundant)
    for score in np.linspace(1, score_threshold, 100):
        while tp < Num_matched:
            if scores_matched[tp] >= score:
                tp += 1
            else:
                break
        while fp < Num_redundant:
            if redundant[fp] >= score:
                fp += 1
            else:
                break
        precision = tp / (tp + fp + 1e-6)
        recalls = np.append(recalls, tp)
        precisions = np.append(precisions, precision)

    recalls /= gts_total
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def summarize_ap(
    all_info_dict: dict,
    all_result_dict: List,
    cid: int,
    category: str,
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
    all_info_list = all_info_dict["all_info_list"]
    num_frame = len(all_info_list)
    gt_count = 0
    for info in all_info_list:
        for _, frame_info in info.items():
            all_ins_info = frame_info["object_info"]
            for ins_info in all_ins_info:
                if ins_info["attr"] == "FN":
                    if cid == "all" or ins_info["vcs_discobj_cls"] == cid:
                        gt_count += 1
                    continue
                if cid == "all" or ins_info["pred_bev_discobj_cls_id"] == cid:
                    if ins_info["attr"] == "FP":
                        is_tp = 0
                    elif ins_info["attr"] == "TP":
                        is_tp = 1
                        gt_count += 1
                    res_all["tp"] += [is_tp]
                    res_all["fp"] += [1 - is_tp]
                    res_all["conf"] += [ins_info["pred_score"]]

    tp = np.array(res_all["tp"])
    fp = np.array(res_all["fp"])
    conf = np.array(res_all["conf"])

    argsort = np.argsort(-conf)
    conf = conf[argsort]
    tp = tp[argsort]
    is_tp = tp.astype(np.bool_)
    fp = fp[argsort]

    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    if gt_count == 0:
        recalls = np.zeros(tp.shape)
    else:
        recalls = tp / float(gt_count)
    assert np.all(0 <= recalls) & np.all(recalls <= 1)

    precisions = tp / (tp + fp + 1e-6)
    assert np.all(0 <= precisions) & np.all(precisions <= 1)
    _, recall, precision = calap(recalls, precisions)
    recall = np.array(recall)
    precision = np.array(precision)

    fppi = fp / float(num_frame)
    fppi += 1e-6
    detection_rate = (tp - fp) / (gt_count + 1e-6)

    results = {}
    results["recalls"] = recall.tolist()
    results["precisions"] = precision.tolist()
    results["fppi"] = fppi.tolist()
    results["conf"] = conf.tolist()
    results["detection_rate"] = detection_rate.tolist()
    results["fp"] = fp.tolist()

    results["scores_lut"] = {}
    thres_rec_pre_tp_fp_gt = []
    for thres in np.unique(conf):
        select = conf >= thres
        sub_tp = np.max(tp[select])
        sub_fp = np.max(fp[select])
        rec = sub_tp / (gt_count + 1e-10)
        pre = sub_tp / (sub_tp + sub_fp + 1e-10)
        thres_rec_pre_tp_fp_gt.append(
            [thres, rec, pre, sub_tp, sub_fp, gt_count]
        )
        results["scores_lut"][float(thres)] = {
            "threshold": thres,
            "recall": rec,
            "precision": pre,
            "num_tp": sub_tp,
            "num_fp": sub_fp,
            "num_gt": gt_count,
        }
    thres_rec_pre_tp_fp_gt = np.array(thres_rec_pre_tp_fp_gt)

    results["target_recalls"] = {}
    if len(thres_rec_pre_tp_fp_gt) == 0:
        pass
    else:
        target_recalls = np.linspace(0.5, 0.95, 9)
        for target_recall in target_recalls:
            valid_idx = np.where(thres_rec_pre_tp_fp_gt[:, 1] > target_recall)[
                0
            ]
            if valid_idx.shape[0] > 0:
                idx = valid_idx[thres_rec_pre_tp_fp_gt[valid_idx, 1].argmin()]
            else:
                idx = np.argmin(target_recall - thres_rec_pre_tp_fp_gt[:, 1])
            results["target_recalls"][target_recall] = {
                "threshold": thres_rec_pre_tp_fp_gt[idx, 0],
                "recall": thres_rec_pre_tp_fp_gt[idx, 1],
                "precision": thres_rec_pre_tp_fp_gt[idx, 2],
                "num_tp": thres_rec_pre_tp_fp_gt[idx, 3],
                "num_fp": thres_rec_pre_tp_fp_gt[idx, 4],
                "num_gt": thres_rec_pre_tp_fp_gt[idx, 5],
            }

        results["target_precisions"] = {}
        target_precisions = np.linspace(0.5, 0.95, 9)
        for target_precision in target_precisions:
            valid_idx = np.where(
                thres_rec_pre_tp_fp_gt[:, 2] > target_precision
            )[0]
            if valid_idx.shape[0] > 0:
                idx = valid_idx[thres_rec_pre_tp_fp_gt[valid_idx, 2].argmin()]
            else:
                idx = np.argmin(
                    target_precision - thres_rec_pre_tp_fp_gt[:, 2]
                )
            results["target_precisions"][target_precision] = {
                "threshold": thres_rec_pre_tp_fp_gt[idx, 0],
                "recall": thres_rec_pre_tp_fp_gt[idx, 1],
                "precision": thres_rec_pre_tp_fp_gt[idx, 2],
                "num_tp": thres_rec_pre_tp_fp_gt[idx, 3],
                "num_fp": thres_rec_pre_tp_fp_gt[idx, 4],
                "num_gt": thres_rec_pre_tp_fp_gt[idx, 5],
            }

        results["target_thresholds"] = {}
        target_thresholds = np.linspace(0.15, 0.85, 14)
        for target_threshold in target_thresholds:
            valid_idx = np.where(
                thres_rec_pre_tp_fp_gt[:, 0] > target_threshold
            )[0]
            if valid_idx.shape[0] > 0:
                idx = valid_idx[thres_rec_pre_tp_fp_gt[valid_idx, 0].argmin()]
            else:
                idx = np.argmin(
                    target_threshold - thres_rec_pre_tp_fp_gt[:, 0]
                )
            results["target_thresholds"][target_threshold] = {
                "threshold": thres_rec_pre_tp_fp_gt[idx, 0],
                "recall": thres_rec_pre_tp_fp_gt[idx, 1],
                "precision": thres_rec_pre_tp_fp_gt[idx, 2],
                "num_tp": thres_rec_pre_tp_fp_gt[idx, 3],
                "num_fp": thres_rec_pre_tp_fp_gt[idx, 4],
                "num_gt": thres_rec_pre_tp_fp_gt[idx, 5],
            }

    all_result_dict[f"{category}_result"].update(results)
    return all_result_dict


def statistic_all_info(
    format_metrics,
    eval_vcs_range,
    id2label,
    depth_intervals,
    dist_intervals,
    gt_max_depth,
    all_res,
    metrics,
    ap_score_threshold,
    score_threshold,
    eps,
    compute_foreground_prec,
):
    all_result_dict = {}
    leaderboard_results = {}
    counts_show = ["tp", "fn", "fp", "Recall", "Precision", "AP"]
    if compute_foreground_prec:
        counts_show.append("Foreground_Precision")

    def get_interval_result(full_intervals, cate_metrics_dep):
        interval_results = {}
        Num_tp, Num_fn, Num_fp, Num_fp_miscls = 0, 0, 0, 0
        for i_dep, dep in enumerate(full_intervals):
            interval_results.setdefault(dep, {})
            if len(cate_metrics_dep[i_dep + 1]["tp"]) == 0:
                for metric_name in metrics:
                    interval_results[dep][metric_name] = 0
                Num_tp_ = 0
                Num_fp_miscls_ = 0
                Num_fn_ = len(cate_metrics_dep[i_dep + 1]["fn"])
                scores_fp = cate_metrics_dep[i_dep + 1]["fp"][:, 0]
                Num_fp_ = np.sum(scores_fp >= score_threshold)
                show_nums_ = [0, Num_fn_, Num_fp_, 0, 0, 0]
                if compute_foreground_prec:
                    show_nums_.append(0)
                for counts_show_num, num in zip(counts_show, show_nums_):
                    interval_results[dep][counts_show_num] = num
            else:
                ap_ = round(
                    voc_ap(
                        cate_metrics_dep[i_dep + 1],
                        ap_score_threshold,
                    ),
                    3,
                )
                scores_tp = cate_metrics_dep[i_dep + 1]["tp"][:, 0]
                scores_fp = cate_metrics_dep[i_dep + 1]["fp"][:, 0]
                fn = cate_metrics_dep[i_dep + 1]["fn"]
                Num_tp_ = np.sum(scores_tp >= score_threshold)
                Num_fp_ = np.sum(scores_fp >= score_threshold)
                gt_totals = len(scores_tp) + len(fn)
                Num_fn_ = gt_totals - Num_tp_
                indices_matched = scores_tp >= score_threshold
                gt_matched = cate_metrics_dep[i_dep + 1]["tp"][indices_matched]

                for i_metric, metric_name in enumerate(metrics):
                    val = round(
                        np.sum(gt_matched[:, i_metric + 1]) / (Num_tp_ + eps),
                        3,
                    )
                    interval_results[dep][metric_name] = val
                recall_ = round(Num_tp_ / (gt_totals + eps), 3)
                precision_ = round(Num_tp_ / (Num_tp_ + Num_fp_ + eps), 3)
                show_nums_ = [
                    Num_tp_,
                    Num_fn_,
                    Num_fp_,
                    recall_,
                    precision_,
                    ap_,
                ]
                if compute_foreground_prec:
                    Num_fp_miscls_ = len(
                        cate_metrics_dep[i_dep + 1]["fp_miscls"]
                    )
                    fg_prec_ = round(
                        Num_tp_ / (Num_tp_ + Num_fp_miscls_ + eps), 3
                    )
                    show_nums_.append(fg_prec_)
                for counts_show_num, num in zip(counts_show, show_nums_):
                    interval_results[dep][counts_show_num] = num
            Num_tp += Num_tp_
            Num_fp += Num_fp_
            Num_fn += Num_fn_
            if compute_foreground_prec:
                Num_fp_miscls += Num_fp_miscls_
        return Num_tp, Num_fn, Num_fp, interval_results, Num_fp_miscls

    for cid, cate_metrics in format_metrics.items():
        if (
            len(cate_metrics.get("tp", [])) + len(cate_metrics.get("fn", []))
            == 0
        ):
            continue

        if cid == "all":
            category = cid
            cls_eval_vcs_range = eval_vcs_range.get(id2label[0], None)
            cls_depth_intervals = depth_intervals.get(id2label[0], None)
        else:
            category = id2label[cid]
            cls_eval_vcs_range = eval_vcs_range.get(id2label[cid], None)
            cls_depth_intervals = depth_intervals.get(id2label[0], None)
        all_result_dict.setdefault(f"{category}_result", {})
        all_result_dict = summarize_ap(all_res, all_result_dict, cid, category)

        if cls_eval_vcs_range:
            cls_depth_intervals = (
                [cls_eval_vcs_range[0]]
                + list(cls_depth_intervals)
                + [cls_eval_vcs_range[2]]
            )
        else:
            cls_depth_intervals = [0] + [20, 50, 70] + [gt_max_depth]
        full_depth_intervals = [
            f"({start},{end})"
            for start, end in zip(
                cls_depth_intervals[:-1], cls_depth_intervals[1:]
            )
        ]
        cate_metrics_dep = collect_result_by_dep(
            cate_metrics, cls_depth_intervals
        )

        Num_tp, Num_fn, Num_fp, dep_rsts, Num_fp_miscls = get_interval_result(
            full_depth_intervals, cate_metrics_dep
        )
        recall = round(Num_tp / (Num_tp + Num_fn + eps), 3)
        precision = round(Num_tp / (Num_tp + Num_fp + eps), 3)
        ap = round(voc_ap(cate_metrics, ap_score_threshold), 3)
        if cid == "all":
            leaderboard_results.update(
                dict(  # noqa
                    AP=ap,
                    Recall=recall,
                    Precision=precision,
                )
            )
        show_nums = [Num_tp, Num_fn, Num_fp, recall, precision, ap]
        if compute_foreground_prec:
            fg_prec = round(Num_tp / (Num_tp + Num_fp_miscls + eps), 3)
            show_nums.append(fg_prec)
            if cid == "all":
                leaderboard_results.update(
                    {
                        "Foreground_Precision": fg_prec,
                    }
                )
        for counts_show_num, num in zip(counts_show, show_nums):
            all_result_dict[f"{category}_result"].setdefault(
                "aps", {"threshold": score_threshold}
            )
            all_result_dict[f"{category}_result"]["aps"][counts_show_num] = num
        # update results with max detection rate to aps info.
        det_rate = all_result_dict["all_result"]["detection_rate"]
        max_det_rate = max(det_rate)
        max_det_rate_index = det_rate.index(max_det_rate)
        max_rate_score = all_result_dict["all_result"]["conf"][
            max_det_rate_index
        ]
        max_rate_pre = all_result_dict["all_result"]["precisions"][
            max_det_rate_index
        ]
        max_rate_rec = all_result_dict["all_result"]["recalls"][
            max_det_rate_index
        ]
        all_result_dict[f"{category}_result"]["aps"]["MaxDetRate"] = round(
            max_det_rate, 3
        )
        all_result_dict[f"{category}_result"]["aps"][
            "Threshod@MaxDetRate"
        ] = round(max_rate_score, 3)
        all_result_dict[f"{category}_result"]["aps"][
            "Precison@MaxDetRate"
        ] = round(max_rate_pre, 3)
        all_result_dict[f"{category}_result"]["aps"][
            "Recall@MaxDetRate"
        ] = round(max_rate_rec, 3)

        all_result_dict[f"{category}_result"].setdefault("dep_interval", {})
        all_result_dict[f"{category}_result"]["dep_interval"].update(dep_rsts)
        if dist_intervals:
            dist_bottom = round(
                math.sqrt(
                    cls_eval_vcs_range[0] ** 2
                    + max(
                        abs(cls_eval_vcs_range[1]), abs(cls_eval_vcs_range[3])
                    )
                    ** 2
                ),
                1,
            )
            dist_top = round(
                math.sqrt(
                    cls_eval_vcs_range[2] ** 2
                    + max(
                        abs(cls_eval_vcs_range[1]), abs(cls_eval_vcs_range[3])
                    )
                    ** 2
                ),
                1,
            )
            dist_max = max(dist_bottom, dist_top)
            cls_dist_intervals = dist_intervals.get(id2label[0], None)
            cls_dist_intervals = [0] + list(cls_dist_intervals) + [dist_max]
            full_dist_intervals = [
                f"({start},{end})"
                for start, end in zip(
                    cls_dist_intervals[:-1], cls_dist_intervals[1:]
                )
            ]
            cate_metrics_dist = collect_result_by_dist(
                cate_metrics, cls_dist_intervals
            )
            _, _, _, dist_rsts, _ = get_interval_result(
                full_dist_intervals, cate_metrics_dist
            )
            all_result_dict[f"{category}_result"].setdefault(
                "dist_interval", {}
            )
            all_result_dict[f"{category}_result"]["dist_interval"].update(
                dist_rsts
            )

    return all_result_dict, leaderboard_results


def gather_all_cls(
    all_res,
    eval_category_ids,
    collect_result_list,
    eval_vcs_range,
    id2label,
    depth_intervals,
    dist_intervals,
    gt_max_depth,
    metrics,
    ap_score_threshold,
    score_threshold,
    eps,
    compute_foreground_prec,
):
    format_metrics = {}
    format_metrics.setdefault("all", {})
    for cid in eval_category_ids:
        format_metrics.setdefault(cid, {})
        val = all_res.get(f"{cid}")
        valid_ind = [len(v) > 0 for v in val]
        if np.sum(np.array(valid_ind)) == 0:
            continue
        first_id = valid_ind.index(1)
        val_npy = val[first_id]
        for v in val[first_id + 1 :]:
            if len(v) == 0:
                continue
            val_npy = np.append(val_npy, v, axis=0)

        fn_index = np.sum(np.abs(val_npy[:, 2:]), axis=1) < 1e-6
        fn = val_npy[fn_index, :2]
        val_npy = val_npy[fn_index == 0]
        fp_index = np.sum(np.abs(val_npy[:, 3:]), axis=1) < 1e-6
        fp = val_npy[fp_index, :3]
        tp = val_npy[fp_index == 0]

        tmp_res = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        if compute_foreground_prec:
            assert f"{cid}_misfg_prec_data" in all_res
            val = all_res.get(f"{cid}_misfg_prec_data")
            if len(val) == 0:
                val_npy = np.zeros((0, 3), dtype=np.float32)
            else:
                valid_ind = [len(v) > 0 for v in val]
                if np.sum(np.array(valid_ind)) == 0:
                    val_npy = np.zeros((0, 3), dtype=np.float32)
                else:
                    first_id = valid_ind.index(1)
                    val_npy = val[first_id]
                    for v in val[first_id + 1 :]:
                        if len(v) == 0:
                            continue
                        val_npy = np.append(val_npy, v, axis=0)
            tmp_res["fp_miscls"] = val_npy

        for name in collect_result_list:
            if name not in format_metrics[cid]:
                format_metrics[cid][name] = tmp_res[name]
            else:
                format_metrics[cid][name] = np.append(
                    format_metrics[cid][name], tmp_res[name], axis=0
                )

            if name not in format_metrics["all"]:
                format_metrics["all"][name] = tmp_res[name]
            else:
                format_metrics["all"][name] = np.append(
                    format_metrics["all"][name], tmp_res[name], axis=0
                )

    all_result_dict, leaderboard_results = statistic_all_info(
        format_metrics=format_metrics,
        eval_vcs_range=eval_vcs_range,
        id2label=id2label,
        depth_intervals=depth_intervals,
        dist_intervals=dist_intervals,
        gt_max_depth=gt_max_depth,
        all_res=all_res,
        metrics=metrics,
        ap_score_threshold=ap_score_threshold,
        score_threshold=score_threshold,
        eps=eps,
        compute_foreground_prec=compute_foreground_prec,
    )
    assert "all_info_list" in all_res
    all_result_dict["all_info_list"] = all_res.pop("all_info_list")
    return all_result_dict, leaderboard_results
