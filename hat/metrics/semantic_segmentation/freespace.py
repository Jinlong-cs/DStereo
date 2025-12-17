# Copyright (c) Horizon Robotics. All rights reserved.
# Each label in freespace can be merged into one class,
# ref to: https://horizonrobotics.feishu.cn/wiki/wikcnFCRJbcfUzZ5vBXQy0GyoQh
import collections

import numpy as np
from sklearn.metrics import confusion_matrix

from .get_files import get_distance_transform, get_line


def cvt_freespace(conf, freespace_id):
    """Calculate confusion matrix according to freespace_id.

    Args:
        conf: confusion matrix of pred and gt.
        freespace_id: list of freespace id.
    """
    try:
        conf.shape
    except BaseException:
        return 0
    # clsnum = conf.shape[0]
    sum_c = conf.sum(0)
    sum_c_freespace = conf[freespace_id, :].sum(0)
    sum_c_nofreespace = sum_c - sum_c_freespace
    conf_f = np.zeros((2, 2))
    conf_f[0, 0] = sum_c_freespace[freespace_id].sum()
    conf_f[0, 1] = sum_c_freespace.sum() - conf_f[0, 0]
    conf_f[1, 0] = sum_c_nofreespace[freespace_id].sum()
    conf_f[1, 1] = sum_c_nofreespace.sum() - conf_f[1, 0]
    return conf_f


def freespace_category_metrics(confusion_matrix, labels_dict, freespace_id):
    """Calculate metrics according to freespace_id.

    Args:
        confusion_matrix: confusion matrix of pred and gt.
        labels_dict: labels dict. The key means label, the value
            describes the label.
        freespace_id: list of the freespace id.
    """
    num_gt_per_category = confusion_matrix.sum(axis=1)
    num_pred_per_category = confusion_matrix.sum(axis=0)
    categorys_recalls = np.diag(confusion_matrix) / num_gt_per_category.astype(
        np.float64
    )
    categorys_precisions = np.diag(
        confusion_matrix
    ) / num_pred_per_category.astype(np.float64)
    category_f1_scores = (
        2
        * categorys_recalls
        * categorys_precisions
        / (categorys_recalls + categorys_precisions)
    )

    freespace_category_metrics_dict = dict()  # noqa: C408
    label_id = 0
    for idx, label_info in labels_dict.items():
        if idx not in freespace_id:
            freespace_category_metrics_dict[
                label_info["name"]
            ] = dict()  # noqa: C408
            freespace_category_metrics_dict[label_info["name"]][
                "precision"
            ] = round(categorys_precisions[label_id], 4)
            freespace_category_metrics_dict[label_info["name"]][
                "recall"
            ] = round(categorys_recalls[label_id], 4)
            freespace_category_metrics_dict[label_info["name"]]["F1"] = round(
                category_f1_scores[label_id], 4
            )
            freespace_category_metrics_dict[label_info["name"]][
                "num_gt"
            ] = int(num_gt_per_category[label_id])
            freespace_category_metrics_dict[label_info["name"]][
                "num_pred"
            ] = int(num_pred_per_category[label_id])
            label_id += 1

    return freespace_category_metrics_dict


def eval_freespace(
    gt_label,
    pred_label,
    clsnum,
    conf,
    freespace_dict,
    margin_flag,
    no_freespace_id,
):
    freespace_eval_dict = collections.OrderedDict()
    assert "freespace_id" in freespace_dict
    freespace_id = freespace_dict["freespace_id"]
    anno_map = list(map(lambda x: 1 if x < clsnum else 255, range(256)))

    for i in freespace_id:
        anno_map[i] = 0
    anno_map = np.array(anno_map)
    ignore = -1
    # draw gt_line of freespace, replace gt ignore region with pred label
    gt_no_ignore = gt_label.copy()
    gt_no_ignore[np.where(gt_label == 255)] = pred_label[
        np.where(gt_label == 255)
    ]
    gt_line, gt_line_categorys = get_line(gt_no_ignore, anno_map, ignore)
    pred_line, pred_line_categorys = get_line(pred_label, anno_map, ignore)
    gt_line_categorys[pred_line_categorys == ignore] = ignore

    conf_f = cvt_freespace(conf, freespace_id)
    class_pixels_f = conf_f.sum(1)
    class_pixels_f += class_pixels_f == 0  # avoid zero in denominator
    TP_f = np.diag(conf_f)
    # TP = conf[xrange(num_classes), xrange(num_classes)]
    FN_f = class_pixels_f - TP_f
    FP_f = conf_f.sum(0) - TP_f
    cls_ious_f = TP_f / (FN_f + TP_f + FP_f + 0.0)
    cls_accs_f = TP_f / (TP_f + FP_f + 0.0)
    conf_f_categorys = confusion_matrix(
        gt_line_categorys, pred_line_categorys, labels=no_freespace_id
    )

    freespace_eval_dict["conf_f"] = conf_f
    freespace_eval_dict["conf_f_categorys"] = conf_f_categorys
    freespace_eval_dict["cls_ious_f"] = cls_ious_f
    freespace_eval_dict["cls_accs_f"] = cls_accs_f
    freespace_eval_dict["TP_f"] = TP_f
    freespace_eval_dict["FN_f"] = FN_f
    freespace_eval_dict["FP_f"] = FP_f
    freespace_eval_dict["margin_flag"] = margin_flag
    freespace_eval_dict["gt_line"] = gt_line
    freespace_eval_dict["pred_line"] = pred_line
    freespace_eval_dict["pred_line_categorys"] = pred_line_categorys
    freespace_eval_dict["gt_line_categorys"] = gt_line_categorys
    freespace_eval_dict["no_freespace_id"] = no_freespace_id

    if margin_flag == 1:
        margin_list = freespace_dict["margin"]
        freespace_eval_dict["margin_list"] = margin_list
        assert len(margin_list) > 0
        dist_vertical = abs(pred_line - gt_line) * np.array(gt_line >= 0)
        gt_distance_map = get_distance_transform(
            gt_line, gt_label.shape[0], gt_label.shape[1]
        )
        dist_around = []
        for i in range(gt_label.shape[1]):
            dist_around.append(gt_distance_map[int(pred_line[i]) - 1, i])
        dist_around = dist_around * np.array(gt_line >= 0)
        vertical_margins_acc = [
            (dist_vertical <= m).sum() / float(len(dist_vertical))
            for m in margin_list
        ]
        around_margins_acc = [
            (dist_around <= m).sum() / float(len(dist_around))
            for m in margin_list
        ]
        gt_line_categorys_with_margin = [
            gt_line_categorys[dist_vertical <= m] for m in margin_list
        ]
        pred_line_categorys_with_margin = [
            pred_line_categorys[dist_vertical <= m] for m in margin_list
        ]
        conf_f_categorys_with_margin = []
        for m in range(len(margin_list)):
            if (
                len(gt_line_categorys_with_margin[m]) == 0
                or len(pred_line_categorys_with_margin[m]) == 0
            ):
                conf_margin = np.zeros(
                    (len(no_freespace_id), len(no_freespace_id)),
                    dtype=np.int64,
                )
            elif np.sum(gt_line_categorys_with_margin[m] == 255) == len(
                gt_line_categorys_with_margin[m]
            ) or np.sum(pred_line_categorys_with_margin[m] == 255) == len(
                pred_line_categorys_with_margin[m]
            ):
                # avoid error if all gt labels are 255 for confusion_matrix()
                conf_margin = np.zeros(
                    (len(no_freespace_id), len(no_freespace_id)),
                    dtype=np.int64,
                )
            else:
                conf_margin = confusion_matrix(
                    gt_line_categorys_with_margin[m],
                    pred_line_categorys_with_margin[m],
                    labels=no_freespace_id,
                )
            conf_f_categorys_with_margin.append(conf_margin)
        freespace_eval_dict["dist_vertical"] = dist_vertical
        freespace_eval_dict["dist_around"] = dist_around
        freespace_eval_dict[
            "conf_f_categorys_with_margin"
        ] = conf_f_categorys_with_margin
    else:
        vertical_margins_acc = None
        around_margins_acc = None
    freespace_eval_dict["vertical_margins_acc"] = vertical_margins_acc
    freespace_eval_dict["around_margins_acc"] = around_margins_acc

    return freespace_eval_dict


def get_freespace_res(
    TPs_f,
    FNs_f,
    FPs_f,
    conf_all_f,
    dist_vertical_all,
    dist_around_all,
    has_margin,
    has_threshold,
    margin_list,
    vertical_margins_acc_all,
    around_margins_acc_all,
    conf_all_f_categorys_with_margin,
    labels_dict,
    freespace_dict,
    threshold_list,
    reslist,
):
    freespace_res_dict = dict()  # noqa: C408
    cls_ious_all_f = TPs_f / (FNs_f + TPs_f + FPs_f + 0.0).astype(np.float64)
    cls_accs_all_f = TPs_f / (FPs_f + TPs_f + 0.0).astype(np.float64)
    cls_recs_all_f = TPs_f / (FNs_f + TPs_f + 0.0).astype(np.float64)

    # freespace general result
    freespace_general_dict = dict()  # noqa: C408
    freespace_general_dict["cls_ious_all_f"] = cls_ious_all_f
    freespace_general_dict["cls_accs_all_f"] = cls_accs_all_f
    freespace_general_dict["cls_recs_all_f"] = cls_recs_all_f
    freespace_general_dict["conf_all_f"] = conf_all_f
    freespace_res_dict["freespace_general_res"] = freespace_general_dict

    # freespace distance statistic
    freespace_distance_dict = dict()  # noqa: C408
    freespace_distance_dict["vertical_distance"] = {
        "max": int(np.max(dist_vertical_all)),
        "min": int(np.min(dist_vertical_all)),
        "mean": round(np.mean(dist_vertical_all), 4),
        "var": round(np.var(dist_vertical_all), 4),
        "std": round(np.std(dist_vertical_all), 4),
        "90 percentile": int(np.percentile(dist_vertical_all, 90)),
        "95 percentile": int(np.percentile(dist_vertical_all, 95)),
        "98 percentile": int(np.percentile(dist_vertical_all, 98)),
    }
    freespace_distance_dict["around_distance"] = {
        "max": int(np.max(dist_around_all)),
        "min": int(np.min(dist_around_all)),
        "mean": round(np.mean(dist_around_all), 4),
        "var": round(np.var(dist_around_all), 4),
        "std": round(np.std(dist_around_all), 4),
        "90 percentile": int(np.percentile(dist_around_all, 90)),
        "95 percentile": int(np.percentile(dist_around_all, 95)),
        "98 percentile": int(np.percentile(dist_around_all, 98)),
    }
    freespace_res_dict["freespace_distance_res"] = freespace_distance_dict

    # freespace margin result
    freespace_margin_res_dict = dict()  # noqa: C408
    if has_margin == 1:
        freespace_margin_res_dict["margin_list"] = margin_list
        vertical_margins_acc_all = np.array(vertical_margins_acc_all)
        around_margins_acc_all = np.array(around_margins_acc_all)
        freespace_vertical_margin_acc = vertical_margins_acc_all.mean(axis=0)
        freespace_around_margin_acc = around_margins_acc_all.mean(axis=0)
        freespace_margin_res_dict[
            "freespace_vertical_margin_acc"
        ] = freespace_vertical_margin_acc
        freespace_margin_res_dict[
            "freespace_around_margin_acc"
        ] = freespace_around_margin_acc
        freespace_margin_res_dict["category_metrics_dict"] = []
        for i in range(len(margin_list)):
            freespace_margin_res_dict["category_metrics_dict"].append(
                freespace_category_metrics(
                    conf_all_f_categorys_with_margin[i],
                    labels_dict,
                    freespace_dict["freespace_id"],
                )
            )

        if has_threshold == 1:
            freespace_margin_res_dict["threshold_list"] = threshold_list
            freespace_vertical_precision = np.zeros(
                (len(margin_list), len(threshold_list))
            )
            freespace_around_precision = np.zeros(
                (len(margin_list), len(threshold_list))
            )
            for i_thred in range(len(threshold_list)):
                for j_margin in range(len(margin_list)):
                    freespace_vertical_precision[j_margin, i_thred] = len(
                        np.where(
                            vertical_margins_acc_all[:, j_margin]
                            >= threshold_list[i_thred]
                        )[0]
                    ) / (len(reslist) + 0.0)
                    freespace_around_precision[j_margin, i_thred] = len(
                        np.where(
                            around_margins_acc_all[:, j_margin]
                            >= threshold_list[i_thred]
                        )[0]
                    ) / (len(reslist) + 0.0)
            freespace_margin_res_dict[
                "freespace_vertical_precision"
            ] = freespace_vertical_precision
            freespace_margin_res_dict[
                "freespace_around_precision"
            ] = freespace_around_precision
    freespace_res_dict["freespace_margin_res"] = freespace_margin_res_dict
    return freespace_res_dict
