# Copyright (c) Horizon Robotics. All rights reserved.
# Generate results like AR、AP and so on according GT and DET.

import json
import os
import shutil
from io import BytesIO
from multiprocessing import Pool

import matplotlib
import numpy as np
from tqdm import tqdm

from .generate_data import gen_det_cls_one_data
from .utils import calap, calar, get_optimal_metrics

matplotlib.use("Agg")


def gen_results(
    data,
    result_json,
    tables_json,
    errors_json,
    target_recalls=None,
    target_precisions=None,
    target_thresholds=None,
    prefix="",
    suffix="",
    reset_ignore_class=False,
):
    if target_recalls is None:
        target_recalls = []
    if target_precisions is None:
        target_precisions = []
    if target_thresholds is None:
        target_thresholds = []
    images = data["images"]

    gts = []
    dets = []
    for image_key in images:
        gts.extend(data["gts_dict"][image_key])
        dets.extend(data["dets_dict"][image_key])

    num_image = len(images)
    num_gt = len(
        [
            1
            for gt in gts
            if gt["gt_type"] == "normal" and gt["eval_type"] in ["FN", "TP"]
        ]
    )

    tp = np.array([int(det["eval_type"] == "TP") for det in dets])
    fp = np.array([int(det["eval_type"] == "FP") for det in dets])
    fn = np.array(
        [
            int(gt["gt_type"] == "normal" and gt["eval_type"] == "FN")
            for gt in gts
        ]
    )
    ignore = np.array([int(det["eval_type"] == "IGNORE") for det in dets])
    for tp_, fp_, ignore_ in zip(tp, fp, ignore):
        assert tp_ + fp_ + ignore_ == 1

    conf = np.array([det["score"] for det in dets])
    # sort_by_score_then_area = np.array(
    #     [det['score'] * 1E10 -
    #      (det['bbox']['x2']-det['bbox']['x1'])*
    #      (det['bbox']['y2']-det['bbox']['y1'])
    #      for det in dets])
    # idx = np.argsort(-sort_by_score_then_area, axis=0)
    idx = np.argsort(-conf, axis=0)
    conf = conf[idx]
    tp = np.require(tp[idx], dtype=np.float64)
    fp = np.require(fp[idx], dtype=np.float64)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    fn = np.cumsum(fn)
    recall = tp / (num_gt + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    fppi = fp / float(num_image)
    fppi += 1e-15
    detection_rate = (tp - fp) / (num_gt + 1e-10)
    ar = calar(fppi, recall)
    ap, recall, precision = calap(recall, precision)
    recall = np.array(recall)
    precision = np.array(precision)

    reset_ignore_class = reset_ignore_class and not len(recall)
    results = {}
    results["aps"] = {
        "ap": ap,
        "rec": recall[-1]
        if len(recall) > 0
        else 0.0
        if reset_ignore_class
        else np.inf,
        "detection_rate": detection_rate[-1]
        if len(detection_rate) > 0
        else 0.0
        if reset_ignore_class
        else np.inf,
        "precision": precision[-1]
        if len(precision) > 0
        else 0.0
        if reset_ignore_class
        else np.inf,
        "ar": ar,
        "num_image": num_image,
        "num_gt": num_gt,
        "num_tp": tp[-1] if len(tp) else 0,
        "num_fp": fp[-1] if len(fp) else 0,
        "num_fn": fn[-1] if len(fn) else 0,
    }
    results["recall"] = recall.tolist()
    results["precision"] = precision.tolist()
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
        rec = sub_tp / (num_gt + 1e-10)
        pre = sub_tp / (sub_tp + sub_fp + 1e-10)
        thres_rec_pre_tp_fp_gt.append(
            [thres, rec, pre, sub_tp, sub_fp, num_gt]
        )
        results["scores_lut"][float(thres)] = {
            "threshold": thres,
            "recall": rec,
            "precision": pre,
            "num_tp": sub_tp,
            "num_fp": sub_fp,
            "num_gt": num_gt,
        }
    thres_rec_pre_tp_fp_gt = np.array(thres_rec_pre_tp_fp_gt)

    (
        optimal_thres,
        max_det_rate,
        optiaml_recall,
        optiaml_precision,
    ) = get_optimal_metrics(results["scores_lut"])
    results["aps"].update(
        {
            "MaxDetRate": max_det_rate if not reset_ignore_class else 0.0,
            "Threshold@MaxDetRate": optimal_thres
            if not reset_ignore_class
            else 0.0,
            "Recall@MaxDetRate": optiaml_recall
            if not reset_ignore_class
            else 0.0,
            "Precision@MaxDetRate": optiaml_precision
            if not reset_ignore_class
            else 0.0,
        }
    )

    results["target_recalls"] = {}
    if len(thres_rec_pre_tp_fp_gt) == 0:
        pass
    else:
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

    num_gt_normal = len(list(filter(lambda g: g["gt_type"] == "normal", gts)))
    num_gt_hard = len(list(filter(lambda g: g["gt_type"] == "hard", gts)))
    num_gt_ignore = len(list(filter(lambda g: g["gt_type"] == "ignore", gts)))
    num_gt_tp = len(
        list(
            filter(
                lambda g: g["eval_type"] == "TP" and g["gt_type"] == "normal",
                gts,
            )
        )
    )
    num_gt_fn = len(
        list(
            filter(
                lambda g: g["eval_type"] == "FN" and g["gt_type"] == "normal",
                gts,
            )
        )
    )
    num_det_fp = len(list(filter(lambda d: d["eval_type"] == "FP", dets)))
    num_det_tp = len(list(filter(lambda d: d["eval_type"] == "TP", dets)))
    num_det_ign = len(list(filter(lambda d: d["eval_type"] == "IGNORE", dets)))

    results["gt"] = {
        "NORMAL": num_gt_normal,
        "HARD": num_gt_hard,
        "IGNORE": num_gt_ignore,
    }
    results["gt_eval"] = {"TP": num_gt_tp, "FN": num_gt_fn}
    results["det_eval"] = {
        "TP": num_det_tp,
        "FP": num_det_fp,
        "IGNORE": num_det_ign,
    }

    errors = {}
    for det in dets:
        if det["eval_type"] == "TP":
            for error_type in det["error"]:
                errors.setdefault(error_type, []).append(
                    det["error"][error_type]
                )

    results["errors"] = {}
    for error_type in sorted(errors):
        tmp = np.array(errors[error_type])
        mean = np.mean(tmp)
        percentage = "_err" in error_type
        d = {
            "name": "%s (%%)" % error_type if percentage else error_type,
            "mean": round(mean * 100, 4) if percentage else round(mean, 4),
            "std": round(np.std(tmp), 4),
            "p0.9": round(np.percentile(tmp, 90) * 100, 4)
            if percentage
            else round(np.percentile(tmp, 90), 4),
        }
        results["errors"][error_type] = d

    with open(result_json, "w") as f:
        json.dump(results, f, cls=NpEncoder)

    old_tables = (
        list(json.load(open(tables_json)))
        if os.path.exists(tables_json)
        else []
    )
    tables = old_tables + generate_tables(results, prefix, suffix)
    with open(tables_json, "w") as f:
        json.dump(tables, f, indent=2, cls=NpEncoder)

    with open(errors_json, "w") as f:
        error_list = list(
            map(lambda k: {"name": k, "data": errors[k]}, errors)
        )
        json.dump(error_list, f, indent=2, cls=NpEncoder)

    summary = {"AP": ap, "AR": ar}
    return summary


def generate_tables(results, prefix="", suffix=""):
    tables = []
    if len(prefix):
        prefix = prefix + " "
    if len(suffix):
        suffix = " " + suffix
    aps_table = {
        "name": prefix + "overview" + suffix,
        "header": list(results["aps"].keys()),
        "data": [results["aps"]],
    }
    tables.append(aps_table)
    if results.get("max_detection_rate", {}):
        max_detection_rate_table = {
            "name": prefix + "max detection_rate overview" + suffix,
            "header": list(results["max_detection_rate"].keys()),
            "data": [results["max_detection_rate"]],
        }
        tables.append(max_detection_rate_table)
    if results.get("errors", {}):
        errors = results["errors"]
        errors_table = {
            "name": prefix + "errors" + suffix,
            "header": ["name", "mean", "std", "p0.9"],
            "data": list(
                map(lambda error_type: errors[error_type], sorted(errors))
            ),
        }
        tables.append(errors_table)

    if results.get("target_precisions", {}):
        target_precisions = sorted(results["target_precisions"])
        header = ["target_precision"] + list(
            results["target_precisions"][target_precisions[0]].keys()
        )
        target_precision_table = {
            "name": prefix + "target precision" + suffix,
            "header": header,
            "data": [],
        }
        for target_precision in sorted(results["target_precisions"]):
            row_dict = results["target_precisions"][target_precision]
            row_dict["target_precision"] = target_precision
            target_precision_table["data"].append(row_dict)
        tables.append(target_precision_table)

    if results.get("target_recalls", {}):
        target_recalls = sorted(results["target_recalls"])
        header = ["target_recall"] + list(
            results["target_recalls"][target_recalls[0]].keys()
        )
        target_recall_table = {
            "name": prefix + "target recall" + suffix,
            "header": header,
            "data": [],
        }
        for target_recall in sorted(results["target_recalls"]):
            row_dict = results["target_recalls"][target_recall]
            row_dict["target_recall"] = target_recall
            target_recall_table["data"].append(row_dict)
        tables.append(target_recall_table)

    if results.get("target_thresholds", {}):
        target_thresholds = sorted(results["target_thresholds"])
        header = ["target_threshold"] + list(
            results["target_thresholds"][target_thresholds[0]].keys()
        )
        target_threshold_table = {
            "name": prefix + "target threshold" + suffix,
            "header": header,
            "data": [],
        }
        for target_threshold in sorted(results["target_thresholds"]):
            row_dict = results["target_thresholds"][target_threshold]
            row_dict["target_threshold"] = target_threshold
            target_threshold_table["data"].append(row_dict)
        tables.append(target_threshold_table)

    if results.get("tp_errors_by_depth", {}):
        depth_ranges = list(results["tp_errors_by_depth"].keys())
        header = ["tp_errors_by_depth"] + list(
            results["tp_errors_by_depth"][depth_ranges[0]].keys()
        )
        tp_errors_table = {
            "name": prefix + "tp_errors_by_depth" + suffix,
            "header": header,
            "data": [],
        }
        for depth_range in depth_ranges:
            row_dict = results["tp_errors_by_depth"][depth_range]
            row_dict["tp_errors_by_depth"] = depth_range
            tp_errors_table["data"].append(row_dict)
        tables.append(tp_errors_table)

    return tables


def draw_compare_curves(eval_files, eval_names):
    """Draw compare curves of multi result file.

    Args:
        eval_files: array-like, a list of file objects
        eval_names: array-like, a list of eval names
    """
    import pylab

    fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=(25, 7))
    min_threshold = 0
    max_threshold = 1
    for eval_file, fn in zip(eval_files, eval_names):
        results = json.load(eval_file)
        recall = results["recall"]
        precision = results["precision"]
        fppi = results["fppi"]
        conf = results["conf"]
        detection_rate = (
            results["detection_rate"]
            if results.get("detection_rate") is not None
            else results["accuracy"]
        )

        ax1.plot(
            recall,
            precision,
            label="{} ap:{:0.4f} rec: {:0.4f} detection_rate:{:0.4f} num_gt:{}".format(  # noqa: E501
                fn,
                results["aps"]["ap"],
                results["aps"]["rec"],
                results["aps"]["detection_rate"]
                if results["aps"].get("detection_rate") is not None
                else results["aps"]["acc"],
                results["aps"]["num_gt"],
            ),
        )
        if "num_scence" in results["aps"]:
            text = "num_scence"
        else:
            text = "num_image"
        ax2.semilogx(
            fppi,
            recall,
            label="{} ar:{:0.4f} {}:{}".format(
                fn, results["aps"]["ar"], text, results["aps"][text]
            ),
        )
        ax3.plot(conf, precision, label="{}: precision".format(fn))
        ax3.plot(conf, recall, label="{}: recall".format(fn))
        ax3.plot(conf, detection_rate, label="{}: detection_rate".format(fn))
        min_threshold = min(conf + [min_threshold])
        max_threshold = max(conf + [max_threshold])

    ax1.grid(True)
    ax1.legend(loc="lower left", borderaxespad=0.0, fontsize="xx-small")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xticks(np.arange(0.0, 1, 0.05))
    ax1.set_yticks(np.arange(0.0, 1, 0.05))
    ax1.set_title("recall vs precision")

    ax2.grid(True)
    ax2.legend(loc="lower right", borderaxespad=0.0, fontsize="xx-small")
    ax2.set_xlabel("fppi")
    ax2.set_ylabel("recall")
    ax2.set_xlim([0.001, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xticks(np.power(10, np.arange(-3, -0.2, 0.2)))
    ax2.set_yticks(np.arange(0.0, 1.0, 0.05))
    ax2.set_title("fppi vs recall")

    ax3.grid(True)
    ax3.legend(loc="lower left", borderaxespad=0.0, fontsize="xx-small")
    ax3.set_xlabel("threshold")
    ax3.set_ylabel("recall & precision & detection_rate")
    ax3.set_xlim([min_threshold, max_threshold])
    ax3.set_ylim([0, 1])
    ax3.set_xticks(
        np.arange(
            min_threshold,
            max_threshold + 0.00001,
            (max_threshold - min_threshold) / 20.0,
        )
    )
    ax3.set_yticks(np.arange(0.0, 1.0, 0.05))
    ax3.set_title("thr vs recall & precision & detection_rate")

    for ax in fig.axes:
        matplotlib.pyplot.sca(ax)
        pylab.xticks(rotation=45)

    b_io = BytesIO()
    fig.savefig(b_io)
    b_io.seek(0)
    pylab.close(fig)
    return b_io


def draw_curves(eval_files, eval_names, output_file):
    b_io = draw_compare_curves(
        [open(eval_file, "r") for eval_file in eval_files], eval_names
    )
    with open(output_file, "wb") as f:
        f.write(b_io.read())
    return b_io


def write_eval_stats(
    data,
    output_dir,
    target_recalls=None,
    target_precisions=None,
    target_thresholds=None,
    mode="DET",
):
    if target_recalls is None:
        target_recalls = []
    if target_precisions is None:
        target_precisions = []
    if target_thresholds is None:
        target_thresholds = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_json = os.path.join(output_dir, f"{mode}_all.json")
    with open(output_json, "w") as f:
        json.dump(data, f)

    result_json = os.path.join(output_dir, f"{mode}_result.json")
    tables_json = os.path.join(output_dir, f"{mode}_tables.json")
    errors_json = os.path.join(output_dir, f"{mode}_errors.json")
    summary = gen_results(
        data,
        result_json=result_json,
        tables_json=tables_json,
        errors_json=errors_json,
        target_recalls=target_recalls,
        target_precisions=target_precisions,
        target_thresholds=target_thresholds,
    )
    draw_curves(
        [result_json],
        ["all"],
        os.path.join(output_dir, f"{mode}_result.png"),
    )
    return summary


def _write_eval_stats_worker(
    data,
    det_cls,
    output_dir,
    target_recalls,
    target_precisions,
    target_thresholds,
    mode,
    keep_ignore_data=True,
):
    data_new = gen_det_cls_one_data(
        data,
        det_cls,
        keep_ignore_data=keep_ignore_data,
    )

    prefix = f"{mode}_{det_cls}"
    output_json = os.path.join(output_dir, f"{prefix}_all.json")
    with open(output_json, "w") as f:
        json.dump(data_new, f)

    result_json = os.path.join(output_dir, f"{prefix}_result.json")
    tables_json = os.path.join(output_dir, f"{prefix}_tables.json")
    errors_json = os.path.join(output_dir, f"{prefix}_errors.json")
    result_png = os.path.join(output_dir, f"{prefix}_result.png")
    summary = gen_results(
        data_new,
        result_json=result_json,
        tables_json=tables_json,
        errors_json=errors_json,
        target_recalls=target_recalls,
        target_precisions=target_precisions,
        target_thresholds=target_thresholds,
        prefix=det_cls if mode != "DET_CLS_SINGLE" else "",
        reset_ignore_class=not keep_ignore_data,
    )
    draw_curves([result_json], [det_cls], result_png)

    ap = summary["AP"]
    ar = summary["AR"]

    results = json.load(open(result_json))
    aps_data = {"icon": "", "category": det_cls, **results["aps"]}
    aps_data["samples"] = ""

    target_thresholds_data = results.get("target_thresholds", {})

    res = {
        "ap": ap,
        "ar": ar,
        "eval_type": det_cls,
        "aps_data": aps_data,
        "target_thresholds": target_thresholds_data,
        "output_json": output_json,
        "result_json": result_json,
        "tables_json": tables_json,
        "errors_json": errors_json,
        "result_png": result_png,
    }
    return res


def _warp_write_eval_stats_worker(args):
    return _write_eval_stats_worker(*args)


def write_eval_stats_sep(
    eval_types,
    data,
    output_dir,
    target_recalls=None,
    target_precisions=None,
    target_thresholds=None,
    num_workers=0,
    mode="DET_CLS_SEP",
    keep_ignore_data=True,
):
    if target_recalls is None:
        target_recalls = []
    if target_precisions is None:
        target_precisions = []
    if target_thresholds is None:
        target_thresholds = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ap_list = []
    aps_data = []
    target_thresholds_data = {}

    args = [
        (
            data,
            det_cls,
            output_dir,
            target_recalls,
            target_precisions,
            target_thresholds,
            mode,
            keep_ignore_data,
        )
        for det_cls in eval_types
    ]
    if num_workers > 1:
        with Pool(num_workers) as p:
            stats_results = list(
                p.imap(
                    _warp_write_eval_stats_worker,
                    tqdm(
                        args, desc="eval per cls (thread %d) ..." % num_workers
                    ),
                )
            )
    else:
        stats_results = list(
            map(
                lambda x: _write_eval_stats_worker(*x),
                tqdm(args, desc="eval per cls (thread %d) ..." % num_workers),
            )
        )

    output_json = os.path.join(output_dir, f"{mode}_all.json")
    result_json = os.path.join(output_dir, f"{mode}_result.json")
    tables_json = os.path.join(output_dir, f"{mode}_tables.json")
    errors_json = os.path.join(output_dir, f"{mode}_errors.json")
    result_png = os.path.join(output_dir, f"{mode}_result.png")

    tables = []
    for stats_res in stats_results:
        ap_list.append(stats_res["ap"])
        aps_data.append(stats_res["aps_data"])
        target_thresholds_per_cls = stats_res["target_thresholds"]

        tables_json_per_cls = stats_res["tables_json"]
        old_tables = (
            json.load(open(tables_json_per_cls))
            if os.path.exists(tables_json_per_cls)
            else []
        )
        tables = tables + old_tables

        for target_threshold in target_thresholds_per_cls:
            target_thresholds_data.setdefault(target_threshold, {})
            for target_type in target_thresholds_per_cls[target_threshold]:
                target_thresholds_data[target_threshold].setdefault(
                    target_type, []
                )
                target_thresholds_data[target_threshold][target_type].append(
                    target_thresholds_per_cls[target_threshold][target_type]
                )

    if len(stats_results) > 1:
        mean_ap = float(sum(ap_list)) / len(ap_list)
        mean_precision = sum([ap["precision"] for ap in aps_data]) / len(
            aps_data
        )
        mean_recall = sum([ap["rec"] for ap in aps_data]) / len(aps_data)
        map_table = {
            "name": "mAP",
            "header": ["mAP", "mRecall", "mPrecision"],
            "data": [
                {
                    "mAP": mean_ap,
                    "mRecall": mean_recall,
                    "mPrecision": mean_precision,
                }
            ],
        }
        aps_table = {
            "name": "total aps overview",
            "header": list(aps_data[0].keys()),
            "data": aps_data,
        }
        mean_recall_precision_table = {
            "name": "mean recall precision overview",
            "header": [
                "target_threshold",
                "threshold",
                "mRecall",
                "mPrecision",
                "num_tp",
                "num_fp",
                "num_gt",
            ],
            "data": [],
        }
        for target_threshold in target_thresholds_data:
            target_threshold_row = {
                "target_threshold": target_threshold,
                **target_thresholds_data[target_threshold],
            }
            for target_type in target_thresholds_data[target_threshold]:
                target_data = target_threshold_row[target_type]
                if target_type == "threshold":
                    target_threshold_row[target_type] = min(target_data)
                if target_type in ["recall", "precision"]:
                    target_threshold_row.pop(target_type)
                    target_threshold_row[f"m{target_type.capitalize()}"] = sum(
                        target_data
                    ) / len(stats_results)
                if target_type in ["num_tp", "num_fp", "num_gt"]:
                    target_threshold_row[target_type] = sum(target_data)
            mean_recall_precision_table["data"].append(target_threshold_row)

        tables = [map_table, aps_table, mean_recall_precision_table] + tables

        res = {
            "mAP": mean_ap,
            "mRecall": mean_recall,
            "mPrecision": mean_precision,
        }
    else:
        res = {"AP": stats_results[0]["ap"], "AR": stats_results[0]["ar"]}
        shutil.copy(stats_results[0]["output_json"], output_json)
        shutil.copy(stats_results[0]["result_json"], result_json)
        shutil.copy(stats_results[0]["errors_json"], errors_json)
        shutil.copy(stats_results[0]["result_png"], result_png)

    with open(tables_json, "w") as f:
        json.dump(tables, f, indent=2)

    return res


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding="utf-8")
        else:
            return super(NpEncoder, self).default(obj)
