import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from hat.metrics.detection2d.generate_result import (
    NpEncoder,
    draw_curves,
    generate_tables,
)


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj) -> Any:
        return (
            super().encode(bool(obj))
            if isinstance(obj, np.bool_)
            else super().default(obj)
        )


def cal_percentile_vaule(error_data, ratios):
    percentile_vaule = []
    if len(error_data) != 0:
        error_data_sorted = sorted(error_data)
    for ratio in ratios:
        if len(error_data) == 0:
            percentile_vaule.append(np.mean(error_data))
        else:
            index = int(len(error_data_sorted) * ratio)
            percentile_vaule.append(error_data_sorted[index])
    return percentile_vaule


def generate_error_tables(
    metrics_res: dict,
    metrics_type: List[str],
    dep_thresh: List[int],
    lidar_error: bool,
    y_thresh: List[int],
    prefix: str = "3d",
) -> List:
    if len(prefix):
        prefix = prefix + " "
    tables = []
    range_info = (
        [(0, dep_thresh[0])] if dep_thresh[0] > 0 else [(-120, dep_thresh[0])]
    )  # noqa
    for i in range(0, len(dep_thresh) - 1):
        range_info += [(dep_thresh[i], dep_thresh[i + 1])]

    img_count = metrics_res["counts"]["img_count"]
    TP = np.sum(metrics_res["counts"]["gt_matched"]).astype(np.int32)
    FP = metrics_res["counts"]["redundant_det"]
    FN = metrics_res["counts"]["gt_missed"]
    max_det_rate = metrics_res["max_det_rate"]["max_det_rate"]
    max_det_rate_score = metrics_res["max_det_rate"]["score"]
    detction_rate_table = {
        "name": prefix + "overview extend (based on 3d projection)",
        "header": [
            "valid images",
            "TP",
            "FP",
            "FN",
            "Recall",
            "Precision",
            "Max Det Rate",
            "Max Det Rate Score",
        ],  # noqa
        "data": [
            {
                "valid images": img_count,
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "Recall": round(float(TP / (TP + FN)), 4),
                "Precision": round(float(TP / (TP + FP)), 4),
                "Max Det Rate": round(float(max_det_rate), 4),
                "Max Det Rate Score": round(float(max_det_rate_score), 4),
            }
        ],
    }
    tables.append(detction_rate_table)
    # depth range table
    range_name = "depth_range" if not lidar_error else "x_range"
    metric_names = [range_name, "gt_matched"] + metrics_type.copy()
    # attention metric order
    metric_names += ["dxp@90%", "dxp@95%"]
    metric_names += ["dy@90%", "dy@95%"]

    metric_names += ["dxyp@90%", "dxyp@95%"]
    metric_names += ["drot@std", "drot@2sigma", "drot@90%", "drot@95%"]

    metric_items = [range_info, metrics_res["counts"]["gt_matched"]] + [
        metrics_res[name] for name in metrics_type
    ]
    depth_range_table = {
        "name": prefix + "error (based on 3d projection)",
        "header": metric_names,
        "data": [],
    }
    for items in zip(*metric_items):
        cur_items = [items[0], items[1]]
        ratios = [0.9, 0.95]
        extend_metric_value = []
        for error_name, error_data in zip(metrics_type, items[2:]):
            cur_items.append(np.mean(error_data))

            if error_name == "dxp":
                percentile_vaule = cal_percentile_vaule(error_data, ratios)
                extend_metric_value.extend(percentile_vaule)

            if error_name == "dy":
                percentile_vaule = cal_percentile_vaule(error_data, ratios)
                extend_metric_value.extend(percentile_vaule)

            if error_name == "dxyp":
                percentile_vaule = cal_percentile_vaule(error_data, ratios)
                extend_metric_value.extend(percentile_vaule)

            if error_name == "drot":
                mean = np.mean(error_data)
                std = np.std(error_data)
                sigma2 = mean + 2 * std
                extend_metric_value += [std, sigma2]

                percentile_vaule = cal_percentile_vaule(error_data, ratios)
                extend_metric_value.extend(percentile_vaule)
        cur_items.extend(extend_metric_value)
        detail_items = {
            k: round(v, 4) if not isinstance(v, tuple) else v
            for k, v in zip(metric_names, cur_items)
        }
        depth_range_table["data"].append(detail_items)
    tables.append(depth_range_table)
    # y range table
    if y_thresh is not None:
        y_range_info = [(-50, y_thresh[0])]
        for i in range(0, len(y_thresh) - 1):
            y_range_info += [(y_thresh[i], y_thresh[i + 1])]
        y_range_info += [(y_thresh[-1], 50)]
        metric_names = ["y_range", "gt_matched"] + metrics_type.copy()
        metric_names += ["dxp@90%", "dxp@95%"]
        metric_names += ["dy@90%", "dy@95%"]

        metric_names += ["dxyp@90%", "dxyp@95%"]
        metric_names += ["drot@std", "drot@2sigma", "drot@90%", "drot@95%"]

        metric_items = [
            y_range_info,
            metrics_res["metrics_y"]["counts"]["y_gt_matched"],  # noqa
        ] + [
            metrics_res["metrics_y"][name] for name in metrics_type
        ]  # noqa
        y_range_table = {
            "name": prefix + "error (based on 3d projection) along y axis",
            "header": metric_names,
            "data": [],
        }
        for items in zip(*metric_items):
            cur_items = [items[0], items[1]]
            ratios = [0.9, 0.95]
            extend_metric_value = []
            for error_name, error_data in zip(metrics_type, items[2:]):
                cur_items.append(np.mean(error_data))

                if error_name == "dxp":
                    percentile_vaule = cal_percentile_vaule(error_data, ratios)
                    extend_metric_value.extend(percentile_vaule)

                if error_name == "dy":
                    percentile_vaule = cal_percentile_vaule(error_data, ratios)
                    extend_metric_value.extend(percentile_vaule)

                if error_name == "dxyp":
                    percentile_vaule = cal_percentile_vaule(error_data, ratios)
                    extend_metric_value.extend(percentile_vaule)

                if error_name == "drot":
                    mean = np.mean(error_data)
                    std = np.std(error_data)
                    sigma2 = mean + 2 * std
                    extend_metric_value += [std, sigma2]

                    percentile_vaule = cal_percentile_vaule(error_data, ratios)
                    extend_metric_value.extend(percentile_vaule)

            cur_items.extend(extend_metric_value)
            detail_items = {
                k: round(v, 4) if not isinstance(v, tuple) else v
                for k, v in zip(metric_names, cur_items)
            }
            y_range_table["data"].append(detail_items)
        tables.append(y_range_table)

    return tables


def generate_all_json(metrics_res: List[str], prefix: str = "") -> Dict:
    # all.json is used to draw image.
    all_json = {
        "images": [],
        "img_meta": {},
        "gts_dict": defaultdict(list),
        "dets_dict": defaultdict(list),
    }
    #
    score_thresh = metrics_res["max_det_rate"]["score"] * 0.8
    for data in metrics_res["img_fps"]:
        image_key = data["meta"]["image_key"]
        all_json["images"].append(image_key)
        all_json["gts_dict"][image_key] = []
        all_json["dets_dict"][image_key] = []
        all_json["img_meta"][image_key] = {
            "image_height": data["meta"]["img_height"],
            "image_width": data["meta"]["img_width"],
            "distCoeffs": data["meta"]["distCoeffs"],
            "calib": data["meta"]["calib"],
            "ignore_mask": data["meta"]["ignore_mask"],
            "camera_model": data["meta"]["camera_model"],
        }
        for gt in data["gts"]:
            gt["class"] = 1
            all_json["gts_dict"][image_key].append(gt)
        max_score_det = None
        max_score = -1000000
        for det, f in zip(data["dets"], data["fp"]):
            if "eval_type" not in det.keys():
                det.update({"eval_type": "FP"}) if f == 1 else det.update(
                    {"eval_type": "ignore"}
                )  # noqa
            det["class"] = 0
            if det["score"] <= score_thresh:
                if det["score"] > max_score:
                    max_score_det = det
                    max_score = det["score"]
                continue
            all_json["dets_dict"][image_key].append(det)
        if len(all_json["dets_dict"][image_key]) == 0 and max_score_det:
            all_json["dets_dict"][image_key].append(max_score_det)
    return all_json


def generate_auto_result(
    results: Dict,
    metrics_res: Dict,
    metrics_type: List[str],
    dep_thresh: List[int],
    lidar_error: bool,
    y_thresh: List[int],
    output_dir: str,
    result_json: str = "result_auto.json",
    result_png: str = "result_auto.png",
    table_json: str = "tables.json",
    all_json: str = "all.json",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # write results json
    result_json_file = os.path.join(output_dir, result_json)
    with open(result_json_file, "w") as f:
        json.dump(results, f, cls=NpEncoder)
    # draw pr
    result_json_file = os.path.join(output_dir, result_json)
    draw_curves(
        [result_json_file], ["all"], os.path.join(output_dir, result_png)
    )
    # write table
    tables_json_file = os.path.join(output_dir, table_json)
    old_tables = (
        list(json.load(open(tables_json_file)))
        if os.path.exists(tables_json_file)
        else []
    )
    tables = old_tables + generate_tables(
        results, prefix="3d", suffix=" (based on 3d projection)"
    )  # noqa
    with open(tables_json_file, "w") as f:
        json.dump(tables, f, indent=2, cls=NpEncoder)

    # write error table
    if output_dir is not None:
        tables_json_file = os.path.join(output_dir, table_json)
        old_tables = (
            list(json.load(open(tables_json_file)))
            if os.path.exists(tables_json_file)
            else []
        )
        tables = old_tables + generate_error_tables(
            metrics_res,
            metrics_type,
            dep_thresh,
            lidar_error,
            y_thresh,
            prefix="3d",
        )
        with open(tables_json_file, "w") as f:
            json.dump(tables, f, indent=2, cls=NpEncoder)

        dump_jons_to = os.path.join(output_dir, all_json)
        all_json_file = generate_all_json(metrics_res)
        json_data = json.dumps(all_json_file, cls=CustomJSONizer)
        with open(dump_jons_to, "w") as json_file:  # noqa
            json_file.write(json_data)


# ------------------- Nuscenes Metric -------------------


def generate_nuscenes_result(
    results: Dict,
    output_dir: str,
    result_json="result.json",
    result_png="result.png",
    table_json="tables.json",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # write results json
    results = results["all_results"]
    result_json_file = os.path.join(output_dir, result_json)
    with open(result_json_file, "w") as f:
        json.dump(results, f, cls=NpEncoder)
    # draw pr
    result_json_file = os.path.join(output_dir, result_json)
    draw_curves(
        [result_json_file], ["all"], os.path.join(output_dir, result_png)
    )
    # write table
    tables_json_file = os.path.join(output_dir, table_json)
    old_tables = (
        list(json.load(open(tables_json_file)))
        if os.path.exists(tables_json_file)
        else []
    )
    tables = old_tables + generate_tables(
        results, prefix="3d", suffix=" (based on 3d iou)"
    )  # noqa
    with open(tables_json_file, "w") as f:
        json.dump(tables, f, indent=2, cls=NpEncoder)
