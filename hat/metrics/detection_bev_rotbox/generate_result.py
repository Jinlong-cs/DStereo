import copy
import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
import numpy as np


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


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj) -> Any:
        return (
            super().encode(bool(obj))
            if isinstance(obj, np.bool_)
            else super().default(obj)
        )


def generate_tables(results, prefix="", suffix=""):
    tables = []
    if len(prefix):
        prefix = prefix + " "
    if len(suffix):
        suffix = " " + suffix
    aps_table = {
        "name": prefix + "Overview" + suffix,
        "header": list(results["aps"].keys()),
        "data": [results["aps"]],
    }
    tables.append(aps_table)
    if results.get("dep_interval", {}):
        dep_interval = results["dep_interval"]
        header = ["dep_interval"] + list(
            dep_interval[list(dep_interval.keys())[0]].keys()
        )
        target_precision_table = {
            "name": prefix + suffix,
            "header": header,
            "data": [],
        }
        for interval in dep_interval.keys():
            row_dict = dep_interval[interval]
            row_dict["dep_interval"] = interval
            target_precision_table["data"].append(row_dict)
        tables.append(target_precision_table)
    if results.get("dist_interval", {}):
        dist_interval = results["dist_interval"]
        header = ["dist_interval"] + list(
            dist_interval[list(dist_interval.keys())[0]].keys()
        )
        target_precision_table = {
            "name": prefix + suffix,
            "header": header,
            "data": [],
        }
        for interval in dist_interval.keys():
            row_dict = dist_interval[interval]
            row_dict["dist_interval"] = interval
            target_precision_table["data"].append(row_dict)
        tables.append(target_precision_table)

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
    return tables


def gen_results(
    results: Dict,
    output_dir: str,
    eval_vcs_range: tuple,
    eval_multi_category: dict,
    result_json: Optional[str] = "result.json",
    result_png: Optional[str] = "result.png",
    table_json: Optional[str] = "tables.json",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # write results json
    result_json_file = os.path.join(output_dir, result_json)
    with open(result_json_file, "w") as f:
        json.dump(results, f, cls=NpEncoder)
    if eval_multi_category is None:
        # draw pr
        draw_curves(
            [result_json_file], ["all"], os.path.join(output_dir, result_png)
        )
    else:
        cls_json_list = []
        tmp_eval_multi_category = copy.deepcopy(eval_multi_category)
        tmp_eval_multi_category.insert(0, "all")
        cls_json_list.append(os.path.join(output_dir, result_json))
        for cls in eval_multi_category:
            cls_results = results.get(f"{cls}_result", {})
            if len(cls_results) == 0:
                tmp_eval_multi_category.remove(cls)
                continue
            result_json_one_cls_file = os.path.join(
                output_dir, f"{cls}_" + result_json
            )
            cls_json_list.append(result_json_one_cls_file)
            with open(result_json_one_cls_file, "w") as f:
                json.dump(cls_results, f, cls=NpEncoder)
        draw_curves(
            cls_json_list,
            tmp_eval_multi_category,
            os.path.join(output_dir, result_png),
        )

    # write table
    tables_json_file = os.path.join(output_dir, table_json)
    old_tables = (
        list(json.load(open(tables_json_file)))
        if os.path.exists(tables_json_file)
        else []
    )
    tables = old_tables + generate_tables(
        results["all_result"],
        prefix="Catrgory: All",
        suffix="",
    )  # noqa
    if eval_multi_category is not None:
        for cls in eval_multi_category:
            cls_results = results.get(f"{cls}_result", {})
            if not cls_results:
                continue
            vcs_range = eval_vcs_range[cls]
            tables += generate_tables(
                cls_results,
                prefix=f"Catrgory: {cls}",
                suffix=f"Eval_vcs_range: {vcs_range}",
            )
    with open(tables_json_file, "w") as f:
        json.dump(tables, f, indent=2, cls=NpEncoder)


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
    for eval_file, eval_name in zip(eval_files, eval_names):
        results = json.load(eval_file)
        if "all_result" in results:
            results = results.pop("all_result")
        recall = results["recalls"]
        precision = results["precisions"]
        fppi = results["fppi"]
        conf = results["conf"]
        detection_rate = results["detection_rate"]

        ax1.plot(
            recall,
            precision,
            label="{} AP:{:0.4f} Recall:{:0.4f} Precision: {:0.4f} tp:{} fn:{} fp:{}".format(  # noqa: E501
                eval_name,
                results["aps"]["AP"],
                results["aps"]["Recall"],
                results["aps"]["Precision"],
                results["aps"]["tp"],
                results["aps"]["fn"],
                results["aps"]["fp"],
            ),
        )
        ax2.semilogx(
            fppi,
            recall,
            label="{} Recall:{:0.4f} tp:{} fn:{} fp:{}".format(
                eval_name,
                results["aps"]["Recall"],
                results["aps"]["tp"],
                results["aps"]["fn"],
                results["aps"]["fp"],
            ),
        )
        ax3.plot(conf, precision, label="{}: precision".format(eval_name))
        ax3.plot(conf, recall, label="{}: recall".format(eval_name))
        ax3.plot(
            conf, detection_rate, label="{}: detection_rate".format(eval_name)
        )
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


def draw_all_curves(eval_files_path: List[str], eval_names: List[str]):
    """Draw bev rotbox eval curve.

    Args:
        eval_files: List of eval result files.
        eval_names: List of prediction names.
    """
    assert len(eval_files_path) == len(eval_names)
    output_dirs = [os.path.dirname(out_dir) for out_dir in eval_files_path]
    eval_files = [open(file_path) for file_path in eval_files_path]
    eval_result_list = [json.load(file) for file in eval_files]
    [file.seek(0) for file in eval_files]
    eval_multi_category = (
        True
        if np.any(["all_result" in k for k in eval_result_list[0].keys()])
        else False
    )
    if eval_multi_category:
        curve = []
        for key in eval_result_list[0].keys():
            if "all_result" in key:
                cls = key.split("_result")[0]
                result_one_cls_file_list = [
                    os.path.join(output_dirs[idx], "result.json")
                    for idx in range(len(eval_files))
                ]
                result_file_exists = [
                    os.path.exists(result_cls_file)
                    for result_cls_file in result_one_cls_file_list
                ]
                if not np.all(result_file_exists):
                    continue
                cls_legend_list = [name + f" {cls}" for name in eval_names]
                cls_curve = draw_compare_curves(
                    [open(file) for file in result_one_cls_file_list],
                    cls_legend_list,
                )
                curve.append(cls_curve)
        curve_list = [
            cv2.imdecode(np.fromstring(c.getvalue(), np.uint8), 1)
            for c in curve
        ]
        curve_cat = np.concatenate(curve_list, axis=0)
        _, curve = cv2.imencode(".png", curve_cat)
        curve = BytesIO(curve.tostring())
    else:
        curve = draw_compare_curves(eval_files, eval_names)
    return curve
