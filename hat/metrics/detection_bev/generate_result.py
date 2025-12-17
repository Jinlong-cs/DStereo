import copy
import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from hat.metrics.detection2d.generate_result import (
    NpEncoder,
    draw_compare_curves,
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


def draw_all_curves(eval_files_path: List[str], eval_names: List[str]):
    """Draw bev 3d eval curcv.

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
        if np.any(["results" in k for k in eval_result_list[0].keys()])
        else False
    )
    if eval_multi_category:
        curve = []
        for key in eval_result_list[0].keys():
            if "results" in key:
                cls = key.split("_results")[0]
                result_one_cls_file_list = [
                    os.path.join(output_dirs[idx], f"{cls}_eval_result.json")
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


def generate_bev_det_result(
    results: Dict,
    output_dir: str,
    gt_result_json: Optional[str] = "gts_result.json",
    pred_result_json: Optional[str] = "preds_result.json",
    eval_result_json: Optional[str] = "eval_result.json",
    result_png: Optional[str] = "result.png",
    table_json: Optional[str] = "tables.json",
    eval_multi_category: Optional[List[str]] = None,
    eval_match_method: Optional[str] = "3d iou",
):
    # delete unnecessary item
    def _del_key(data, key):
        assert isinstance(data, dict)
        if key in data:
            data.pop(key)

    for gt in results["gts_list"]:
        for v in gt["image_objects"].values():
            _del_key(v, "ignore_mask")
            _del_key(v, "objects_3d")
            _del_key(v["meta"], "prelabel_objects_2d")
    for pred in results["preds_list"]:
        for v in pred["image_objects"].values():
            _del_key(v, "objects_3d")
            _del_key(v["meta"], "prelabel_objects_2d")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write gt/pred/eval results json
    gts_list = results.pop("gts_list")
    preds_list = results.pop("preds_list")
    gt_result_json_file = os.path.join(output_dir, gt_result_json)
    with open(gt_result_json_file, "w") as f:
        json.dump(gts_list, f, cls=NpEncoder)
    pred_result_json_file = os.path.join(output_dir, pred_result_json)
    with open(pred_result_json_file, "w") as f:
        json.dump(preds_list, f, cls=NpEncoder)
    eval_result_json_file = os.path.join(output_dir, eval_result_json)
    with open(eval_result_json_file, "w") as f:
        json.dump(results, f, cls=NpEncoder)

    # write PR curves
    if eval_multi_category is None:
        draw_curves(
            [eval_result_json_file],
            ["all"],
            os.path.join(output_dir, result_png),
        )
    else:
        cls_json_list = [eval_result_json_file]
        for cls in eval_multi_category:
            cls_results = results[f"{cls}_results"]
            result_json_one_cls_file = os.path.join(
                output_dir, f"{cls}_" + eval_result_json
            )
            cls_json_list.append(result_json_one_cls_file)
            with open(result_json_one_cls_file, "w") as f:
                json.dump(cls_results, f, cls=NpEncoder)
        tmp_eval_multi_category = copy.deepcopy(eval_multi_category)
        tmp_eval_multi_category.insert(0, "all")
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
        results,
        prefix="3d",
        suffix=f" (based on {eval_match_method})",
    )  # noqa
    if eval_multi_category is not None:
        for cls in eval_multi_category:
            cls_results = results[f"{cls}_results"]
            tables += generate_tables(
                cls_results,
                prefix=f"3d sub_type:{cls}",
                suffix=f" (based on {eval_match_method})",
            )
    with open(tables_json_file, "w") as f:
        json.dump(tables, f, indent=2, cls=NpEncoder)


def generate_temporal_bev_det_result(
    results: Dict,
    output_dir: str,
    gt_result_json: Optional[str] = "gts_temporal_result.json",
    pred_result_json: Optional[str] = "preds_temporal_result.json",
    table_json: Optional[str] = "tables.json",
    eval_match_method: Optional[str] = "3d iou",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_tracklets_list = results.pop("gt_tracklets")
    pred_tracklets_list = results.pop("pred_tracklets")
    gt_result_json_file = os.path.join(output_dir, gt_result_json)
    with open(gt_result_json_file, "w") as f:
        json.dump(gt_tracklets_list, f, cls=NpEncoder)
    pred_result_json_file = os.path.join(output_dir, pred_result_json)
    with open(pred_result_json_file, "w") as f:
        json.dump(pred_tracklets_list, f, cls=NpEncoder)

    tables_json_file = os.path.join(output_dir, table_json)
    old_tables = (
        list(json.load(open(tables_json_file)))
        if os.path.exists(tables_json_file)
        else []
    )
    temporal_table = [
        {
            "name": f" Temporal results overview (based on {eval_match_method})",  # noqa
            "header": list(results["eval_results"].keys()),
            "data": [results["eval_results"]],
        }
    ]
    tables = old_tables + temporal_table
    with open(tables_json_file, "w") as f:
        json.dump(tables, f, indent=2, cls=NpEncoder)

    return None
