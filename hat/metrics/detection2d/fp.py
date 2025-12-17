# Copyright (c) Horizon Robotics. All rights reserved.

import json
import os

import numpy as np

from hat.core.eval_setting.parameter_def import DetectionSettingConfig
from hat.metrics.detection2d.fp_draw_charts import (
    draw_det_pie,
    draw_gt_pie,
    draw_histogram,
)
from hat.registry import OBJECT_REGISTRY
from .single import Detection2dSingle


@OBJECT_REGISTRY.register
class Detection2dFP(Detection2dSingle):
    """Single-frame image detection fp metrics.

    Args:
        config: config setting.
        save_dir: path used to save json file.
    """

    def __init__(self, config: DetectionSettingConfig, save_dir: str):
        super().__init__(config, save_dir)
        self.save_dir = save_dir

    def get(self):
        super().get()
        output_dir = self.save_dir
        fp_config = self.config

        det_eval_result_path = os.path.join(output_dir, "all.json")
        with open(det_eval_result_path) as fin:
            det_eval_result = json.load(fin)

        samples, fp_count = parse_result(det_eval_result, fp_config)

        histogram_io = draw_histogram(fp_count)
        histogram_file = os.path.join(output_dir, "histogram.png")
        with open(histogram_file, "wb") as fout:
            fout.write(histogram_io.read())

        gt_pie_io = draw_gt_pie(fp_count)
        gt_pie_file = os.path.join(output_dir, "gt_pie.png")
        with open(gt_pie_file, "wb") as fout:
            fout.write(gt_pie_io.read())

        det_pie_io = draw_det_pie(fp_count)
        det_pie_file = os.path.join(output_dir, "det_pie.png")
        with open(det_pie_file, "wb") as fout:
            fout.write(det_pie_io.read())

        samples_file = os.path.join(output_dir, "samples.json")
        with open(samples_file, "w") as fout:
            json.dump(samples, fout, indent=2)

        fp_count_file = os.path.join(output_dir, "fp_count.json")
        with open(fp_count_file, "w") as fout:
            json.dump(fp_count, fout, indent=2)

        result_tables_file = os.path.join(output_dir, "tables.json")
        with open(result_tables_file, "w") as fout:
            tables = get_result_tables(samples, fp_count)
            json.dump(tables, fout, indent=2)

        total_fp_gt = sum(list(map(lambda fp: fp["fp_gt"], fp_count)))
        total_fp_det = sum(list(map(lambda fp: fp["fp_det"], fp_count)))
        micro_fp_ratio = (
            total_fp_det / float(total_fp_gt) if total_fp_gt else "NaN"
        )
        macro_fp_ratio = (
            np.mean(list(map(lambda fp: fp["fp_ratio"], fp_count)))
            if len(fp_count)
            else "NaN"
        )
        return {"micro FP": micro_fp_ratio, "macro FP": macro_fp_ratio}


def parse_result(evaluation_all_result, config):
    image_keys = evaluation_all_result["images"]
    gts_dict = evaluation_all_result["gts_dict"]
    dets_dict = evaluation_all_result["dets_dict"]

    fp_count = {}
    samples = []

    for image_key in image_keys:
        fp_type = None
        matched_gt_id = None
        image_samples = {"image_key": image_key, "dets": [], "gts": []}
        samples.append(image_samples)
        dets = dets_dict[image_key]
        gts = gts_dict[image_key]
        for det in dets:
            if det["eval_type"] == "TP":
                eval_type = "FP"
                f = []
                for gt in gts:
                    if gt["id"] == det["matched_gt_id"]:
                        gt["det"] = det
                        f.append(1)
                        fp_type = gt["gt_attrs"][config["negative_key"]]
                assert np.sum(f) == 1
                # is_fp = True
                matched_gt_id = det["matched_gt_id"]
            elif det["eval_type"] == "IGNORE":
                eval_type = "IGNORE"
                matched_gt_id = None
                fp_type = None
            elif det["eval_type"] == "FP":
                eval_type = "UNKNOWN"
                matched_gt_id = None
                fp_type = None
            else:
                raise Exception("Invalid eval_type: %s" % det["eval_type"])
            det = {
                "id": det["id"],
                "bbox": det["bbox"],
                "bbox_type": det["bbox_type"],
                "score": det["score"],
                "eval_type": eval_type,
                "fp_type": fp_type,
                "matched_gt_id": matched_gt_id,
            }
            image_samples["dets"].append(det)
        for gt in gts:
            if gt["gt_type"] == "normal":
                fp_type = gt["gt_attrs"][config["negative_key"]]
                if fp_type not in fp_count:
                    fp_count[fp_type] = {"fp_gt": 0, "fp_det": 0}
                fp_count[fp_type]["fp_gt"] += 1
                if gt["eval_type"] == "TP":
                    fp_count[fp_type]["fp_det"] += 1
                    eval_type = "MATCHED"
                elif gt["eval_type"] == "FN":
                    eval_type = "UNMATCHED"
                else:
                    raise Exception("Invalid eval_type: %s" % gt["eval_type"])
            elif gt["gt_type"] == "hard":
                fp_type = None
                eval_type = "HARD"
            elif gt["gt_type"] == "ignore":
                fp_type = None
                eval_type = "IGNORE"
            elif gt["gt_type"] == "remove":
                fp_type = None
                eval_type = "REMOVE"
                matched_gt_id = None
            else:
                raise Exception("Invalid gt_type: %s" % gt["gt_type"])
            gt = {
                "id": gt["id"],
                "bbox": gt["bbox"],
                "bbox_type": gt["bbox_type"],
                "fp_type": fp_type,
                "eval_type": eval_type,
                "matched_gt_id": matched_gt_id,
                "gt_attrs": gt["gt_attrs"],
            }
            image_samples["gts"].append(gt)

    fp_count = list(
        map(
            lambda fp_type: {
                "fp_type": fp_type,
                "fp_gt": fp_count[fp_type]["fp_gt"],
                "fp_det": fp_count[fp_type]["fp_det"],
                "fp_ratio": fp_count[fp_type]["fp_det"]
                / float(fp_count[fp_type]["fp_gt"])
                if fp_count[fp_type]["fp_gt"]
                else "NaN",
            },
            fp_count,
        )
    )
    fp_count.sort(key=lambda f: f["fp_gt"], reverse=True)
    return samples, fp_count


def get_result_tables(samples, fp_count):
    # total result table
    num_images = len(samples)
    total_fp_gt = sum(list(map(lambda fp: fp["fp_gt"], fp_count)))
    total_fp_det = sum(list(map(lambda fp: fp["fp_det"], fp_count)))
    micro_fp_ratio = (
        total_fp_det / float(total_fp_gt) if total_fp_gt else "NaN"
    )
    macro_fp_ratio = (
        np.mean(list(map(lambda fp: fp["fp_ratio"], fp_count)))
        if len(fp_count)
        else "NaN"
    )
    fppi = total_fp_det / float(len(samples))
    total_result_table = {
        "header": [
            "num_images",
            "total_fp_gt",
            "total_fp_det",
            "micro_fp_ratio",
            "macro_fp_ratio",
            "fppi",
        ],
        "data": [
            {
                "num_images": num_images,
                "total_fp_gt": total_fp_gt,
                "total_fp_det": total_fp_det,
                "micro_fp_ratio": micro_fp_ratio,
                "macro_fp_ratio": macro_fp_ratio,
                "fppi": fppi,
            }
        ],
        "name": "total fp result",
    }
    types_result_table = {
        "header": ["fp_type", "fp_gt", "fp_det", "fp_ratio"],
        "data": fp_count,
        "name": "fp result of each types",
    }
    tables = {"tables": [total_result_table, types_result_table]}
    return tables
