# Copyright (c) Horizon Robotics. All rights reserved.
import json
import os
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from hat.core.eval_setting.parameter_def import ClassificationSettingConfig
from hat.metrics.classification.curves import (
    draw_all_precision_recall_curves,
    draw_confusion_matrix,
    draw_precision_recall_curve,
)
from hat.metrics.classification.utils import (
    parse_gts,
    parse_gts_original,
    parse_preds,
    parse_preds_original,
)
from hat.metrics.classification.visualize import draw_confusion_matrix_samples
from hat.metrics.detection2d.generate_result import NpEncoder
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class ClassificationMetric(EvalMetric):
    """Single-frame image classification metrics.

    Args:
        config: config setting.
        output_dir: path used to save json file.
        image_dir: path for image dir..
    """

    def __init__(
        self,
        config: ClassificationSettingConfig,
        output_dir: str,
        image_dir: Optional[str] = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.image_dir = image_dir

        self.all_gts = {}
        self.all_preds = {}
        self.all_original_gts = {}
        self.all_original_preds = {}
        self.confusion_pred = (
            config["pred_to_gt_allow_confusion"] is not None
        ) and (len(config["pred_to_gt_allow_confusion"]) > 0)

    def update(self, gt, pred):
        self.all_gts.update(parse_gts(gt, self.config))
        self.all_preds.update(parse_preds(pred, self.config))

        if self.confusion_pred:
            self.all_original_gts.update(parse_gts_original(gt, self.config))
            self.all_original_preds.update(
                parse_preds_original(pred, self.config)
            )

    def get(self):
        return self.evaluation()

    def evaluation(self):
        config = self.config
        gts = self.all_gts
        preds = self.all_preds
        eval_categorys = config["eval_categorys"]
        eval_category_map = {
            category: label_id
            for label_id, category in enumerate(eval_categorys)
        }
        label_ids = range(len(eval_categorys))
        results = {
            "categorys": list(
                map(lambda category: {"name": category}, eval_categorys)
            )
        }
        obj_keys = gts.keys()
        gt_ids = np.array(
            list(
                map(
                    lambda obj_key: eval_category_map[
                        gts[obj_key]["category"]
                    ],
                    obj_keys,
                )
            )
        )
        pred_ids = list(
            map(lambda obj_key: preds[obj_key]["pred_id"], obj_keys)
        )

        con_matrix = confusion_matrix(gt_ids, pred_ids, labels=label_ids)

        if self.confusion_pred:
            original_gts = self.all_original_gts
            original_preds = self.all_original_preds

            less_acc_pred_item = config["pred_to_gt_allow_confusion"].keys()

            for obj_id in original_gts:
                if (
                    original_preds[obj_id]["pred_id"] in less_acc_pred_item
                    and original_gts[obj_id]["category"]
                    in config["pred_to_gt_allow_confusion"][
                        original_preds[obj_id]["pred_id"]
                    ]
                ):
                    # fix confusion matrix
                    gt_id = eval_category_map[gts[obj_id]["category"]]
                    pred_id = preds[obj_id]["pred_id"]
                    con_matrix[gt_id][pred_id] -= 1
                    con_matrix[gt_id][gt_id] += 1

        num_gt_per_category = con_matrix.sum(axis=1)
        num_pred_per_category = con_matrix.sum(axis=0)
        categorys_recalls = np.diag(con_matrix) / num_gt_per_category.astype(
            np.float64
        )
        categorys_precisions = np.diag(
            con_matrix
        ) / num_pred_per_category.astype(np.float64)
        category_f1_scores = (
            2
            * categorys_recalls
            * categorys_precisions
            / (categorys_recalls + categorys_precisions)
        )
        results.update(
            {
                "num_gt": con_matrix.sum(),
                "confusion_matrix": con_matrix.tolist(),
                "accuracy": np.diag(con_matrix).sum()
                / con_matrix.sum().astype(np.float64),
                "mean_recall": categorys_recalls[
                    ~np.isnan(categorys_recalls)
                ].mean(),
                "mean_precision": categorys_precisions[
                    ~np.isnan(categorys_precisions)
                ].mean(),
                "mean_f1_score": category_f1_scores[
                    ~np.isnan(category_f1_scores)
                ].mean(),
            }
        )

        min_gt_num = config["min_gt_num"]
        if min_gt_num > 0:
            invalid_category_idxs = []
            num_gt_per_category = con_matrix.sum(axis=1)
            for i, num_gt in enumerate(num_gt_per_category):
                if num_gt < min_gt_num:
                    invalid_category_idxs.append(i)
            valid_con_matrix = np.delete(
                con_matrix, invalid_category_idxs, axis=0
            )
            valid_con_matrix = np.delete(
                valid_con_matrix, invalid_category_idxs, axis=1
            )
            # eval_categorys = np.delete(eval_categorys, invalid_category_idxs).to[]  # noqa
            valid_num_gt_per_category = valid_con_matrix.sum(axis=1)
            valid_num_pred_per_category = valid_con_matrix.sum(axis=0)
            valid_categorys_recalls = np.diag(
                valid_con_matrix
            ) / valid_num_gt_per_category.astype(np.float64)
            valid_categorys_precisions = np.diag(
                valid_con_matrix
            ) / valid_num_pred_per_category.astype(np.float64)
            valid_category_f1_scores = (
                2
                * valid_categorys_recalls
                * valid_categorys_precisions
                / (valid_categorys_recalls + valid_categorys_precisions)
            )
            results.update(
                {
                    "num_gt": valid_con_matrix.sum(),
                    "confusion_matrix": valid_con_matrix.tolist(),
                    "accuracy": np.diag(valid_con_matrix).sum()
                    / valid_con_matrix.sum().astype(np.float64),
                    "mean_recall": valid_categorys_recalls[
                        ~np.isnan(valid_categorys_recalls)
                    ].mean(),
                    "mean_precision": valid_categorys_precisions[
                        ~np.isnan(valid_categorys_precisions)
                    ].mean(),
                    "mean_f1_score": valid_category_f1_scores[
                        ~np.isnan(valid_category_f1_scores)
                    ].mean(),
                }
            )

        for label_id, _ in enumerate(eval_categorys):
            results["categorys"][label_id].update(
                {
                    "recall": categorys_recalls[label_id],
                    "precision": categorys_precisions[label_id],
                    "num_gt": num_gt_per_category[label_id],
                    "num_pred": num_pred_per_category[label_id],
                    "f1_score": category_f1_scores[label_id],
                }
            )

        precisions_list = []
        recalls_list = []
        ap_list = []
        curves_output_dir = os.path.join(self.output_dir, "curves")
        os.makedirs(curves_output_dir, exist_ok=True)
        if config["with_scores"]:
            scores_matrix = np.array(
                list(
                    map(
                        lambda obj_key: list(preds[obj_key]["scores"]),
                        obj_keys,
                    )
                )
            )
            pr_categorys = []
            for label_id, category in enumerate(eval_categorys):
                if min_gt_num > 0:
                    if label_id in invalid_category_idxs:
                        continue
                tps = label_id == gt_ids
                if not tps.sum():
                    results["categorys"][label_id][
                        "average_precision"
                    ] = float("nan")
                    continue
                category_scores = scores_matrix[:, label_id]
                if self.confusion_pred and (
                    label_id in config["pred_to_gt_allow_confusion"]
                ):
                    fuzzy_gt_id = list(
                        map(
                            lambda category_name: eval_category_map[
                                category_name
                            ]
                            if category_name in eval_category_map
                            else eval_category_map[
                                config["gt_to_eval_category"][category_name]
                            ],
                            config["pred_to_gt_allow_confusion"][label_id],
                        )
                    )
                    fuzzy_gt_id.append(label_id)
                    tps = np.array(
                        list(map(lambda gt_id: gt_id in fuzzy_gt_id, gt_ids))
                    )
                    category_scores = np.array(
                        list(
                            map(
                                lambda item_score, tp: max(
                                    list(
                                        map(
                                            lambda gt_id: item_score[gt_id],
                                            fuzzy_gt_id,
                                        )
                                    )
                                ),
                                scores_matrix,
                                tps,
                            )
                        )
                    )
                ap = average_precision_score(tps, category_scores)
                precisions, recalls, thresholds = precision_recall_curve(
                    tps, category_scores
                )
                thresholds = np.hstack((0, thresholds))
                thresholds = [0] + thresholds
                thresholds = np.sort(thresholds)
                pr_curve_io = draw_precision_recall_curve(
                    precisions, recalls, thresholds, ap, category
                )
                with open(
                    os.path.join(
                        curves_output_dir, "pr_category_%s.png" % category
                    ),
                    "wb",
                ) as fout:
                    fout.write(pr_curve_io.read())
                pr_categorys.append(category)
                results["categorys"][label_id]["average_precision"] = ap
                ap_list.append(ap)
                precisions_list.append(precisions)
                recalls_list.append(recalls)

            all_pr_curves_io = draw_all_precision_recall_curves(
                pr_categorys, precisions_list, recalls_list, ap_list
            )
            with open(
                os.path.join(curves_output_dir, "pr_all.png"), "wb"
            ) as fout:
                fout.write(all_pr_curves_io.read())

            results["mean_average_precision"] = np.mean(ap_list)

        # remove invalid categorys
        if config["use_valid_categorys"]:
            # results
            valid_category_names = []
            invalid_items = []
            for _, category in enumerate(results["categorys"]):
                if "num_gt" in category:
                    num_gt = category["num_gt"]
                    num_pred = category["num_pred"]
                    if num_gt == 0 and num_pred == 0:
                        invalid_items.append(category)
                    else:
                        valid_category_names.append(category["name"])
                else:
                    invalid_items.append(category)
            for item in invalid_items:
                results["categorys"].remove(item)
            # confusion_matrix
            delete_idxs = []
            for _, category in enumerate(eval_categorys):
                if category not in valid_category_names:
                    delete_idxs.append(i)
            con_matrix = np.delete(con_matrix, delete_idxs, 0)
            con_matrix = np.delete(con_matrix, delete_idxs, 1)

        cm_x_label_rot = config["cm_x_label_rot"]

        if config["use_valid_categorys"]:
            cm_io = draw_confusion_matrix(
                valid_category_names,
                con_matrix,
                "Confusion Matrix",
                cm_x_label_rot,
            )  # noqa
        else:
            cm_io = draw_confusion_matrix(
                eval_categorys, con_matrix, "Confusion Matrix", cm_x_label_rot
            )  # noqa
        out_con_img_path = os.path.join(
            curves_output_dir, "confusion_matrix.png"
        )
        with open(out_con_img_path, "wb") as fout:
            fout.write(cm_io.read())

        with open(os.path.join(self.output_dir, "result.json"), "w") as fout:
            json.dump(results, fout, indent=2, cls=NpEncoder)

        objs = list(
            map(
                lambda idx_obj_key: {
                    "image_key": gts[idx_obj_key[1]]["image_key"],
                    "obj_id": gts[idx_obj_key[1]]["obj_id"],
                    "bbox": gts[idx_obj_key[1]]["bbox"],
                    "gt_id": eval_category_map[
                        gts[idx_obj_key[1]]["category"]
                    ],
                    "pred_id": pred_ids[idx_obj_key[0]],
                    "scores": preds[idx_obj_key[1]]["scores"]
                    if config["with_scores"]
                    else None,
                    "score": preds[idx_obj_key[1]]["score"],
                },
                enumerate(obj_keys),
            )
        )

        detailed_result = {"categorys": eval_categorys, "objects": objs}
        with open(
            os.path.join(self.output_dir, "detailed_result.json"), "w"
        ) as fout:
            json.dump(detailed_result, fout, indent=2, cls=NpEncoder)

        if (
            self.image_dir is not None
            and config["show_confusion_matrix_samples"]
        ):
            samples_output_dir = os.path.join(
                self.output_dir, "confusion_matrix_samples"
            )
            os.mkdir(samples_output_dir)
            draw_confusion_matrix_samples(
                detailed_result,
                self.image_dir,
                samples_output_dir,
                config["output_json"],
            )
        return {
            "mPre": results["mean_precision"],
            "mRec": results["mean_recall"],
            "ACC": results["accuracy"],
        }
