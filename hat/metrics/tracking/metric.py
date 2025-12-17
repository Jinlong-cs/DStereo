# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import warnings
from typing import Dict

import numpy as np

from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from .compute_entry import cal_tp_error, statistic_error_by_filter
from .generate_result import generate_tracking_result
from .nuscenes_tracking_metric import NUSC_EXTRA_ARGS, NuscenesTrackingMetric

logger = logging.getLogger(__name__)

__all__ = ["TrackingMetric"]


@OBJECT_REGISTRY.register
class TrackingMetric(EvalMetric):
    """Tracking metric.

    Args:
        save_dir: Output dir to save file.
        eval_metric_type: Metric function type.
        cfg: Metric option dict.
    """

    def __init__(
        self,
        save_dir: str,
        eval_metric_type: str,
        cfg: Dict,
    ):
        assert eval_metric_type in ["Nuscenes"]
        self.save_dir = save_dir
        self.all_gts = {}
        self.all_preds = {}
        self.cfg = cfg
        self.eval_vis_cfg = self.cfg.pop("eval_vis_cfg", {})
        self.eval_metric_type = eval_metric_type

    def update(self, preds: dict, gts: dict):
        # check whether global trans and rot exists
        def check_global(frames):
            for _, frame in frames.items():
                percs = frame.get_messages("overall")[0].get_instances()
                for inst in percs:
                    # TODO: support global pos from odom info when missing
                    assert (
                        len(inst.get_attributes(topics=["global_translation"]))
                        > 0
                    )
                    assert (
                        len(inst.get_attributes(topics=["global_rotation"]))
                        > 0
                    )

        check_global(gts)
        check_global(preds)
        self.all_gts.update(gts)
        self.all_preds.update(preds)

    def get(self):
        if self.eval_metric_type == "Nuscenes":
            extra_cfg = {
                arg: self.cfg[arg]
                for arg in NUSC_EXTRA_ARGS
                if arg in self.cfg
            }
            metric = NuscenesTrackingMetric(
                self.save_dir,
                self.all_gts,
                self.all_preds,
                self.cfg["eval_classes"],
                self.cfg["eval_ranges"],
                self.cfg["dist_fcn"],
                self.cfg["dist_th_tp"],
                **extra_cfg,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                results = metric.main(render_curves=False)
                # process results to allcls and class-wise for table
                result_summary, result_raw = metric.post_process(
                    self.all_gts, self.all_preds
                )
            leaderboard_results = dict(  # noqa
                AMOTA=round(results["amota"], 4),
                AMOTP=round(results["amotp"], 4),
            )
            eval_classes = self.cfg["eval_classes"]
            result_raw["eval_vis_cfg"] = self.eval_vis_cfg
        else:
            # TODO: support evs tracklet asso metric
            raise NotImplementedError(
                f"{self.eval_metric_type} tracking metric not supported"
            )

        # cal obs error tables
        tp_error = {}
        for sample_id, gt in result_raw["gts"].items():
            pred = result_raw["preds"][sample_id]
            error_res = cal_tp_error(pred, gt)
            for k, v in error_res.items():
                if k not in tp_error:
                    tp_error[k] = []
                tp_error[k].append(v)
        tp_error = {k: np.concatenate(v, axis=0) for k, v in tp_error.items()}
        filters = self.cfg["eval_filters"]
        eval_error = self.cfg.get("eval_error", None)
        error_res = statistic_error_by_filter(tp_error, filters, eval_error)
        result_summary["overall"]["errors"] = error_res
        for cls in self.cfg["eval_classes"]:
            for error_filter in filters:
                error_filter["class"] = cls
            error_res = statistic_error_by_filter(
                tp_error, filters, eval_error
            )
            result_summary[cls]["errors"] = error_res

        # dump to file for table and viz
        generate_tracking_result(
            results=result_summary,
            results_raw=result_raw,
            output_dir=self.save_dir,
            result_json="result.json",
            result_png="result.png",
            table_json="tables.json",
            eval_classes=eval_classes,
        )

        logger.info("eval done")
        logger.info(leaderboard_results)
        return leaderboard_results
