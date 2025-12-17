# Copyright (c) Horizon Robotics. All rights reserved.
import datetime
import logging
import os
import time
from os import path as osp
from typing import List

import numpy as np
import torch.nn.functional as F
from prettytable import PrettyTable

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info
from .metric import EvalMetric

__all__ = ["FasSigmoidAccuracy", "FasTARTRR"]

logger = logging.getLogger(__name__)

eps = 1e-5


@OBJECT_REGISTRY.register
class FasSigmoidAccuracy(EvalMetric):
    """Computes fas accuracy classification score.

    Args:
        thr: Thresh in accuracy calculation.
        name: Name of this metric instance for display.
    """

    def __init__(self, thr: float = 0.5, name: str = "fas_accuracy"):
        super().__init__(f"fas_accuracy({name})")
        self.thr = thr
        self.name = name

    def update(self, preds, fas_label, car_cls):
        batch_num = fas_label.shape[0]
        self.num_inst += batch_num
        for i in range(len(preds)):
            cur_samples = car_cls == i
            if sum(cur_samples) == 0:
                continue
            pred = preds[i].squeeze()
            pred = F.sigmoid(pred[cur_samples])
            cur_label = fas_label[cur_samples]
            pred = pred > self.thr
            num_correct = sum(pred == cur_label)
            self.sum_metric += num_correct


@OBJECT_REGISTRY.register
class FasTARTRR(EvalMetric):
    """Evaluation in Fas protocol.

    TAR: Live acceptance rate.
    TRR: Inactive rejection rate.

    Not ready for distributed environment.
    Should not be used together with DistributedSampler.

    Args:
        database_names: Car name list.
        val_interval: Evaluation interval. Defaults to 1.
        name: Name of this metric instance for display.
            Defaults to "FasTARTRR".
        save_results: Whether to save results in txt. Defaults to False.
        save_prefix: Path to save result. Defaults to "./WORKSPACE/results".
        use_small_step: Whether to use small thresh step. Defaults to False.
    """

    def __init__(
        self,
        database_names: List[str],
        val_interval: int = 1,
        name: str = "FasTARTRR",
        save_results: bool = False,
        save_prefix: str = "./WORKSPACE/results",
        use_small_step: bool = False,
    ):

        super().__init__(name)

        self.iter = 0
        self.val_interval = val_interval
        self.database_names = database_names
        self.save_results = save_results
        rank, world_size = get_dist_info()
        self.save_prefix = save_prefix + str(rank) + "_" + str(world_size)
        self.get_times = 0
        self.name = name
        self.thr_num = 999 if use_small_step else 99
        self.thr_min = 0.009 if use_small_step else 0.09
        self.thr_max = 0.996 if use_small_step else 0.96

        if self.save_results:
            try:
                os.makedirs(osp.expanduser(self.save_prefix), exist_ok=True)
            except Exception:
                pass
            t = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            self.t = t
            self.results_filename = osp.abspath(
                osp.join(
                    osp.expanduser(self.save_prefix),
                    "_".join(self.database_names)
                    + f"_{t}_{self.get_times}_results.txt",
                )
            )

        self.error = [
            np.zeros((self.thr_num, 4))
            for i in range(len(self.database_names))
        ]

    def reset(self):
        self.error = [
            np.zeros((self.thr_num, 4))
            for i in range(len(self.database_names))
        ]

    def get(self):
        acc = []
        threshes = np.arange(1, self.thr_num + 1) / (self.thr_num + 1)
        beg = time.time()
        for database_label, database_name in enumerate(self.database_names):
            results = np.zeros((self.thr_num, 2))
            table = PrettyTable(["Thresh", "STAR", "STRR"])  # noqa
            table.float_format = "2.3"
            for idx, i in enumerate(range(1, self.thr_num + 1)):
                thresh = i / (self.thr_num + 1)
                current_error = self.error[database_label][idx]
                tar = (
                    np.around(
                        current_error[0]
                        / (current_error[0] + current_error[2] + eps),
                        4,
                    )
                    * 100
                )
                trr = (
                    np.around(
                        current_error[3]
                        / (current_error[3] + current_error[1] + eps),
                        4,
                    )
                    * 100
                )

                table.add_row([thresh, tar, trr])
                if i % 50 == 0:
                    acc.append(
                        np.around(
                            (current_error[0] + current_error[3])
                            / (np.sum(current_error) + eps),
                            4,
                        )
                        * 100
                    )
                results[idx] = [tar, trr]
            result_str = table.get_string()
            if self.save_results:
                with open(self.results_filename, "a+") as f:
                    f.write(f"database {database_label}: {database_name}\n")
                    f.write(result_str)
                    f.write("\n")
            logger.info(f"database {database_label}: {database_name}\n")
            logger.info(result_str)

            star = results[:, 0]
            strr = results[:, 1]
            deviation_rate = np.maximum(95 - star, 0) + np.maximum(
                95 - strr, 0
            )
            # moving average
            mean_deviation_rate = (
                np.convolve(deviation_rate, np.ones(19), mode="same") / 19
            )
            mean_star = np.convolve(star, np.ones(19), mode="same") / 19
            mean_strr = np.convolve(strr, np.ones(19), mode="same") / 19
            inds_overall = np.lexsort(
                (-mean_strr, -mean_star, mean_deviation_rate)
            )

            result_mean = [
                inds_overall[0] / (self.thr_num + 1),
                mean_deviation_rate[inds_overall[0]],
                mean_star[inds_overall[0]],
                mean_strr[inds_overall[0]],
            ]
            table = PrettyTable(
                ["Thresh", "mean_deviation_rate", "mean_star", "mean_strr"]
            )
            table.float_format = "2.3"
            table.add_row(result_mean)
            result_str = table.get_string()
            if self.save_results:
                with open(self.results_filename, "a+") as f:
                    f.write(result_str)
            logger.info(result_str)

            inds_good = (
                (deviation_rate == 0)
                & (threshes > self.thr_min)
                & (threshes < self.thr_max)
            )
            if np.sum(inds_good) > 0:
                inds_good_final = np.lexsort(
                    (-strr[inds_good], -star[inds_good])
                )
                result_good_final = [
                    threshes[inds_good][inds_good_final[0]],
                    deviation_rate[inds_good][inds_good_final[0]],
                    star[inds_good][inds_good_final[0]],
                    strr[inds_good][inds_good_final[0]],
                ]
                table = PrettyTable(
                    ["Thresh", "deviation_rate", "star", "strr"]
                )
                table.float_format = "2.3"
                table.add_row(result_good_final)
                result_str = table.get_string()
                if self.save_results:
                    with open(self.results_filename, "a+") as f:
                        f.write(result_str)
                logger.info(result_str)
            else:
                if self.save_results:
                    with open(self.results_filename, "a+") as f:
                        f.write("\nNO GOOD RESULT.\n")
                logger.info(result_str)

            elapse = time.time() - beg
            logger.info(f"elpased: {elapse}s\n")

        self.get_times += 1
        self.reset()
        return list(map(lambda x: x + "_acc", self.database_names)), acc

    def update(self, preds, fas_label, car_cls):
        pred_multi_head = [
            F.sigmoid(pred.squeeze()).cpu().numpy() for pred in preds
        ]
        targets = fas_label.cpu().numpy()
        database_labels = car_cls.cpu().numpy()

        for database_label in range(len(self.database_names)):
            scores = pred_multi_head[database_label]
            cur_samples = database_labels == database_label
            if sum(cur_samples) == 0:
                continue

            if self.save_results:
                cur_dataname = self.database_names[database_label]
                self.predict_filename = osp.abspath(
                    osp.join(
                        osp.expanduser(self.save_prefix),
                        f"{cur_dataname}_{self.t}_{self.get_times}_pred.txt",
                    )
                )
                with open(self.predict_filename, "a+") as f:
                    f.write("\n".join(map(str, scores[cur_samples].tolist())))
                    f.write("\n")

            for idx, i in enumerate(range(1, (self.thr_num + 1))):
                thresh = i / (self.thr_num + 1)
                tp = np.sum((scores > thresh) * targets * cur_samples)
                fp = np.sum((scores > thresh) * (targets == 0) * cur_samples)
                fn = np.sum((scores <= thresh) * targets * cur_samples)
                tn = np.sum((scores <= thresh) * (targets == 0) * cur_samples)
                self.error[database_label][idx] += np.array([tp, fp, fn, tn])
