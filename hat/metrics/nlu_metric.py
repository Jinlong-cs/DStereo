# Copyright (c) Horizon Robotics. All rights reserved.

import os

import numpy as np
from sklearn.metrics import classification_report, f1_score

from hat.core.nlu.nlu_protocol_eval import Normalizer, ProtocolEvaluator
from hat.core.nlu.nlu_utils import (
    export_eval_report,
    get_diff_samples,
    transfer_batch_sample,
)
from hat.metrics.nlu_sequence_metric import (
    classification_report as seq_classification_report,
)
from hat.metrics.nlu_sequence_metric import f1_score as seq_f1_score
from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["NluEvalMetric"]


@OBJECT_REGISTRY.register
class NluEvalMetric(EvalMetric):
    """Compute metric for nlu {domain|intent|slots}."""

    def __init__(
        self,
        name=None,
        report=False,
        out_path=None,
        average="macro",
        use_protocol=False,
    ):
        super(NluEvalMetric, self).__init__(name, warn_without_compute=False)
        self.name = name
        self.report = report
        self.out_path = out_path
        self.average = average
        self.use_protocol = use_protocol
        self.reset()

    def _init_states(self):
        self.domain_pred = []
        self.domain_label = []
        self.intent_pred = []
        self.intent_label = []
        self.slots_pred = []
        self.slots_label = []
        self.batch_query = []
        self.losses = []

    def update(self, preds, targets):
        self.domain_pred.extend(preds[0])
        self.domain_label.extend(targets[0])
        self.intent_pred.extend(preds[1])
        self.intent_label.extend(targets[1])
        self.slots_pred.extend(preds[2])
        self.slots_label.extend(targets[2])
        self.batch_query.extend(preds[3])
        self.losses.extend([preds[4].item()])

    def reset(self):
        self.domain_pred = []
        self.domain_label = []
        self.intent_pred = []
        self.intent_label = []
        self.slots_pred = []
        self.slots_label = []
        self.batch_query = []
        self.losses = []

    def compute(self):
        domain_f1 = f1_score(
            self.domain_pred,
            self.domain_label,
            average=self.average,
        )
        intent_f1 = f1_score(
            self.intent_pred,
            self.intent_label,
            average=self.average,
        )
        slots_f1 = seq_f1_score(
            self.slots_pred,
            self.slots_label,
            average=self.average,
        )
        multi_task_f1 = (domain_f1 + intent_f1 + slots_f1) / 3

        # 输出评估报告
        if self.report:
            assert self.out_path is not None
            out_dir = os.path.abspath(os.path.join(self.out_path, ".."))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            loss = np.mean(self.losses)

            true_batch_sample = transfer_batch_sample(
                self.batch_query,
                self.domain_label,
                self.intent_label,
                self.slots_label,
            )
            pred_batch_sample = transfer_batch_sample(
                self.batch_query,
                self.domain_pred,
                self.intent_pred,
                self.slots_pred,
            )

            (
                batch_miss_recall,
                batch_false_recall,
                batch_other_diff,
            ) = get_diff_samples(true_batch_sample, pred_batch_sample)

            diff_num = (
                len(batch_miss_recall)
                + len(batch_false_recall)
                + len(batch_other_diff)
            )
            total_num = len(self.domain_label)

            stat_dic = {
                "loss": str(loss),
                "domain_f1": str(domain_f1),
                "intent_f1": str(intent_f1),
                "slots_f1": str(slots_f1),
                "multi_task_f1": str(multi_task_f1),
                "domain_report": classification_report(
                    self.domain_label, self.domain_pred
                ),
                "intent_report": classification_report(
                    self.intent_label, self.intent_pred
                ),
                "slots_report": seq_classification_report(
                    self.slots_label, self.slots_pred
                ),
                "total_samples_num": str(total_num),
                "match_num": str(total_num - diff_num),
                "diff_num": str(diff_num),
                "miss_recall_num": str(len(batch_miss_recall)),
                "false_recall_num": str(len(batch_false_recall)),
                "other_diff_num": str(len(batch_other_diff)),
                "match_ratio(accuracy)": str(
                    (total_num - diff_num) / total_num
                ),
                "miss_recall_ratio": str(len(batch_miss_recall) / total_num),
                "false_recall_ratio": str(len(batch_false_recall) / total_num),
                "miss_recall_samples": batch_miss_recall,
                "false_recall_samples": batch_false_recall,
                "other_diff_samples": batch_other_diff,
            }

            if self.use_protocol:
                normalizer = Normalizer()
                protocol_evaluator = ProtocolEvaluator(normalizer)
                protocol_false_recall_samples = [
                    sample
                    for sample in batch_false_recall
                    if protocol_evaluator.is_legal(sample["pred_label"])
                ]
                stat_dic["protocol_false_recall_num"] = str(
                    len(protocol_false_recall_samples)
                )
                stat_dic[
                    "protocol_false_recall_samples"
                ] = protocol_false_recall_samples

            export_eval_report(self.out_path, stat_dic)

        return domain_f1, intent_f1, slots_f1, multi_task_f1
