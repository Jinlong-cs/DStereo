import json
import os

import numpy as np
import torch.distributed as dist

try:
    from aidisdk.experiment import Table
except ImportError:
    Table = None

from hat.metrics.bev.bev_e2e import BEVE2Eval
from hat.metrics.bev_3d import BEVDetEval, BEVDetTagEval
from hat.metrics.mean_iou import MeanIOU
from hat.metrics.metric_3dv_utils import NpEncoder
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import EvalResult
from hat.utils.apply_func import _as_list

__all__ = ["ANCMeanIOUV2", "ANCBEVDetEvalV2", "ANCBEVDetTagEvalV2"]


def reverse_table(columns, data_list):
    new_columns = []
    new_data_list = []
    new_columns_key_info = list(data_list[0].keys())[0]
    new_columns = ["item"] + [
        data_info[new_columns_key_info] for data_info in data_list
    ]

    table_data = [list(data_info.values())[1:] for data_info in data_list]
    new_table_data = np.array(table_data).T.tolist()

    for name, row_data_list in zip(columns[1:], new_table_data):
        table_info = {"item": name}
        for new_columns_key, row_data in zip(new_columns[1:], row_data_list):
            table_info[new_columns_key] = row_data
        new_data_list.append(table_info)
    return new_columns, new_data_list


@OBJECT_REGISTRY.register
class ANCMeanIOUV2(MeanIOU):
    """Temporary implementation to save the metric result.

    Args:
        task_name: Name of task to be saved.
        gt_name: Key of groundtruth in batch inputs.
        pred_name: Key of prediction in model outputs.
        save_path: Path to save metric results.
        verbose: Whether to compute detail metric.
        result_prefix: Prefix of model results.
    """

    def __init__(
        self,
        task_name: str,
        gt_name: str,
        pred_name: str,
        save_path: str,
        verbose: bool,
        result_prefix: str,
        **kwargs,
    ):
        super(ANCMeanIOUV2, self).__init__(verbose=verbose, **kwargs)
        self.task_name = task_name
        self.gt_name = gt_name
        self.pred_name = pred_name
        self.save_path = save_path
        self.verbose = verbose
        self.result_prefix = result_prefix

    def compute(self):
        result = super(ANCMeanIOUV2, self).compute()
        if self.save_path:
            assert self.verbose
            miou, macc, all_acc, iou, acc, seg_class = result
            save_result = {}
            save_result["miou"] = miou.cpu().numpy().tolist()
            save_result["macc"] = macc.cpu().numpy().tolist()
            save_result["all_acc"] = all_acc.cpu().numpy().tolist()
            save_result["iou"] = iou.cpu().numpy().tolist()
            save_result["acc"] = acc.cpu().numpy().tolist()
            save_result["seg_class"] = seg_class
            save_dir, _ = os.path.split(self.save_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(self.save_path, "w") as fw:
                json.dump(save_result, fw)

        summary, tables = None, None
        if self.verbose:
            mIoU = result[0].cpu().item()
            mAcc = result[1].cpu().item()
            aAcc = result[2].cpu().item()
            summary = {
                f"{self.result_prefix} mIoU": mIoU,
                f"{self.result_prefix} mAcc": mAcc,
                f"{self.result_prefix} aAcc": aAcc,
            }
            columns = ["Class", "IoU", "Acc"]
            data = []
            for iou, acc, seg_class in zip(result[3], result[4], result[5]):
                IoU = iou.cpu().item()
                Acc = acc.cpu().item()
                data.append(
                    {
                        "Class": seg_class,
                        "IoU": IoU,
                        "Acc": Acc,
                    }
                )
            tables = [
                Table(
                    name=self.result_prefix + "_" + self.name,
                    columns=columns,
                    data=data,
                )
            ]
        else:
            mIoU = result.cpu().item()
            summary = {f"{self.result_prefix} mIoU": mIoU}

        return EvalResult(
            summary=summary,
            tables=tables,
        )

    def update(self, label: dict, preds: dict):
        pred = _as_list(preds[self.pred_name])[0]
        label = label[self.task_name][self.gt_name].reshape(pred.shape)
        return super().update(label=label, preds=pred)


@OBJECT_REGISTRY.register
class ANCBEVDetEvalV2(BEVDetEval):
    """Temporary implementation to save the metric result.

    It will be removed when aidieval is ready.
    """

    def __init__(self, save_metric_path, result_prefix, **kwargs):
        super(ANCBEVDetEvalV2, self).__init__(**kwargs)
        self.save_metric_path = save_metric_path
        self.result_prefix = result_prefix

    def compute(self):
        name, values, result = super(ANCBEVDetEvalV2, self).compute()
        if self.save_metric_path is not None and dist.get_rank() == 0:
            save_dir, _ = os.path.split(self.save_metric_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(self.save_metric_path, "w") as f:
                json.dump(result, f, cls=NpEncoder)

            summary, tables = {}, []
            for cid, value in result.items():
                label = self.id2label[cid] if cid in self.id2label else ""
                if label == "vehicle":
                    summary.update(
                        {
                            f"{self.result_prefix} {label} Recall": value[
                                "Recall"
                            ],
                            f"{self.result_prefix} {label} Precision": value[
                                "Precision"
                            ],
                            f"{self.result_prefix} {label} AP": value["AP"],
                        }
                    )
                data = []
                for k, v in value["depthwise_metric"].items():
                    _data = {"name": k}
                    for i in value["all_depth_intervals"]:
                        _data.update({i: v[i]})
                    data.append(_data)
                columns, data = reverse_table(
                    ["name"] + value["all_depth_intervals"], data
                )
                tables.append(
                    Table(
                        name=self.result_prefix + "_" + label,
                        columns=columns,
                        data=data,
                    )
                )

            return EvalResult(
                summary=summary,
                tables=tables,
            )


@OBJECT_REGISTRY.register
class BEVE2Evalv2(BEVE2Eval):
    """Temporary implementation to save the metric result.

    It will be removed when aidieval is ready.
    """

    def __init__(self, result_prefix, **kwargs):
        super(BEVE2Evalv2, self).__init__(**kwargs)
        self.result_prefix = result_prefix

    def compute(self):
        _, _, result = super(BEVE2Evalv2, self).compute()
        if dist.get_rank() == 0:
            summary, tables = {}, []
            metric_all, traj_metric_norm, traj_metric_center = result
            for cid, value in metric_all.items():
                label = self.id2label[cid] if cid in self.id2label else ""
                summary.update(
                    {
                        f"{self.result_prefix} {label} total_images": value[  # noqa
                            "total_images"
                        ],
                        f"{self.result_prefix} {label} valid_images": value[  # noqa
                            "valid_images"
                        ],
                        f"{self.result_prefix} {label} TP": value["TP"],
                        f"{self.result_prefix} {label} FP": value["FP"],
                        f"{self.result_prefix} {label} FN": value["FN"],
                        f"{self.result_prefix} {label} Recall": value[
                            "Recall"
                        ],
                        f"{self.result_prefix} {label} Precision": value[
                            "Precision"
                        ],
                        f"{self.result_prefix} {label} AP": value["AP"],
                    }
                )
                if "sAMOTA" in value:
                    summary.update(
                        {
                            f"{self.result_prefix} {label} sAMOTA": value[
                                "sAMOTA"
                            ],
                            f"{self.result_prefix} {label} AMOTA": value[
                                "AMOTA"
                            ],
                            f"{self.result_prefix} {label} AMOTP": value[
                                "AMOTP"
                            ],
                        }
                    )

                data = []
                for k, v in value["depthwise_metric"].items():
                    _data = {"name": k}
                    for i in value["all_depth_intervals"]:
                        _data.update({i: v[i]})
                    data.append(_data)
                tables.append(
                    Table(
                        name=self.result_prefix + "_" + label,
                        columns=["name"] + value["all_depth_intervals"],
                        data=data,
                    )
                )
            if traj_metric_norm is not None:
                data = []
                for k in traj_metric_norm["mode"]:
                    _data = {"mode": k}
                    for i in traj_metric_norm["outkeys"]:
                        _data.update({i: traj_metric_norm[k][i]})
                    data.append(_data)
                tables.append(
                    Table(
                        name=self.result_prefix + "_" + "traj_norm",
                        columns=["mode"] + traj_metric_norm["outkeys"],
                        data=data,
                    )
                )
            if traj_metric_center is not None:
                data = []
                for k in traj_metric_center["mode"]:
                    _data = {"mode": k}
                    for i in traj_metric_center["outkeys"]:
                        _data.update({i: traj_metric_center[k][i]})
                    data.append(_data)
                tables.append(
                    Table(
                        name=self.result_prefix + "_" + "traj_norm",
                        columns=["mode"] + traj_metric_center["outkeys"],
                        data=data,
                    )
                )
            return EvalResult(
                summary=summary,
                tables=tables,
            )


@OBJECT_REGISTRY.register
class ANCBEVDetTagEvalV2(BEVDetTagEval):
    """Save the metric result to aidi experiment."""

    def __init__(self, save_metric_path, result_prefix, **kwargs):
        super(ANCBEVDetTagEvalV2, self).__init__(**kwargs)
        self.save_metric_path = save_metric_path
        self.result_prefix = result_prefix

    def compute(self):
        name, values, result = super(ANCBEVDetTagEvalV2, self).compute()
        if self.save_metric_path is not None and dist.get_rank() == 0:
            save_dir, _ = os.path.split(self.save_metric_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(self.save_metric_path, "w") as f:
                json.dump(result, f, cls=NpEncoder)

            summary, tables = {}, []
            for cid, value in result.items():
                label = self.id2label[cid] if cid in self.id2label else ""
                if label == "vehicle":
                    summary.update(
                        {
                            f"{self.result_prefix} {label} Recall": value[
                                "Recall"
                            ],
                            f"{self.result_prefix} {label} Precision": value[
                                "Precision"
                            ],
                            f"{self.result_prefix} {label} AP": value["AP"],
                        }
                    )
                data = []
                for k, v in value["tagwise_metric"].items():
                    _data = {"name": k}
                    for i in value["all_tags"]:
                        _data.update({i: v[i]})
                    data.append(_data)
                columns, data = reverse_table(
                    ["name"] + value["all_tags"], data
                )
                tables.append(
                    Table(
                        name=self.result_prefix + "_" + label,
                        columns=columns,
                        data=data,
                    )
                )

            return EvalResult(
                summary=summary,
                tables=tables,
            )
