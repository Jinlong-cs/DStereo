import os
from dataclasses import fields
from typing import Sequence

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.data_struct.app_struct import DetObjects
from hat.core.data_struct.base_struct import ClsLabels, DetBoxes2D


def mean_pq_metric_reorganize(outs, batch):
    gt_attributes = batch["gt_labels"]
    gt_id_maps = list(batch["orig_gt_seg"])

    pred_attributes = [
        torch.stack(
            [
                torch.arange(1, len(out) + 1).to(
                    out.device
                )  # index start from 1
            ]
            + [
                getattr(out, f.name).cls_idxs
                for f in fields(out)
                if f.type == ClsLabels
            ],
            1,
        )
        for out in outs
    ]
    pred_id_maps = [out.pred_id_map.mask for out in outs]

    return {
        "gt_attributes": gt_attributes,
        "gt_id_maps": gt_id_maps,
        "pred_attributes": pred_attributes,
        "pred_id_maps": pred_id_maps,
    }


def mean_iou_metric_reorganize(outs, batch):
    label = batch["orig_gt_seg"]
    preds = torch.stack([out.mask for out in outs], 0)
    return {"label": label, "preds": preds}


def convert_hat_to_coco(det_objects):
    field = fields(det_objects)
    assert len(field) == 1, field
    field = field[0]
    det_boxes = getattr(det_objects, field.name)
    assert isinstance(det_boxes, DetBoxes2D), type(det_boxes)

    pred_bboxes = torch.cat(
        [
            det_boxes.boxes,
            det_boxes.scores.unsqueeze(1),
            det_boxes.cls_idxs.unsqueeze(1),
        ],
        1,
    )
    return pred_bboxes


def convert_dict_to_coco(det_results):
    boxes = det_results["boxes"]
    scores = det_results["scores"]
    cls_idxs = det_results["cls_idxs"]
    pred_bboxes = torch.cat(
        [
            boxes,
            scores.unsqueeze(1),
            cls_idxs.unsqueeze(1),
        ],
        1,
    )
    return pred_bboxes


def coco_metric_reorganize(outs, batch):
    img_names = batch["img_name"]
    img_ids = batch["img_id"]
    if isinstance(outs, dict):
        pred_bboxes = outs["pred_bboxes"]
    else:
        assert isinstance(outs, Sequence), type(outs)
        if isinstance(outs[0], DetObjects):
            pred_bboxes = [convert_hat_to_coco(out) for out in outs]
        elif isinstance(outs[0], dict):
            pred_bboxes = [convert_dict_to_coco(out) for out in outs]
        else:
            raise NotImplementedError

    return {
        "output": {
            "pred_bboxes": pred_bboxes,
            "img_name": img_names,
            "img_id": img_ids,
        }
    }


def get_update_metric(is_train, metric_reorganize, task_name):
    def update_metric(metrics, batch, model_outs):
        if isinstance(batch, tuple):
            batch = batch[0]
        outs = model_outs[task_name]

        if is_train:
            for metric, out in zip(metrics, outs):
                metric.update(out)
        else:
            for metric in metrics:
                metric.update(**metric_reorganize(outs, batch))

    if task_name.endswith("_classification"):
        return update_metric_using_regex(per_metric_patterns=metric_reorganize)

    return update_metric


def get_metrics(
    task_name,
    attributes=None,
    val_metrics=None,
    log_freq=50,
    metric_reorganize=None,
    ann_files=None,
    seg_class=None,
    ignore_index=255,
    save_root=None,
    attr_type_list=None,
    attr_type_value=None,
    ignore_idxs=None,
):
    # LossShow metrics for training
    train_metrics = [
        dict(
            type="LossShow",
            name="loss",
        )
    ]

    if task_name.endswith("_classification"):
        loss_names = [
            "classification_loss",
        ]
        acc_metrics = []
        acc_pattens = []
        for item in attr_type_list + ["all_attribute"]:
            acc_metrics.append(
                dict(
                    type="AccuracyAttrMultiLabel",
                    name="cls_acc",
                    attr_type_name=item,
                    attr_type_list=attr_type_list,
                    attr_type_numcls=attr_type_value,
                    ignore_idx=ignore_idxs,
                )
            )
            acc_pattens.append(
                dict(
                    label_pattern=f"^.*{task_name}_labels$",
                    pred_pattern=".*pred*",
                )
            )
        train_metrics = [
            dict(type="LossShow", name=name) for name in loss_names
        ] + acc_metrics
        metric_patterns = [  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_{name}$",
            )
            for name in loss_names
        ] + acc_pattens

    if val_metrics is None:
        if task_name.endswith("_detection"):
            if isinstance(ann_files, str):
                ann_files = [
                    ann_files,
                ]
            val_metrics = [
                dict(
                    type="COCODetectionMetric",
                    ann_file=ann_file,
                    val_interval=log_freq,
                    name="COCOMeanAP",
                    save_prefix=os.path.join(
                        save_root,
                        "coco_results",
                        f"Validation_{task_name}_",
                    ),
                )
                for ann_file in ann_files
            ]
        elif task_name.endswith("_segmentation"):
            val_metrics = [
                dict(
                    type="MeanIOU",
                    seg_class=seg_class,
                    name="MeanIOU",
                    ignore_index=ignore_index,
                ),
            ]
        elif task_name.endswith("_instanceseg"):
            val_metrics = [
                dict(
                    type="PanopticQualityWithAttributes",
                    __build_recursive=False,
                    name="PanopticQuality50",
                    attributes=attributes,
                    ignore_index=ignore_index,
                    iou_thr=0.5,
                ),
                # dict(
                #     type="PanopticQualityWithAttributes",
                #     __build_recursive=False,
                #     name="PanopticQuality75",
                #     attributes=attributes,
                #     ignore_index=ignore_index,
                #     iou_thr=0.75,
                # ),
                # dict(
                #     type="PanopticQualityWithAttributes",
                #     __build_recursive=False,
                #     name="PanopticQuality90",
                #     attributes=attributes,
                #     ignore_index=ignore_index,
                #     iou_thr=0.90,
                # )
            ]
        elif task_name.endswith("_classification"):
            val_metrics = acc_metrics
        else:
            raise NotImplementedError(str(task_name))

    # functions for reorganize model outputs
    if metric_reorganize is None:
        if task_name.endswith("_detection"):
            metric_reorganize = coco_metric_reorganize
        elif task_name.endswith("_segmentation"):
            metric_reorganize = mean_iou_metric_reorganize
        elif task_name.endswith("_instanceseg"):
            metric_reorganize = mean_pq_metric_reorganize
        elif task_name.endswith("_classification"):
            metric_reorganize = metric_patterns
        else:
            raise NotImplementedError(str(task_name))

    # metric updater for training
    train_metric_updater = dict(
        type="MetricUpdater",
        metrics=train_metrics,
        metric_update_func=get_update_metric(
            is_train=True,
            metric_reorganize=metric_reorganize,
            task_name=task_name,
        ),
        filter_condition=lambda x: x[1] == task_name,
        log_prefix=task_name,
        step_log_freq=log_freq,
        epoch_log_freq=1,
        reset_metrics_by="log",  # "epoch",
    )

    # metric updaters for validation
    if task_name.endswith("_classification"):
        val_metric_updater = [
            dict(
                type="MetricUpdater",
                metrics=val_metrics,
                metric_update_func=update_metric_using_regex(
                    per_metric_patterns=acc_pattens
                ),
                step_log_freq=25,
                epoch_log_freq=1,
                log_prefix=task_name,
                reset_metrics_by="epoch",
            )
        ]
    else:
        val_metric_updater = [
            dict(
                type="MetricUpdater",
                metrics=[val_metric],
                filter_condition=lambda x: x[1] == task_name,
                metric_update_func=get_update_metric(
                    is_train=False,
                    metric_reorganize=metric_reorganize,
                    task_name=task_name,
                ),
                log_prefix="Validation_" + task_name,
                step_log_freq=-1,
            )
            for val_metric in val_metrics
        ]

    return train_metric_updater, val_metric_updater
