import os
from copy import deepcopy
from functools import partial

import torch
from torchvision.transforms import InterpolationMode

from hat.data.collates.collates import collate_2d
from hat.registry import build_from_registry
from ..common import (
    aidi_eval,
    backbone,
    bifpn_out_strides,
    bn_kwargs,
    compare_version,
    feat_channels,
    job_name,
    log_freq,
    loss_weights,
    model_input_size,
    model_version,
    neck,
    pipeline_test_dump,
    project_id,
    save_prefix,
    tasks_batch_size,
    train_num_workers,
    training_step,
    val_num_workers,
    val_only,
    val_only_transforms,
)
from ..datasets.auto_2d.datasets import PREDICT_TAGS, adas_eval_datadet_id_list
from ..lib.aidieval_helper import reformat_seg_to_aidi_eval
from .vdvru_common import get_2d_detection_aidi_eval_loaders

ignore_index = 255


# -------------------------- fake inputs --------------------------------
height, width = model_input_size[:2]
inputs = dict(
    labels=torch.zeros((1, height, width)),
    img_id=torch.Tensor(
        [
            [0],
        ]
    ),
    img_name=[
        "xxx.jpg",
    ],
    img_height=torch.Tensor(
        [
            [height],
        ]
    ),
    img_width=torch.Tensor(
        [
            [width],
        ]
    ),
    color_space=[
        "yuv",
    ],
    layout=[
        "chw",
    ],
)


val_inputs = deepcopy(inputs)
val_inputs.pop("labels")
val_inputs["before_crop_shape"] = [[height, width, 3]]
val_inputs["crop_offset"] = [[0, 0, 0, 0]]

if training_step == "int_infer":
    test_inputs = dict()
else:
    test_inputs = deepcopy(val_inputs)


def get_inputs(mode):
    inputs_ = dict()
    if val_only:
        inputs_ = val_inputs
    elif mode == "train" or mode == "val":
        inputs_ = inputs
    else:
        inputs_ = test_inputs

    return inputs_


# -------------------------- transforms --------------------------------
train_transforms = [
    dict(
        type="Resize",
        keep_ratio=True,
        img_scale=(height * 2, width * 2),
        ratio_range=(0.75, 1.25),
    ),
    dict(
        type="SegRandomCrop",
        size=(height, width),
        cat_max_ratio=0.5,
        ignore_index=ignore_index,
    ),
    dict(type="Pad", size=(height, width)),
    dict(
        type="SegRandomCutOut",
        prob=0.5,
        n_holes=(2, 6),
        cutout_ratio=[
            (0.05, 0.05),
            (0.02, 0.02),
            (0.07, 0.07),
            (0.1, 0.1),
            (0.2, 0.2),
        ],
    ),
    dict(type="RandomFlip", px=0.5, py=0.0),
    dict(type="ToTensor", to_yuv=False),
    dict(
        type="SegRandomAffine",
        degrees=15,
        label_fill_value=255,
        interpolation=InterpolationMode.BILINEAR,
        rotate_p=0.5,
        translate_p=0,
        scale_p=0,
    ),
    dict(type="CopyKeys", keys=["gt_seg|labels"]),
    dict(type="DeleteKeys", keys=["before_pad_shape"]),
]


# -------------------------- model --------------------------------
metric_loss_name = "semseg_loss"


def get_loss(
    task_name,
    num_classes,
    losses_weight,
    use_auxi_loss,
    class_weight,
    auto_class_weight,
    weight_min,
    weight_noobj,
    num_auxi_layer,
):
    loss_weight = loss_weights[task_name]
    losses = [
        dict(
            type="MixSegLoss",
            losses=[
                dict(
                    type="CEWithWeightMap",  # ce_loss
                    use_sigmoid=False,
                    loss_weight=1,
                    reduction="mean",
                    loss_name="ce_loss_s2",
                    ignore_index=ignore_index,
                    class_weight=class_weight,
                    num_class=num_classes,
                    auto_class_weight=auto_class_weight,
                    weight_min=weight_min,
                    weight_noobj=weight_noobj,
                ),
                dict(
                    type="LovaszSoftmaxLoss",  # lovasz_loss
                    per_image=False,
                    ignore_index=ignore_index,
                    loss_name="lovasz_loss_s2",
                ),
            ],
            losses_weight=[1.0 * loss_weight, 1.0 * loss_weight],
        )
    ]

    if use_auxi_loss:
        for i in range(num_auxi_layer):
            losses.append(
                dict(
                    type="CEWithWeightMap",
                    use_sigmoid=False,
                    class_weight=class_weight,
                    loss_weight=1.0,
                    ignore_index=ignore_index,
                    loss_name="ce_loss_s" + str(pow(2, 3 + i)),
                    num_class=num_classes,
                    auto_class_weight=auto_class_weight,
                    weight_min=weight_min,
                    weight_noobj=weight_noobj,
                )
            )
    loss_ret = dict(
        type="MixSegLossMultipreds",
        node_name=task_name + "_mix_seg_loss_multi_preds",
        losses=losses,
        losses_weight=losses_weight,
        loss_name="semseg_loss",
    )
    return loss_ret


def get_model(
    mode,
    task_name,
    num_classes,
    loss,
    head_out_strides,
    stacked_convs,
    level_indices,
    conv_method,
    share_conv=False,
    use_auxi_loss=False,
    aggregation_method="sum",
):
    return dict(
        type="BMSegmentor",
        backbone=backbone,
        neck=neck,
        head=dict(
            type="MaskcatFeatHead",
            node_name=task_name + "_maskcat_feat_head",
            has_project_layer=True,
            num_classes=num_classes,
            in_strides=bifpn_out_strides,
            out_strides=head_out_strides,
            in_channels=feat_channels,
            out_channels=feat_channels,  # need to be improved
            stacked_convs=stacked_convs,
            level_indices=level_indices,
            bn_kwargs=bn_kwargs,
            group_base=8,
            conv_method=conv_method,
            share_conv=share_conv,
            use_auxi_loss=False if mode == "test" else use_auxi_loss,
            auxi_use_bifpn=False,
            aggregation_method=aggregation_method,
            argmax_output=False,
            dequant_output=True,
            int8_output=True,
            upsample_output_scale=mode == "train",
        ),
        loss=loss if mode == "train" else None,
        postprocess=dict(
            type="SemSegDecoder",
            node_name="SemSegDecoder",
            output_name="semseg_pred",
        )
        if mode != "train" and not aidi_eval
        else None,
    )


# task specific metric
def get_update_metric(task_name, is_train=False):
    def mean_iou_metric_reorganize(out):
        label = preds = None
        for field, data in zip(out._fields, out):
            if "pred_seg" in field:
                preds = data
            if "gt_seg" in field:
                label = data
        return label, preds

    def update_metric(metrics, batch, model_outs):
        if is_train:
            outs = model_outs[0]
            for metric in metrics:
                total_loss = 0.0
                loss_dict = {}
                is_this_task = False
                for field, value in zip(outs._fields, outs):
                    if task_name not in field:
                        continue
                    if metric_loss_name not in field:
                        continue
                    is_this_task = True
                    total_loss += value
                    loss_dict[field.split(f"{metric_loss_name}_")[1]] = value
                if not is_this_task:
                    break
                loss_dict["loss_all"] = total_loss
                metric.update(loss_dict)
        else:
            # reorganize model prediction results
            assert len(metrics) == 1
            metrics[0].update(*mean_iou_metric_reorganize(model_outs[0]))

    return update_metric


def get_metric_updater(task_name):
    metric_updater = dict(
        type="MetricUpdater",
        metrics=[dict(type="LossShow", name="loss_semseg")],
        metric_update_func=get_update_metric(
            task_name=task_name, is_train=True
        ),
        step_log_freq=log_freq,
        epoch_log_freq=1,
        log_prefix=task_name,
    )
    return metric_updater


def get_val_metric_updater(num_class, task_name, label_desc):
    def update_val_metric(metrics, batch, model_outs):
        label = batch[0]["gt_seg"]
        if label.dim() == 4:
            label = label.squeeze(dim=-1)
        preds_label = model_outs[task_name][0]["semseg_pred"]

        for metric in metrics:
            metric.update(label, preds_label)

    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=[
            dict(
                type="MeanIOU",
                seg_class=[
                    label_desc[i]["class_name"][0] for i in range(num_class)
                ],
                ignore_index=ignore_index,
            )
        ],
        metric_update_func=update_val_metric,
        step_log_freq=500,
        epoch_log_freq=1,
        log_prefix="Validation " + task_name,
    )
    return val_metric_updater


def get_parsing_dataset(
    mode,
    rec_paths,
    anno_paths,
    train_sample_weights=None,
):
    datasets_ = [
        dict(
            type="DenseboxDataset",
            data_path=rec_path_i,
            anno_path=anno_path_i,
            task_type="segmentation",
            to_rgb=True,
            transforms=train_transforms
            if mode == "train"
            else val_only_transforms,
        )
        for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
    ]
    if mode == "train":
        concat_dataset = dict(
            type="DistributedComposeRandomDataset",
            sample_weights=train_sample_weights,
            datasets=datasets_,
            shuffle=not pipeline_test_dump,
        )
    elif mode == "val":
        concat_dataset = dict(
            type="ConcatDataset",
            datasets=datasets_,
        )

    return concat_dataset


def get_dataloader(mode, task_name, dataset):
    batch_size_dict = tasks_batch_size[task_name]
    num_workers = train_num_workers if mode == "train" else val_num_workers

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dataset,
        collate_fn=collate_2d,
        batch_size=batch_size_dict[mode],
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=train_num_workers > 0,
        multiprocessing_context=None if pipeline_test_dump else "spawn",
    )

    if mode == "val":
        data_loader["pin_memory"] = True
        data_loader.pop("persistent_workers")
        data_loader.pop("multiprocessing_context")

    return data_loader


get_aidi_eval_loaders = partial(
    get_2d_detection_aidi_eval_loaders,
    transforms=val_only_transforms,
)


def get_aidi_eval_callback(dataset_name, task_name):
    aidi_eval_page_label = task_name
    aidi_eval_dataset_id = adas_eval_datadet_id_list[dataset_name]
    aidi_eval_type = "detection"
    old_prediction = compare_version
    new_prediction = model_version

    aidi_eval_callback = []
    for id in aidi_eval_dataset_id:
        eval_callback = dict(
            type="AIDIEval",
            aidi_eval_dataset_id=id,
            output_root=os.path.join(
                save_prefix, job_name, task_name, str(id)
            ),
            prediction_name=model_version,
            prediction_tags=PREDICT_TAGS,
            project_id=project_id,
            reformat_output_fn=reformat_seg_to_aidi_eval,
            reformat_out_fn_kwargs={
                "decoder": build_from_registry(
                    dict(
                        type="SegDecoder",
                        out_strides=[2],
                        output_names="pred",
                        decode_strides=2,
                        transforms=val_only_transforms,
                        inverse_transform_key=[
                            "scale_factor",
                            "crop_offset",
                            "before_crop_shape",
                            "layout",
                        ],
                    )
                ),
                "task_name": task_name,
                "task_keyword": "pred",
            },
        )
        aidi_eval_callback.append(
            [
                eval_callback,
                dict(
                    type="StatsMonitor",
                    log_freq=log_freq,
                ),
            ]
        )

    return (
        aidi_eval_callback,
        aidi_eval_page_label,
        aidi_eval_dataset_id,
        aidi_eval_type,
        old_prediction,
        new_prediction,
    )
