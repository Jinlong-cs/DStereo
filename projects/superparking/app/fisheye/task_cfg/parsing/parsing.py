import copy
import json
import os

import torch
from torchvision.transforms import InterpolationMode

from hat.metrics.mean_iou import MeanIOU
from ...common import (
    all_task_data_files,
    backbone,
    bifpn_out_strides,
    bn_kwargs,
    bucket_root,
    commit_level_ci,
    config_file_root,
    feat_channels,
    fisheye_input_size,
    get_aidi_eval_info_common,
    input_size,
    is_int_infer,
    log_freq,
    neck,
    parsing_loss_weight,
    save_all_data_paths,
    task_common_transforms,
    tasks_batch_size,
    train_num_workers,
    use_mini_dataset,
    val_only,
)
from ...lib.aidieval_helper import reformat_seg_to_aidi_eval
from ...lib.utils import get_parsing_dataset_list
from ..vdvru_common import get_2d_aidi_eval_loaders

task_name = "parsing"
batch_size_dict = tasks_batch_size[task_name]
# model config
num_classes = 24
class_weight = None
ignore_index = 255
height, width = input_size[:2]

# head config
use_auxi_loss = True
loss_auxi_weight = [0.5, 0.25, 0.125, 0.125]
losses_weight = [1.0, 0.5, 0.25, 0.125]
num_auxi_layer = 3
start_level = 0
end_level = 3
assert (end_level - start_level) == num_auxi_layer
head_out_strides = [2, 8, 16, 32] if use_auxi_loss else [2]
stacked_convs = 1
has_project_layer = True
conv_method = "sep_conv"
share_conv = False
aggregation_method = "sum"
metric_loss_name = "semseg_loss"

# loss config
auto_class_weight = False
weight_min = 0.0
weight_noobj = 0.75

losses = [
    dict(
        type="MixSegLoss",
        losses=[
            dict(
                type="CEWithWeightMap",  # ce_loss
                use_sigmoid=False,
                loss_weight=1,
                reduction="mean",
                loss_name="ce_loss2",
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
                ignore_index=255,
                loss_name="lovasz_loss2",
            ),
        ],
        losses_weight=[1.0 * parsing_loss_weight, 1.0 * parsing_loss_weight],
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
                loss_name="ce_loss" + str(pow(2, 3 + i)),
                num_class=num_classes,
                auto_class_weight=auto_class_weight,
                weight_min=weight_min,
                weight_noobj=weight_noobj,
            )
        )
        # losses_weight.append(loss_auxi_weight[i])  # need improve

# task desc
labels = [
    dict(class_name=["road"], color_map=[128, 64, 128]),
    dict(class_name=["sidewalk"], color_map=[244, 35, 232]),
    dict(class_name=["sky"], color_map=[70, 130, 180]),
    dict(class_name=["terrain"], color_map=[107, 142, 35]),
    dict(class_name=["curb"], color_map=[34, 237, 242]),
    dict(class_name=["fence"], color_map=[190, 153, 153]),
    dict(class_name=["vegetation"], color_map=[152, 251, 152]),
    dict(class_name=["pole"], color_map=[153, 153, 153]),
    dict(class_name=["traffic_sign"], color_map=[220, 220, 0]),
    dict(class_name=["car"], color_map=[0, 0, 142]),
    dict(class_name=["tricycle"], color_map=[0, 0, 255]),
    dict(class_name=["bicycle"], color_map=[119, 11, 32]),
    dict(class_name=["person"], color_map=[220, 20, 60]),
    dict(class_name=["rider"], color_map=[237, 162, 13]),
    dict(class_name=["trolley"], color_map=[150, 50, 150]),
    dict(class_name=["traffic_cone"], color_map=[111, 74, 0]),
    dict(class_name=["bollard"], color_map=[70, 70, 70]),
    dict(class_name=["folding_warning_sign"], color_map=[220, 220, 220]),
    dict(class_name=["single_water_barrier"], color_map=[0, 80, 90]),
    dict(class_name=["untraversable"], color_map=[150, 120, 120]),
    dict(class_name=["parking_rod"], color_map=[100, 5, 180]),
    dict(class_name=["parking_lock"], color_map=[237, 159, 241]),
    dict(class_name=["column"], color_map=[0, 128, 128]),
    dict(class_name=["background"], color_map=[0, 0, 0]),
]

task_desc = [dict(size=(width // 2, height // 2))]
desc_task_name = f"fisheye_{task_name}"
add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task=desc_task_name,
                num_classes=num_classes,
                labels=labels,
                **desc,
            )
        )
        for desc in task_desc
    ],
    node_name="parsing_desc",
)


def get_model(mode):

    postprocess = dict(
        type="SemSegDecoder",
        node_name="SemSegDecoder",
        output_name="semseg_pred",
    )

    if (is_int_infer and not val_only) or (mode == "train"):
        postprocess = None

    return dict(
        type="BMSegmentor",
        backbone=backbone,
        neck=neck,
        head=dict(
            type="MaskcatFeatHead",
            node_name="MaskcatFeatHead",
            has_project_layer=True,
            num_classes=num_classes,
            in_strides=bifpn_out_strides,
            out_strides=head_out_strides,
            in_channels=feat_channels,
            out_channels=feat_channels,  # need to be improved
            stacked_convs=stacked_convs,
            start_level=start_level,
            end_level=end_level,
            bn_kwargs=bn_kwargs,
            group_base=8,
            conv_method=conv_method,
            share_conv=share_conv,
            use_auxi_loss=False if mode == "test" else use_auxi_loss,
            auxi_use_bifpn=False,
            aggregation_method=aggregation_method,
            argmax_output=is_int_infer and (not val_only),
            dequant_output=val_only or (not is_int_infer),
            int8_output=True,
            upsample_output_scale=True,
        ),
        loss=dict(
            type="MixSegLossMultipreds",
            node_name="MixSegLossMultipreds",
            losses=losses,
            losses_weight=losses_weight,
            loss_name=metric_loss_name,
        )
        if mode == "train"
        else None,
        postprocess=postprocess,
        desc=add_desc_pp if (is_int_infer and not val_only) else None,
    )


# -------------------------- dataloader --------------------------------

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


val_inputs = copy.deepcopy(inputs)
val_inputs.pop("labels")

if is_int_infer and not val_only:
    test_inputs = dict()
else:
    test_inputs = copy.deepcopy(val_inputs)


def get_inputs(mode):
    if mode == "train":
        inputs_ = inputs
    elif mode == "val":
        inputs_ = val_inputs
    else:
        inputs_ = test_inputs

    return inputs_


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
    dict(type="ToTensor", to_yuv=False),
    dict(type="RandomFlip", px=0.5, py=0.0),
    dict(
        type="SegRandomAffine",
        degrees=15,
        label_fill_value=ignore_index,
        interpolation=InterpolationMode.BILINEAR,
        rotate_p=0.5,
        translate_p=0,
        scale_p=0,
    ),
    dict(type="ToTensor", to_yuv=False),
    dict(type="CopyKeys", keys=["gt_seg|labels"]),
    dict(type="DeleteKeys", keys=["before_pad_shape"]),
]

val_transforms = [
    dict(
        type="Resize",
        img_scale=input_size,
        keep_ratio=False,
    ),
    dict(type="ToTensor", to_yuv=False),
    dict(type="RenameKeys", keys=["imgs|img"]),
] + task_common_transforms

transforms = {"train": train_transforms, "val": val_transforms}

data_paths = get_parsing_dataset_list(
    bucket_root=bucket_root,
    data_root=os.path.join(config_file_root, "datasets/parsing"),
    attribution="superparking",
    model_version="v0.9.0" if not use_mini_dataset else "ci_test",
    version_yaml="parsing_dataset_version.yaml",
    dataset_yaml="parsing_all_datasets.yaml",
    commit_level_ci=commit_level_ci,
)
train_rec_paths, train_anno_paths = data_paths[:2]
val_rec_paths, val_anno_paths = data_paths[-2:]


def get_dataset(mode):
    rec_paths = eval(mode + "_rec_paths")
    anno_paths = eval(mode + "_anno_paths")
    if save_all_data_paths:
        with open(all_task_data_files, "at") as fw:
            fw.writelines([p + "\n" for p in rec_paths + anno_paths])

    concat_dataset = dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="DenseboxDataset",
                data_path=rec_path_i,
                anno_path=anno_path_i,
                task_type="segmentation",
                to_rgb=True,
                transforms=transforms[mode],
            )
            for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
        ],
    )
    return concat_dataset


train_dataset = dict(
    type="RepeatDataset",
    dataset=get_dataset("train"),
    times=3,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_dict["train"],
    shuffle=True,
    num_workers=train_num_workers,
    pin_memory=True,
)

val_dataset = get_dataset("val")


def mean_iou_metric_reorganize(out):
    label = preds = None
    for field, data in zip(out._fields, out):
        if "pred_seg" in field:
            preds = data
        if "gt_seg" in field:
            label = data
    return label, preds


# task specific metric
def get_update_metric(is_train=False):
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


metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name="loss_semseg")],
    metric_update_func=get_update_metric(is_train=True),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=get_dataset("val"),
    batch_size=batch_size_dict["val"],
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    drop_last=False,
)


def update_val_metric(metrics, batch, model_outs):
    label = batch[0]["gt_seg"]
    preds_label = torch.argmax(model_outs[0][0], dim=1)

    for metric in metrics:
        metric.update(label, preds_label)


# callback config in validation stage
val_miou_metric = MeanIOU(
    seg_class=[labels[i]["class_name"][0] for i in range(num_classes)],
    ignore_index=ignore_index,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_miou_metric],
    metric_update_func=update_val_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

# -----------------------aidi eval-----------------------------
(
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_callback,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_aidi_eval_info_common(
    task_name,
    "semantic_segmentation",
    reformat_seg_to_aidi_eval,
    {
        "original_shape": fisheye_input_size,
    },
    # extra_tags=["wo_fs"], # try not to eval freespace
    extra_tags=None,
)

# since there are images with (1280, 1920), bpu_transforms do not work
# aidi_transforms = copy.deepcopy(bpu_transforms)
# aidi_transforms[1].pop("scale_wh")
# aidi_transforms[1]["target_hw"] = input_size
aidi_eval_loader = get_2d_aidi_eval_loaders(
    batch_size_dict["val"],
    task_name,
    transforms=val_transforms,
    return_img_buf=False,
)
