import copy
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from lib.aidi_eval_helper import (
    TASK2AIDI_EVAL_DATA_PATH,
    TASK2AIDI_EVAL_IDS,
    reformat_seg_to_aidi_eval,
)
from torchvision.transforms import InterpolationMode

from hat.metrics.mean_iou import MeanIOU
from hat.utils import Config

task_name = "parsing"

cfg_dir = os.path.dirname(__file__)
BASE_CONFIG = Config.fromfile(
    os.path.join(os.path.dirname(cfg_dir), "base.py")
)

save_prefix = BASE_CONFIG.save_prefix
job_name = BASE_CONFIG.job_name
aidi_eval = BASE_CONFIG.aidi_eval
project_id = BASE_CONFIG.project_id
predition_tags = BASE_CONFIG.get("predition_tags", [])
training_step = BASE_CONFIG.training_step
val_num_workers = BASE_CONFIG.val_num_workers[task_name]

# task desc
labels = [
    dict(class_name=["road"], color_map=[128, 64, 128]),  # 0
    dict(class_name=["traffic_line"], color_map=[230, 150, 140]),  # 1
    dict(class_name=["parking_line"], color_map=[18, 145, 170]),  # 2
    dict(class_name=["parking_space"], color_map=[0, 0, 230]),  # 3
    dict(class_name=["traffic_arrow"], color_map=[220, 220, 0]),  # 4
    dict(class_name=["guide_line"], color_map=[220, 20, 60]),  # 5
    dict(class_name=["crosswalk_line"], color_map=[70, 130, 180]),  # 6
    dict(class_name=["no_parking_sign_line"], color_map=[0, 0, 110]),  # 7
    dict(class_name=["stop_line"], color_map=[0, 80, 100]),  # 8
    dict(class_name=["speed_bump"], color_map=[190, 153, 153]),  # 9
    dict(class_name=["sign_line"], color_map=[224, 35, 232]),  # 10
    dict(class_name=["parking_lock_open"], color_map=[70, 70, 70]),  # 11
    dict(class_name=["parking_lock_closed"], color_map=[129, 187, 89]),  # 12
    dict(class_name=["traffic_cone"], color_map=[153, 153, 153]),  # 13
    dict(class_name=["parking_rod"], color_map=[230, 123, 34]),  # 14
    dict(class_name=["curb"], color_map=[34, 237, 242]),  # 15
    dict(class_name=["column"], color_map=[102, 102, 156]),  # 16
    dict(class_name=["immovable_obstacle"], color_map=[150, 100, 100]),  # 17
    dict(class_name=["movable_obstacle"], color_map=[111, 74, 0]),  # 18
    dict(class_name=["other"], color_map=[193, 17, 101]),  # 19
    dict(class_name=["sidewalk"], color_map=[200, 200, 128]),  # 20
]

# train config
task_loss_weight = BASE_CONFIG.parsing_loss_weight
batch_size_per_gpu = BASE_CONFIG.batch_size_per_gpu[task_name]
dataloader_workers = BASE_CONFIG.train_num_workers[task_name]
val_dataloader_workers = BASE_CONFIG.val_num_workers[task_name]
local_train = BASE_CONFIG.local_train
train_paths = BASE_CONFIG.parsing_train_paths
train_weights = BASE_CONFIG.parsing_train_weights
val_paths = BASE_CONFIG.parsing_val_paths

height = BASE_CONFIG.height
width = BASE_CONFIG.width
bn_kwargs = BASE_CONFIG.bn_kwargs

task_desc = BASE_CONFIG.task_desc[task_name]
log_freq = BASE_CONFIG.log_freq

# model config
num_classes = BASE_CONFIG.parsing_num_classes
feat_channels = BASE_CONFIG.feat_channels
bifpn_out_strides = BASE_CONFIG.bifpn_out_strides

# head config
use_auxi_loss = BASE_CONFIG.parsing_use_auxi_loss
num_auxi_layer = BASE_CONFIG.parsing_head_num_auxi_layer
share_conv = BASE_CONFIG.parsing_head_share_conv
head_out_strides = BASE_CONFIG.parsing_head_out_strides
stacked_convs = BASE_CONFIG.parsing_head_stacked_convs
start_level = BASE_CONFIG.parsing_head_start_level
end_level = BASE_CONFIG.parsing_head_end_level
group_base = BASE_CONFIG.parsing_head_group_base
conv_method = BASE_CONFIG.parsing_head_conv_method
aggregation_method = BASE_CONFIG.parsing_head_aggregation_method


# loss config
ignore_index = BASE_CONFIG.parsing_ignore_index
class_weight = BASE_CONFIG.parsing_class_weight
losses_weight = BASE_CONFIG.parsing_losses_weight

auto_class_weight = BASE_CONFIG.parsing_atuo_class_weight
weight_min = BASE_CONFIG.parsing_weight_min
weight_noobj = BASE_CONFIG.parsing_weight_noobj
losses = [
    dict(
        type="MixSegLoss",
        losses=[
            dict(
                type="CEWithWeightMap",  # CrossEntropyLossV2
                use_sigmoid=False,
                loss_weight=1,
                reduction="mean",
                loss_name="ce_loss",
                ignore_index=255,
                class_weight=class_weight,
                num_class=num_classes,
                auto_class_weight=auto_class_weight,
                weight_min=weight_min,
                weight_noobj=weight_noobj,
            ),
            dict(
                type="LovaszSoftmaxLoss",  # lovasz_loss
                loss_weight=1.0,
                per_image=False,
                ignore_index=255,
                loss_name="lovasz_loss",
            ),
        ],
        losses_weight=[1.0 * task_loss_weight, 1.0 * task_loss_weight],
    )
]
if use_auxi_loss:
    for i in range(num_auxi_layer):
        losses.append(
            dict(
                type="CEWithWeightMap",  # CrossEntropyLossV2
                use_sigmoid=False,
                class_weight=class_weight,
                loss_weight=1.0 * task_loss_weight,
                ignore_index=ignore_index,
                loss_name="ce_loss" + str(pow(2, 3 + i)),
                num_class=num_classes,
                auto_class_weight=auto_class_weight,
                weight_min=weight_min,
                weight_noobj=weight_noobj,
            )
        )

# -------------------------- model --------------------------
inputs = dict(
    gt_seg=torch.randint(0, 255, (1, height, width), dtype=torch.uint8),
    img_id=torch.Tensor([[0]]),  # noqa
    img_name=["xxx.jpg"],  # noqa
    img_height=torch.Tensor([[height]]),
    img_width=torch.Tensor(
        [
            [width],
        ]
    ),
    color_space=["yuv"],  # noqa
    layout=["chw"],  # noqa
    img_shape=[
        torch.Tensor([height]),
        torch.Tensor([width]),
        torch.Tensor([3]),
    ],  # noqa
    pad_shape=[
        torch.Tensor([height]),
        torch.Tensor([width]),
        torch.Tensor([3]),
    ],  # noqa
    scale_factor=torch.Tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
        ]
    ),
    scale=None,
    scale_idx=None,
    resized_shape=None,
    keep_ratio=torch.Tensor(
        [
            True,
        ]
    ),
    data_path="",
)

val_inputs = dict(
    gt_seg=torch.randint(0, 255, (1, height, width), dtype=torch.uint8),
    img_id=torch.Tensor([[0]]),  # noqa
    img_name=["xxx.jpg"],  # noqa
    img_height=torch.Tensor([[height]]),
    img_width=torch.Tensor(
        [
            [width],
        ]
    ),
    color_space=["yuv"],  # noqa
    layout=["chw"],  # noqa
    img_shape=[
        torch.Tensor([height]),
        torch.Tensor([width]),
        torch.Tensor([3]),
    ],  # noqa
    pad_shape=[
        torch.Tensor([height]),
        torch.Tensor([width]),
        torch.Tensor([3]),
    ],
)
if aidi_eval:
    val_inputs.pop("gt_seg")
    val_inputs["orig_hw"] = [
        torch.Tensor([height]),
        torch.Tensor([width]),
    ]

if training_step == "int_infer":
    test_inputs = dict()
else:
    test_inputs = copy.deepcopy(val_inputs)
    if "gt_seg" in test_inputs:
        test_inputs.pop("gt_seg")


def topo_builder(nodes, _inputs, feats, mode):
    # filter by inputs keys
    if mode == "train":
        inner_inputs = {k: _inputs[k] for k in inputs}
    elif mode == "val":
        inner_inputs = {k: _inputs[k] for k in val_inputs}
    elif mode == "test":
        inner_inputs = {k: _inputs[k] for k in test_inputs}
    else:
        raise Exception("error mode")
    name2out = OrderedDict()
    out_module = nodes[f"{task_name}_head"]
    name2out.update({task_name: out_module(feats, inner_inputs)})
    return name2out


out_module = dict(
    type="OutputModule",
    head=dict(
        type="MaskcatFeatHead",
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
        group_base=group_base,
        conv_method=conv_method,
        share_conv=share_conv,
        use_auxi_loss=use_auxi_loss,
        aggregation_method=aggregation_method,
        argmax_output=False,
        dequant_output=True,
        int8_output=True,
    ),
    head_parser=dict(
        type="IPMHeadParser",
        out_strides=head_out_strides,
    ),
    target=dict(
        type="IPMSegTarget",
        label_name="gt_seg",
    ),
    loss=dict(
        type="SegLoss",
        loss=[
            dict(
                type="MixSegLossMultipreds",
                losses=losses,
                losses_weight=losses_weight,
            ),
        ],
    ),
    prefix="head",
    keep_name=True,
)

val_out_module = copy.deepcopy(out_module)
val_out_module["head_parser"] = None
val_out_module["target"] = None
val_out_module["loss"] = None
val_out_module["postprocess"] = dict(
    type="SemSegDecoder",
    output_name="pred_seg",
)

test_out_module = copy.deepcopy(val_out_module)
test_out_module["postprocess"] = None
test_out_module["head"]["use_auxi_loss"] = False
add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task=task_name, num_classes=num_classes, labels=labels, **desc
            )
        )
        for desc in task_desc
    ],
)
if training_step == "int_infer":
    test_out_module["head"]["argmax_output"] = True
    test_out_module["head"]["dequant_output"] = False
    test_out_module["postprocess"] = add_desc_pp
    out_module["head"]["use_auxi_loss"] = False

nodes = {f"{task_name}_head": out_module}
val_nodes = {f"{task_name}_head": val_out_module}
test_nodes = {f"{task_name}_head": test_out_module}


# data augmentation
mapping = np.arange(0, 256)
mapping[20] = 0
transforms = [
    dict(
        type="RandomDownSample",
        data_shape=(3, height, width),
        min_downsample_width=0.7 * width,
        p=0.1,
        inter_method=0,
    ),
    dict(type="GaussianBlur", kernel_size_min=3, kernel_size_max=7, p=0.1),
    dict(type="JPEGCompress", min_quality=80, max_quality=100, p=0.08),
    dict(type="MotionBlur", length_min=3, length_max=7, p=0.15),
    dict(
        type="HueSaturationValue",
        hue_range=(-10, 10),
        sat_range=(-10, 10),
        val_range=(-10, 10),
        p=0.08,
    ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=(-0.2, 0.2),
        contrast_limit=(-0.2, 0.2),
        p=0.25,
    ),
    dict(
        type="Resize",
        keep_ratio=True,
        img_scale=(height, width),
        ratio_range=(0.75, 1.25),
    ),
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
    dict(
        type="SegRandomCrop",
        size=(height, width),
        cat_max_ratio=0.5,
        ignore_index=ignore_index,
    ),
    dict(type="Pad", size=(height, width)),
    dict(type="RandomFlip", px=0.5, py=0.0),
    dict(type="ToTensor", to_yuv=False),
    dict(type="LabelRemap", mapping=mapping.tolist()),
    dict(
        type="SegRandomAffine",
        degrees=10,
        label_fill_value=ignore_index,
        interpolation=InterpolationMode.BILINEAR,
        rotate_p=0.5,
        translate_p=0,
        scale_p=0,
    ),
    dict(type="ToTensor", to_yuv=True),
    dict(type="ImageNormalize", mean=128.0, std=128.0),
    dict(type="DeleteKeys", keys=["before_pad_shape", "padded_img"]),
]
# dataset config
train_dataset = dict(
    type="DistributedComposeRandomDataset",
    datasets=[
        dict(
            type="DenseboxDataset",
            data_path=os.path.join(path, "train.rec"),
            anno_path=os.path.join(path, "train.json"),
            task_type="segmentation",
            rec_idx_file_path=os.path.join(path, "train.rec.idx"),
            transforms=transforms,
            to_rgb=True,
        )
        for path in train_paths
    ],
    sample_weights=list(map(float, train_weights)),
    shuffle=True,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    batch_size=batch_size_per_gpu,
    num_workers=dataloader_workers,
    pin_memory=True,
    persistent_workers=dataloader_workers > 0,
    multiprocessing_context="spawn",
)

val_transforms = [
    dict(type="ToTensor", to_yuv=True),
    dict(type="LabelRemap", mapping=mapping.tolist()),
    dict(type="ImageNormalize", mean=128.0, std=128.0),
]

val_dataset = dict(
    type="ConcatDataset",
    datasets=[
        dict(
            type="DenseboxDataset",
            data_path=os.path.join(path, "val.rec"),
            anno_path=os.path.join(path, "val.json"),
            task_type="segmentation",
            rec_idx_file_path=os.path.join(path, "val.rec.idx"),
            transforms=val_transforms,
            to_rgb=True,
        )
        for path in val_paths
    ],
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=val_dataloader_workers,
    pin_memory=True,
)


# -------------------------- solver --------------------------
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
            for metric, out in zip(metrics, model_outs[0]):
                if is_train:
                    metric.update(out)
        else:
            # reorganize model prediction results
            assert len(metrics) == 1
            metrics[0].update(*mean_iou_metric_reorganize(model_outs[0]))

    return update_metric


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="ce_loss4"),
    ],
    metric_update_func=get_update_metric(is_train=True),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
    filter_condition=lambda x: x[1] == task_name,
)

# callback config in validation stage
val_miou_metric = MeanIOU(
    seg_class=[str(i) for i in range(num_classes)], ignore_index=ignore_index
)

val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_miou_metric],
    metric_update_func=get_update_metric(is_train=False),
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
    filter_condition=lambda x: x[1] == task_name,
)

aidi_val_transforms = [
    dict(type="ToTensor", to_yuv=True),
    dict(type="ImageNormalize", mean=128.0, std=128.0),
]

aidi_eval_datasets = TASK2AIDI_EVAL_DATA_PATH[task_name]

aidi_eval_datasets = [
    dict(
        type="Auto2dFromImage",
        data_path=path,
        to_rgb=True,
        transforms=aidi_val_transforms,
        return_img_buf=False,
        return_orig_hw=True,
        infer_model_type=None,
    )
    for path in aidi_eval_datasets
]

aidi_eval_loaders = [
    dict(
        type=torch.utils.data.DataLoader,
        dataset=ds,
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=False,
        drop_last=False,
    )
    for ds in aidi_eval_datasets
]

aidi_eval_page_label = task_name
aidi_eval_dataset_id = TASK2AIDI_EVAL_IDS[task_name][0:1]
aidi_eval_type = "semantic_segmentation"
old_prediction = "IPM_Multitask_parsing_v10.1.0"
new_prediction = job_name

aidi_eval_callbacks = []
for dataset_id in TASK2AIDI_EVAL_IDS[task_name]:
    aidi_eval_callback = dict(
        type="AIDIEval",
        output_root=os.path.join(save_prefix, job_name, "prediction"),
        prediction_tags=predition_tags,
        project_id=str(project_id),
        prediction_name=job_name,
        aidi_eval_dataset_id=[dataset_id],
        reformat_output_fn=reformat_seg_to_aidi_eval,
        reformat_out_fn_kwargs={},
    )
    aidi_eval_callbacks.append(
        [
            aidi_eval_callback,
            dict(
                type="StatsMonitor",
                log_freq=log_freq,
            ),
        ]
    )


def tb_update_func(writer, epoch_id, **kwargs):
    name, value = val_miou_metric.get()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = [value.item()]
        else:
            value = value.cpu().numpy().tolist()
    elif not isinstance(value, list):
        value = [value]
    else:
        if isinstance(value[0], torch.Tensor):
            for i in range(len(value)):
                value[i] = value[i].item()
    if not isinstance(name, list):
        name = [name]
    for k, v in zip(name, value):
        writer.add_scalar(k, v, global_step=epoch_id)
