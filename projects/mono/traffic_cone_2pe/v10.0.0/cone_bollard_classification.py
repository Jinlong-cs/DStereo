import copy
import os
from collections import OrderedDict

import torch

from hat.data.collates.collates import collate_2d
from hat.utils import Config

# -------------------------- common --------------------------
cfg_dir = os.path.dirname(__file__)
BASE_CONFIG = Config.fromfile(os.path.join(cfg_dir, "auto2d_base.py"))

alpha = BASE_CONFIG.alpha
bn_kwargs = BASE_CONFIG.bn_kwargs
backbone_out_channels = BASE_CONFIG.backbone_out_channels
norm_len = BASE_CONFIG.norm_len
norm_method = BASE_CONFIG.norm_method
log_freq = BASE_CONFIG.log_freq
input_size = BASE_CONFIG.input_size
task_loss_weights = BASE_CONFIG.task_loss_weights
compile_model = BASE_CONFIG.compile_model
get_classification_model_desc = BASE_CONFIG.get_classification_model_desc
test_image_dir = BASE_CONFIG.test_image_dir
train_num_workers = BASE_CONFIG.train_num_workers
val_num_workers = BASE_CONFIG.val_num_workers
test_num_workers = BASE_CONFIG.test_num_workers
input_sequence_length = BASE_CONFIG.input_sequence_length

dataset_val_img_path = BASE_CONFIG.dataset_val_img_path
dataset_train_rect_path = BASE_CONFIG.dataset_train_rect_path
annotations_train_path = BASE_CONFIG.annotations_train_path
train_batch_size_per_gpu = BASE_CONFIG.train_batch_size_per_gpu
val_batch_size_per_gpu = BASE_CONFIG.val_det_batch_size_per_gpu
test_batch_size_per_gpu = BASE_CONFIG.test_batch_size_per_gpu


# -------------------------- task --------------------------
dataset_index = 0
task_type = "classification"
task_name = f"cone_bollard_{task_type}"
divisor = 64
num_classes = 6
task_loss_weight = task_loss_weights[task_name]


# -------------------------- data --------------------------
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="DenseboxDataset",
        data_path=dataset_train_rect_path[dataset_index],
        anno_path=annotations_train_path[dataset_index],
        task_type=task_type,
        class_id=range(1, num_classes + 1),
        category=range(1, num_classes + 1),
        to_rgb=False,
        ignore_hard=True,
        version="v2",
        # note: Normalize needs to be after ToTensor if `to_yuv`==True
        transforms=[
            dict(
                type="RoiTransformer",
                roi_crop_parm=dict(
                    norm_len=norm_len,
                    norm_method=norm_method,
                    output_wh=input_size,
                    input_wh=None,
                    min_crop_scale=0.9,
                    max_crop_scale=1.1,
                    max_coord_jitter_ratio=0.05,
                    img_min_scale=0.01,
                    img_max_scale=100,
                    padd_val=0,
                    random_roi_ratio=0.0,
                    restrict_roi_in_center=False,
                    flip_ratio=0,
                ),
                img_crop_parm=dict(
                    target_wh=input_size,
                    inter_method=10,
                    use_pyramid=True,
                    pyramid_min_step=0.7,
                    pyramid_max_step=0.8,
                    pixel_center_aligned=False,
                ),
                bbox_ts_parm=dict(
                    clip=False,
                    min_valid_area=100,
                    min_valid_clip_area_ratio=0.02,
                    min_edge_size=10,
                    label_type=task_type,
                ),
            ),
            dict(type="RandomFlip", px=0.5),
            dict(type="ToTensor", to_yuv=True),
            dict(type="Normalize", mean=128.0, std=128.0),
            dict(
                type="Batchify",
                size=input_size[::-1],
                divisor=divisor,
                repeat=input_sequence_length,
            ),
            dict(type="RenameKeys", keys=["imgs|img"]),
        ],
    ),
    # sampler=dict(type=torch.utils.data.DistributedSampler),
    # collate_fn=collate_2d,
    batch_size=train_batch_size_per_gpu,
    # shuffle=True,
    num_workers=train_num_workers,
    pin_memory=True,
    persistent_workers=train_num_workers > 0,
)
data_loader = BASE_CONFIG.change_dataset(data_loader, task_name, "mono")


val_transforms = [
    dict(
        type="RoiTransformer",
        roi_crop_parm=dict(
            norm_len=norm_len,
            norm_method=norm_method,
            output_wh=input_size,
            input_wh=None,
            min_crop_scale=1.0,
            max_crop_scale=1.0,
            max_coord_jitter_ratio=0.0,
            img_min_scale=0.01,
            img_max_scale=100,
            padd_val=0,
            random_roi_ratio=0.0,
            restrict_roi_in_center=False,
            flip_ratio=0,
        ),
        img_crop_parm=dict(
            target_wh=input_size,
            inter_method=10,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            pixel_center_aligned=False,
        ),
        bbox_ts_parm=dict(
            clip=False,
            min_valid_area=100,
            min_valid_clip_area_ratio=0.02,
            min_edge_size=10,
            label_type=task_type,
        ),
    ),
    dict(type="ToTensor"),
]

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Auto2dFromImage",
        data_path=dataset_val_img_path[0],
        to_rgb=True,
        transforms=val_transforms,
        infer_model_type="full_image_2pe",
    ),
    batch_size=val_batch_size_per_gpu,
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=val_num_workers,
    pin_memory=True,
    drop_last=False,
)


test_transforms = copy.deepcopy(val_transforms)

test_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Auto2dFromImage",
        data_path=test_image_dir,
        to_rgb=True,
        transforms=test_transforms,
    ),
    batch_size=test_batch_size_per_gpu,
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=test_num_workers,
    pin_memory=True,
    drop_last=False,
)


# -------------------------- model --------------------------
inputs = dict(
    # # TODO(min.du): use imgs instead of img #
    img_name=None,
    img_id=None,
    img_height=None,
    img_width=None,
    gt_classes=None,
    color_space=None,
    layout=None,
    img_shape=None,
    pad_shape=None,
    # ig_bboxes=None,
    crop_roi=None,
)

val_inputs = copy.deepcopy(inputs)
val_inputs.pop("gt_classes")
test_inputs = dict()


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
        type="TinyVarGNetV2ClassificationHead",
        input_channels=backbone_out_channels[-1],
        disable_quanti_input=True,
        num_classes=num_classes,
        gc_group_base=8,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
        cls_pooling_stride=2,
        factor=2,
        export_model=True if compile_model else False,
    ),
    target=lambda x, _: x.get("gt_classes").reshape(-1),
    loss=dict(
        type=torch.nn.CrossEntropyLoss,
        label_smoothing=0.01,
        ignore_index=-1,
    ),
    head_parser=None,
    prefix="head",
)

add_desc_pp = get_classification_model_desc(
    task_name=task_name,
    output_name=task_name,
    class_name="traffic_cone",
    desc_id=str(num_classes),
)


val_out_module = copy.deepcopy(out_module)

val_out_module["loss"] = None
val_out_module["target"] = None
val_out_module["postprocess"] = add_desc_pp
test_out_module = copy.deepcopy(val_out_module)


nodes = {f"{task_name}_head": out_module}
val_nodes = {f"{task_name}_head": val_out_module}
test_nodes = {f"{task_name}_head": test_out_module}


# -------------------------- solver --------------------------
def update_metric(metrics, batch, model_outs):
    losses, preds, _ = model_outs[0]
    for metric in metrics:
        metric.update(losses)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    metrics=[dict(type="LossShow", name="loss_cls")],
    epoch_log_freq=1,
    log_prefix=task_name,
    filter_condition=lambda x: x[1] == task_name,
    step_log_freq=log_freq,
    reset_metrics_by="log",
)
