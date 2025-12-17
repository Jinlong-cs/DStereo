import os

import torch
import yaml
from common import (
    backbone,
    batch_size_per_gpu,
    bn_kwargs,
    drop_last,
    factor,
    group_base,
    local_train,
    log_freq,
    neck,
    num_workers,
    pin_memory,
    stride2channels,
    unet_out_strides,
)
from description import face_detection_desc

from hat.callbacks.metric_updater import update_metric_using_regex

task_name = "face_detection"
# -------------------------------- Dataset configuration ---------------------------------- """  # noqa
yaml_path = os.path.join(os.path.dirname(__file__), "dataset.yaml")
yaml_file = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
bucket_root = "/horizon-bucket" if local_train else "/bucket/input"
task = task_name.split("_")[0]
train_rec_paths = yaml_file[task]["train"]["rec"]
for idx in range(len(train_rec_paths)):
    train_rec_paths[idx] = os.path.join(bucket_root, train_rec_paths[idx])

# -------------------------------- Head configuration ---------------------------------- """  # noqa
head_in_channels = 16
head_channels = dict(hm=1, wh=2)
sep_conv = True
use_varg = False
stack = 1

# -------------------------------- Loss configuration ---------------------------------- """  # noqa
loss_weights = dict(
    hm=1.0,
    wh=1.0,
)
heatmap_type = dict(wh="point")
loss_names = []
for name in head_channels.keys():
    loss_names.append("face_" + name + "_loss")

# -------------------------------- Dataloader ---------------------------------- """  # noqa
size = (960, 192)
transforms = [
    dict(
        type="PresetCrop",
        crop_top=220,
        crop_bottom=128,
        crop_left=0,
        crop_right=0,
    ),
    dict(type="RandomFlip", px=0.5),
    dict(
        type="CenterNetTargetGenerator",
        input_size=size,
        head_channels=dict(hm=1, wh=2),
        down_stride=4,
    ),
]
dataset = dict(
    type="PbRec2DDataset",
    paths=train_rec_paths,
    transforms=transforms,
    to_rgb=False,
)

dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dataset,
    collate_fn=None,
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=drop_last,
    sampler=dict(
        type=torch.utils.data.distributed.DistributedSampler,
        shuffle=True,
    ),
)
# -------------------------------- Model ---------------------------------- """  # noqa
head = dict(
    type="Real3DHead",
    in_strides=unet_out_strides,
    out_strides=unet_out_strides,
    in_channels=head_in_channels,
    head_channels=head_channels,
    feature_channels=stride2channels,
    sep_conv=sep_conv,
    use_varg=use_varg,
    stack=stack,
    bn_kwargs=bn_kwargs,
    factor=factor,
    use_bias=True,
    group_base=group_base,
    interpolate_kwargs=None,
    node_name=f"{task_name}_head",
)

loss = dict(
    type="CenterNetLoss",
    task="face",
    loss_weights=loss_weights,
    heatmap_type=heatmap_type,
    reg_keys=["wh"],
    node_name=f"{task_name}_loss",
)
model = dict(
    type="SegmentorV2",
    backbone=backbone,
    neck=neck,
    head=head,
    target=None,
    loss=loss,
    desc=None,
    postprocess=None,
)

deploy_model = dict(
    type="SegmentorV2",
    backbone=backbone,
    neck=neck,
    head=head,
    target=None,
    loss=None,
    desc=dict(
        type="AddDesc",
        per_tensor_desc=face_detection_desc(),
        node_name=f"{task_name}_desc",
    ),
    postprocess=None,
)

# -------------------------------- Metric ---------------------------------- """  # noqa
metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=name) for name in loss_names],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[
            {"label_pattern": None, "pred_pattern": f"^.*{task_name}_{name}$"}
            for name in loss_names
        ]
    ),
    filter_condition=None,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)
# -------------------------------- Inputs ---------------------------------- """  # noqa
inputs = dict(
    train=dict(
        labels=dict(
            hm=torch.ones([1, 1, 48, 240]),
            wh=torch.ones([1, 2, 48, 240]),
            ignore_mask=torch.ones([1, 1, 48, 240]),
        ),
    ),
    val=dict(),
    test=dict(),
)
