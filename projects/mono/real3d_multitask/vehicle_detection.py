import math
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
from description import vehicle_detection_desc

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.datasets.real3d_dataset import Real3DDataset

task_name = "vehicle_detection"
# -------------------------------- Dataset configuration ---------------------------------- """  # noqa
image_size = (2160, 3840)
input_size = (960, 192)
center_shift = (0, 184.0)  # input_size == 960, 192
# center_shift = (0, -56)  # input_size == 960, 512
pre_resize_scale = -1.0
num_classes = 1
num_dist = 8
max_objs = 100
select_sample = False
category_id_dict = Real3DDataset.get_category_id_dict(num_classes)
track_params = [0, 0, 0, 1]
view = "front"
down_stride = 4
pop_anno = True
yaml_path = os.path.join(os.path.dirname(__file__), "dataset.yaml")
yaml_file = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
bucket_root = "/horizon-bucket" if local_train else "/bucket/input"
task = task_name.split("_")[0]
train_rec_paths = yaml_file[task]["train"]
for idx in range(len(train_rec_paths)):
    train_rec_paths[idx] = os.path.join(bucket_root, train_rec_paths[idx])

# -------------------------------- Head configuration ---------------------------------- """  # noqa
head_in_channels = 32
head_channels = dict(hm=1, dep=1, rot=12, dim=3, loc_offset=2, wh=2)
sep_conv = True
use_varg = False
stack = 1
# -------------------------------- Loss configuration ---------------------------------- """  # noqa
max_dep = 80
use_dynamic_weight = True
if use_dynamic_weight:
    loss_weights = dict(
        hm=1.0,
        rot=10.0,
        dep=3.0,
        dim=1.0,
        loc_offset=3.0,
        wh=0.01,
        std=10.0,
    )
else:
    loss_weights = dict(
        hm=1.5,
        rot=5.0,
        dep=1.5,
        dim=1.0,
        loc_offset=2.0,
        wh=0.02,
        std=1.0,
    )
heatmap_type = dict(wh="point", dep="point", loc_offset="point", dim="point")
norm_type = "l1"
use_multibin = True
multibin_centers = (0.0, math.pi / 2, math.pi, -math.pi / 2)
loss_names = []
for name in head_channels.keys():
    loss_names.append(name + "_loss")
    if use_dynamic_weight:
        loss_names.append(name + "_weight_loss")
# -------------------------------- Decoder configuration ---------------------------------- """  # noqa
focal_length_default = 2411.0
topk = 40
max_pooling_kernel_size = 3
undistort = True
fisheye = False
nms_kwargs = dict(iou_threshold=0.7, replace=True)


# -------------------------------- Dataloader ---------------------------------- """  # noqa
transforms = [
    dict(
        type="ImageTransform",
        size=input_size,
        center_shift=center_shift,
        pre_resize_scale=pre_resize_scale,
    ),
    dict(type="ImageToTensor", from_numpy=True),
    dict(
        type="ConvertLayout",
        hwc2chw=True,
        keys=["img"],
    ),
    dict(
        type="Real3dTargetGenerator",
        num_classes=num_classes,
        focal_length_default=focal_length_default,
        input_size=input_size,
        category_id_dict=category_id_dict,
        origin_image_shape=image_size,
        head_channels=head_channels,
        down_stride=down_stride,
        max_objs=max_objs,
        center_shift=center_shift,
        max_depth=max_dep,
        pop_anno=pop_anno,
        undistort=undistort,
        fisheye=fisheye,
        pre_resize_scale=pre_resize_scale,
        min_wh=0.0,
        multibin_centers=multibin_centers,
    ),
]

dataset = dict(
    type="Real3DDatasetRec",
    paths=train_rec_paths,
    num_classes=num_classes,
    transforms=transforms,
    select_sample=select_sample,
    num_dist=num_dist,
    view=view,
    track_params=track_params,
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
    type="Real3DLoss",
    max_dep=max_dep,
    loss_weights=loss_weights,
    heatmap_type=heatmap_type,
    use_dynamic_weight=use_dynamic_weight,
    norm_type=norm_type,
    use_multibin=use_multibin,
    num_multibin=len(multibin_centers),
    node_name=f"{task_name}_loss",
)

decoder = dict(
    type="Real3DDecoder",
    focal_length_default=focal_length_default,
    topk=topk,
    max_pooling_kernel_size=max_pooling_kernel_size,
    center_shift=center_shift,
    undistort=undistort,
    fisheye=fisheye,
    pre_resize_scale=pre_resize_scale,
    nms_kwargs=nms_kwargs,
    use_multibin=use_multibin,
    multibin_centers=multibin_centers,
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
        per_tensor_desc=vehicle_detection_desc(
            focal_length_default=focal_length_default
        ),
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
        calibration=torch.ones([1, 3, 4]),
        dist_coeffs=torch.ones([1, 8]),
        valid=torch.ones([1, 1]),
        Tr_vel2cam=torch.ones([1, 4, 4]),
        target=dict(
            hm=torch.ones([1, 1, 48, 240]),
            dep=torch.ones([1, 1, 48, 240]),
            rot=torch.ones([1, 12, 48, 240]),
            dim=torch.ones([1, 3, 48, 240]),
            loc_offset=torch.ones([1, 2, 48, 240]),
            wh=torch.ones([1, 2, 48, 240]),
            ignore_mask=torch.ones([1, 1, 48, 240]),
            weight_hm=torch.ones([1, 1, 48, 240]),
            ind_=torch.ones([1, 100], dtype=torch.long),
            dim_=torch.ones([1, 100, 3]),
            loc_=torch.ones([1, 100, 3]),
            ind_mask_=torch.ones([1, 100]),
            rot_y_=torch.ones([1, 100]),
            alpha_z_=torch.ones([1, 100]),
            bin_cls_=torch.ones([1, 100, 4]),
            bin_offset_=torch.ones([1, 100, 4]),
        ),
    ),
    val=dict(),
    test=dict(),
)
