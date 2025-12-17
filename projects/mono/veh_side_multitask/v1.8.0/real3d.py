import copy
import json
import math
import os
from collections import OrderedDict

import torch
import yaml
from common import (
    batch_size_factor,
    debug_mode,
    global_desc,
    log_freq,
    out_stride2channels,
    out_strides,
    pipeline_test,
    training_step,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.datasets.real3d_dataset import Real3DDataset

# dataloader
train_num_workers = 0 if debug_mode else 2 if pipeline_test else 4
val_num_workers = 0 if debug_mode else 2 if pipeline_test else 2
test_num_workers = 0
train_batch_size_per_gpu = max(1, int(19 * batch_size_factor))
test_batch_size_per_gpu = max(1, int(6 * batch_size_factor))
val_batch_size_per_gpu = max(1, int(15 * batch_size_factor))
input_sequence_length = 1
resize_hw_for_val = [540, 960]

task_name = "real3d"
task_loss_weight = 0.1
loss_weight = 1.0

yaml_path = os.path.join(
    os.path.dirname(__file__),
    "dataset_test.yaml" if debug_mode or pipeline_test else "dataset.yaml",
)
dataset_dict = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
data_paths = dataset_dict[task_name]

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


# -------------------------------- Head configuration ---------------------------------- """  # noqa
head_in_channels = 32
head_channels = dict(hm=1, dep=1, rot=2, dim=3, loc_offset=2, wh=2)
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
use_multibin = False
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
        use_random_shift=False,
        random_shift_px=50.0,
    ),
    # dict(type="ImageToTensor", from_numpy=True),
    # dict(
    #     type="ConvertLayout",
    #     hwc2chw=True,
    #     keys=["img"],
    # ),
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
    dict(type="ToTensor", to_yuv=True),
    dict(type="Normalize", mean=128.0, std=128.0),
]
dataset = dict(
    type="Real3DDatasetRec",
    paths=data_paths,
    num_classes=num_classes,
    transforms=transforms,
    select_sample=select_sample,
    num_dist=num_dist,
    view=view,
    track_params=track_params,
)
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dataset,
    collate_fn=None,
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(
        type=torch.utils.data.distributed.DistributedSampler,
        shuffle=True,
    ),
)

# -------------------------------- Model ---------------------------------- """  # noqa
head = dict(
    type="Real3DHead",
    in_strides=out_strides,
    out_strides=[4],
    in_channels=head_in_channels,
    head_channels=head_channels,
    feature_channels=out_stride2channels,
    sep_conv=sep_conv,
    use_varg=use_varg,
    stack=stack,
    bn_kwargs={"eps": 1e-05, "momentum": 0.1},
    factor=2,
    use_bias=True,
    group_base=8,
    interpolate_kwargs=None,
)
val_data_loader = None
test_data_loader = None
loss = dict(
    type="Real3DLoss",
    max_dep=max_dep,
    loss_weights=loss_weights,
    heatmap_type=heatmap_type,
    use_dynamic_weight=use_dynamic_weight,
    norm_type=norm_type,
    use_multibin=use_multibin,
    num_multibin=len(multibin_centers),
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


def vehicle_detection_desc(
    focal_length_default,
    undistort_point_method="Pinhole",
    use_multibin=True,
    multibin_centers=(0.0, math.pi / 2, math.pi, -math.pi / 2),
):

    per_tensor_desc = [
        {
            "task": "camera_3d_detection",
            "output_name": "heatmap_output",
            "focal_length_default": focal_length_default,
            "input_resize_ratio": 1.0,
            "score_threshold": 0.3,
            "topk": 40,
            "kernel_size": [3, 3],
            "properties": [{"channel_labels": ["vehicle"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "depth_output",
            "properties": [{"channel_labels": ["depth"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "rot_output",
            "use_multibin": 0,
            "properties": [{"channel_labels": ["sin", "cos"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "dim_output",
            "properties": [{"channel_labels": ["height", "width", "length"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "loc_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **global_desc,
        },
        {
            "task": "camera_3d_detection",
            "output_name": "wh_output",
            "properties": [{"channel_labels": ["bbox_w", "bbox_h"]}],
            **global_desc,
        },
    ]
    for idx, desc in enumerate(per_tensor_desc):
        if desc["output_name"] == "heatmap_output" and undistort_point_method:
            per_tensor_desc[idx][
                "undistort_point_method"
            ] = undistort_point_method
        if desc["output_name"] == "rot_output" and use_multibin:
            per_tensor_desc[idx]["use_multibin"] = 1
            per_tensor_desc[idx]["multibin_centers"] = multibin_centers
            per_tensor_desc[idx]["properties"] = [
                {
                    "channel_labels": [
                        f"rot{i}" for i in range(len(multibin_centers) * 3)
                    ]
                }
            ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=vehicle_detection_desc(
        focal_length_default=focal_length_default,
        use_multibin=use_multibin,
    ),
    node_name=f"{task_name}_desc",
)

out_module = dict(
    type="OutputModule",
    head=head,
    head_parser=None,
    target=None,
    loss=loss,
    prefix=task_name + "_head",
)

val_out_module = copy.deepcopy(out_module)
val_out_module["loss"] = None
val_out_module["postprocess"] = decoder
test_out_module = copy.deepcopy(val_out_module)
test_out_module["target"] = None
test_out_module["postprocess"] = None

if training_step != "int_infer":
    test_out_module["postprocess"] = decoder
else:
    test_out_module["postprocess"] = add_desc_pp

nodes = {f"{task_name}_head": out_module}
val_nodes = {f"{task_name}_head": val_out_module}
test_nodes = {f"{task_name}_head": test_out_module}
# -------------------------------- Metric ---------------------------------- """  # noqa
metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=name) for name in loss_names],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[
            {"label_pattern": None, "pred_pattern": f"^.*{name}$"}
            for name in loss_names
        ]
    ),
    filter_condition=None,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


inputs = dict(
    image_name=None,
    image_height=None,
    image_width=None,
    imgs=None,
    layout=None,
    color_space=None,
    calibration=None,
    dist_coeffs=None,
    Tr_vel2cam=None,
    image_id=None,
    ignore_mask=None,
    image_transform=None,
    target=None,
    view=None,
    index=None,
    valid=True,
    mat_vcsgnd2img=None,
    raw_img=None,
    Tr_vcs2cam=None,
    # side_img=None,
    img_shape=None,
    pad_shape=None,
    # before_pad_shape=None,
    # padded_img=None,
    # ig_bboxes=None,
)

val_inputs = copy.deepcopy(inputs)
val_inputs["annotations"] = None
if training_step == "int_infer":
    test_inputs = dict()
    test_out_module["postprocess"] = add_desc_pp
else:
    test_inputs = dict(
        img_name=None,
        calibration=None,
        dist_coeffs=None,
        image_transform=None,
    )


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
