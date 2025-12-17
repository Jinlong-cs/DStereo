import copy
import json
import os
from collections import OrderedDict

import torch
from lib.aidi_eval_helper import TASK2AIDI_EVAL_IDS, reformat_psd_to_aidi_eval

from hat.data.collates.collates import collate_psd
from hat.metrics.psd_metric import PSDMetric
from hat.utils import Config
from hat.utils.apply_func import is_list_of_type

task_name = "psd"

cfg_dir = os.path.dirname(__file__)
BASE_CONFIG = Config.fromfile(
    os.path.join(os.path.dirname(cfg_dir), "base.py")
)

log_freq = BASE_CONFIG.log_freq

save_prefix = BASE_CONFIG.save_prefix
job_name = BASE_CONFIG.job_name
aidi_eval = BASE_CONFIG.aidi_eval
project_id = BASE_CONFIG.project_id
predition_tags = BASE_CONFIG.get("predition_tags", [])
training_step = BASE_CONFIG.training_step
val_num_workers = BASE_CONFIG.val_num_workers[task_name]
# train config
task_loss_weight = BASE_CONFIG.psd_loss_weight
batch_size_per_gpu = BASE_CONFIG.batch_size_per_gpu["psd"]
dataloader_workers = BASE_CONFIG.train_num_workers["psd"]
val_dataloader_workers = BASE_CONFIG.val_num_workers["psd"]
local_train = BASE_CONFIG.local_train

height = BASE_CONFIG.height
width = BASE_CONFIG.width
bn_kwargs = BASE_CONFIG.bn_kwargs


train_data_path = BASE_CONFIG.psd_train_paths
train_data_weight = BASE_CONFIG.psd_train_weights
val_data_path = BASE_CONFIG.psd_val_paths
task_desc = BASE_CONFIG.task_desc[task_name]

# model config
feat_channels = BASE_CONFIG.feat_channels
bifpn_out_strides = BASE_CONFIG.bifpn_out_strides

# head config
num_slot_type = BASE_CONFIG.psd_num_slot_type
local_stride_idx = BASE_CONFIG.psd_local_stride_idx
global_stride_idx = BASE_CONFIG.psd_global_stride_idx

# loss config
gaussian_loss_alpha = BASE_CONFIG.psd_gaussian_loss_alpha
gaussian_loss_gamma = BASE_CONFIG.psd_gaussian_loss_gamma
local_loss_radius = BASE_CONFIG.psd_local_loss_radius
local_loss_weights = BASE_CONFIG.psd_local_loss_weights
local_loss_weights = [i * task_loss_weight for i in local_loss_weights]
local_loss_feat_size = BASE_CONFIG.psd_local_loss_feat_size
local_loss_far_weight = BASE_CONFIG.psd_local_loss_far_weight

# global loss config
global_loss_weights = BASE_CONFIG.psd_global_loss_weights
global_loss_weights = [i * task_loss_weight for i in global_loss_weights]
global_loss_feat_size = BASE_CONFIG.psd_global_loss_feat_size

# slot loss config
slot_weight_dict = BASE_CONFIG.psd_slot_weight_dict

# metric config
metric_global_threshold = BASE_CONFIG.metric_global_threshold
metric_global_topk = BASE_CONFIG.metric_global_topk
metric_global_downsample_factor = BASE_CONFIG.metric_global_downsample_factor
metric_global_max_distance = BASE_CONFIG.metric_global_max_distance
metric_global_kernel_size = BASE_CONFIG.metric_global_kernel_size
metric_local_threshold = BASE_CONFIG.metric_local_threshold
metric_local_topk = BASE_CONFIG.metric_local_topk
metric_local_downsample_factor = BASE_CONFIG.metric_local_downsample_factor
metric_local_max_distance = BASE_CONFIG.metric_local_max_distance
metric_local_kernel_size = BASE_CONFIG.metric_local_kernel_size
metric_nms_distance_threshold = BASE_CONFIG.metric_nms_distance_threshold
metric_threshold_fuse_distance = BASE_CONFIG.metric_threshold_fuse_distance
metric_iou_threshold = BASE_CONFIG.metric_iou_threshold

# -------------------------- model --------------------------
inputs = dict(
    ori_img=torch.randn((1, 3, height, width)),
    label=[],
    color_space=["yuv"],
    layout=["chw"],
    img_name=["xxx."],
    img_scene=[],
    img_shape=[
        torch.Tensor([height]),
        torch.Tensor([width]),
        torch.Tensor([3]),
    ],  # noqa
    pad_shape=[
        torch.Tensor([height]),
        torch.Tensor(
            [
                width,
            ]
        ),
        torch.Tensor([3]),
    ],  # noqa
)

val_inputs = copy.deepcopy(inputs)

if aidi_eval and False:  # not ready until image path bug fixed
    val_inputs = dict(
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
        orig_hw=[
            torch.Tensor([height]),
            torch.Tensor([width]),
        ],
    )

if training_step == "int_infer":
    test_inputs = dict()
else:
    test_inputs = copy.deepcopy(val_inputs)
    test_inputs.pop("label")


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
        type="SuperPSDHead",
        global_head=dict(
            type="SuperPSDGlobalCPNHead",
            num_cascade=4,
            num_slot_type=num_slot_type,
            in_channels=32,
            out_shape=(28, 28),
            out_channels=32,
        ),
        local_head=dict(
            type="SuperPSDLocalCPNHead",
            num_cascade=4,
            in_channels=32,
            out_shape=(224, 224),
            out_channels=32,
        ),
    ),
    head_parser=None,
    target=dict(
        type="SuperPSDTarget",
        global_target=dict(
            type="SuperPSDGlobalTarget",
            input_size=height,
            feature_size=global_loss_feat_size,
            slot_weight_dict=slot_weight_dict,
        ),
        local_target=dict(
            type="SuperPSDLocalTarget",
            input_size=height,
            feature_size=local_loss_feat_size,
            radius=local_loss_radius,
            far_weight=local_loss_far_weight,
            slot_weight_dict=slot_weight_dict,
        ),
        is_val=False,
    ),
    loss=dict(
        type="SegLoss",
        loss=[
            dict(
                type="SuperPSDLoss",
                local_loss=dict(
                    type="SuperPSDLocalLoss",
                    classification_loss=dict(
                        type="GaussianFocalLoss",
                        alpha=gaussian_loss_alpha,
                        gamma=gaussian_loss_gamma,
                    ),
                    offset_loss=dict(type="SmoothL1Loss", reduction="none"),
                    sline_angle_loss=dict(
                        type=torch.nn.MSELoss, reduction="none"
                    ),
                    point_type_loss=dict(
                        type=torch.nn.BCELoss, reduction="none"
                    ),
                    loss_weights=local_loss_weights,
                ),
                global_loss=dict(
                    type="SuperPSDGlobalLoss",
                    classification_loss=dict(
                        type="FocalLossV2", reduction="none"
                    ),
                    offset_loss=dict(type="SmoothL1Loss", reduction="none"),
                    occupancy_loss=dict(
                        type=torch.nn.BCELoss, reduction="none"
                    ),
                    slot_type_loss=dict(
                        type=torch.nn.BCELoss, reduction="none"
                    ),
                    direction_loss=dict(
                        type=torch.nn.MSELoss, reduction="none"
                    ),
                    loss_weights=global_loss_weights,
                ),
            ),
        ],
    ),
    prefix="head",
    keep_name=True,
)

val_out_module = copy.deepcopy(out_module)
val_out_module["target"] = None
val_out_module["loss"] = None

val_out_module["postprocess"] = dict(
    type="PSDPostprocess",
    input_size=[height, width],
    threshold_globalslot_feature=metric_global_threshold,
    topk_globalslot_feature=metric_global_topk,
    downsample_factor_globalslot_feature=metric_global_downsample_factor,
    global_kernel_size=metric_global_kernel_size,
    threshold_localjunction_feature=metric_local_threshold,
    topk_localjunction_feature=metric_local_topk,
    downsample_factor_localjunction_feature=metric_local_downsample_factor,
    local_kernel_size=metric_local_kernel_size,
    threshold_fuse_distance=metric_threshold_fuse_distance,
    nms_distance_threshold=metric_nms_distance_threshold,
)

add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        json.dumps(dict(task=task_name, **desc)) for desc in task_desc
    ],
)
test_out_module = copy.deepcopy(val_out_module)
test_out_module["postprocess"] = None
if training_step == "int_infer":
    test_out_module["postprocess"] = add_desc_pp

nodes = {f"{task_name}_head": out_module}
val_nodes = {f"{task_name}_head": val_out_module}
test_nodes = {f"{task_name}_head": test_out_module}

# dataset config
transforms = [
    dict(type="ToTensor", to_yuv=True),
    dict(type="ImageNormalize", mean=128.0, std=128.0),
]
train_dataset = dict(
    type="DistributedComposeRandomDataset",
    datasets=[
        dict(
            type="PSDSlotDataset",
            path=path,
            input_size=(width, height),
            transforms=transforms,
        )
        for path in train_data_path
    ],
    sample_weights=list(map(float, train_data_weight)),
    shuffle=True,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    batch_size=batch_size_per_gpu,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_psd,
    persistent_workers=dataloader_workers > 0,
    multiprocessing_context="spawn",
)

val_dataset = dict(
    type="ConcatDataset",
    datasets=[
        dict(
            type="PSDSlotDataset",
            path=path,
            input_size=(width, height),
            transforms=transforms,
        )
        for path in val_data_path
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
    collate_fn=collate_psd,
)


# -------------------------- solver --------------------------
def val_flat_condition(key, values):
    """
    Validation need label and flating it will result in a bug with more
    than 255 parameters. So this function will not flat label.
    """
    flag = None
    if "label" in key:
        flag = False
    elif isinstance(values, dict) or not is_list_of_type(values, torch.Tensor):
        flag = True
    else:
        flag = False
    return flag


def psd_metric_reorganize(out):
    labels = preds = None
    result_dict = {}
    for field, data in zip(out._fields, out):
        # import hat.utils.forkedpdb as pdb;pdb.set_trace()
        if "preds_label" in field:
            preds = data
            result_dict["preds_label"] = preds
        if "label" in field:
            labels = data
            result_dict["labels"] = labels
        if "img_scene" in field:
            img_scene = data
            result_dict["img_scene"] = img_scene
    return result_dict


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
            metrics[0].update(psd_metric_reorganize(model_outs[0]))

    return update_metric


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="global_cls"),
        dict(type="LossShow", name="global_offset"),
        dict(type="LossShow", name="global_occupancy"),
        dict(type="LossShow", name="global_slot"),
        dict(type="LossShow", name="global_dir"),
        dict(type="LossShow", name="local_cls"),
        dict(type="LossShow", name="local_offset"),
        dict(type="LossShow", name="local_angle"),
        dict(type="LossShow", name="loss_point_type"),
    ],
    metric_update_func=get_update_metric(is_train=True),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
    filter_condition=lambda x: x[1] == task_name,
)

# callback config in validation stage
val_psd_metric = PSDMetric(
    global_max_distance=metric_global_max_distance,
    local_max_distance=metric_local_max_distance,
    is_training=True,
    iou_threshold=metric_iou_threshold,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_psd_metric],
    metric_update_func=get_update_metric(is_train=False),
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
    filter_condition=lambda x: x[1] == task_name,
)


def tb_update_func(writer, epoch_id, **kwargs):
    name, value = val_psd_metric.get(log=False)
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


aidi_val_transforms = [
    dict(type="ToTensor", to_yuv=True),
    dict(type="ImageNormalize", mean=128.0, std=128.0),
]

aidi_eval_page_label = task_name
aidi_eval_dataset_id = TASK2AIDI_EVAL_IDS[task_name][0:1]
aidi_eval_type = "Ipm_Psd"
old_prediction = "IPM_Multitask_20220921v10.1.0"
new_prediction = job_name

aidi_eval_loaders = [val_data_loader]  # there are some invalid image paths

aidi_eval_callbacks = []
for dataset_id in TASK2AIDI_EVAL_IDS[task_name]:
    aidi_eval_callback = dict(
        type="AIDIEval",
        output_root=os.path.join(save_prefix, job_name, "prediction"),
        prediction_tags=predition_tags,
        project_id=str(project_id),
        prediction_name=job_name,
        aidi_eval_dataset_id=[dataset_id],
        reformat_output_fn=reformat_psd_to_aidi_eval,
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
