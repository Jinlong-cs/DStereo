import copy
import json
import os

import torch

from hat.data.collates.collates import collate_2d
from ...common import (
    aidi_eval,
    all_task_data_files,
    backbone,
    bifpn_stride2levels,
    bpu_transforms,
    compare_version,
    datapaths,
    feat_channels,
    input_size,
    is_int_infer,
    log_freq,
    model_version,
    neck,
    save_all_data_paths,
    sod_loss_weight,
    tasks_batch_size,
    test_image_dir,
    val_only,
)
from ...lib.aidieval_helper import TASK2AIDI_EVAL_IDS
from ...lib.detection2d.config_utils import adjust_multitask_sample_weight
from ...lib.detection2d.vis_utils import VisualizationCallBack
from ...lib.utils import reorganize_sod_batch_results
from ..detection_common import (
    get_2d_detection_dataloader,
    get_train_transform,
    val_transforms,
)
from ..vdvru_common import (
    EVAL_TYPE_2D_DET,
    get_2d_aidi_eval_loaders,
    get_2d_det_aidi_eval_callback,
)

task_name = "sod"
version = "v1.0"
# task description
desc_task_name = "centernet_static_obstacle_detection"
num_classes = 6
classnames = [
    "parking_column",
    "traffic_bollard",
    "aframe_sign",
    "traffic_cone",
    "parking_lock_open",
    "parking_lock_close",
]
desc_class_names = [f"sod_{s}" for s in classnames]
task_class_ids = [12, 13, 14, 9, 15, 16]
common_dataset_names = {
    "parking_column": [0],
    "traffic_bollard": [1, 2, 3],
    "aframe_sign": [1, 2, 3],
    "traffic_cone": [1, 2, 3],
    "parking_lock_open": [4, 5],
    "parking_lock_close": [4, 5],
}
class_id_group = list(range(num_classes))  # map classname to dataset
# mapped_id = [0, 1, 1, 1, 4, 5]
mapped_id = [0, 1, 2, 3, 4, 5]  # merge class ids
classnames = {
    cn: list(common_dataset_names.keys())[class_id_group[i]]
    for i, cn in enumerate(classnames)
}
class_maps = {}
task_list = []
for _, (gi, classname) in enumerate(zip(class_id_group, classnames)):
    if gi not in class_maps:
        class_maps[gi] = {}
        task_list.append(datapaths.get(classnames[classname]))
    for real_labeled_id in common_dataset_names[classname]:
        class_maps[gi][task_class_ids[real_labeled_id]] = real_labeled_id

if not is_int_infer or val_only:
    nms_score_thresholds = [0] * len(classnames)
else:
    nms_score_thresholds = [0.42, 0.42, 0.42, 0.44, 0.42, 0.44]
nms_iou_thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
metric_iou_threshold = 0.5
metric_score_threshold = 0
color_map = (
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 0),
    (255, 0, 255),
    (0, 0, 110),
    (50, 150, 255),
)
desc_out_names = [
    "heatmap",
    "wh",
    "offset",
    "maxheat",
]  # in order of FilterByThreshold's output order  # noqa
divisor = 64
use_iou_replace_ctrness = True
batch_size_dict = tasks_batch_size[task_name]
# set sub-batch_size for each class, currently equally distributed,
# could be adjusted by sample distribution
# set this by prior distribution assuming batch size is multiple of 8
sub_task_bs_weight = [1, 1, 2, 1, 2, 1]
assert len(sub_task_bs_weight) == len(common_dataset_names)
base_batch_size = 8
train_bs = batch_size_dict["train"]
assert train_bs % base_batch_size == 0, (
    f"training batch_size should be multiple of {base_batch_size}"
    + f"but get {train_bs}"
)
bs_factor = train_bs // base_batch_size
compose_batchsize = (
    [bs_factor * w for w in sub_task_bs_weight]
    if bs_factor > 0
    else [train_bs]
)  # considering batch size != 8 condition
use_ignore = True  # whether to use ignore regions in dataset
in_strides = 4
cls_loss_weight = sod_loss_weight * torch.tensor([1, 1, 1, 1, 1, 1])
task_class_ids = [12, 13, 14, 9, 15, 16]
test_cfg = dict(
    topk=100,
    local_maximum_kernel=3,
    max_per_img=100,
    nms="nms",
    score_thresholds=nms_score_thresholds,
    iou_thresholds=nms_iou_thresholds,
)


def get_model(mode):
    model = dict(
        type="SingleStageDetector",
        backbone=backbone,
        neck=neck,
        box_module=dict(
            type="OutputModule",
            node_name=f"{task_name}_out_module",
            head=dict(
                type="CenterNetHead",
                node_name=f"{task_name}_ctnet_head",
                num_classes=num_classes,
                strides2levels=bifpn_stride2levels,
                in_strides=in_strides,
                feat_channels=feat_channels,
                conv_type="sep_conv",
            ),
            head_parser=dict(
                type="CenterNetHeadParser",
            )
            if mode == "train"
            else None,
            target=dict(
                type="CenterNetTarget",
                num_classes=num_classes,
                strides2levels=bifpn_stride2levels,
                in_strides=in_strides,
                img_shape=input_size,
                use_ignore=use_ignore,
                gaussian_radius_alpha=0.5,
            )
            if mode == "train"
            else None,
            loss=dict(
                type="CenterNetIOULoss",
                heatmap_loss=dict(
                    type="CenterNetFocalLoss",
                    loss_name="loss_heatmap",
                    alpha=2.0,
                    gamma=4.0,
                    loss_weight=cls_loss_weight,
                ),
                iou_loss=dict(
                    type="CIoULoss",
                    loss_name="loss_ciou",
                    loss_weight=sod_loss_weight,
                ),
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="CenterNetDecoder",
                num_classes=num_classes,
                img_shape=input_size,
                test_cfg=test_cfg,
                with_nms=True,
                class_map=mapped_id,
            )
            if mode != "train"
            else None,
        ),
    )
    add_desc_pp = dict(
        type="AddDesc",
        per_tensor_desc=[
            json.dumps(
                dict(
                    task=desc_task_name,
                    output_name=desc_out_names[0],  # heatmap
                    properties=[
                        dict(
                            channel_label=desc_class_names,
                            stride=in_strides,
                            channel_score_threshold=nms_score_thresholds,
                            channel_iou_threshold=nms_iou_thresholds,
                        )
                    ],
                )
            )
        ]
        + [
            json.dumps(
                dict(
                    task=desc_task_name,
                    output_name=outname,
                    properties=[dict(stride=in_strides)],
                )
            )
            for outname in desc_out_names[1:]
        ],
    )
    if is_int_infer and mode == "test":
        model["box_module"]["postprocess"] = add_desc_pp

    return model


# -------------------------- data --------------------------
def get_sub_class_dataset(
    sub_rec_paths,
    sub_anno_paths,
    class_id_map,
    mode,
    use_mosaic_aug=False,
    use_mixup_aug=None,
):
    if save_all_data_paths:
        with open(all_task_data_files, "at") as fw:
            fw.writelines([p + "\n" for p in sub_rec_paths + sub_anno_paths])
    datasets = [
        dict(
            type="MulticlassDenseboxDataset",
            data_path=rec_path_i,
            anno_path=anno_path_i,
            class_id_map=class_id_map,
            to_rgb=True,
            transforms=get_train_transform(use_mosaic_aug, use_mixup_aug)
            if mode == "train"
            else val_transforms,
            ignore_hard=True,  # if mode == "val" else False,
            use_ignore=use_ignore,
        )
        for rec_path_i, anno_path_i in zip(sub_rec_paths, sub_anno_paths)
    ]
    if mode == "train":
        return datasets
    else:
        sub_dataset = dict(
            type="ConcatDataset",
            datasets=datasets,
        )
        return sub_dataset


use_mosaic_aug_list = [True, True, True, True, True, True]
# ["yolox", "yolov5", False / None / ""]
use_mixup_aug_list = [None, "yolox", None, "yolox", "yolox", "yolox", None]


def get_dataset(mode):
    rec_paths = []
    anno_paths = []
    sample_weights = []
    for ds in task_list:
        rec_paths.append([d["rec_path"] for d in ds[f"{mode}_data_paths"]])
        anno_paths.append([d["anno_path"] for d in ds[f"{mode}_data_paths"]])
        sample_weights.append(
            [d["sample_weight"] for d in ds[f"{mode}_data_paths"]]
        )
    sample_weights = adjust_multitask_sample_weight(
        compose_batchsize, sample_weights
    )

    datasets = [
        get_sub_class_dataset(
            sub_rec_paths,
            sub_anno_paths,
            class_maps[di],
            mode,
            use_mosaic_aug_list[di],
            use_mixup_aug_list[di],
        )
        for di, (
            sub_rec_paths,
            sub_anno_paths,
        ) in enumerate(zip(rec_paths, anno_paths))
    ]
    if mode == "train":
        concat_dataset = []
        for dataset in datasets:
            concat_dataset.extend(dataset)
        concat_dataset = dict(
            type="DistributedComposeRandomDataset",
            datasets=concat_dataset,
            sample_weights=sample_weights,
        )
    else:
        concat_dataset = dict(
            type="ConcatDataset",
            datasets=[ds for ds in datasets if len(ds["datasets"]) > 0],
        )
    return concat_dataset


data_loader = get_2d_detection_dataloader(
    "train", get_dataset("train"), batch_size_dict
)

val_data_loader = get_2d_detection_dataloader(
    "val", get_dataset("val"), batch_size_dict
)

test_transforms = copy.deepcopy(val_transforms)
has_test_image_dir = os.path.exists(test_image_dir)
test_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Auto2dFromImage",
        data_path=test_image_dir,
        to_rgb=True,
        transforms=test_transforms,
    )
    if has_test_image_dir
    else get_dataset("val"),
    batch_size=batch_size_dict["test"],
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)

# -------------------------- inputs --------------------------
inputs = dict(
    img_id=torch.Tensor([[1]]),
    img_name=["xxx.jpg"],
    img_height=torch.Tensor([[input_size[0]]]),
    img_width=torch.Tensor([[input_size[1]]]),
    gt_bboxes=[torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4]])],
    ig_bboxes=[torch.Tensor([[6, 7, 8, 9]])],
    gt_classes=[torch.Tensor([0, 0])],
    gt_labels=[torch.Tensor([0, 1])],
    color_space=["yuv"],
    layout=["hwc"],
    scale_factor=torch.randn((1, 4)),
    img_shape=[[input_size[0], input_size[1], 3]],
    pad_shape=[[input_size[0], input_size[1], 3]],
    keep_ratio=torch.Tensor([False]),
    # unneccesay key-value
    scale=[torch.Tensor([input_size[0]]), torch.Tensor([input_size[1]])],
    scale_idx=torch.Tensor([0]),
    crop_offset=[[1, 2, 3, 4]],
)
if use_ignore:
    inputs["ig_bboxes"] = [torch.Tensor([[6, 7, 8, 9]])]

val_inputs = copy.deepcopy(inputs)
val_inputs.pop("pad_shape")
val_inputs.pop("crop_offset")
val_inputs.pop("scale_factor")
val_inputs.pop("keep_ratio")
val_inputs.pop("scale")
val_inputs.pop("scale_idx")
if has_test_image_dir or aidi_eval:
    val_inputs.pop("gt_bboxes")
    if "ig_bboxes" in val_inputs:
        val_inputs.pop("ig_bboxes")
    if "gt_labels" in val_inputs:
        val_inputs.pop("gt_labels")
    val_inputs.pop("gt_classes")

test_inputs = dict()


def get_inputs(mode):
    if mode == "train":
        return inputs
    elif mode == "val":
        return val_inputs
    else:
        return test_inputs


# -------------------------- metric --------------------------
task_filter_confition = lambda x: x[1] == task_name

focal_v = 221  # pixel focal length in vertical direction
sense_dists = [20.0, 10.0, 10.0, 10.0, 8.0, 8.0]  # max sensing distance
size_difficulties = [2.5, 0.6, 0.6, 0.6, 0.3, 0.1]  # min object height
size_difficulties = [
    size_difficulties[i] * focal_v / sense_dists[i]
    for i in range(len(size_difficulties))
]


def get_update_metric(
    is_train=False, size_diff=size_difficulties, filter_gt=True
):
    def update_metric(metrics, batch, model_outs):
        if is_train:
            for metric, out in zip(metrics, model_outs[0]):
                if is_train:
                    metric.update(out)
        else:
            model_outs = reorganize_sod_batch_results(
                batch,
                model_outs,
                size_diff,
                filter_pred=False,
                filter_gt=filter_gt,
            )
            for m in metrics:
                m.update(copy.deepcopy(model_outs))

    return update_metric


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="loss_heatmap"),
        dict(type="LossShow", name="loss_ciou"),
    ],
    metric_update_func=get_update_metric(is_train=True),
    filter_condition=task_filter_confition,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

anno_file = None
val_map_metric = None

classname_to_idx_map = {}
for class_idx, class_name in enumerate(classnames):
    if mapped_id[class_idx] != class_idx:
        continue
    classname_to_idx_map[class_name] = class_idx
classnames = list(classnames.keys())

vstg = 0.025  # val_score_thresh_gap
ststart = 0.3  # score thresh start
stend = 0.475  # score thresh end
val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="VOCMApMetric",
            num_classes=num_classes,
            class_names=classnames,
            iou_thresh=metric_iou_threshold,
            score_threshs=tuple(
                [
                    ststart + vstg * i
                    for i in range(int((stend - ststart) / vstg))
                ]
            ),
        ),
    ],
    filter_condition=task_filter_confition,
    metric_update_func=get_update_metric(
        is_train=False, size_diff=size_difficulties, filter_gt=True
    ),
    log_prefix="Validation " + task_name,
    step_log_freq=-1,
)

vis_callback = VisualizationCallBack(
    task_name,
    draw_results=True,
    color_map=color_map,
    filter_condition=task_filter_confition,
    difficulties=size_difficulties,
    show_gt=not has_test_image_dir,
    class_names=classnames,
    save_dir=os.path.join(test_image_dir, f"sod_result_{version}")
    if has_test_image_dir
    else None,
)

val_metric_updater_list = [
    # vis_callback,
    val_metric_updater,
]

aidi_eval_page_label = task_name
aidi_eval_dataset_id = []
aidi_eval_callback = []
aidi_eval_type = EVAL_TYPE_2D_DET
old_prediction = compare_version
new_prediction = model_version
for ii, classname in enumerate(classnames):
    dataset_id = TASK2AIDI_EVAL_IDS[task_name][ii]
    aidi_eval_callback.append(
        [
            get_2d_det_aidi_eval_callback(
                dataset_id=dataset_id,
                task_name=task_name,
                classname=classname,
                category_ids=[ii],
            ),
            dict(
                type="StatsMonitor",
                log_freq=log_freq,
            ),
        ]
    )
    aidi_eval_dataset_id.append(dataset_id)

aidi_transforms = copy.deepcopy(bpu_transforms)
aidi_transforms.pop(1)
aidi_transforms.insert(1, {"type": "ImgBufToYUV444"})
aidi_eval_loader = get_2d_aidi_eval_loaders(
    batch_size_dict["val"], task_name, transforms=aidi_transforms
)
