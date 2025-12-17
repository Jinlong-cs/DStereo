from collections import OrderedDict

from ...common import (
    datasets,
    log_freq,
    model_input_size,
    tasks_batch_size,
    val_decoders,
)
from ..vdvru_common import get_inputs  # noqa
from ..vdvru_common import (
    enable_auto_assign,
    get_2d_detection_aidi_eval_callback,
    get_2d_detection_aidi_eval_loaders,
    get_2d_detection_dataloader,
    get_2d_detection_dataset,
    get_update_metric,
    use_iou_replace_ctrness,
)
from .common import get_model, object_type  # noqa

task_type = "detection"
task_class_id = 1
task_name = f"{object_type}_{task_type}"
batch_size = tasks_batch_size[task_name]
dataset_name = object_type

model_thresh = None
val_decoders[object_type] = (
    [task_name],
    dict(
        type="RoIDecoder",
        task_descs=OrderedDict([(task_name, ("pred_boxes", task_type))]),
        im_hw=model_input_size,
        clip_by_im_hw=True,
        min_filter_hw=(2, 2),
        nms_threshold=0.5,
        score_threshold=model_thresh["det_thresh"][object_type]
        if model_thresh
        else 0.1,
        node_name=f"{object_type}_decoder",
    ),
)

# -------------------------- data --------------------------
# data
ds = datasets[dataset_name]
train_rec_paths = [d["rec"] for d in ds["train"]]
train_anno_paths = [d["pbrec"] for d in ds["train"]]
train_sample_weights = [d["sample_weight"] for d in ds["train"]]

val_rec_paths = [d["rec"] for d in ds["val"]]
val_anno_paths = [d["pbrec"] for d in ds["val"]]
val_sample_weights = [d["sample_weight"] for d in ds["val"]]


# -------------------------- transform --------------------------

data_loader = get_2d_detection_dataloader(
    "train",
    get_2d_detection_dataset(
        "train",
        train_rec_paths,
        train_anno_paths,
        task_class_id,
        train_sample_weights,
    ),
    batch_size,
)

val_data_loader = get_2d_detection_dataloader(
    "val",
    get_2d_detection_dataset(
        "val",
        val_rec_paths,
        val_anno_paths,
        task_class_id,
    ),
    batch_size,
)

task_filter_confition = lambda x: x[1] == task_name
metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="loss_cls"),
        dict(type="LossShow", name="loss_bbox"),
        dict(
            type="LossShow",
            name="loss_centerness"
            if not use_iou_replace_ctrness
            else "loss_iou",
        ),
    ]
    if not enable_auto_assign
    else [
        dict(type="LossShow", name="loss_pos"),
        dict(type="LossShow", name="loss_neg"),
        dict(type="LossShow", name="loss_center"),
    ],
    metric_update_func=get_update_metric(is_train=True),
    filter_condition=task_filter_confition,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

classnames = [object_type]
classname_to_idx_map = {}
for class_idx, class_name in enumerate(classnames):
    classname_to_idx_map[class_name] = class_idx

vstg = 0.025  # val_score_thresh_gap
ststart = 0.5  # score thresh start
stend = 0.7  # score thresh end
val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="VOCMApMetric",
            num_classes=1,
            class_names=classnames,
            iou_thresh=0.5,
            score_threshs=tuple(
                [
                    ststart + vstg * i
                    for i in range(int((stend - ststart) / vstg))
                ]
            ),
        ),
    ],
    filter_condition=task_filter_confition,
    metric_update_func=get_update_metric(is_train=False, task_name=task_name),
    log_prefix="Validation " + task_name,
    step_log_freq=-1,
)

(
    aidi_eval_callback,
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_2d_detection_aidi_eval_callback(dataset_name, task_name)

aidi_eval_loader = get_2d_detection_aidi_eval_loaders(
    batch_size["val"], dataset_name
)
