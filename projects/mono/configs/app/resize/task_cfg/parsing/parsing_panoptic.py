from functools import partial

from ...common import datasets, tasks_batch_size
from ..parsing_common import (  # noqa
    get_aidi_eval_callback,
    get_aidi_eval_loaders,
    get_dataloader,
    get_inputs,
    get_loss,
    get_metric_updater,
    get_model,
    get_parsing_dataset,
    get_val_metric_updater,
)

task_name = "parsing_panoptic"
dataset_name = "parsing"
task_group = None
batch_size_dict = tasks_batch_size[task_name]
# model config
num_classes = 16
class_weight = None

# head config
use_auxi_loss = False
loss_auxi_weight = [0.5, 0.25, 0.125, 0.125]
losses_weight = [1.0, 0.5, 0.25, 0.125]
num_auxi_layer = 1
level_indices = [0]
head_out_strides = [2, 8, 16, 32] if use_auxi_loss else [2]
stacked_convs = 1
has_project_layer = True
conv_method = "sep_conv"
share_conv = False
aggregation_method = "sum"

# loss config
auto_class_weight = False
weight_min = 0.0
weight_noobj = 0.75


# -------------------------- deploy desc --------------------------------
labels = [
    dict(class_name=["road"], color_map=[128, 64, 128]),
    dict(class_name=["background"], color_map=[0, 0, 0]),
    dict(class_name=["fence"], color_map=[190, 153, 153]),
    dict(class_name=["pole"], color_map=[153, 153, 153]),
    dict(class_name=["traffic"], color_map=[220, 220, 0]),
    dict(class_name=["person"], color_map=[220, 20, 60]),
    dict(class_name=["vehicle"], color_map=[0, 0, 142]),
    dict(class_name=["two_wheel"], color_map=[119, 11, 32]),
    dict(class_name=["lane_marking"], color_map=[244, 35, 232]),
    dict(class_name=["crosswalk"], color_map=[107, 142, 35]),
    dict(class_name=["traffic_arrow"], color_map=[34, 237, 242]),
    dict(class_name=["sign_line"], color_map=[152, 251, 152]),
    dict(class_name=["guide_line"], color_map=[0, 0, 255]),
    dict(class_name=["traffic_cone"], color_map=[111, 74, 0]),
    dict(class_name=["stop_line"], color_map=[237, 162, 13]),
    dict(class_name=["speed_bump"], color_map=[150, 50, 150]),
]


# -------------------------- model --------------------------------
loss = get_loss(
    task_name=task_name,
    num_classes=num_classes,
    losses_weight=losses_weight,
    use_auxi_loss=use_auxi_loss,
    class_weight=class_weight,
    auto_class_weight=auto_class_weight,
    weight_min=weight_min,
    weight_noobj=weight_noobj,
    num_auxi_layer=num_auxi_layer,
)

get_model = partial(
    get_model,
    task_name=task_name,
    loss=loss,
    num_classes=num_classes,
    head_out_strides=head_out_strides,
    stacked_convs=stacked_convs,
    level_indices=level_indices,
    conv_method=conv_method,
    aggregation_method=aggregation_method,
    share_conv=share_conv,
    use_auxi_loss=use_auxi_loss,
)

# -------------------------- datasets --------------------------------
dataset = datasets["semantic_parsing"]
train_rec_paths = [d["rec"] for d in dataset["train"]]
train_anno_paths = [d["pbrec"] for d in dataset["train"]]
train_sample_weights = [d["sample_weight"] for d in dataset["train"]]
val_rec_paths = [d["rec"] for d in dataset["val"]]
val_anno_paths = [d["pbrec"] for d in dataset["val"]]


# -------------------------- dataloaders --------------------------------
data_loader = get_dataloader(
    "train",
    task_name,
    get_parsing_dataset(
        "train", train_rec_paths, train_anno_paths, train_sample_weights
    ),
)

val_data_loader = get_dataloader(
    "val", task_name, get_parsing_dataset("val", val_rec_paths, val_anno_paths)
)

aidi_eval_loader = get_aidi_eval_loaders(batch_size_dict["val"], dataset_name)


# -------------------------- metric updater --------------------------------
metric_updater = get_metric_updater(task_name)

val_metric_updater = get_val_metric_updater(num_classes, task_name, labels)


# -------------------------- callbacks --------------------------------
(
    aidi_eval_callback,
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_aidi_eval_callback(dataset_name, task_name)
