from common import (
    backbone,
    bn_kwargs,
    device_ids,
    get_classification_model_desc,
    get_train_step,
    input_hw,
    log_freq,
    lt_id,
    march,
    pipeline_test,
    rb_id,
    torch,
)
from datasets import datapaths, parse_dataset
from horizon_plugin_pytorch.march import March

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.classification import _get_classification_labels

# -------------------------------------------------------
# task define
# ------------------------------------------------------
task_name = "traffic_sign_cn_category_classification"
num_classes = 258
class_name_4pe = "traffic_sign"
training_step = get_train_step()

# ------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------
inputs = dict(
    train=dict(gt_classes=torch.zeros((1,)).long()),
    val=dict(),
    test=dict(),
)

# ------------------------------------------------------------------------------
# data loader
# ------------------------------------------------------------------------------
classname2idxs = list([i for i in range(1, num_classes + 1)])  # noqa
channel_labels = _get_classification_labels(task_name, str(num_classes))
data_desc = parse_dataset(datapaths, task_name)
train_data_desc = data_desc.train_data_desc
val_data_desc = data_desc.val_data_desc
# use val dataset to boost pipeline-test, noqa
train_data_desc = train_data_desc if not pipeline_test else val_data_desc
train_batch_size = (
    train_data_desc.batch_size_per_ctx if not pipeline_test else 10
)
rec_paths = train_data_desc.rec_paths
anno_paths = train_data_desc.anno_paths
roi_list_paths = train_data_desc.roi_list_paths
sample_weights = (
    train_data_desc.sample_weights
    if not pipeline_test
    else [float(len(device_ids))] * len(rec_paths)
)
num_worker = 16
# True for debug, con't train model with True
show_image_info = False
show_origin_class_id = False
show_origin_image = False
show_origin_bbox = False

base_transformer = dict(
    albumentations_type="Compose",
    p=0.5,
    transforms=[
        # dict(
        #     albumentations_type="GaussianBlur",
        #     blur_limit=(3, 5),
        #     sigma_limit=0,
        #     p=0.2,
        # ),
        # dict(
        #     albumentations_type="MotionBlur",
        #     blur_limit=3,
        #     allow_shifted=True,
        #     p=0.2,
        # ),
        dict(
            albumentations_type="GaussNoise",
            var_limit=(0.0, 20.0),
            mean=0,
            per_channel=True,
            p=0.3,
        ),
        dict(
            albumentations_type="Affine",
            rotate=(-10, 10),
            shear=(-10, 10),
            cval=(128, 128, 128),
            p=0.3,
        ),
    ],
)

color_transformer = dict(
    albumentations_type="Compose",
    transforms=[
        dict(
            albumentations_type="ColorJitter",
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            # hue=0.2,
            hue=0.0,
            p=0.5,
        ),
        dict(
            albumentations_type="RGBShift",
            r_shift_limit=(-10, 10),
            g_shift_limit=(-10, 10),
            b_shift_limit=(-10, 10),
            p=0.5,
        ),
        # dict(
        #     albumentations_type="ChannelShuffle",
        #     p=0.2,
        # ),
    ],
)

flip_transformer = dict(
    albumentations_type="Compose",
    transforms=[
        dict(albumentations_type="HorizontalFlip", p=0.5),
    ],
)

datasets = [
    dict(
        type="TrafficSignDenseBoxImageRecordDataset",
        rec_path=rec_path_i,
        anno_path=anno_path_i,
        to_rgb=True,
        task_type="classification",
        transforms=[
            dict(
                type="DecodeDenseBoxDatasetToDetFormatWithImageInfo",
                selected_class_ids=classname2idxs,
                lt_point_id=lt_id,
                rb_point_id=rb_id,
                show_image_info=show_image_info,
                class_id_key="class_id",
            ),
            dict(
                type="TrafficSignROICropResizeTransform",
                crop_method="RandomCropNearBBoxV2",
                max_jitter_ratio=(0.1, 0.1, 0.1, 0.1),
                random_jitter_bilateral=True,
                target_wh=input_hw[::-1],
                filter_valid_roi=True,
                min_valid_area=16,
                min_edge_size=5,
                random_roi_class_id=-1,
                show_origin_image=show_origin_image,
                show_origin_bbox=show_origin_bbox,
                show_debug_info=False,
            ),
            dict(
                type="TrafficSignImageAugmentation",
                base_transformer=base_transformer,
                color_transformer=color_transformer,
                flip_transformer=flip_transformer,
                flip_trans_all=False,
                flip_label_mapping=None,
                color_trans_all=False,
                color_trans_types=None,
                classname2idxs=dict(zip(channel_labels, classname2idxs)),
            ),
            dict(
                type="ClassIdRemap",
                selected_class_ids=classname2idxs,
            ),
            dict(
                type="ClassificationLabelTs",
                to_tensor=True,
                to_yuv=False,
                task_type="classification",
                # True for debug, con't train model with True
                show_image_info=show_image_info,
                show_origin_class_id=show_origin_class_id,
                show_origin_image=show_origin_image,
                show_origin_bbox=show_origin_bbox,
            ),
        ],
    )
    for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    drop_last=True,
    num_workers=num_worker,
    persistent_workers=True,
    batch_size=train_batch_size,
    dataset=dict(
        type="DistributedComposeRandomDataset",
        sample_weights=sample_weights,
        multi_sample_output=True,
        datasets=datasets,
    ),
)


# -------------------------------------------------------
# models
# ------------------------------------------------------
def get_model(mode):
    model = dict(
        type="TrafficSignClassifierMultitask",
        backbone=backbone,
        backbone_extra=torch.nn.Identity(),
        prediction_head=dict(
            type="TinyVarGNetV2ClassificationHead",
            input_channels=int(
                backbone["channel_list"][-1] * backbone["alpha"]
            ),
            disable_quanti_input=True,
            num_classes=num_classes,
            gc_group_base=8,
            bn_kwargs=bn_kwargs,
            alpha=0.5,
            cls_pooling_stride=2,
            factor=2,
            export_model=True if training_step == "int_infer" else False,
            output_by_argmax=False if march == March.BAYES else True,
            node_name=f"{task_name}_prediction_head",
        ),
        losses=dict(
            type="CEWithLabelSmooth",
            smooth_alpha=0.01,
            ignore_index=-1,
            node_name=f"{task_name}_cls_loss",
        ),
        desc=get_classification_model_desc(
            task_name=task_name,
            output_name=task_name,
            class_name=class_name_4pe,
            desc_id=str(num_classes),
        )
        if mode != "train"
        else None,
    )
    return model


# ------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------

metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="classification_loss"),
        dict(type="Accuracy", name="cls_acc"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_classification_loss$",
            ),
            dict(
                label_pattern=f"^.*{task_name}_gt_classes$",
                pred_pattern=f".*{task_name}_pred*",
            ),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)
