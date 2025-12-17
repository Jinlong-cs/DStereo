import torch
from common import (
    backbone,
    backbone_channels,
    batch_size,
    bn_kwargs,
    datapaths,
    get_classification_model_desc,
    input_size,
    inter_method,
    log_freq,
    march,
    min_valid_clip_area_ratio,
    model_type,
    norm_length,
    norm_method,
    num_worker,
    pixel_center_aligned,
    val_decoders,
)

from hat.callbacks.metric_updater import update_metric_using_regex

task_type = "classification"
classnames = ["scene"]
object_type = "_".join(classnames)
task_name = f"{object_type}_{task_type}"
work_condition_scene_sub_classnames = [
    "Highway",
    "National_road",
    "Urban",
    "Rural",
    "Mountain",
    "Tunnel",
    "Tunnel_Entry",
    "Charge_Station",
    "Bridge",
    "Indoor_Parking_Lot_Entrance",
    "Indoor_Parking_Lot",
    "Parking_Lot_Ramp",
    "Openair_Parking_Lot_Entrance",
    "Openair_Parking_Lot",
]

# model
num_classes = len(work_condition_scene_sub_classnames)


def get_model(mode):
    model = dict(
        type="TrafficLightClassifier",
        backbone=backbone,
        backbone_extra=torch.nn.Identity(),
        prediction_head=dict(
            type="TinyVarGNetV2ClassificationHead",
            input_channels=backbone_channels[-1],
            disable_quanti_input=True,
            num_classes=num_classes,
            gc_group_base=8,
            bn_kwargs=bn_kwargs,
            alpha=2,
            cls_pooling_stride=2,
            cls_pooling_padding=(1, 0),
            avg_pool_size=2,
            factor=2,
            node_name=f"{task_name}_prediction_head",
        ),
        losses=dict(
            type="CEWithLabelSmooth",
            smooth_alpha=0.01,
            ignore_index=-1,
            node_name=f"{task_name}_cls_loss",
        )
        if mode == "train"
        else None,
        postprocess=dict(
            type="HorizonAdasClsPostProcessor",
            data_name=task_name,
            dim=1,
            march=march,
            node_name=f"{task_name}_channel_argmax_output",
        )
        if mode == "test"
        else None,
        desc=get_classification_model_desc(
            task_name=task_name,
            output_name=task_name,
            class_name=task_name,
            desc_id=str(num_classes),
        )
        if mode != "train"
        else None,
    )
    return model


val_decoders[model_type] = (
    [task_name],
    dict(
        type="WkPredDecoder",
        model_name="work_condition_mtl_hat",
        task_descs={task_name: ("pred_cls", task_type)},
        name_dict={task_name: work_condition_scene_sub_classnames},
        node_name=f"{object_type}_decoder",
    ),
)

inputs = dict(
    train=dict(gt_classes=torch.zeros((1, 1)).long()),
    val=dict(),
    test=dict(),
)


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


ds = datapaths.scene_classification
rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
pb_rec_paths = [d["anno_path"] for d in ds["train_data_paths"]]
anno_idx_paths = [rec_p.replace(".rec", ".rec.idx") for rec_p in rec_paths]
sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    drop_last=True,
    num_workers=num_worker,
    persistent_workers=True,
    batch_size=batch_size,
    dataset=dict(
        type="DistributedComposeRandomDataset",
        sample_weights=sample_weights,
        multi_sample_output=True,
        datasets=[
            dict(
                type="DenseboxDataset2PE",
                data_path=rec_path,
                anno_path=pb_rec_path,
                rec_idx_file_path=anno_idx_path,
                task_type=task_type,
                task_name="work_condition_multitask",
                class_id=range(0, num_classes),
                category=range(0, num_classes),
                to_rgb=False,
                read_only=True,
                ignore_hard=True,
                transforms=[
                    dict(
                        type="RoiTransformer",
                        roi_crop_parm=dict(
                            norm_len=norm_length,
                            norm_method=norm_method,
                            output_wh=input_size[::-1],
                            input_wh=None,
                            min_crop_scale=0.95,
                            max_crop_scale=1.1,
                            max_coord_jitter_ratio=0.1,
                            img_min_scale=0.01,
                            img_max_scale=100,
                            padd_val=0,
                            random_roi_ratio=0,
                            restrict_roi_in_center=False,
                            flip_ratio=0.5,
                        ),
                        img_crop_parm=dict(
                            target_wh=input_size[::-1],
                            inter_method=inter_method,
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            pixel_center_aligned=pixel_center_aligned,
                        ),
                        bbox_ts_parm=dict(
                            clip=False,
                            min_valid_area=100,
                            min_valid_clip_area_ratio=min_valid_clip_area_ratio,  # noqa
                            min_edge_size=10,
                            label_type=task_type,
                        ),
                    ),
                    dict(
                        type="DeleteKeys",
                        keys=[
                            "affine_aug_param",
                        ],
                    ),
                    dict(
                        type="ToTensor",
                    ),
                ],
            )
            for rec_path, pb_rec_path, anno_idx_path in zip(
                rec_paths, pb_rec_paths, anno_idx_paths
            )
        ],
    ),
)
