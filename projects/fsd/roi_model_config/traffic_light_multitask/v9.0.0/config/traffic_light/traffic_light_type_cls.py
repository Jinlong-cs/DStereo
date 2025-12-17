import torch
from common import (
    backbone,
    backbone_channels,
    batch_size,
    bn_kwargs,
    cls_pooling_padding,
    datapaths,
    get_classification_model_desc,
    input_preprocess,
    input_size,
    inter_method,
    log_freq,
    mask_in_bpu,
    min_valid_clip_area_ratio,
    model_type,
    norm_length,
    norm_method,
    num_worker,
    pixel_center_aligned,
)
from traffic_light_color_cls import val_decoders

from hat.callbacks.metric_updater import update_metric_using_regex

task_type = "classification"
classnames = ["traffic_light_category"]
object_type = "_".join(classnames)
task_name = f"{object_type}_{task_type}"
traffic_light_category_sub_classnames = [
    "L_Circle",
    "L_Forward",
    "L_Left",
    "L_Right",
    "L_Return",
    "L_Pedestrain",
    "L_Non_Motor",
    "L_Time",
    "L_Other",
    "L_left_and_return",
    "L_Forward_and_Left",
    "L_Forward_and_Right",
    "L_No_Drive_into",
    "L_Allow_Drive_into",
    "L_unknown",
    "text_of_allow_ped",
    "sign_of_allow_ped",
    "text_of_forbid_ped",
    "sign_of_forbid_ped",
]

# model
num_classes = len(traffic_light_category_sub_classnames)


def get_model(mode):
    model = dict(
        type="TrafficLightClassifier",
        input_preprocess=input_preprocess,
        backbone=backbone,
        backbone_extra=torch.nn.Identity(),
        prediction_head=dict(
            type="TinyVarGNetV2ClassificationHead",
            input_channels=backbone_channels[-1],
            disable_quanti_input=True,
            num_classes=num_classes,
            gc_group_base=8,
            bn_kwargs=bn_kwargs,
            alpha=0.5,
            cls_pooling_stride=2,
            cls_pooling_padding=cls_pooling_padding,
            factor=2,
            export_model=True if mode == "test" else False,
            node_name=f"{task_name}_prediction_head",
        ),
        losses=dict(
            type="CEWithLabelSmooth",
            smooth_alpha=0.01,
            ignore_index=-1,
            loss_weight=2.0,
            node_name=f"{task_name}_cls_loss",
        )
        if mode == "train"
        else None,
        desc=get_classification_model_desc(
            task_name=task_name,
            output_name=task_name,
            class_name=task_name,
            desc_id=str(num_classes),
        )
        if mode != "train"
        else None,
        postprocess=dict(
            type="HorizonAdasClsPostProcessor",
            data_name=task_name,
            dim=1,
            node_name=f"{task_name}_channel_argmax_output",
        )
        if mode == "test"
        else None,
        mask_in_bpu=mask_in_bpu,
    )
    return model


assert model_type in val_decoders
val_decoders[model_type][0].append(task_name)
val_decoders[model_type][1]["task_descs"][task_name] = (
    "pred_cls",
    task_type,
)

inputs = dict(
    train=dict(
        gt_classes=torch.zeros((1, 1)).long(),
    ),
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


ds = datapaths.traffic_light_category
classname2idxs = list(range(1, num_classes + 1))
rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
anno_paths = [rec_p.replace(".rec", ".json") for rec_p in rec_paths]
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
                class_id=range(1, num_classes + 1),
                category=range(1, num_classes + 1),
                to_rgb=False,
                ignore_hard=True,
                # note: Normalize needs to be after ToTensor if `to_yuv`==True
                transforms=[
                    dict(
                        type="RoiTransformer",
                        roi_crop_parm=dict(
                            norm_len=norm_length,
                            norm_method=norm_method,
                            output_wh=input_size,
                            input_wh=None,
                            min_crop_scale=0.95,
                            max_crop_scale=1.1,
                            max_coord_jitter_ratio=0.1,
                            img_min_scale=0.01,
                            img_max_scale=100,
                            padd_val=0,
                            random_roi_ratio=0,
                            restrict_roi_in_center=False,
                            flip_ratio=0,
                        ),
                        img_crop_parm=dict(
                            target_wh=input_size,
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
                        convert_roi=True,
                    ),
                    dict(
                        type="MaskOutsideRegion",
                        input_wh=input_size,
                        extend_ratio=0.1,
                        mask_in_bpu=mask_in_bpu,
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
