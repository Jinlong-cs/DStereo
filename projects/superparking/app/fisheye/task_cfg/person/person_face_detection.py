import copy

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from ...common import (
    backbone,
    datapaths,
    input_size,
    is_int_infer,
    log_freq,
    neck,
    tasks_batch_size,
    val_only,
)
from ..vdvru_common import (
    bifpn_desc,
    bifpn_quant,
    get_2d_aidi_eval_loaders,
    get_mtfcos3d_module,
    get_rcnn_aidi_eval_info,
    pick_stride_neck,
)
from .common import (
    add_bbox_desc_pp,
    get_roi_module4det_task,
    mtfos3d_params,
    object_type,
    roi_args,
    val_decoders,
)

task_name = "person_face_detection"
task_type = "detetcion"

sub_classnames = ["person_face_detection"]
num_classes = len(sub_classnames)
batch_size = tasks_batch_size[task_name]

task_roi_args = copy.deepcopy(roi_args)
task_roi_args.update(num_fg_classes=num_classes)


def get_model(mode):

    if (not is_int_infer) or (val_only):
        return dict(
            type="TwoStageDetector",
            backbone=backbone,
            neck=dict(
                type="ExtSequential",
                modules=[neck, pick_stride_neck, bifpn_quant],
            ),
            remap_label_key=True,
            rpn_module=get_mtfcos3d_module(
                object_type, mode, **mtfos3d_params
            ),
            roi_module=get_roi_module4det_task(
                mode, sub_classnames, task_name, task_roi_args
            ),
        )
    else:
        return dict(
            type="TraceRcnnModule",
            obj_type=object_type,
            quant_module=bifpn_quant,
            fpn_desc=bifpn_desc,
            roi_desc=add_bbox_desc_pp,
            roi_module=get_roi_module4det_task(
                mode, sub_classnames, task_name, task_roi_args
            ),
        )


assert object_type in val_decoders
val_decoders[object_type][0].append(task_name)
val_decoders[object_type][1]["task_descs"][task_name] = (
    "pred_bboxes",
    task_type,
)


# inputs
inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
        gt_boxes_num=torch.zeros(1),
        ig_regions=torch.zeros((1, 110, 5)),
        ig_regions_num=torch.zeros(1),
        im_hw=torch.zeros((1, 2)),
        parent_gt_boxes=torch.zeros((1, 10, 5)),
        parent_gt_boxes_num=torch.zeros(1),
        parent_ig_regions=torch.zeros((1, 110, 5)),
        parent_ig_regions_num=torch.zeros(1),
    ),
    val=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
    ),
    test=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
    ),
)


def get_inputs(mode):
    return inputs[mode]


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="label_map_loss"),
        dict(type="LossShow", name="offset_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_label_map_loss$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_offset_loss$",
            ),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

# --------------------------- val_metric ---------------------------
def get_update_metric(is_train=False):
    def update_metric(metrics, batch, model_outs):
        if is_train:
            for metric, out in zip(metrics, model_outs[0]):
                if is_train:
                    metric.update(out)
        else:
            children_gt_boxes_ = batch[0]["gt_boxes"]  # (b, 200, 5)
            children_gt_boxes_num = batch[0]["gt_boxes_num"]
            parent_gt_bboxes_ = batch[0]["parent_gt_boxes"]  # (b, 200, 5)
            parent_gt_bboxes_num = batch[0]["parent_gt_boxes_num"]
            children_gt_boxes_num = children_gt_boxes_num.flatten()
            parent_gt_bboxes_num = parent_gt_bboxes_num.flatten()

            # get valid gt_boxes,filter zero boxes
            children_gt_boxes = []
            parent_gt_bboxes = []
            for idx, gn in enumerate(children_gt_boxes_num):
                children_gt_boxes.append(children_gt_boxes_[idx, : int(gn)])
            for idx, gn in enumerate(parent_gt_bboxes_num):
                parent_gt_bboxes.append(parent_gt_bboxes_[idx, : int(gn)])

            gt_labels = [
                torch.zeros_like(box[..., 0]) for box in parent_gt_bboxes
            ]

            out = model_outs[0]
            if len(out) == 0:
                return
            pred_bboxes = [
                data
                for field, data in zip(out._fields, out)
                if "pred_bboxes" in field
            ][0]
            parent_rois = [
                data
                for field, data in zip(out._fields, out)
                if "parent_rois" in field
            ][0]
            rois_num = [
                data
                for field, data in zip(out._fields, out)
                if "rois_num" in field
            ][0]

            rois_num = rois_num.cpu().detach().numpy().tolist()

            if pred_bboxes is None:
                return
            pred_bboxes = pred_bboxes.split(rois_num, dim=0)
            # gt
            targets = {
                "gt_bboxes": children_gt_boxes,  # List[Tensor]
                "parent_gt_bboxes": parent_gt_bboxes,  # List[Tensor]
                "gt_classes": gt_labels,  # List[Tensor]
                "gt_difficult": None,
            }
            preds = {
                "pred_rcnn_bboxes": pred_bboxes,
                "pred_parent_rois": parent_rois,
            }
            # if VAL_VISUAL:
            #     visual_rcnn_boxes_pred(batch, preds, task_name)
            for m in metrics:
                m.update(copy.deepcopy(targets), copy.deepcopy(preds))

    return update_metric


task_classnames = ["person_face"]
classname_to_idx_map = {}
for class_idx, class_name in enumerate(task_classnames):
    classname_to_idx_map[class_name] = class_idx

val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="RcnnDetectionRecallPrec",
            class_dict=classname_to_idx_map,
            score_threshs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        ),
    ],
    filter_condition=lambda x: x[1] == task_name,
    metric_update_func=get_update_metric(is_train=False),
    log_prefix="Validation " + task_name,
    step_log_freq=-1,
)

# ------------------------ dataloader ---------------------------
ds = datapaths.person_face_detection
rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

val_rec_paths = [d["rec_path"] for d in ds["val_data_paths"]]
val_anno_paths = [d["anno_path"] for d in ds["val_data_paths"]]
val_sample_weights = [d["sample_weight"] for d in ds["val_data_paths"]]

classname2idxs = [10]
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1
# dataloader
use_parent = True
parent_id = 1

data_loader = dict(
    type="MultiCachedDataLoader",
    __build_recursive=False,
    last_batch="pad",
    batch_size=batch_size["train"],
    num_workers=1,
    shuffle=True,
    chunk_size=32,
    min_prefetch=1,
    max_prefetch=2,
    min_chunk_num=4,
    max_chunk_num=8,
    batched_transform=True,
    skip_batchify=False,
    prefetcher_using_thread=True,
    dataset=dict(
        type="MultiFusedIterableDataset",
        dataset=[
            dict(
                type="SplitDataset",
                dataset=dict(
                    type="LegacyDenseBoxImageRecordDataset",
                    rec_path=rec_path_i,
                    anno_path=anno_path_i,
                    read_only=True,
                    with_seg_label=False,
                    to_rgb=True,
                    as_nd=False,
                ),
                even_split=False,
            )
            for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
        ],
        prob=sample_weights,
        balance=True,
    ),
    transform=[
        dict(
            type="LegacyDenseBoxImageRecordDatasetDecoder",
            to_rgb=True,
            as_nd=False,
        ),
        dict(
            type="DecodeDenseBoxDatasetToPersonHeadDetFormat",
            selected_class_ids=classname2idxs,
            lt_point_id=10,
            rb_point_id=12,
            parent_lt_point_id=0,
            parent_rb_point_id=2,
            use_parent=use_parent,
            parent_id=parent_id,
        ),
        dict(
            type="ROIDetectionIterableDetRoITransform",
            # roi transform
            target_wh=input_size[::-1],
            resize_wh=None if input_size is None else input_size[::-1],
            img_scale_range=(0.5, 2.0),
            roi_scale_range=(0.7, 1.0 / 0.7),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=inter_method,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            min_valid_area=8,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=2,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            clip_bbox=False,
            # person center and head center must in image
            allow_outside_center=True,  # True is better than False
            pixel_center_aligned=pixel_center_aligned,
            keep_aspect_ratio=True,
        ),
        dict(
            type="PadPersongHeadData",
            target_wh=input_size[::-1],
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
    ],
)

val_data_loader = dict(
    type="MultiCachedDataLoader",
    __build_recursive=False,
    last_batch="keep",
    batch_size=batch_size["val"],
    num_workers=1,
    shuffle=False,
    chunk_size=32,
    min_prefetch=1,
    max_prefetch=2,
    min_chunk_num=4,
    max_chunk_num=8,
    batched_transform=True,
    skip_batchify=False,
    prefetcher_using_thread=True,
    dataset=dict(
        type="MultiFusedIterableDataset",
        dataset=[
            dict(
                type="SplitDataset",
                dataset=dict(
                    type="LegacyDenseBoxImageRecordDataset",
                    rec_path=rec_path_i,
                    anno_path=anno_path_i,
                    read_only=True,
                    with_seg_label=False,
                    to_rgb=True,
                    as_nd=False,
                ),
                even_split=False,
            )
            for rec_path_i, anno_path_i in zip(val_rec_paths, val_anno_paths)
        ],
        prob=val_sample_weights,
        balance=True,
    ),
    transform=[
        dict(
            type="LegacyDenseBoxImageRecordDatasetDecoder",
            to_rgb=True,
            as_nd=False,
        ),
        dict(
            type="DecodeDenseBoxDatasetToPersonHeadDetFormat",
            selected_class_ids=classname2idxs,
            lt_point_id=10,
            rb_point_id=12,
            parent_lt_point_id=0,
            parent_rb_point_id=2,
            use_parent=use_parent,
            parent_id=parent_id,
        ),
        dict(
            type="ROIDetectionIterableDetRoITransform",
            # roi transform
            target_wh=input_size[::-1],
            resize_wh=None if input_size is None else input_size[::-1],
            img_scale_range=(1.0, 1.0),
            roi_scale_range=(1.0, 1.0),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=inter_method,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            min_valid_area=8,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=2,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.0,
            clip_bbox=False,
            # person center and head center must in image
            allow_outside_center=True,  # True is better than False
            pixel_center_aligned=pixel_center_aligned,
            keep_aspect_ratio=True,
        ),
        dict(
            type="PadPersongHeadData",
            target_wh=input_size[::-1],
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
    ],
)

(
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_callback,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_rcnn_aidi_eval_info("person_face_detection", "person")

aidi_eval_loader = get_2d_aidi_eval_loaders(
    batch_size["val"], "person_face_detection", infer_model_type="rcnn"
)
