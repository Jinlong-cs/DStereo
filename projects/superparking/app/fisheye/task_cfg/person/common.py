import copy
import json
from collections import OrderedDict
from functools import partial

from hat.core.proj_spec.descs import (
    frcnn_classification_desc,
    frcnn_roi_det_with_subscore_desc,
)
from ...common import input_size, model_thresh
from ..vdvru_common import (
    _roi_feat_extractor,
    _roi_head,
    bn_kwargs,
    get_model_mtf3d,
)

classnames = ["person"]
object_type = "_".join(classnames)
num_classes = 1
roi_head = copy.deepcopy(_roi_head)
roi_head["node_name"] = f"{object_type}_roi_share_head"
roi_feat_extractor = copy.deepcopy(_roi_feat_extractor)
roi_feat_extractor["node_name"] = f"{object_type}_roi_feat_extractor"
disentangled_corner3D = False

roi_args = dict(
    class_agnostic_reg=True,
    exclude_background=True,
    feat_len=96,
    with_tracking_feat=False,
    num_fg_classes=num_classes,
    roi_zoom_scale_wh=(1.0, 1.0),
)
data_args = dict(legacy_bbox=True)


task_type = "det"
task_name = classnames[0]

val_decoders = {}
val_decoders[object_type] = (
    [task_name],
    dict(
        type="RoIDecoder",
        task_descs=OrderedDict([(task_name, ("pred_boxes", task_type))]),
        im_hw=input_size,
        clip_by_im_hw=True,
        min_filter_hw=(2, 2),
        nms_threshold=0.5,
        score_threshold=model_thresh["det_thresh"][object_type]
        if model_thresh
        else 0.1,
        node_name=f"{object_type}_decoder",
    ),
)

mtfos3d_params = dict(
    object_type=object_type,
    loss_2d_2dBatch_weight=10.0,
    lr_factor=1.0,
    reg_2d_factor=5.0,
    disentangled_corner3D=disentangled_corner3D,
)

get_model = partial(get_model_mtf3d, **mtfos3d_params)

mtfos3d_params.pop("object_type")

# bbox desc
add_bbox_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(task="fcos_person_rois", size=[100, 4], class_name=["person"])
        )
    ],
    node_name="roi_person_desc",
)


def get_roi_module4cls_task(mode, sub_cls_names, rcnn_task_name, roi_args):
    sub_num_classes = len(sub_cls_names)

    postprocess = dict(
        type="SoftmaxRoIClsDecoder",
        cls_name_mapping={
            i: cls_name for i, cls_name in enumerate(sub_cls_names)
        },
        node_name=f"{rcnn_task_name}_roi_decoder",
    )

    split_index = rcnn_task_name.find("_")
    desc_name = "fcos_person_roi" + rcnn_task_name[split_index:]
    return dict(
        type="RoIModule",
        output_head_out=True,
        roi_key="pred_bboxes",
        roi_feat_extractor=roi_feat_extractor,
        head=dict(
            type="ExtSequential",
            modules=[
                roi_head,
                dict(
                    type="RCNNVarGNetSplitHead",
                    num_fg_classes=roi_args["num_fg_classes"],
                    bn_kwargs=bn_kwargs,
                    with_background=not roi_args["exclude_background"],
                    in_channel=128,
                    pw_num_filter2=roi_args["feat_len"],
                    with_box_reg=False,
                    with_tracking_feat=roi_args["with_tracking_feat"],
                    node_name=f"{rcnn_task_name}_roi_split_head",
                ),
            ]
            + (
                [
                    dict(
                        type="AddDesc",
                        per_tensor_desc=frcnn_classification_desc(
                            task_name=desc_name,
                            output_name=rcnn_task_name,
                            class_names=classnames,
                            desc_id=str(sub_num_classes),
                            with_tracking_feat=roi_args["with_tracking_feat"],
                        ),
                        node_name=f"{rcnn_task_name}_roi_desc",
                    )
                ]
                if "test" in mode
                else []
            ),
        ),
        target=dict(
            type="ProposalTarget",
            matcher=dict(
                type="MaxIoUMatcher",
                pos_iou=0.7,
                neg_iou=0.7,
                allow_low_quality_match=False,
                low_quality_match_iou=0.3,
                legacy_bbox=data_args["legacy_bbox"],
            ),
            ig_region_matcher=dict(
                type="IgRegionMatcher",
                num_classes=roi_args["num_fg_classes"] + 1,
                ig_region_overlap=0.5,
                legacy_bbox=data_args["legacy_bbox"],
                exclude_background=roi_args["exclude_background"],
            ),
            label_encoder=dict(
                type="MatchLabelSepEncoder",
                class_encoder=dict(
                    type="OneHotClassEncoder",
                    num_classes=roi_args["num_fg_classes"] + 1,
                    class_agnostic_neg=False,
                    exclude_background=roi_args["exclude_background"],
                ),
                cls_use_pos_only=True,
                cls_on_hard=False,
            ),
            node_name=f"{rcnn_task_name}_roi_target",
        )
        if mode == "train"
        else None,
        loss=dict(
            type="RCNNCLSLoss",
            cls_loss=dict(
                type="SoftmaxCELoss",
                dim=1,
                reduction="mean",
                loss_weight=0.25,
            ),
            node_name=f"{rcnn_task_name}_roi_loss",
        )
        if mode == "train"
        else None,
        postprocess=postprocess if "val" in mode else None,
    )


def get_roi_module4det_task(mode, sub_classnames, rcnn_task_name, roi_args):
    num_classes = len(sub_classnames)

    postprocess = dict(
        type="HeatmapBox2dDecoder",
        score_threshold=model_thresh["roi_det_thresh"][object_type]
        if model_thresh
        else 0.1,
        roi_wh_zoom_scale=roi_args["roi_zoom_scale_wh"],
        node_name=f"{rcnn_task_name}_roi_decoder",
    )

    split_index = rcnn_task_name.find("_")
    desc_name = "fcos_person_roi" + rcnn_task_name[split_index:]
    return dict(
        type="RoIModule",
        output_head_out=True,
        roi_key="pred_bboxes",
        target_keys=["gt_boxes", "parent_gt_boxes"],
        target_opt_keys=[
            "parent_gt_boxes_num",
            "parent_ig_regions",
            "parent_ig_regions_num",
        ],
        roi_feat_extractor=roi_feat_extractor,
        head=dict(
            type="ExtSequential",
            modules=[
                dict(
                    type="RCNNVarGNetHead",
                    with_box_reg=True,
                    num_fg_classes=roi_args["num_fg_classes"],
                    bn_kwargs=bn_kwargs,
                    class_agnostic_reg=roi_args["class_agnostic_reg"],
                    with_background=not roi_args["exclude_background"],
                    roi_out_channel=32,
                    dw_num_filter=128,
                    pw_num_filter=128,
                    pw_num_filter2=96,
                    group_base=8,
                    factor=1,
                    stride_step1=1,
                    ksize_step2=1,
                    upscale=False,
                    node_name=f"{rcnn_task_name}_roi_head",
                )
            ]
            + (
                [
                    dict(
                        type="AddDesc",
                        per_tensor_desc=frcnn_roi_det_with_subscore_desc(
                            task_name=desc_name,
                            class_names=["person"],
                            sub_class_name="head",
                            label_output_name=f"{rcnn_task_name}_label",
                            offset_output_name=f"{rcnn_task_name}_offset",
                            subbox_score_thresh=model_thresh["roi_det_thresh"][
                                object_type
                            ]
                            if model_thresh
                            else 0.1,
                        ),
                        node_name=f"{rcnn_task_name}_roi_desc",
                    )
                ]
                if "test" in mode
                else []
            ),
        ),
        target=dict(
            type="ProposalTargetBinDet",
            matcher=dict(
                type="MaxIoUMatcher",
                pos_iou=0.5,
                neg_iou=0.5,
                allow_low_quality_match=False,
                low_quality_match_iou=0.3,
                legacy_bbox=data_args["legacy_bbox"],
            ),
            ig_region_matcher=dict(
                type="IgRegionMatcher",
                num_classes=roi_args["num_fg_classes"] + 1,
                legacy_bbox=data_args["legacy_bbox"],
                ig_region_overlap=0.5,
                exclude_background=roi_args["exclude_background"],
            ),
            label_encoder=dict(
                type="RCNNBinDetLabelFromMatch",
                roi_h_zoom_scale=roi_args["roi_zoom_scale_wh"][1],
                roi_w_zoom_scale=roi_args["roi_zoom_scale_wh"][0],
                feature_w=8,
                feature_h=8,
                num_classes=num_classes,
                cls_on_hard=False,
                allow_low_quality_heatmap=True,
            ),
            node_name=f"{rcnn_task_name}_roi_target",
        )
        if mode == "train"
        else None,
        loss=dict(
            type="RCNNBinDetLoss",
            cls_loss=dict(
                type="SmoothL1Loss",
                reduction="mean",
                hard_neg_mining_cfg=dict(
                    keep_pos=True,
                    neg_ratio=0.5,
                    hard_ratio=0.5,
                ),
            ),
            reg_loss=dict(type="SmoothL1Loss", reduction="mean"),
            node_name=f"{rcnn_task_name}_roi_loss",
        )
        if mode == "train"
        else None,
        postprocess=postprocess if "val" in mode else None,
    )
