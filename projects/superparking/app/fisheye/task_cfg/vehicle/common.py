import copy
import json
from collections import OrderedDict
from functools import partial

from hat.core.proj_spec.descs import frcnn_classification_desc, frcnn_kps_desc
from ...common import input_size
from ..vdvru_common import (
    _roi_feat_extractor,
    _roi_head,
    bn_kwargs,
    get_model_mtf3d,
)

classnames = ["vehicle"]
object_type = "_".join(classnames)
num_classes = 1

roi_head = copy.deepcopy(_roi_head)
roi_head["node_name"] = f"{object_type}_roi_share_head"
roi_feat_extractor = copy.deepcopy(_roi_feat_extractor)
roi_feat_extractor["node_name"] = f"{object_type}_roi_feat_extractor"

disentangled_corner3D = True

roi_args = dict(
    exclude_background=True,
    num_fg_classes=None,
    feat_len=96,
    with_tracking_feat=False,
)
data_args = dict(legacy_bbox=True)

task_type = "det"
task_name = classnames[0]
model_thresh = {"det_thresh": {"vehicle": 0.5, "person": 0.5}}

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
    loss_2d_2dBatch_weight=1.0,
    lr_factor=10.0,
    reg_2d_factor=1.0,
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
            dict(
                task="fcos_vehicle_rois", size=[100, 4], class_name=["vehicle"]
            )
        )
    ],
    node_name="roi_vehicle_desc",
)


def get_roi_module4cls_task(
    mode,
    classnames,
    roi_feat_extractor,
    sub_cls_names,
    rcnn_task_name,
    roi_args,
):
    sub_num_classes = len(sub_cls_names)

    postprocess = dict(
        type="SoftmaxRoIClsDecoder",
        cls_name_mapping={
            i: cls_name for i, cls_name in enumerate(sub_cls_names)
        },
        node_name=f"{rcnn_task_name}_roi_decoder",
    )
    split_index = rcnn_task_name.find("_")
    desc_name = "fcos_vehicle_roi" + rcnn_task_name[split_index:]

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
                            feature_size=(
                                1000,
                                1,
                                1,
                                roi_args["feat_len"],
                            ),
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
                pos_iou=0.5,
                neg_iou=0.5,
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
            cls_loss=dict(type="SoftmaxCELoss", dim=1, reduction="mean"),
            node_name=f"{rcnn_task_name}_roi_loss",
        )
        if mode == "train"
        else None,
        postprocess=postprocess if "val" in mode else None,
    )


def get_roi_module4kps_task(
    mode, classnames, roi_feat_extractor, rcnn_task_name, roi_args
):
    roi_head_kps = copy.deepcopy(roi_head)
    roi_head_kps.update(stride=1, node_name=f"{rcnn_task_name}_roi_head")

    postprocess = dict(
        type="KpsDecoder",
        num_kps=2,
        pos_distance=1,
        roi_expand_param=roi_args["expand_param"],
        node_name=f"{rcnn_task_name}_roi_decoder",
    )

    return dict(
        type="RoIModule",
        output_head_out=True,
        roi_key="pred_bboxes",
        roi_feat_extractor=roi_feat_extractor,
        head=dict(
            type="ExtSequential",
            modules=[
                roi_head_kps,
                dict(
                    type="RCNNKPSSplitHead",
                    in_channel=128,
                    points_num=2,
                    node_name=f"{rcnn_task_name}_roi_split_head",
                ),
            ]
            + (
                [
                    dict(
                        type="AddDesc",
                        per_tensor_desc=frcnn_kps_desc(
                            task_name="fcos_vehicle_roi_wheel_kps",
                            class_names=classnames,
                            label_output_name="wheel_kps_detection_label",
                            offset_output_name="wheel_kps_detection_offset",
                            roi_expand_param=roi_args["expand_param"],
                        ),
                        node_name=f"{rcnn_task_name}_roi_desc",
                    ),
                ]
                if "test" in mode
                else []
            ),
        ),
        target=dict(
            type="ProposalTarget",
            matcher=dict(
                type="MaxIoUMatcher",
                pos_iou=0.5,
                neg_iou=0.5,
                allow_low_quality_match=False,
                low_quality_match_iou=0.5,
                legacy_bbox=data_args["legacy_bbox"],
            ),
            label_encoder=dict(
                type="RCNNKPSLabelFromMatch",
                feat_h=8,
                feat_w=8,
                kps_num=2,
                ignore_labels=(0, 3),
                roi_expand_param=roi_args["expand_param"],
            ),
            node_name=f"{rcnn_task_name}_roi_target",
        )
        if mode == "train"
        else None,
        loss=dict(
            type="RCNNKPSLoss",
            kps_num=2,
            cls_loss=dict(
                type="SmoothL1Loss", loss_weight=1, reduction="mean"
            ),
            reg_loss=dict(
                type="SmoothL1Loss", loss_weight=1, reduction="mean"
            ),
            feat_height=8,
            feat_width=8,
            node_name=f"{rcnn_task_name}_roi_loss",
        )
        if mode == "train"
        else None,
        postprocess=postprocess if "val" in mode else None,
    )
