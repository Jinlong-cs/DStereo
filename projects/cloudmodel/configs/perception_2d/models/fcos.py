from typing import Dict, Optional


def build_detector_fcos(
    backbone: Dict,
    neck: Dict,
    task_name: str,
    task_loss_weight: float = 1.0,
    decoder_transforms: Optional[Dict] = None,
    mode: str = "train",
    head_cfg: Optional[Dict] = None,
):
    r"""Build the fcos detector according to the definition of FCOS structure
    in hat/models/structures/detectors/fcos.py.

    Args:
        backbone: the backbone of this fcos detector, e.g., EfficientNet.
        neck: the neck of this fcos detector, e.g. BiFPN.
        task_name: the name of a specific task, e.g., vehicle_detection, used
            to distinguish nodes of different tasks in MultitaskGraphModel.
        task_loss_weight: the loss weight of each task.
        decoder_transforms: transforms in decoder.
        mode: the mode of this fcos detector, e.g. "train", "val", "test".
            Different mode corresponds to different detector composition.
        head_cfg: the head parameters.

    Returns:
        A complete fcos detector can be used as a training model, a validation
        model or a test model.
    """
    head_cfg = {} if head_cfg is None else head_cfg

    in_strides = head_cfg.get("in_strides", [4, 8, 16, 32, 64])
    out_strides = head_cfg.get("out_strides", [8, 16, 32, 64, 128])
    stride2channels = head_cfg.get(
        "stride2channels",
        {s: 256 for s in [4, 8, 16, 32, 64]},
    )
    feat_channels = stride2channels[in_strides[0]]
    add_stride = head_cfg.get("add_stride", True)
    stacked_convs = head_cfg.get("stacked_convs", 4)
    num_classes = head_cfg.get("num_classes", 1)
    score_thr = head_cfg.get("score_thr", 0.05)
    cls_name_mapping = head_cfg.get(
        "cls_name_mapping", {0: "_".join(task_name.split("_")[:-1])}
    )

    fcos_detector = dict(
        type="BMFCOS",
        backbone=backbone,
        neck=neck,
        head=dict(
            type="FCOSHead",
            num_classes=num_classes,
            in_strides=in_strides,
            out_strides=out_strides,
            stride2channels=stride2channels,
            upscale_bbox_pred=True,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            share_conv=True,
            use_sigmoid=True,
            share_bn=True,
            dequant_output=True,
            int8_output=False,
            use_plain_conv=True,
            use_gn=True,
            use_scale=True,
            add_stride=add_stride,
            node_name=f"{task_name}_head",
        ),
        target=dict(
            type="DynamicFcosTarget",
            center_sampling=True,
            center_sampling_radius=2.5,
            strides=out_strides,
            cls_out_channels=num_classes,
            background_label=num_classes,
            topK=10,
            loss_cls=dict(
                type="FocalLoss",
                loss_name="cls",
                num_classes=num_classes + 1,
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                reduction="none",
            ),
            loss_reg=dict(
                type="CIoULoss",
                loss_name="reg",
                loss_weight=3.0,
                reduction="none",
            ),
            node_name=f"{task_name}_target",
        )
        if mode == "train"
        else None,
        loss=dict(
            type="FCOSLoss",
            cls_loss=dict(
                type="FocalLoss",
                loss_name="loss_cls",
                num_classes=num_classes + 1,
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0 * task_loss_weight,
            ),
            reg_loss=dict(
                type="CIoULoss",
                loss_name="loss_bbox",
                loss_weight=2.0 * task_loss_weight,
            ),
            centerness_loss=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_name="loss_iou",
                loss_weight=1.0 * task_loss_weight,
            ),
            node_name=f"{task_name}_loss",
        )
        if mode == "train"
        else None,
        postprocess=dict(
            type="MultiInputSequential",
            modules=[
                dict(
                    type="FCOSDecoder",
                    num_classes=num_classes,
                    strides=out_strides,
                    nms_use_centerness=True,
                    nms_sqrt=False,
                    filter_score_mul_centerness=True,
                    transforms=decoder_transforms,
                    inverse_transform_key=["scale_factor"],
                    test_cfg=dict(
                        score_thr=score_thr,
                        nms_pre=1000,
                        nms=dict(name="nms", iou_threshold=0.65),
                        max_per_img=100,
                    ),
                    node_name=f"{task_name}_postprocess",
                ),
                dict(
                    type="FCOSConverter",
                    task_name=task_name,
                    cls_name_mapping=cls_name_mapping,
                    node_name=f"{task_name}_converter",
                ),
            ],
        )
        if mode != "train"
        else None,
    )

    return fcos_detector
