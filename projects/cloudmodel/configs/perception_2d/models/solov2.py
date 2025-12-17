from typing import Dict, Optional


def build_segmentor_solov2(
    backbone: Dict,
    neck: Dict,
    task_name: str,
    mode: str = "train",
    head_cfg: Optional[Dict] = None,
):
    """Build the solov2 segmentor.

    Args:
        backbone: the backbone of this fcos detector, e.g., EfficientNet.
        neck: the neck of this solov2 segmentor, e.g. BiFPN.
        task_name: the name of a specific task, e.g., lane_instanceseg, used
            to distinguish nodes of different tasks in MultitaskGraphModel.
        mode: the mode of this fcos detector, e.g. "train", "val", "test".
            Different mode corresponds to different segmentor composition.

    Returns:
        A complete solov2 segmentor can be used as a training model,
        a validation model or a test model.
    """

    head_cfg = {} if head_cfg is None else head_cfg

    desc_attributes = head_cfg.get("desc_attributes")
    cls_name_mapping = head_cfg.get("cls_name_mapping")
    attr_name2num = head_cfg.get("attr_name2num")
    num_classes = head_cfg.get("num_classes")

    pos_scale = head_cfg.get("pos_scale", 0.15)
    strides = head_cfg.get("strides", [4, 8, 16])
    scale_ranges = head_cfg.get(
        "scale_ranges", ((1, 96), (64, 192), (128, 2048))
    )
    num_grids = head_cfg.get("num_grids", [48, 40, 32])

    stacked_convs = head_cfg.get("stacked_convs", 3)
    in_channels = head_cfg.get("in_channels", 256)
    feat_channels_attr = head_cfg.get("feat_channels_attr", 128)
    feat_channels = head_cfg.get("feat_channels", 256)
    use_dw_conv = head_cfg.get("use_dw_conv", False)

    dynamic_conv_size = head_cfg.get("dynamic_conv_size", 1)
    mask_out_channels = head_cfg.get("mask_out_channels", 128)
    mask_feat_channels = head_cfg.get("mask_feat_channels", 128)
    mask_stacked_convs = head_cfg.get("mask_stacked_convs", 1)
    upsample_mask_feat = head_cfg.get("upsample_mask_feat", False)
    upsample_mask_logit = head_cfg.get("upsample_mask_logit", True)
    mask_stride = head_cfg.get("mask_stride", 4)
    max_pos_num = head_cfg.get("max_pos_num", 100)
    use_ignore_region = head_cfg.get("use_ignore_region", False)

    kernel_out_channels = mask_out_channels * (dynamic_conv_size ** 2)
    num_levels = len(strides)

    solov2_segmentor = dict(
        type="BMSegmentor",
        backbone=backbone,
        neck=neck,
        head=dict(
            node_name=f"{task_name}_head",
            type="SOLOV2Head",
            __build_recursive=False,
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            feat_channels_attr=feat_channels_attr,
            num_grids=num_grids,
            mask_stride=mask_stride,
            mask_feature_head_updater=dict(
                feat_channels=mask_feat_channels,
                out_channels=mask_out_channels,
                stacked_convs=mask_stacked_convs,
                upsample_mask_feat=upsample_mask_feat,
            ),
            upsample_mask_logit=upsample_mask_logit,
            stacked_convs=stacked_convs,
            num_levels=num_levels,
            kernel_out_channels=kernel_out_channels,
            attr_name2num=attr_name2num,
            use_dw_conv=use_dw_conv,
        ),
        postprocess=dict(
            node_name=f"{task_name}_decoder",
            type="SOLOV2Decoder",
            __build_recursive=False,
            dynamic_conv_size=dynamic_conv_size,
            kernel_out_channels=kernel_out_channels,
            mask_stride=mask_stride,
            num_classes=num_classes,
            num_grids=num_grids,
            strides=strides,
            upsample_mask_logit=upsample_mask_logit,
            test_cfg_updater=dict(
                nms_pre=500,
                score_thr=0.2,  # 0.1, #
                mask_thr=0.5,
                filter_thr=0.2,  # 0.15,  # 0.05, #
                kernel="gaussian",  # gaussian/linear
                sigma=2.0,
                max_per_img=50,
            ),
            combine=True,
            object_name="lane",
            attr_names=list(desc_attributes.keys()),
            cls_name_mapping=cls_name_mapping,
        )
        if mode != "train"
        else None,
        target=dict(
            node_name=f"{task_name}_target",
            type="SOLOV2Target",
            __build_recursive=False,
            num_classes=num_classes,
            scale_ranges=scale_ranges,
            pos_scale=pos_scale,
            num_grids=num_grids,
            mask_stride=mask_stride,
            dynamic_conv_size=dynamic_conv_size,
            upsample_mask_logit=upsample_mask_logit,
            attr_name2num=attr_name2num,
            ignore_index=255,
            max_pos_num=max_pos_num,
            use_ignore_region=use_ignore_region,
        )
        if mode == "train"
        else None,
        loss=dict(
            node_name=f"{task_name}_loss",
            type="SOLOV2Loss",
            mask_loss=dict(
                type="MaskLoss",
                use_sigmoid=True,
                activate=True,
                loss_weight=3.0,
                loss_types=("dice", "focal"),
            ),
            cls_loss=dict(
                type="FocalLossV2",
                alpha=0.25,
                gamma=2.0,
                from_logits=False,
            ),
            attr_loss=[
                dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                    loss_weight=1.0,
                    ignore_index=255,
                )
            ]
            * len(attr_name2num),
            mask_loss_name="loss_mask",
            cls_loss_name="loss_cls",
            attr_loss_names=[f"loss_ce_{k}" for k in attr_name2num.keys()],
        )
        if mode == "train"
        else None,
    )

    return solov2_segmentor
