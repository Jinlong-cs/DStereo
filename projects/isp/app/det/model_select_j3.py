from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels

# model settings
bn_kwargs = dict(eps=1e-5, momentum=0.1)
alpha = 0.5
feat_channels = 64
num_classes = 1
size_divisor = 64
task_head_stack_nums = {
    "person": 4,
    "vehicle": 2,
    "traffic_light": 2,
}

neck = dict(
    type="BiFPN",
    fpn_name="bifpn_sum",
    in_strides=[2, 4, 8, 16, 32],
    out_strides=[4, 8, 16, 32, 64],
    stride2channels=get_vargnetv2_stride2channels(alpha),
    out_channels=feat_channels,
    stack=2,
    start_level=1,
    end_level=-1,
    num_outs=5,
)

loss_cls = dict(
    type="FocalLoss",
    loss_name="cls",
    num_classes=num_classes + 1,
    alpha=0.25,
    gamma=2,
    loss_weight=1.0,
)

loss_centerness = dict(
    type="CrossEntropyLoss", loss_name="centerness", use_sigmoid=True
)

loss_reg = dict(
    type="GIoULoss",
    loss_name="reg",
    loss_weight=1.0,
)


def get_model_j3(
    det_task,
    me_type,
    preprocess_type,
    me_channels=3,
    lutnet=None,
    deploy=False,
    split_transform=False,
):
    backbone = dict(
        type="VargNetV2",
        num_classes=1000,
        alpha=alpha,
        bn_kwargs=bn_kwargs,
        group_base=8,
        include_top=False,
        model_type="VargNetV2",
        factor=2,
        bias=True,
        extend_features=False,
        disable_quanti_input=False,
        flat_output=False,
        input_channels=me_channels,
        input_sequence_length=1,
        head_factor=1,
    )

    head = dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=(4, 8, 16, 32, 64),
        out_strides=(4, 8, 16, 32, 64),
        stride2channels={
            4: feat_channels,
            8: feat_channels,
            16: feat_channels,
            32: feat_channels,
            64: feat_channels,
        },
        feat_channels=feat_channels,
        upscale_bbox_pred=True,
        stacked_convs=task_head_stack_nums[det_task],
        int8_output=True,
        dequant_output=True,
    )

    targets = dict(
        type="DynamicFcosTarget",
        strides=[4, 8, 16, 32, 64],
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
            type="GIoULoss",
            loss_name="reg",
            loss_weight=2.0,
            reduction="none",
        ),
    )

    if preprocess_type == "rawpack":
        in_channels = 4
        transforms = [
            dict(type="RawPack", method=preprocess_type),
            dict(type="Resize"),
        ]
    elif preprocess_type == "demosaic" or preprocess_type == "none":
        in_channels = 3
        transforms = [dict(type="Resize")]
    else:
        raise NotImplementedError

    post_process = dict(
        type="FCOSDecoder",
        num_classes=num_classes,
        strides=[4, 8, 16, 32, 64],
        nms_use_centerness=True,
        nms_sqrt=True,
        transforms=transforms,
        inverse_transform_key=["scale_factor"],
        test_cfg=dict(
            nms_pre=1000,
            nms=dict(name="nms", iou_threshold=0.6, max_per_img=100),
            score_thr=0.3,
            min_bbox_size=0,
        ),
        filter_score_mul_centerness=True,
    )

    if deploy:
        fcos = dict(
            type="FCOS",
            backbone=backbone,
            neck=neck,
            head=head,
        )
    else:
        fcos = dict(
            type="FCOS",
            backbone=backbone,
            neck=neck,
            head=head,
            targets=targets,
            post_process=post_process,
            loss_cls=loss_cls,
            loss_centerness=loss_centerness,
            loss_reg=loss_reg,
        )

    if me_type == "genisp":
        me = dict(
            type="GenISP",
            in_channels=3,
            out_channels=me_channels,
            split_transform=split_transform,
        )
    elif me_type == "nisp":
        me = dict(
            type="NISP",
            in_channels=in_channels,
            out_channels=me_channels,
            block_num=2,
        )
    elif me_type == "none":
        me = None

    if me_type == "baseline":
        model = fcos
    else:
        if lutnet is not None:
            model = dict(
                type="LutMeDetector",
                lutnet=lutnet,
                me=me,
                detector=fcos,
                deploy=deploy,
            )
        else:
            model = dict(
                type="IspMeDetector",
                me=me,
                detector=fcos,
            )
    return model
