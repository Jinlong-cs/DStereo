from hat.models.backbones.mixvargenet import (
    MixVarGENetConfig,
    get_mixvargenet_stride2channels,
)

# model settings
bn_kwargs = dict(eps=1e-5, momentum=0.1)
num_classes = 1
input_sequence_length = 1
input_resize_scale = None
unet_out_strides = [4, 8, 16, 32, 64]

task_head_stack_nums = {
    "person": 4,
    "vehicle": 2,
    "traffic_light": 2,
}

head_out_strides_list = {
    "person": [8, 16, 32, 64],
    "vehicle": [8, 16, 32, 64],
    "traffic_light": [4, 8, 16, 32, 64],
}


def get_mixvargenet_config(
    net_config, output_list, input_channels=3, disable_quanti_input=False
):
    mixvargenet_backbone = dict(
        type="MixVarGENet",
        net_config=net_config,
        output_list=output_list,
        input_channels=input_channels,
        input_sequence_length=input_sequence_length,
        input_resize_scale=input_resize_scale,
        num_classes=1000,
        include_top=False,
        bn_kwargs=bn_kwargs,
        bias=True,
        disable_quanti_input=disable_quanti_input,
    )

    return mixvargenet_backbone


def get_unet_config(stride2channels):
    unet_neck = dict(
        type="Unet",
        in_strides=[2, 4, 8, 16, 32, 64],
        out_strides=unet_out_strides,
        stride2channels=stride2channels,
        factor=2,
        use_bias=False,
        bn_kwargs=bn_kwargs,
        group_base=32,
    )

    return unet_neck


mixvargenet_stride64_example = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=2,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f2_r", "mixvarge_k3k3_f2"],
            stack_factor=2,
            stride=2,
            fusion_strides=[2],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f2_r"],
            stack_factor=2,
            stride=2,
            fusion_strides=[2, 4],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_r_gb16",
                "mixvarge_k3k3_f2_gb16",
                "mixvarge_f2_r_gb16",
                "mixvarge_k3k3_f2_gb16",
                "mixvarge_f2_r_gb16",
            ],
            stack_factor=2,
            stride=2,
            fusion_strides=[4, 8],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_r_gb16"],
            stack_factor=2,
            stride=2,
            fusion_strides=[8, 16],
            extra_downsample_num=1,
        ),  # noqa
    ],  # stride 32
    [
        MixVarGENetConfig(
            in_channels=160,
            out_channels=320,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_r_gb16"],
            stack_factor=2,
            stride=2,
            fusion_strides=[16, 32],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 64
]

large_mixvargenet_stride2channels = get_mixvargenet_stride2channels(
    net_config=mixvargenet_stride64_example
)

neck = get_unet_config(large_mixvargenet_stride2channels)


loss = dict(
    type="AutoAssignLoss",
    loss_name=["loss_pos", "loss_neg", "loss_center"],
    pos_loss=dict(
        type="PosLoss",
        reg_loss=dict(
            type="CIoULoss",
            loss_name="loss_bbox",
            loss_weight=5.0,
            reduction="none",
        ),
        loss_weight=0.25,
    ),
    neg_loss=dict(
        type="NegLoss",
        loss_weight=0.75,
    ),
    center_loss=dict(
        type="CenterLoss",
        loss_weight=0.75,
    ),
)


def get_model_j5(
    det_task,
    me_type,
    preprocess_type,
    me_channels=3,
    lutnet=None,
    deploy=False,
    split_transform=False,
):
    backbone = get_mixvargenet_config(
        net_config=mixvargenet_stride64_example,
        output_list=[0, 1, 2, 3, 4, 5],
        input_channels=me_channels,
    )

    head = dict(
        type="AutoAssignHead",
        num_classes=num_classes,
        in_strides=unet_out_strides,
        out_strides=head_out_strides_list[det_task],
        stride2channels={
            4: 32,
            8: 64,
            16: 96,
            32: 160,
            64: 320,
        },
        feat_channels=32,
        stacked_convs=task_head_stack_nums[det_task],
        use_sigmoid=True,
        share_bn=False,
        share_conv=False,
        upscale_bbox_pred=True,
    )

    target = dict(
        type="AutoAssignTarget",
        strides=head_out_strides_list[det_task],
        cls_out_channels=num_classes,
        center_prior=dict(
            type="CenterPrior",
            force_topk=False,
            num_classes=num_classes,
            strides=head_out_strides_list[det_task],
        ),
        norm_on_bbox=True,
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
        strides=head_out_strides_list[det_task],
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
            type="AutoAssignFCOS",
            backbone=backbone,
            neck=neck,
            head=head,
        )
    else:
        fcos = dict(
            type="AutoAssignFCOS",
            backbone=backbone,
            neck=neck,
            head=head,
            targets=target,
            loss=loss,
            post_process=post_process,
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
