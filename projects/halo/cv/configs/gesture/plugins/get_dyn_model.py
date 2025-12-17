from torch import nn

from hat.models.backbones.mixvargenet import (
    MixVarGENetConfig,
    get_mixvargenet_stride2channels,
)


def get_dyn_vargnet(num_classes, alpha=0.5, input_channels=17, deploy=False):
    if deploy:
        loss = None
    else:
        loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    return dict(
        type="DynGestureClassifier",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            input_channels=input_channels,
            include_top=False,
            alpha=alpha,  # 0.5
            bn_kwargs={},  # default: eps 1e-5, m: 0.1
        ),
        head=dict(
            type="DynGestureHead",
            num_classes=num_classes,
            flat_output=not deploy,
        ),
        loss=loss,
    )


def get_dyn_mixvargnet(num_classes, input_channels=17, deploy=False):
    if deploy:
        loss = None
    else:
        loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    mixvarg_config = [
        [
            MixVarGENetConfig(
                in_channels=input_channels,
                out_channels=32,
                head_op="mixvarge_f2",
                stack_ops=[],
                stack_factor=1,
                stride=1,
            ),  # noqa
        ],  # stride 2
        [
            MixVarGENetConfig(
                in_channels=32,
                out_channels=64,
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
            ),  # noqa
        ],  # stride 4
        [
            MixVarGENetConfig(
                in_channels=64,
                out_channels=128,
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
            ),  # noqa
        ],  # stride 8
        [
            MixVarGENetConfig(
                in_channels=128,
                out_channels=128,
                head_op="mixvarge_f2_gb16",
                stack_ops=[
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                ],
                stack_factor=1,
                stride=2,
            ),  # noqa
        ],  # stride 16
        [
            MixVarGENetConfig(
                in_channels=128,
                out_channels=128,
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                stack_factor=1,
                stride=2,
            ),  # noqa
        ],  # stride 32
    ]

    stride2channels = get_mixvargenet_stride2channels(mixvarg_config)

    in_strides = [2, 4, 8, 16, 32]
    in_channels = []
    for _s in in_strides:
        in_channels.append(stride2channels[_s])

    return dict(
        type="DynGestureClassifier",
        backbone=dict(
            type="MixVarGENet",
            net_config=mixvarg_config,
            num_classes=10,
            input_channels=input_channels,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
            output_list=[],
            include_top=False,
            flat_output=False,
            input_resize_scale=1,
        ),
        head=dict(
            type="DynGestureHead",
            num_classes=num_classes,
            flat_output=not deploy,
        ),
        loss=loss,
    )


def get_snpe_mbv2_vargnet(
    num_classes, in_chls=None, out_chls=None, input_channels=17, deploy=False
):
    in_chls = [
        [32],
        [16, 24],
        [24, 32, 32],
        [32] + [64] * 4 + [96] * 2,
        [96] + [128] * 3,
    ]
    out_chls = [
        [16],
        [24, 24],
        [32, 32, 32],
        [64] * 4 + [96] * 3,
        [128] * 3 + [128],
    ]
    if deploy:
        loss = None
    else:
        loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    return dict(
        type="DynGestureClassifier",
        backbone=dict(
            type="SNDRMobileNetV2",
            num_classes=1000,
            input_channels=input_channels,
            include_top=False,
            alpha=1,
            bn_kwargs={},  # default: eps 1e-5, m: 0.1
            in_chls=in_chls,
            out_chls=out_chls,
        ),
        head=dict(
            type="DynGestureHead",
            use_pool=False,
            num_classes=num_classes,
            flat_output=not deploy,
        ),
        loss=loss,
    )
