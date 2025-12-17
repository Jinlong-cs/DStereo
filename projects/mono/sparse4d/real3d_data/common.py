from hat.models.backbones.mixvargenet import (
    MixVarGENetConfig,
    get_mixvargenet_stride2channels,
)

bn_kwargs = dict(eps=1e-5, momentum=0.1)


def get_mixvargenet_config(
    net_config, output_list, input_channels=3, disable_quanti_input=False
):
    mixvargenet_backbone = dict(
        type="MixVarGENet",
        net_config=net_config,
        output_list=output_list,
        input_channels=input_channels,
        input_sequence_length=1,
        input_resize_scale=None,
        num_classes=1000,
        include_top=False,
        bn_kwargs=bn_kwargs,
        bias=True,
        disable_quanti_input=disable_quanti_input,
    )

    return mixvargenet_backbone


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


# LargeMixVarGENet
large_mixvargenet_stride2channels = get_mixvargenet_stride2channels(
    net_config=mixvargenet_stride64_example
)
large_mixvargenet_backbone = get_mixvargenet_config(
    net_config=mixvargenet_stride64_example, output_list=[0, 1, 2, 3, 4, 5]
)
backbone = large_mixvargenet_backbone
