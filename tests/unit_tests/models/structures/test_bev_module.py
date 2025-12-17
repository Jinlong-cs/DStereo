from collections import OrderedDict, namedtuple

import pytest
import torch

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.models.structures.bev_module import (
    BEVSplitModuleWrapper,
    ListInputModelWraper,
    ListInputPreprocess,
)
from hat.registry import build_from_registry

try:
    import hatbc
except ImportError:
    hatbc = None

from tests.unit_tests.models.base import qat_test


class ToyModuleInWraper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x["img"] * 1.2


@pytest.mark.parametrize(
    [
        "mode",
    ],
    [
        pytest.param("tensor"),
        pytest.param("list"),
    ],
)
def test_list_input_model_wraper(mode):
    view_num = 2

    processor = ListInputPreprocess(need_quant=True)
    model = ToyModuleInWraper()

    wraper = ListInputModelWraper(
        seq_len=view_num, preprocess=processor, model=model
    )

    if mode == "tensor":
        x = {"img": torch.rand(view_num, 3, 640, 960)}
    elif mode == "list":
        x = {"img_%d" % i: torch.rand(1, 3, 640, 960) for i in range(view_num)}

    y = wraper(x)

    assert isinstance(y, torch.Tensor)

    qat_test(wraper, x, with_quantized=False)


class ToyStage1ModuleInWrapper(torch.nn.Module):
    def __init__(self, out_name):
        super().__init__()
        self.out_name = out_name

    def forward(self, x, tasks=None):
        Feat = namedtuple("Outputs_stage1", [self.out_name])
        return [Feat([x["img"] * 1.2])]


class ToyStage2ModuleInWrapper(torch.nn.Module):
    def __init__(self, in_name, out_name, view):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.view = view

    def forward(self, x, tasks=None):
        Result = namedtuple("Outputs_stage2", [self.out_name])
        res = sum(x["%s_%d" % (self.in_name, i)] for i in range(self.view))
        return [Result([res])]


def test_split_module_wrapper():
    view_num = 2
    mid_feature_name = "x_frame0"
    final_feature_name = "res_feat0"
    stage1 = ToyStage1ModuleInWrapper(mid_feature_name)
    stage2 = ToyStage2ModuleInWrapper(
        mid_feature_name, final_feature_name, view_num
    )
    wrapper = BEVSplitModuleWrapper(mid_feature_name, stage1, stage2, view_num)
    x = {
        "img_%d" % i: torch.rand(view_num, 3, 640, 960)
        for i in range(view_num)
    }
    res = wrapper(x)
    assert (
        len(res[0]) == 1 and res[0]._fields[0] == final_feature_name
    ), "wrapper fails to merge all inputs"

    qat_test(wrapper, x, with_quantized=False)


@pytest.mark.skipif(hatbc is None, reason="need habtc")
def test_bev_module():

    view_num = 5
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    bevfusion_output_size = (512, 512)
    ipm_output_size = (256, 256)
    seg_class = 16
    grid_quant_scale = 0.015625

    mixvargenet_config = [
        [
            MixVarGENetConfig(
                in_channels=32,
                out_channels=32,
                head_op="mixvarge_f2",
                stack_ops=[],
                stack_factor=1,
                stride=1,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 2
        [
            MixVarGENetConfig(
                in_channels=32,
                out_channels=32,
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 4
        [
            MixVarGENetConfig(
                in_channels=32,
                out_channels=64,
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 8
        [
            MixVarGENetConfig(
                in_channels=64,
                out_channels=96,
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
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 16
        [
            MixVarGENetConfig(
                in_channels=96,
                out_channels=160,
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 32
    ]

    mixvargenet_2x_config = [
        [
            MixVarGENetConfig(
                in_channels=32,
                out_channels=64,
                head_op="mixvarge_f2",
                stack_ops=[],
                stack_factor=1,
                stride=1,
                fusion_strides=[],
                extra_downsample_num=0,
            )
        ],  # stride 2
        [
            MixVarGENetConfig(
                in_channels=64,
                out_channels=64,
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            )
        ],  # stride 4
        [
            MixVarGENetConfig(
                in_channels=64,
                out_channels=128,
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            )
        ],  # stride 8
        [
            MixVarGENetConfig(
                in_channels=128,
                out_channels=256,
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
                fusion_strides=[],
                extra_downsample_num=0,
            )
        ],  # stride 16
        [
            MixVarGENetConfig(
                in_channels=256,
                out_channels=256,
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            )
        ],  # stride 32
    ]

    config = dict(
        type="MultiViewTwoStageBEVModule",
        multi_view_module=dict(
            img=dict(
                type="BEVStageOneModule",
                backbone=dict(
                    type="MixVarGENet",
                    net_config=mixvargenet_config,
                    output_list=[0, 1, 2, 3, 4],
                    input_channels=3,
                    num_classes=1000,
                    bn_kwargs=bn_kwargs,
                    include_top=False,
                    bias=True,
                    disable_quanti_input=False,
                ),
                neck=dict(
                    type="PAFPN",
                    in_channels=[32, 32, 64, 96, 160],
                    out_channels={4: 32, 8: 64, 16: 96, 32: 160, 64: 320},
                    out_strides=[4, 8, 16, 32, 64],
                    start_level=1,
                    add_extra_convs="on_output",  # use P5
                    num_outs=5,
                    relu_before_extra_convs=True,
                ),
                head=dict(
                    type="OutputModule",
                    keep_name=True,
                    head=dict(
                        type="PixelHead",
                        # feature_name="feats",
                        in_strides=[4, 8, 16, 32, 64],
                        out_strides=[4],
                        stride2channels={
                            4: 32,
                            8: 64,
                            16: 96,
                            32: 160,
                            64: 320,
                        },  # noqa
                        forward_frame_idxs=[0],
                        out_nums=seg_class,
                        output_name="front_pred_segs",
                        quanti_last_conv=True,
                        dequant_out=False,
                        bn_kwargs=bn_kwargs,
                        group_base=8,
                        batchnorm_output=True,
                    ),
                ),
            ),
        ),
        bev_fusion_module=dict(
            type="BEVStageTwoModule",
            bevfusion=dict(
                type="ANCBEVFusionModule",
                grid_quant_scale=grid_quant_scale,
                random_rotation_cfg=None,
                drop_view_prob=0.0,
                views=view_num,
                ipm_output_size=ipm_output_size,
                use_homo_offset=True,
                generate_offset_module=None,
                homo_offset_info_keys=None,
            ),
            backbone=dict(
                type="MixVarGENet",
                net_config=mixvargenet_2x_config,
                output_list=[0, 1, 2, 3, 4],
                input_channels=seg_class,
                input_sequence_length=1,
                input_resize_scale=None,
                num_classes=1000,
                include_top=False,
                bn_kwargs=bn_kwargs,
                bias=True,
                disable_quanti_input=True,
            ),
            neck=dict(
                type="Unet",
                in_strides=[2, 4, 8, 16, 32],
                out_strides=[2, 4, 8, 16, 32],
                stride2channels={
                    2: 64,
                    4: 64,
                    8: 128,
                    16: 256,
                    32: 256,
                    64: 256,
                    128: 256,
                    256: 256,
                },
                out_stride2channels={
                    2: 48,
                    4: 48,
                    8: 96,
                    16: 192,
                    32: 192,
                    64: 192,
                    128: 192,
                    256: 192,
                },
                factor=2,
                use_bias=False,
                group_base=8,
            ),
            head=dict(
                type="OutputModule",
                head=dict(
                    type="ANCBEV3DHead",
                    in_strides=[2, 4, 8, 16, 32],
                    out_strides=[4],
                    in_channels=48,
                    forward_frame_idx=0,
                    head_channels=OrderedDict(
                        bev3d_hm=1,
                        bev3d_dim=3,  # h, w, l
                        bev3d_rot=2,  # cos, sin
                        bev3d_ct_offset=2,  # x, y
                        bev3d_loc_z=1,  # vcs z axis
                    ),
                    use_bias=False,
                    bn_kwargs=bn_kwargs,
                    dw_with_relu=True,
                    pw_with_relu=False,
                    factor=2,
                    group_base=8,
                ),
                head_parser=None,
                loss=None,
                postprocess=None,
                target=None,
            ),
            bev_fusion_upsample=dict(
                type="ResizeParser",
                use_plugin_interpolate=True,
                dequant_out=False,
                resize_kwargs=dict(
                    size=bevfusion_output_size, mode="bilinear"
                ),
            ),
        ),
    )

    two_stage_bev_model = build_from_registry(config)

    x = {
        "img": torch.rand(view_num, 3, 640, 960),
        "homo_offset": torch.randn((view_num, 256, 256, 2)),
        "meta_info": dict(
            homo_offset=torch.randn((view_num * 4,) + ipm_output_size + (2,)),
            homography=torch.randn((1, view_num * 4, 3, 3)),
        ),
    }

    y = two_stage_bev_model(x)

    assert isinstance(y, dict)
