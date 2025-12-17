from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from hat.models.backbones.snapdragon import SNDRMixDWNet
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
)


@dataclass
class MixDWNetConfig:
    in_channels: int
    out_channels: int
    head_op: str
    stack_ops: List[str]
    stride: int
    stack_factor: Optional[int] = 1
    fusion_strides: Optional[Union[Tuple[int], List[int]]] = ()
    extra_downsample_num: Optional[int] = 0


channel_list = [32, 32, 32, 64, 64, 96]
net_config = [
    [
        MixDWNetConfig(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            head_op="mixdw_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixDWNetConfig(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            head_op="mixdw_f4",
            stack_ops=["mixdw_f4", "mixdw_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixDWNetConfig(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            head_op="mixdw_f4",
            stack_ops=["mixdw_f4", "mixdw_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixDWNetConfig(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            head_op="mixdw_f2",
            stack_ops=[
                "mixdw_f2",
                "mixdw_f2",
                "mixdw_f2",
                "mixdw_f2",
                "mixdw_f2",
                "mixdw_f2",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixDWNetConfig(
            in_channels=channel_list[4],
            out_channels=channel_list[5],
            head_op="mixdw_f2_dw",
            stack_ops=["mixdw_f2_dw", "mixdw_f2_dw"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]


class TestSNDRMixDWNet(BackboneTemplate):
    def setup(self):
        super(TestSNDRMixDWNet, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        bn_kwargs = dict(eps=1e-5, momentum=0.1)
        self.model = SNDRMixDWNet(
            net_config=net_config,
            num_classes=1000,
            bn_kwargs=bn_kwargs,
            include_top=False,
            bias=True,
        )
        self.build_model()
