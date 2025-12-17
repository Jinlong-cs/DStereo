# Copyright (c) Horizon Robotics. All rights reserved.

from itertools import chain
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.models.base_modules.basic_mixvargenet_module import MixVarGEBlock
from hat.models.base_modules.basic_vargnet_module import (
    OnePathResUnit,
    SeparableGroupConvModule2d,
    TwoPathResUnit,
)
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.embeddings import PositionalEmbedding
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class UFPN(nn.Module):
    """Unet FPN neck. (optional) with position embedding.

    Args:
        group_base: Group base of group conv.
        in_strides: strides of each input feature map.
        in_channels: channels of each input feature map, the
            length of in_channels should be equal to in_strides.
        out_channels: channels of each output feature maps, the
            length of out_channels should be equal to in_channels.
        bn_kwargs: Dict for Bn layer. No Bn layer if bn_kwargs=None.
        factor: Factor of group conv.
        out_strides: contains the strides of feature maps the neck output.
        is_with_relu: whether to use the activation function relu.
        pe_stride: feature map stride of position embedding.
        pe_channel: channel num of position embedding.
        ds_blocks: Config list of mixvarge downsample blocks.
        bottom_proj_blocks: Config list of mixvarge projection blocks
            for bottomup fusion.
        up_proj_blocks: Config list of mixvarge projection blocks for upscale.
        fusion_kernel_size: Kernel size of BottomUp fuison block.
    """

    def __init__(
        self,
        group_base: int,
        in_strides: List[int],
        in_channels: List[int],
        out_channels: List[int],
        bn_kwargs: Dict,
        factor: float = 1.0,
        output_strides: List[int] = None,
        is_with_relu: bool = True,
        pe_stride: Optional[int] = None,
        pe_channel: int = 3,
        ds_blocks: List[MixVarGENetConfig] = None,
        bottom_proj_blocks: List[MixVarGENetConfig] = None,
        up_proj_blocks: List[MixVarGENetConfig] = None,
        fusion_kernel_size: int = 3,
    ):
        super().__init__()

        self.in_strides = in_strides
        self.output_strides = output_strides

        assert (
            len(in_strides) == len(in_channels) == len(out_channels)
        ), f"{in_strides} vs. f{in_channels} vs. f{out_channels}"

        assert (
            (ds_blocks is None or len(ds_blocks) == len(in_strides) - 1)
            and (
                bottom_proj_blocks is None
                or len(bottom_proj_blocks) == len(in_strides) - 1
            )
            and (
                up_proj_blocks is None
                or len(up_proj_blocks) == len(in_strides) - 1
            )
        ), "Custom mixvarge blocks not matched with input strides"

        stride2channels = {s: c for s, c in zip(in_strides, out_channels)}

        self.down_sample = nn.ModuleDict()
        self.conv1x1 = nn.ModuleDict()
        self.qat_adds = nn.ModuleDict()

        for i, (s, in_channel, out_channel) in enumerate(
            zip(in_strides, in_channels, out_channels)
        ):
            if i != 0:
                self.conv1x1[f"stride_{s}"] = ConvModule2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(in_channel, **bn_kwargs),
                    act_layer=None,
                )
                self.qat_adds[f"stride_{s}"] = nn.quantized.FloatFunctional()

            if i != len(in_channels) - 1:
                if ds_blocks is not None:
                    self.down_sample[f"stride_{s}"] = MixVarGEBlock(
                        in_ch=in_channel,
                        block_ch=in_channels[i + 1],
                        head_op=ds_blocks[i].head_op,
                        stack_ops=ds_blocks[i].stack_ops,
                        stack_factor=ds_blocks[i].stack_factor,
                        stride=2,
                        bias=True,
                        fusion_channels=(),
                        downsample_num=0,
                        output_downsample=False,
                        bn_kwargs=bn_kwargs,
                    )
                else:
                    self.down_sample[f"stride_{s}"] = TwoPathResUnit(
                        input_channel=in_channel,
                        dw_num_filter=out_channel,
                        group_base=group_base,
                        pw_num_filter=in_channels[i + 1],
                        pw_num_filter2=in_channels[i + 1],
                        bn_kwargs=bn_kwargs,
                        stride=2,
                        is_dim_match=False,
                        use_bias=True,
                        pw_with_act=False,
                        factor=factor,
                    )

        self.fusion_blocks = nn.ModuleDict()
        for i, s in enumerate(in_strides[-2::-1]):
            top_stride = s * 2
            bottom_stride = s
            block = BottomUpFusion(
                up_c=stride2channels[top_stride],
                bottom_c=stride2channels[bottom_stride],
                out_c=stride2channels[bottom_stride],
                dw_group_base=group_base,
                bn_kwargs=bn_kwargs,
                linear_out=True,
                factor=factor,
                use_bias=True,
                is_relu_after_add=is_with_relu,
                bottom_proj_block=bottom_proj_blocks[-i - 1]
                if bottom_proj_blocks is not None
                else None,
                up_proj_block=up_proj_blocks[-i - 1]
                if up_proj_blocks is not None
                else None,
                kernel_size=fusion_kernel_size,
            )
            self.fusion_blocks[f"stride_{s}"] = block

        self.pe_stride = pe_stride
        if pe_stride is not None:
            if pe_stride not in in_strides:
                raise ValueError(f"{pe_stride} not in {in_strides}")
            out_channel = out_channels[in_strides.index(pe_stride)]
            self.pos_embedding = PositionalEmbedding(
                bn_kwargs=bn_kwargs,
                pe_channel=pe_channel,
                is_with_relu=is_with_relu,
            )
            self.cat = nn.quantized.FloatFunctional()
            self.reduction_conv = ConvModule2d(
                in_channels=pe_channel + out_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(out_channel, **bn_kwargs),
                act_layer=None if not is_with_relu else nn.ReLU(inplace=True),
            )

    def forward(
        self,
        x: List[torch.Tensor],
        coordinate_map: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Forward func of ufpn neck.

        Args:
            x (List[torch.Tensor]):
                list of input feature maps.
            coordinate_map (Optional[torch.Tensor], optional):
                position encoding map. float32, NCHW.  Defaults to None.

        Returns:
            List[torch.Tensor]: list of feature maps after ufpn neck.
        """

        down_outputs = [x[0]]

        for i, s in enumerate(self.in_strides[:-1]):
            cs = self.in_strides[i + 1]
            down_outputs.append(
                self.qat_adds[f"stride_{cs}"].add(
                    self.conv1x1[f"stride_{cs}"](x[i + 1]),
                    self.down_sample[f"stride_{s}"](down_outputs[-1]),
                )
            )

        up_outputs = [down_outputs[-1]]

        for i, s in enumerate(self.in_strides[-2::-1]):
            feat_map = self.fusion_blocks[f"stride_{s}"](
                up_outputs[-1], down_outputs[-i - 2]
            )
            if self.pe_stride is not None and self.pe_stride == s:
                position_embedding = self.pos_embedding(coordinate_map)
                feat_map = self.cat.cat([feat_map, position_embedding], dim=1)
                feat_map = self.reduction_conv(feat_map)

            up_outputs.append(feat_map)

        if not self.output_strides:
            up_outputs = up_outputs[::-1]
        else:
            up_outputs = [
                up_outputs[::-1][self.in_strides.index(i)]
                for i in self.output_strides
            ]

        return up_outputs

    def fuse_model(self):
        from horizon_plugin_pytorch import quantization

        for m in chain(
            self.down_sample.values(),
            self.fusion_blocks.values(),
        ):
            m.fuse_model()

        for k in self.conv1x1:
            torch.quantization.fuse_modules(
                self,
                [f"conv1x1.{k}.0", f"conv1x1.{k}.1", f"qat_adds.{k}"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

        if self.pe_stride is not None:
            self.pos_embedding.fuse_model()
            self.reduction_conv.fuse_model()


class BottomUpFusion(nn.Module):
    def __init__(
        self,
        up_c,
        bottom_c,
        out_c,
        dw_group_base,
        bn_kwargs,
        linear_out=True,
        use_bias=True,
        factor=2.0,
        is_relu_after_add=True,
        bottom_proj_block: MixVarGENetConfig = None,
        up_proj_block: MixVarGENetConfig = None,
        kernel_size: int = 3,
    ):
        super().__init__()

        assert bottom_c % dw_group_base == 0

        self.upscale = Upscale(
            in_c=up_c,
            out_c=bottom_c,
            gc_group_base=dw_group_base,
            bn_kwargs=bn_kwargs,
            linear_out=linear_out,
            use_bias=use_bias,
            factor=factor,
            up_proj_block=up_proj_block,
        )
        if bottom_proj_block is not None:
            self.bottom_proj = MixVarGEBlock(
                in_ch=bottom_c,
                block_ch=out_c,
                head_op=bottom_proj_block.head_op,
                stack_ops=bottom_proj_block.stack_ops,
                stack_factor=bottom_proj_block.stack_factor,
                stride=1,
                bias=True,
                fusion_channels=(),
                downsample_num=0,
                output_downsample=False,
                bn_kwargs=bn_kwargs,
            )
        else:
            self.bottom_proj = OnePathResUnit(
                dw_num_filter=bottom_c,
                group_base=dw_group_base,
                pw_num_filter=bottom_c,
                pw_num_filter2=out_c,
                stride=(1, 1),
                is_dim_match=True,
                use_bias=use_bias,
                bn_kwargs=bn_kwargs,
                pw_with_act=not linear_out,
                factor=factor,
            )
        self.fusion = SeparableGroupConvModule2d(
            in_channels=int(bottom_c * factor),
            out_channels=out_c,
            groups=int(bottom_c / dw_group_base),
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(int(kernel_size / 2), int((kernel_size - 1) / 2)),
            bias=use_bias,
            dw_act_layer=nn.ReLU(inplace=True),
            pw_act_layer=None,
            pw_norm_layer=nn.BatchNorm2d(int(bottom_c * factor), **bn_kwargs),
            dw_norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
        )
        self.qat_add = nn.quantized.FloatFunctional()
        self.is_relu_after_add = is_relu_after_add
        self.qat_fuse_ops = ["fusion.1.0", "fusion.1.1", "qat_add"]
        if is_relu_after_add:
            self.relu = nn.ReLU(inplace=True)
            self.qat_fuse_ops.append("relu")

    def forward(self, up, bottom):
        upscale = self.upscale(up)
        out = self.qat_add.add(self.fusion(upscale), self.bottom_proj(bottom))

        if self.is_relu_after_add:
            out = self.relu(out)

        return out

    def fuse_model(self):

        from horizon_plugin_pytorch import quantization

        self.upscale.fuse_model()
        self.bottom_proj.fuse_model()

        getattr(self.fusion, "0").fuse_model()

        torch.quantization.fuse_modules(
            self,
            self.qat_fuse_ops,
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


class Upscale(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        gc_group_base,
        bn_kwargs,
        linear_out=True,
        use_bias=True,
        factor=2.0,
        up_proj_block: MixVarGENetConfig = None,
    ):
        super(Upscale, self).__init__()

        if up_proj_block is not None:
            self.proj_in = MixVarGEBlock(
                in_ch=in_c,
                block_ch=out_c,
                head_op=up_proj_block.head_op,
                stack_ops=up_proj_block.stack_ops,
                stack_factor=up_proj_block.stack_factor,
                stride=1,
                bias=True,
                fusion_channels=(),
                downsample_num=0,
                output_downsample=False,
                bn_kwargs=bn_kwargs,
            )
        else:
            self.proj_in = OnePathResUnit(
                in_filter=in_c,
                dw_num_filter=in_c,
                group_base=gc_group_base,
                pw_num_filter=in_c,
                pw_num_filter2=out_c,
                stride=(1, 1),
                is_dim_match=False,
                use_bias=use_bias,
                bn_kwargs=bn_kwargs,
                pw_with_act=not linear_out,
                factor=factor,
            )

        self.upsample = Interpolate(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )

    def forward(self, x):
        return self.upsample(self.proj_in(x))

    def fuse_model(self):
        self.proj_in.fuse_model()
