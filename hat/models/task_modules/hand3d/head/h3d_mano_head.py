# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Tuple

from torch import Tensor, nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class H3DLinear(nn.Module):
    """Linear for hand pose estimation.

    Args:
        input_channels: Channels of input.
        output_channels: Channels of output.
        bn_kwargs: Dict for BN layer.
        disable_act: Whether to keep batch norm and act layer.
        is_out_layer: Whether it is the last layer of the network.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bn_kwargs: Dict,
        disable_act: bool = False,
        is_out_layer: bool = False,
    ):
        super(H3DLinear, self).__init__()

        self.disable_act = disable_act
        self.is_out_layer = is_out_layer
        self.linear = ConvModule2d(
            input_channels,
            output_channels,
            1,
            padding=0,
            stride=1,
            bias=True,
            norm_layer=None
            if disable_act
            else nn.BatchNorm2d(output_channels, **bn_kwargs),
            act_layer=None if disable_act else nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return x

    def fuse_model(self) -> None:
        self.linear.fuse_model()

    def set_qconfig(self, is_out=False) -> None:
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if is_out:
            self.linear.qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class H3DHeatmapHead(nn.Module):
    """
    A resblock for heatmap head.

    Args:
        bn_kwargs: Dict for Bn layer.
        num_joints: Output channels.
        stride: Stride for first conv.
        bias: Whether to use bias in module.
        has_conv: Whether to use conv before heatmap head.
    """

    def __init__(
        self,
        bn_kwargs: Dict,
        num_joints: int = 21,
        stride: int = 1,
        bias: bool = True,
        has_conv: bool = False,
    ):
        super(H3DHeatmapHead, self).__init__()

        self.has_conv = has_conv
        if self.has_conv:
            self.heatmap_conv = BasicResBlock(
                in_channels=num_joints,
                out_channels=num_joints,
                bn_kwargs=bn_kwargs,
                stride=stride,
                bias=bias,
            )

        self.heatmap_head = ConvModule2d(
            num_joints,
            num_joints,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(num_joints, **bn_kwargs),
            act_layer=None,
        )

        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        if self.has_conv:
            x = self.heatmap_conv(x)
        return self.dequant(self.heatmap_head(x))

    def fuse_model(self) -> None:
        if hasattr(self, "heatmap_conv") and hasattr(
            self.heatmap_conv, "fuse_model"
        ):
            self.heatmap_conv.fuse_model()

        if self.heatmap_head is not None and hasattr(
            self.heatmap_head, "fuse_model"
        ):
            self.heatmap_head.fuse_model()

    def set_qconfig(self) -> None:
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self, "heatmap_conv") and hasattr(
            self.heatmap_conv, "set_qconfig"
        ):
            self.heatmap_conv.set_qconfig()


@OBJECT_REGISTRY.register
class H3DManoMultiFcHead(nn.Module):
    """MultiFc layer for hand pose estimation network head.

    Args:
        encoding_channel: Number of channels in encoding.
        bn_kwargs: Dict for BN layer.
        pose_neurons: Neurons of pose head. Defaults to (256, 128, 90).
        shape_neurons: Neurons of shape head. Defaults to (256, 10).
        camera_base_neurons:
            Neurons of camera extrinsic matrix common head.
            Defaults to (256,).
        rot_neurons: Neurons of rotation head. Defaults to (256, 6).
        trans_neurons: Neurons of translation head. Defaults to (256, 3).
        scale_neurons: Neurons of scale head. Defaults to (256, 128, 1).
        text_fc_neurons: Neurons of texture fc head.
            Defaults to (256, 256, 128).
        text_reg_neurons: Neurons of texture reg head.
            Defaults to (1024, 778, 778).
        head_heatmap: Head module of hand keypoints heatmap.
            Defaults to None.
        head_heatmap_latent: Whether to output heatmap latents.
            Defaults to True.
        enable_render_head: Whether to use texture render head.
            Defaults to False.
        enable_handscale_head: Whether to use hand scale head.
            Defaults to False.
        num_joints: Number of hand landmarks. Defaults to 21.
        deploy: Whether in deploy mode. Defaults to False.
    """

    def __init__(
        self,
        encoding_channel: int,
        bn_kwargs: Dict,
        pose_neurons: List = (256, 128, 90),
        shape_neurons: List = (256, 10),
        camera_base_neurons: List = (256,),
        rot_neurons: List = (256, 6),
        trans_neurons: List = (256, 3),
        scale_neurons: List = (256, 128, 1),
        text_fc_neurons: List = (256, 256, 128),
        text_reg_neurons: List = (1024, 778, 778),
        head_heatmap: nn.Module = None,
        head_heatmap_latent: bool = True,
        enable_shape_head: bool = True,
        enable_render_head: bool = False,
        enable_handscale_head: bool = False,
        num_joints: int = 21,
        deploy: bool = False,
    ):

        super(H3DManoMultiFcHead, self).__init__()
        self.deploy = deploy

        self.head_heatmap = head_heatmap
        self.enable_shape_head = enable_shape_head
        self.enable_render_head = enable_render_head
        self.enable_handscale_head = enable_handscale_head
        self.latent_head = None
        if head_heatmap_latent:
            self.latent_head = ConvModule2d(
                num_joints,
                num_joints,
                1,
                padding=0,
                stride=1,
                bias=True,
                norm_layer=None,
                act_layer=None,
            )

        self.dequant = DeQuantStub()
        self.pose_neurons = pose_neurons
        self.shape_neurons = shape_neurons
        self.rot_neurons = rot_neurons
        self.trans_neurons = trans_neurons
        self.scale_neurons = scale_neurons
        self.text_fc_neurons = text_fc_neurons
        self.text_reg_neurons = text_reg_neurons

        # mano
        # Pose layers
        pose_neurons = [encoding_channel] + list(pose_neurons)
        self.pose_reg = self._make_multifc(
            pose_neurons, bn_kwargs, set_last_layer_output=True
        )

        # camera
        camera_base_neurons = [encoding_channel] + list(camera_base_neurons)
        self.camera_base_fc = self._make_multifc(
            camera_base_neurons, bn_kwargs, set_last_layer_output=False
        )

        # R
        rot_neurons = [camera_base_neurons[-1]] + list(rot_neurons)
        self.rot_reg = self._make_multifc(
            rot_neurons, bn_kwargs, set_last_layer_output=True
        )

        # T
        trans_neurons = [camera_base_neurons[-1]] + list(trans_neurons)
        self.trans_reg = self._make_multifc(
            trans_neurons, bn_kwargs, set_last_layer_output=True
        )

        if self.enable_shape_head:
            # Shape layers
            shape_neurons = [encoding_channel] + list(shape_neurons)
            self.shape_reg = self._make_multifc(
                shape_neurons, bn_kwargs, set_last_layer_output=True
            )
        else:
            self.shape_reg = None

        if self.enable_handscale_head:
            scale_neurons = [encoding_channel] + list(scale_neurons)
            self.scale_reg = self._make_multifc(
                scale_neurons, bn_kwargs, set_last_layer_output=True
            )
        else:
            self.scale_reg = None

        if self.enable_render_head:
            multi_fc_layers: List[nn.Module] = [
                ConvModule2d(
                    inp_neurons,
                    out_neurons,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=True,
                    norm_layer=None,
                    act_layer=None,
                )
                for inp_neurons, out_neurons in zip(
                    text_fc_neurons[:-1],
                    text_fc_neurons[1:],
                )
            ]
            self.text_fc = nn.Sequential(*multi_fc_layers)
            self.text_reg = self._make_multifc(
                text_reg_neurons, bn_kwargs, set_last_layer_output=True
            )
        else:
            self.text_fc = None
            self.text_reg = None

    def _make_multifc(
        self, neurons, bn_kwargs: Dict, set_last_layer_output=True
    ):
        multi_fc_layers = [
            H3DLinear(
                input_channels=inp_neurons,
                output_channels=out_neurons,
                bn_kwargs=bn_kwargs,
                disable_act=True
                if set_last_layer_output and id == len(neurons) - 2
                else False,
                is_out_layer=True
                if set_last_layer_output and id == len(neurons) - 2
                else False,
            )
            for id, (inp_neurons, out_neurons) in enumerate(
                zip(neurons[:-1], neurons[1:])
            )
        ]
        return nn.Sequential(*multi_fc_layers)

    def forward(self, encodings: Tuple) -> Dict:
        # encodings:
        # shape_encoding, confusion_encoding, heatmap_encoding

        pose_mano = self.dequant(self.pose_reg(encodings[1]))
        camera_base_feature = self.camera_base_fc(encodings[1])
        trans = self.dequant(self.trans_reg(camera_base_feature))
        rot = self.dequant(self.rot_reg(camera_base_feature))

        pose_mano = pose_mano.reshape(-1, self.pose_neurons[-1])
        trans = trans.reshape(-1, self.trans_neurons[-1])
        rot = rot.reshape(-1, self.rot_neurons[-1])

        if self.enable_shape_head:
            shape_mano = self.dequant(self.shape_reg(encodings[0]))
            shape_mano = shape_mano.reshape(-1, self.shape_neurons[-1])

        if self.enable_handscale_head:
            scale_intr = self.dequant(self.scale_reg(encodings[1]))
            scale_intr = scale_intr.reshape(-1, self.scale_neurons[-1])

        if self.enable_render_head:
            text = self.text_fc(encodings[3])
            text = text.reshape(
                [-1, self.text_reg_neurons[0], self.text_fc_neurons[-1]]
            )
            text = self.dequant(self.text_reg(text.permute(0, 2, 1)))

        if self.deploy:
            if self.enable_shape_head:
                return pose_mano, trans, rot, shape_mano
            else:
                return pose_mano, trans, rot

        output = {
            "pose_mano": pose_mano,
            "trans": trans,
            "rot": rot,
        }

        if self.enable_shape_head:
            output["shape_mano"] = shape_mano

        if self.enable_handscale_head:
            output["scale_intr"] = scale_intr

        if self.enable_render_head:
            output["text"] = text.permute(0, 2, 1)

        if self.latent_head is not None:
            heatmap_latents = self.dequant(self.latent_head(encodings[2]))
            output["heatmap_latents"] = heatmap_latents

        if self.head_heatmap is not None:
            heatmap_hand = self.head_heatmap(encodings[2])
            output["heatmap_hand"] = heatmap_hand

        return output

    def fuse_model(self) -> None:
        if self.head_heatmap is not None and hasattr(
            self.head_heatmap, "fuse_model"
        ):
            self.head_heatmap.fuse_model()

        if self.latent_head is not None and hasattr(
            self.latent_head, "fuse_model"
        ):
            self.latent_head.fuse_model()

        for mod in [
            self.camera_base_fc,
            self.text_fc,
            self.pose_reg,
            self.shape_reg,
            self.scale_reg,
            self.trans_reg,
            self.rot_reg,
            self.text_reg,
        ]:
            if mod is None:
                continue
            for m in mod:
                if m is not None and hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self) -> None:
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        # disable output quantization for last quanti layer.
        for mod in [
            self.camera_base_fc,
            self.text_fc,
            self.pose_reg,
            self.shape_reg,
            self.scale_reg,
            self.trans_reg,
            self.rot_reg,
            self.text_reg,
        ]:
            if mod is None:
                continue
            for m in mod:
                if m is not None and hasattr(m, "set_qconfig"):
                    m.set_qconfig(is_out=m.is_out_layer)

        if self.head_heatmap is not None and hasattr(
            self.head_heatmap, "set_qconfig"
        ):
            self.head_heatmap.set_qconfig()

        if self.latent_head is not None and hasattr(
            self.latent_head, "set_qconfig"
        ):
            self.latent_head.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
