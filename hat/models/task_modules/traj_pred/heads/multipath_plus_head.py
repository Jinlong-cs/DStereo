# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Tuple

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.traj_pred.heads.multipath_head import (  # noqa: E501
    MultiPathHead,
)
from hat.models.utils import _take_features
from hat.registry import OBJECT_REGISTRY

__all__ = ["MultiPathPlusHead"]


class MCGBlock(nn.Module):
    """Multi Context Gating Block.

    a basic block for building MCG module.

    Args:
        block_channel (int): input channel of mlp.
        pool_kernel (Tuple[int]): kernel size of max pool, the size
            need to be the same as input's spatial shape.
    """

    def __init__(
        self,
        block_channel: int,
        pool_kernel: Tuple[int],
    ):
        super().__init__()
        self.set_mlp = nn.Sequential(
            *[
                ConvModule2d(
                    block_channel,
                    block_channel * 2,
                    kernel_size=1,
                    bias=False,
                    act_layer=nn.ReLU(inplace=True),
                ),
                ConvModule2d(
                    block_channel * 2, block_channel, kernel_size=1, bias=False
                ),
            ]
        )
        self.context_mlp = nn.Sequential(
            *[
                ConvModule2d(
                    block_channel,
                    block_channel * 2,
                    kernel_size=1,
                    bias=False,
                    act_layer=nn.ReLU(inplace=True),
                ),
                ConvModule2d(
                    block_channel * 2, block_channel, kernel_size=1, bias=False
                ),
            ]
        )

        self.pool = nn.MaxPool2d(pool_kernel, stride=1)
        self.mul_op = nn.quantized.FloatFunctional()

    def forward(self, x, ctx):
        """Forward.

        Args:
            x (torch.Tensor): input feature, [num_obj, c, h, w].
            ctx (torch.Tensor): contect feature, [num_obj, c, 1, 1].

        Returns:
            fuse_x (torch.Tensor): output feature, [num_obj, c, h, w].
            ctx_out (torch.Tensor): output context feature, [num_obj, c, 1, 1].
        """
        set_x = self.set_mlp(x)
        ctx_x = self.context_mlp(ctx)
        fuse_x = self.mul_op.mul(set_x, ctx_x)
        ctx_out = self.pool(fuse_x)
        return fuse_x, ctx_out

    def fuse_model(self):
        for module in self.set_mlp:
            module.fuse_model()
        for module in self.context_mlp:
            module.fuse_model()


class MCG(nn.Module):
    """Multi Context Gating Module.

    An interaction module which is proposed in Multipath++,
    the module is stacked by multiple basic MCGblock.

    Args:
        block_channel (int): input channel of mlp.
        pool_kernel (Tuple[int]): kernel size of max pool, the size
            need to be the same as input's spatial shape.
        stack_num (int): the number of stacked MCGBlock.
    """

    def __init__(
        self,
        block_channel: int,
        pool_kernel: Tuple[int],
        stack_num: int,
    ):
        super().__init__()
        self.block_channel = block_channel
        self.pool_kernel = pool_kernel
        self.stack_num = stack_num

        blocks = []
        for _ in range(stack_num):
            blocks.append(MCGBlock(block_channel, pool_kernel))
        self.blocks = nn.Sequential(*blocks)
        self.add_op = nn.quantized.FloatFunctional()

    def forward(self, x, ctx):
        """Forward.

        Args:
            x (torch.Tensor): input feature, [num_obj, c, h, w].
            ctx (torch.Tensor): contect feature, [num_obj, c, 1, 1].

        Returns:
            out_x (torch.Tensor): output feature, [num_obj, c, h, w].
            out_ctx (torch.Tensor): output context feature, [num_obj, c, 1, 1].
        """
        for i in range(self.stack_num):
            res_x = x
            res_ctx = ctx
            out_x, out_ctx = self.blocks[i](x, ctx)
            x = self.add_op.add(res_x, out_x)
            ctx = self.add_op.add(res_ctx, out_ctx)
        return out_x, out_ctx

    def fuse_model(self):
        for block in self.blocks:
            block.fuse_model()


@OBJECT_REGISTRY.register
class MultiPathPlusHead(MultiPathHead):
    """Multipath Plus head implementation.

    idea from multipath++ paper. The head includes two interaction
    module and a trajectory decoder. The first interaction module
    encode the agents' history trajectories with the rasterized map
    feature, the second interaction module encode the context feature
    with several anchor embeddings, and these embeddings is used to
    predict final trajectories in the decoder.
    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
        stride2channels: Dict,
        input_shape: Tuple[int],
        roi_input_patch: Tuple[int] = (16, 16),
        roi_output_patch: Tuple[int] = (11, 11),
        his_enc_channels: Tuple[int] = (16, 16, 32),
        interact_channel: int = 128,
        embedding_channel: int = 256,
        n_hidden_layers: Tuple[int] = (256, 256),
        use_momentum: bool = False,
        is_qat_model: bool = False,
        is_int_infer_model: bool = False,
        stack_num: int = 5,
        norm_coord_scale: int = None,
        num_anchors: int = 5,
        traj_len: int = 12,
    ):
        """Initialize method.

        Args:
            in_strides (List): a list contains the strides of feature maps from
                backbone or neck.
            out_strides (List): a list contains the strides of this head will
                output.
            stride2channels (Dict): a stride to channel dict.
            input_shape (Tuple): the shape of input context maps.
            roi_input_patch (Tuple): height and weight of cropping roi input
                patches. Defaults to (16, 16).
            roi_output_patch (Tuple): height and weight of cropping roi
                patches. Defaults to (11, 11).
            his_enc_channels (Tuple): channels of history trajectory
                encoding module.
            interact_channel (int): channel of interact module between map and
                historical trajectory.
            embedding_channel (int): channel of anchor embedding.
            n_hidden_layers (list): a list contains the channels of the fc
                layers in the trajectory decoder.
            use_momentum (bool): whether to predict momentum instead of xy
                offsets. Defaults to False.
            is_qat_model (bool): whether the model is for qat.
            is_int_infer_model (bool): whether the model is for int inference.
            stack_num (int): the number of stacked MCG blocks.
            norm_coord_scale (int): the norm scale of the trajectories input
                and output, norm is benefical for qat. Default to None, which
                means no norm.
            num_anchors (int): the number of anchor embeddings.
            traj_len (int): the length of predicted trajecrory.
        """
        nn.Module.__init__(self)
        self.in_strides = in_strides
        self.out_strides = out_strides
        self.stride2channels = stride2channels
        self.input_shape = input_shape
        self.roi_input_patch = roi_input_patch
        self.roi_output_patch = roi_output_patch
        self.use_momentum = use_momentum
        self.is_qat_model = is_qat_model
        self.is_int_infer_model = is_int_infer_model
        self.his_enc_channels = his_enc_channels
        self.interact_channel = interact_channel
        self.embedding_channel = embedding_channel
        self.stack_num = stack_num
        self.traj_len = traj_len
        self.n_hidden_layers = [embedding_channel] + list(n_hidden_layers)
        self.norm_coord_scale = norm_coord_scale

        self.input_channel = self.stride2channels[self.out_strides[0]]
        if self.input_channel != self.interact_channel:
            self.input_proj = nn.Conv2d(
                self.input_channel, self.interact_channel, 1, bias=False
            )

        self.num_anchors = num_anchors
        self.anchor_embedding = nn.Embedding(
            self.num_anchors, embedding_channel
        )
        nn.init.orthogonal_(self.anchor_embedding.weight)

        self.build_basic_graph()

        self.cat_op = nn.quantized.FloatFunctional()
        self.quant = QuantStub(scale=None)
        self.anchor_quant = QuantStub(scale=None)
        self.dequant = DeQuantStub()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def build_basic_graph(self):
        """Build the basic graph of Multipath."""
        # ROI.
        grid_quant_scale = self.cal_roi_quanti_scale(
            self.roi_input_patch, self.roi_output_patch
        )
        self.roi_quant = QuantStub(scale=grid_quant_scale)
        self.rroi_layer = horizon.nn.GridSample(padding_mode="zeros")

        # his traj encoding.
        self.his_enc_channels = [
            3,
        ] + list(self.his_enc_channels)

        his_conv = [
            ConvModule2d(
                in_channel,
                out_channel,
                kernel_size=(1, 2),
                norm_layer=nn.BatchNorm2d(out_channel),
                act_layer=nn.ReLU(inplace=True),
            )
            for in_channel, out_channel in zip(
                self.his_enc_channels[:-1], self.his_enc_channels[1:]
            )
        ]

        his_conv.append(
            ConvModule2d(
                self.his_enc_channels[-1],
                self.interact_channel,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d(self.interact_channel),
                act_layer=nn.ReLU(inplace=True),
            )
        )

        self.his_conv = nn.Sequential(*his_conv)

        # interaction.
        self.Lane2Agent = MCG(
            self.interact_channel,
            (self.roi_output_patch[0], self.roi_output_patch[1]),
            self.stack_num,
        )
        self.Anchor2Ctx = MCG(
            self.embedding_channel, (1, self.num_anchors), self.stack_num
        )

        # FC layers.
        self.predict_conv = [
            ConvModule2d(
                in_channel,
                out_channel,
                kernel_size=1,
                act_layer=nn.ReLU(inplace=True),
            )
            for in_channel, out_channel in zip(
                self.n_hidden_layers[:-1], self.n_hidden_layers[1:]
            )
        ]
        self.predict_conv = nn.Sequential(*self.predict_conv)

        self.cls_conv = nn.Conv2d(self.n_hidden_layers[-1], 1, kernel_size=1)
        self.reg_conv = nn.Conv2d(
            self.n_hidden_layers[-1], self.traj_len * 2, kernel_size=1
        )

    def get_mcg_output(
        self,
        out_feat,
        data,
    ):
        if self.is_int_infer_model:
            batch_homographys = data["img_homographys"]
            his_input = self.quant(data["his_input"])
            roi_feat = self._extract_test_model_roi(
                out_feat, batch_homographys
            )
        else:
            batch_homographys = data["img_homographys"]

            ctx_trajs = (
                data["ctx_trajectories"][:, 1:, :]
                - data["ctx_trajectories"][:, :-1, :]
            )

            if self.norm_coord_scale:
                ctx_trajs = ctx_trajs / self.norm_coord_scale

            ctx_masks = data["ctx_masks"][:, :-1]

            his_input = torch.cat((ctx_trajs, ctx_masks.unsqueeze(2)), dim=2)
            his_input = his_input.permute(0, 2, 1).unsqueeze(2)
            his_input = self.quant(his_input)

            batch_img_coords = data["valid_img_coords"]

            roi_feat = self._extract_roi(
                out_feat, batch_img_coords, batch_homographys
            )

        num_obs = roi_feat.shape[0]
        anchor_embedding = self.anchor_embedding.weight.transpose(1, 0)[
            None, :, None, :
        ].repeat(num_obs, 1, 1, 1)
        anchor_embedding = self.anchor_quant(anchor_embedding)

        if self.input_channel != self.interact_channel:
            roi_feat = self.input_proj(roi_feat)

        his_feat = self.his_conv(his_input)
        _, ctx_feat = self.Lane2Agent(roi_feat, his_feat)
        concat_feat = self.cat_op.cat((ctx_feat, his_feat), dim=1)
        anchor_feat, _ = self.Anchor2Ctx(anchor_embedding, concat_feat)

        predict_feat = self.predict_conv(anchor_feat)
        anchor_prob = self.cls_conv(predict_feat)
        anchor_means = self.reg_conv(predict_feat)

        anchor_prob = self.dequant(anchor_prob)
        anchor_means = self.dequant(anchor_means)

        return anchor_prob, anchor_means

    def forward(self, data: Dict):
        """Forward.

        Args:
            data (Dict): the model input dictionary with the following keys: \
                'feats' (List): the features from backbone.
                'valid_img_coords' (List[Tuple]): list of image coordinates \
                    tuples of prediction objects.
                'future_trajectories' (torch.Tensor, [objnum, traj_len, 2]): \
                    the ground-truth trajectories.
                'img_homographys' (torch.Tensor, [objnum, 3, 3]): the \
                    homography matrix.
                'ctx_trajectories' (torch.Tensor, [objnum, ctx_traj_len, 2]): \
                    the agents' context trajectories.
                'ctx_masks': (torch.Tensor, [objnum, ctx_traj_len]): \
                    the agents' context masks.

        Returns:
            results (Dict): the model output dictionary with the following
                keys: \
                'gts', 'probabilities', 'log_anchors_probs', \
                'means', 'track_ids', 'masks', 'resize_ratio', \
                'cur_head_mask'.
        """
        results = {}

        feats = data["feats"]
        features = _take_features(feats, self.in_strides, self.out_strides)
        out_feat = features[0]

        anchor_probs, anchor_means = self.get_mcg_output(out_feat, data)

        if self.is_int_infer_model:
            return anchor_probs, anchor_means
        else:
            gts = data["future_trajectories"]

            means = torch.reshape(
                anchor_means.permute(0, 3, 2, 1),
                (-1, self.num_anchors, self.traj_len, 2),
            )
            if self.use_momentum:
                means = torch.cumsum(means, dim=2)

            if self.norm_coord_scale:
                means = means * self.norm_coord_scale

            results["anchor_probs"] = anchor_probs
            results["anchor_means"] = anchor_means
            anchor_probs = anchor_probs.reshape((-1, self.num_anchors))

            results["gts"] = gts
            results["masks"] = data["valid_masks"]
            results["probabilities"] = anchor_probs
            results["log_anchors_probs"] = self.log_softmax(
                results["probabilities"]
            )
            results["means"] = means
            results["scale_trils"] = None
            results["resize_ratio"] = data["resize_ratio"]
            results["valid_img_coords"] = data["valid_img_coords"]
            if "cur_head_mask" in data:
                results["cur_head_mask"] = data["cur_head_mask"]
        return results

    def fuse_model(self):
        for module in [self.Lane2Agent, self.Anchor2Ctx]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

        for op in self.his_conv:
            op.fuse_model()

        for op in self.predict_conv:
            op.fuse_model()

    def set_qconfig(self):
        self.roi_quant.qconfig = torch.quantization.QConfig(
            activation=horizon.quantization.fake_quantize.default_16bit_fake_quant,  # noqa
            weight=None,
        )
