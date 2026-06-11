import logging
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from torch.fx.proxy import Proxy as FxProxy
from horizon_plugin_pytorch.qtensor import QTensor

from .stereoplus.extractor import Feature
from .stereoplus.submodule import *
from .stereoplus.update import BasicUpdateBlock
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["DStereoPlus"]


# DiscoverStereo rectified intrinsics in the 352x640 crop frame used by
# the current preproc352x640 validation recipe:
# original 1280x1088 -> resize to 640x544 -> center crop to 352x640.
DUST3R_DISCOVER_PREPROC352X640_INTRINSICS = {
    "fx": 186.2146208425,
    "fy": 186.2146208425,
    "cx": 320.0,
    "cy": 176.0,
    "bf": 20.44084855698,
}


class Conv2DInterpolate(nn.Module):
    def __init__(self, inputs_channel=1, scale_factor=2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=inputs_channel,
            out_channels=inputs_channel * (scale_factor**2),
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.scale_factor = scale_factor
        self.inputs_channel = inputs_channel
        self.depth2space = torch.nn.PixelShuffle(scale_factor)
        self._init_weights()
        self.freeze()

    def _init_weights(self):
        conv_weight = torch.zeros(
            self.conv.weight.size(),
            dtype=self.conv.weight.dtype,
        )
        num_conv = conv_weight.shape[0]
        for i_N in range(num_conv):
            i_c = i_N // (self.scale_factor**2)
            conv_weight[i_N, i_c, 1, 1] = 1
        self.conv.weight = torch.nn.Parameter(conv_weight, requires_grad=False)

    def forward(self, x):
        x = self.conv(x)
        out = self.depth2space(x)
        return out

    def freeze(self):
        for param in self.conv.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(Conv2DInterpolate, self).train(mode)
        self.freeze()


class UnfoldConv(nn.Module):
    """
    A unfold module using conv.

    Args:
        in_channels: The channels of inputs.
        kernel_size: The kernel_size of unfold.
    """

    def __init__(self, in_channels: int = 1, kernel_size: int = 3):
        super(UnfoldConv, self).__init__()
        self.kernel_size = kernel_size
        self.unflod_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.kernel_size**2,
            kernel_size=self.kernel_size,
            stride=1,
            bias=False,
        )
        self.pad = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of head module."""

        weight_new = torch.zeros(
            self.unflod_conv.weight.size(), dtype=self.unflod_conv.weight.dtype
        )
        for i in range(self.kernel_size**2):
            wx = i % self.kernel_size
            wy = i // self.kernel_size

            weight_new[i, :, wy, wx] = 1

        self.unflod_conv.weight = torch.nn.Parameter(weight_new, requires_grad=False)
        self.freeze()

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the model."""

        x = self.pad(x)
        x = self.unflod_conv(x)
        return x

    def freeze(self):
        for param in self.unflod_conv.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UnfoldConv, self).train(mode)
        self.freeze()


class hourglass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(
                in_channels,
                in_channels * 2,
                is_3d=False,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=False,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv2 = nn.Sequential(
            BasicConv(
                in_channels * 2,
                in_channels * 4,
                is_3d=False,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=False,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv3 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 6,
                is_3d=False,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 6,
                in_channels * 6,
                is_3d=False,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv3_up = BasicConv(
            in_channels * 6,
            in_channels * 4,
            deconv=True,
            is_3d=False,
            bn=True,
            relu=True,
            kernel_size=(4, 4),
            padding=(1, 1),
            stride=(2, 2),
        )

        self.conv2_up = BasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=False,
            bn=True,
            relu=True,
            kernel_size=(4, 4),
            padding=(1, 1),
            stride=(2, 2),
        )

        self.conv1_up = BasicConv(
            in_channels * 2,
            out_channels,
            deconv=True,
            is_3d=False,
            bn=False,
            relu=False,
            kernel_size=(4, 4),
            padding=(1, 1),
            stride=(2, 2),
        )

        self.agg_0 = nn.Sequential(
            BasicConv(
                in_channels * 8,
                in_channels * 4,
                is_3d=False,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=False,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=False,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.agg_1 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 2,
                is_3d=False,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=False,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=False,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.feature_att_8 = FeatureAtt(in_channels * 2, 128)
        self.feature_att_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_32 = FeatureAtt(in_channels * 6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, 128)

    def forward(self, x, features):

        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


def build_gwc_volume_onnx(refimg_fea, targetimg_fea, maxdisp):
    tmp_volume = []
    for i in range(maxdisp):
        if i > 0:
            cost = refimg_fea[:, :, :, i:] * targetimg_fea[:, :, :, :-i]
            cost = cost.mean(dim=1, keepdim=True)
            cost = torch.nn.functional.pad(cost, (i, 0, 0, 0), "constant", 0)
            tmp_volume.append(cost)
        else:
            cost = refimg_fea * targetimg_fea
            cost = cost.mean(dim=1, keepdim=True)
            tmp_volume.append(cost)

    return torch.cat(
        tmp_volume, dim=1
    )  


class refinement(nn.Module):
    def __init__(self, gru_iters, hidden_dim):
        super().__init__()
        self.gru_iters = gru_iters
        # Compile path can request raw (disp_unfold, up_weights) outputs
        # to keep protocol aligned with PTQ-side postprocess.
        self.return_raw_upsample = False
        self.update_block = BasicUpdateBlock(hidden_dim=hidden_dim)

        self.interp_conv = Conv2DInterpolate(inputs_channel=9)
        self.unfold_conv = UnfoldConv(in_channels=1, kernel_size=3)
        self.spx_2_gru = Conv2x(32, 32, deconv=True, concat=True)
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )

    def context_upsample(self, disp_low, up_weights):
        ###
        # cv (b,1,h,w)
        # sp (b,9,4*h,4*w)
        ###
        b, c, h, w = disp_low.shape
        if torch.onnx.is_in_onnx_export() or self.return_raw_upsample:
            disp_unfold = self.unfold_conv(disp_low)
            # disp_unfold = self.interp_conv(self.interp_conv(disp_unfold))
            return disp_unfold, up_weights
        else:
            # Keep unfold trace-friendly and QTensor-friendly by always using
            # the equivalent conv implementation.
            disp_unfold = self.unfold_conv(disp_low).reshape(b, -1, h, w)
            disp_unfold = F.interpolate(
                disp_unfold, (h * 4, w * 4), mode="nearest"
            ).reshape(b, 9, h * 4, w * 4)
            # HBDK horizon.sum requires keepdim=True on this quantized path.
            disp = (disp_unfold * up_weights).sum(dim=1, keepdim=True)
            return disp

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = torch.softmax(spx_pred, dim=1)
        up_disp = self.context_upsample(disp * 4.0, spx_pred)
        return up_disp

    def forward(self, disp, net, context, geo_encoding_volume, stem_2x):
        disp_preds = []
        for itr in range(self.gru_iters):
            disp = disp.detach()
            net, mask_feat_4, delta_disp = self.update_block(
                net, context, geo_encoding_volume, disp
            )
            disp = disp + delta_disp
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)
        return disp_preds, disp


class prepare_forrefinement(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hnet = nn.Sequential(
            BasicConv(64, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, bias=False),
        )

        self.cnet = BasicConv(64, context_dim, kernel_size=3, stride=1, padding=1)
        self.context_zqr_conv = nn.Conv2d(
            context_dim, context_dim * 3, 3, padding=3 // 2
        )
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, features_left):
        hidden = self.hnet(features_left[0])
        # net = self.relu(hidden)
        net = torch.tanh(hidden)
        context = self.cnet(features_left[0])
        z, q, r = self.context_zqr_conv(context).split(
            split_size=self.hidden_dim,
            dim=1,
        )
        return net, (z, q, r)


class get_initdisp(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp
        # self.classifier = BasicConv(48, 48, kernel_size=3, stride=1, padding=1)
        self.classifier = BasicConv(maxdisp // 4, maxdisp // 4, kernel_size=3, stride=1, padding=1)
        disp_values = torch.arange(
            0,
            maxdisp // 4,
            1,
            dtype=torch.float32,
        ).view(1, maxdisp // 4, 1, 1)
        # Keep a trace-friendly constant without affecting checkpoint keys.
        self.register_buffer("disp_values", disp_values, persistent=False)

    def forward(self, geo_encoding_volume):
        # Init disp from geometry encoding volume
        prob = torch.softmax(self.classifier(geo_encoding_volume), dim=1)
        init_disp = torch.zeros_like(prob[:, :1, :, :])
        for idx in range(self.maxdisp // 4):
            init_disp = init_disp + prob[:, idx : idx + 1, :, :] * float(idx)
        return init_disp


class get_costvolum(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp

    def forward(self, match_left, match_right):
        if (
            torch.onnx.is_in_onnx_export()
            or isinstance(match_left, FxProxy)
            or isinstance(match_right, FxProxy)
        ):
            gwc_volume = build_gwc_volume_onnx(
                match_left, match_right, self.maxdisp // 4
            )
        else:
            gwc_volume = build_gwc_volume(match_left, match_right, self.maxdisp // 4, 1)
        return gwc_volume


class before_costvolum(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.conv = BasicConv(64, 48, kernel_size=3, padding=1, stride=1)

    def forward(self, features_left, features_right):
        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        return match_left, match_right


@OBJECT_REGISTRY.register
class DStereoPlus(nn.Module):
    def __init__(self, backbone, gru_iters, maxdisp, training_stage: str = "float"):
        super().__init__()
        self.backbone = backbone
        self.maxdisp = maxdisp
        self.training_stage = training_stage
        self.hidden_dim = 48
        context_dim = self.hidden_dim

        self.spx = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )
        self.spx_2 = Conv2x(24, 32, deconv=True)
        self.spx_4 = nn.Sequential(
            BasicConv(64, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24),
            nn.ReLU(),
        )
        self.feature = Feature()
        self.cost_agg = hourglass(self.maxdisp // 4, self.maxdisp // 4)
        logger.info(
            "###################### init DStereoPlus done ######################"
        )
        self.get_costvolum = get_costvolum(self.maxdisp)
        self.before_costvolum = before_costvolum()
        self.get_initdisp = get_initdisp(self.maxdisp)
        self.prepare_forrefinement = prepare_forrefinement(
            hidden_dim=32, context_dim=32
        )
        self.refinement = refinement(gru_iters, hidden_dim=32)
        self._pointmap_weight_cache = {}

    def _as_feature_tuple(self, features):
        # MixVarGENet output_list=[0,1,2,3,4]:
        # for input NCHW=(N,3,352,640), feature shapes are
        # x2:  (N, 32, 176, 320)
        # x4:  (N, 32,  88, 160)
        # x8:  (N, 64,  44,  80)
        # x16: (N, 96,  22,  40)
        # x32: (N,160,  11,  20)
        return (
            features[0],
            features[1],
            features[2],
            features[3],
            features[4],
        )

    def _split_feature_tuple(self, features, batch_size: int):
        return (
            features[0][:batch_size, ...],
            features[1][:batch_size, ...],
            features[2][:batch_size, ...],
            features[3][:batch_size, ...],
            features[4][:batch_size, ...],
        ), (
            features[0][batch_size:, ...],
            features[1][batch_size:, ...],
            features[2][batch_size:, ...],
            features[3][batch_size:, ...],
            features[4][batch_size:, ...],
        )

    def _maybe_dequantize(self, value):
        should_dequantize = isinstance(value, QTensor) or (
            isinstance(value, torch.Tensor) and value.is_quantized
        )
        if should_dequantize:
            return value.dequantize()
        return value

    def _squeeze_single_channel_disp(self, value):
        value = self._maybe_dequantize(value)
        if isinstance(value, torch.Tensor) and value.dim() == 4 and value.size(1) == 1:
            return value[:, 0]
        return value

    def _get_pointmap_weight(self, height, width, device, dtype):
        intrinsics = DUST3R_DISCOVER_PREPROC352X640_INTRINSICS
        device_index = device.index if device.type == "cuda" else None
        cache_key = (height, width, device.type, device_index, dtype)
        point_weight = self._pointmap_weight_cache.get(cache_key)
        if point_weight is None:
            u = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
            v = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
            point_weight = (
                ((u - intrinsics["cx"]) / intrinsics["fx"]).abs()
                + ((v - intrinsics["cy"]) / intrinsics["fy"]).abs()
                + 1.0
            )
            self._pointmap_weight_cache[cache_key] = point_weight
        return point_weight

    def _pointmap_l1(self, disp_pred, valid_mask, z_gt, point_weight):
        intrinsics = DUST3R_DISCOVER_PREPROC352X640_INTRINSICS
        disp_pred = disp_pred.clamp_min(1e-3)
        z_pred = intrinsics["bf"] / disp_pred
        safe_mask = valid_mask & torch.isfinite(z_pred) & torch.isfinite(z_gt)
        point_error = (z_pred - z_gt).abs() * point_weight
        point_error = torch.where(safe_mask, point_error, torch.zeros_like(point_error))
        valid_count = safe_mask.to(dtype=point_error.dtype).sum().clamp_min(1.0)
        return point_error.sum() / valid_count

    def forward(self, data):
        self.refinement.return_raw_upsample = self.training_stage == "compile"

        if isinstance(data, FxProxy):
            # int_infer/compile keep PTQ-like dual-input protocol.
            if self.training_stage in ("int_infer", "compile"):
                img = torch.cat((data["infra1"], data["infra2"]), dim=0)
            else:
                img = data["img"]
        elif isinstance(data, dict):
            if "img" in data:
                img = data["img"]
            elif "infra1" in data and "infra2" in data:
                img = torch.cat((data["infra1"], data["infra2"]), dim=0)
            else:
                img = data
        else:
            img = data

        features = self._as_feature_tuple(self.backbone(img))
        features_left, features_right = self._split_feature_tuple(
            features, img.shape[0] // 2
        )

        stem_2x = features_left[0]
        features_left = tuple(
            self.feature(
                features_left[0],
                features_left[1],
                features_left[2],
                features_left[3],
                features_left[4],
            )
        )
        features_right = tuple(
            self.feature(
                features_right[0],
                features_right[1],
                features_right[2],
                features_right[3],
                features_right[4],
            )
        )

        match_left, match_right = self.before_costvolum(features_left, features_right)
        gwc_volume = self.get_costvolum(match_left, match_right)
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)
        init_disp = self.get_initdisp(geo_encoding_volume)

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = torch.softmax(spx_pred, dim=1)

        net, context = self.prepare_forrefinement(features_left)
        disp = init_disp
        disp_preds, disp_4x = self.refinement(
            disp, net, context, geo_encoding_volume, stem_2x
        )
        init_disp_pred = self.refinement.context_upsample(
            init_disp * 4.0, spx_pred.float()
        )
        pred_disp = disp_preds[-1]
        if self.training and (
            (self.training_stage == "qat" and isinstance(data, FxProxy))
            or (isinstance(data, dict) and "gt_disp" in data)
        ):
            if self.training_stage == "qat":
                return {
                    "loss_inputs": (
                        init_disp_pred,
                        tuple(disp_preds),
                        data["gt_disp"],
                    ),
                    "pred_disps": self._maybe_dequantize(pred_disp),
                }
            # Float training keeps in-graph loss behavior.
            losses = self.sequence_loss(init_disp_pred, disp_preds, data["gt_disp"])
            return {
                "losses": losses,
                "pred_disps": self._maybe_dequantize(pred_disp),
            }
        if self.refinement.return_raw_upsample and isinstance(
            pred_disp, (tuple, list)
        ):
            # Keep compile outputs aligned to PTQ deploy protocol:
            # (disp_unfold, up_weights, initdisp, initspx-proxy)
            return pred_disp[0], pred_disp[1], disp_4x, init_disp
        return pred_disp, disp_4x, init_disp

    def sequence_loss(self, agg_pred, iter_preds, disp_gt, loss_gamma=0.9):
        """Loss function defined over sequence of flow predictions"""

        # Keep all operands in [N, H, W] so boolean mask indexing matches shape.
        agg_pred = self._squeeze_single_channel_disp(agg_pred)
        disp_gt = self._squeeze_single_channel_disp(disp_gt)
        iter_preds = [
            self._squeeze_single_channel_disp(pred) for pred in iter_preds
        ]

        n_predictions = len(iter_preds)
        disp_loss = []
        valid = (disp_gt > 0.0) & (disp_gt < self.maxdisp)
        disp_gt = disp_gt.clamp_min(1e-3)
        z_gt = DUST3R_DISCOVER_PREPROC352X640_INTRINSICS["bf"] / disp_gt
        point_weight = self._get_pointmap_weight(
            disp_gt.shape[-2], disp_gt.shape[-1], disp_gt.device, disp_gt.dtype
        )

        disp_loss.append(self._pointmap_l1(agg_pred, valid, z_gt, point_weight))
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        for i in range(n_predictions):
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = self._pointmap_l1(iter_preds[i], valid, z_gt, point_weight)
            disp_loss.append(i_weight * i_loss)

        return disp_loss
