# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import (
    QuantStub,
    get_default_calib_qconfig,
    get_default_qat_qconfig,
)
from torch.nn import BatchNorm2d as BN
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.extend_container import ExtSequential
from hat.models.base_modules.roi_feat_extractors import CropperQAT
from hat.registry import OBJECT_REGISTRY
from hat.utils.tensor_func import take_row

__all__ = [
    "EndpointEncoder",
    "StateEncoder",
    "MaxPointsSampler",
    "HTTP",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class EndpointEncoder(nn.Module):
    """
    Encode the endpoints to feature by full-connect layers(1x1 convolutions).

    Args:
        bn_kwargs: Kwargs for batch normalization layer.
        channels: Output channels of full-connect layers.
        input_channel: Channel of endpoints, default 2.

    Shape:
        - Input: [agent_num x input_channel x 1 x 1] x endpoint_num.
        - Ouput: [agent_num x output_channel x 1 x 1] x endpoint_num.
    """

    def __init__(
        self, bn_kwargs: dict, channels: List[int], input_channel: int = 2
    ):
        super(EndpointEncoder, self).__init__()
        channels = [input_channel] + channels
        fcs = [
            ConvModule2d(
                channels[i],
                channels[i + 1],
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True,
                act_layer=nn.ReLU(inplace=True),
                norm_layer=BN(channels[i + 1], **bn_kwargs),
            )
            for i in range(len(channels) - 1)
        ]
        self.fcs = ExtSequential(fcs)
        self.quant = QuantStub(scale=None)
        self.cat_op = nn.quantized.FloatFunctional()

    def forward(self, endpoints: List[torch.Tensor]) -> torch.Tensor:
        endpoints = [self.quant(x) for x in endpoints]
        agent_num = endpoints[0].shape[0]
        endpoint = self.cat_op.cat(endpoints, dim=0)
        endpoint_features = self.fcs(endpoint).split(agent_num)
        return endpoint_features

    def fuse_model(self):
        self.fcs.fuse_model()


@OBJECT_REGISTRY.register
class StateEncoder(nn.Module):
    """
    Encode the history_state to feature by 1D convolution and full-connect \
    layers, implemented with 2D convolution. \
    (input) -> 1D convs -> fcs -> (output).

    Args:
        bn_kwargs: Kwargs for batch normalization layers.
        channels_seq: Ouput channels of 1D convs.
        channles_fc: Ouput channels of full-connect layers.
        kernel_size: Kernel size of 1D convs.
        input_channel: Channel of the input history_state, defalut 2.

    Shape:
        - Input: agent_num x input_channel x 1 x time_stamp.
        - Output: agent_num x output_channel x 1 x 1.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        channels_seq: List[int],
        channels_fc: List[int],
        kernel_size: List[int],
        input_channel: int = 2,
    ):
        super(StateEncoder, self).__init__()

        channels = [input_channel] + channels_seq + channels_fc
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * len(channels_seq)
        kernel_size += len(channels_fc) * [1]

        convs = [
            ConvModule2d(
                channels[i],
                channels[i + 1],
                kernel_size=(1, kernel_size[i]),
                padding=0,
                stride=1,
                bias=True,
                act_layer=nn.ReLU(inplace=True),
                norm_layer=BN(channels[i + 1], **bn_kwargs),
            )
            for i in range(len(channels) - 1)
        ]
        self.convs = ExtSequential(convs)
        self.quant = QuantStub(scale=None)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = self.quant(state)
        return self.convs(state)

    def fuse_model(self):
        self.convs.fuse_model()


@OBJECT_REGISTRY.register
class MaxPointsSampler(object):
    """
    Sample several points with as much confidence as possible from heatmap.

    Args:
        sample_num: Number of sampling points.
        kernel_size: Spaical size of sampling kernel.
        transformation_func: Convert the coordinates of sampling points from
            heatmap to raster map.
        impossible_threshold: The confidence, which is less than this
            threshold, will be set to 0.
        use_passable_mask: Whether to use passable mask to supperss confidence
            at impassable area.

    Shape:
        - Input:
        * input: agent_num x H x W.
        * passable_mask: agent_num x 1 x H x W.
        * offset: agent_num x 2 x H x W.
        - Output:
        * points: [agent_num x 2] x sample_num.
        * values: [agent_num] x sample_num.
    """

    def __init__(
        self,
        sample_num: int,
        kernel_size: int,
        transformation_func: Callable,
        update_radius: Optional[int] = None,
        impossible_threshold: float = 0.1,
        use_passable_mask: Optional[int] = False,
    ):
        super(MaxPointsSampler, self).__init__()

        self.sample_num = sample_num

        radius = int(kernel_size / 2)
        kernel = np.power(np.arange(-radius, radius + 1)[None], 2) + np.power(
            np.arange(-radius, radius + 1)[:, None], 2
        )
        self.kernel = np.float32(kernel <= radius ** 2)
        self.scale = np.sum(self.kernel)
        self.pad = radius
        self.radius = radius

        self.trf_fn = transformation_func
        self.update_radius = (
            update_radius if update_radius is not None else radius
        )
        self.impossible_threshold = 0.1
        self.use_passable_mask = use_passable_mask

    def __call__(
        self,
        input: torch.Tensor,
        passable_mask: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        heatmap = torch.clone(input).detach()
        if passable_mask is not None and self.use_passable_mask:
            heatmap = heatmap * passable_mask
        N, H, W = heatmap.shape
        heatmap = heatmap.reshape(N, 1, H, W)

        if self.update_radius <= 0:
            update_r = torch.pow(
                (heatmap.reshape(N, -1) > 0.3).sum(dim=-1) / math.pi / 6, 1 / 2
            )

        heatmap = torch.where(
            heatmap < self.impossible_threshold,
            heatmap.new_tensor(-1e-3),
            heatmap,
        )

        if offset is not None:
            offset = torch.clone(offset).detach()
            offset = offset.movedim(1, 3).contiguous().reshape(N, H * W, 2)

        kernel = heatmap.new_tensor(self.kernel[None, None])

        points = []
        values = []
        for i in range(self.sample_num):

            temp = nn.functional.conv2d(
                heatmap, kernel, padding=self.pad
            ).reshape(N, -1)

            value, idx = torch.max(temp, dim=1)

            if offset is not None:
                delta = take_row(offset, idx[:, None])[:, 0]
                idx = torch.stack([idx // W, idx % W], axis=1) + delta
            else:
                idx = torch.stack([idx // W, idx % W], axis=1)

            values.append(value / self.scale)
            points.append(idx)

            if i == self.sample_num - 1:
                break
            update_mask = (
                torch.pow(
                    torch.arange(H, device=heatmap.device)[None, :]
                    - idx[:, :1],
                    2,
                )[..., None]
                + torch.pow(
                    torch.arange(W, device=heatmap.device)[None, :]
                    - idx[:, 1:],
                    2,
                )[:, None]
            )
            update_mask = torch.sqrt(update_mask)[:, None]

            if self.update_radius <= 0:
                heatmap = torch.where(
                    update_mask > update_r[:, None, None, None],
                    heatmap,
                    heatmap * 0.1,
                )
            else:
                heatmap = torch.where(
                    update_mask > self.update_radius,
                    heatmap,
                    heatmap * torch.pow(update_mask / self.update_radius, 4),
                )

        points = [
            self.trf_fn(x, inverse=True).to(dtype=heatmap.dtype)
            for x in points
        ]
        return points, values


@OBJECT_REGISTRY.register
class HTTP(nn.Module):
    """
    The structure model of HeaTmap-based Trajectory Predictor, the specific \
    model structure is as follows: \
    (history_state) -> state_encoder -> (state_feature) \
    (rasterized_dyn_data) -> backbone_extra              (state_feature) \
                     |            |                            |  \
    (raster_map) -> backbone -> neck -> feature_cropper -> heatmap_head -> (heatmap) -> sampler  # noqa \
                                              |                                            |  # noqa \
                     (state_feature) -> trajectory_head <- endpoint_encoder <- (endpoints) <  # noqa \
                                              | \
                                        (trajectory).

    Args:
        backbone: Backbone module. If the extra_backbone is None, it will
            perform feature extraction on raster_map and rasterized_dyn_data,
            else it will only do on raster_map.
        neck: Neck module, input multi-scale feature maps and output
            multi-scale feature maps.
        feature_cropper: Crop feature from multi-scale feature.
        backbone_extra: Perform feature extraction on rasterized_dyn_data.
        state_encoder: Encode history_state.
        heatmap_head: Input agent_feature, and ouput heatmap and offset.
        sampler: Sample several points based on heatmap.
        endpoint_encoder: Encoder endpoints.
        trajectory_head: Input agent_feature and endpoint_feature, output
            trajectory.
        need_backward: Whether to calculate losses.
        heatmap_loss: Loss for heatmap.
        heatmap_offset_loss: Loss for offset from heatmap_head.
        traj_reg_loss: Loss for trajectory.
        traj_confidence_loss: Loss for confidence from trajectory_head.
        trajectory_decoder: Decoder for model outputs.
        teacher_num: If it is greater than 0, ground truth of endpoint will be
            send to stage_2nd.
        teacher_loss_weight: Loss weight for the teacher training endpoints.
        output_keys: Names of local variable to return.
        trace_stage: "full", "stage_1st" or "stage_2nd".
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        feature_cropper: nn.Module,
        backbone_extra: Optional[nn.Module] = None,
        state_encoder: Optional[nn.Module] = None,
        heatmap_head: Optional[nn.Module] = None,
        sampler: Optional[Callable] = None,
        endpoint_encoder: Optional[nn.Module] = None,
        trajectory_head: Optional[nn.Module] = None,
        need_backward: bool = True,
        heatmap_loss: Optional[nn.Module] = None,
        heatmap_offset_loss: Optional[nn.Module] = None,
        traj_reg_loss: Optional[nn.Module] = None,
        traj_confidence_loss: Optional[nn.Module] = None,
        trajectory_decoder: Optional[Callable] = None,
        teacher_num: int = 5,
        teacher_loss_weight: float = -1,
        output_keys: Tuple[str] = (
            "heatmap",
            "endpoint",
            "trajectory_pred",
            "traj_pred_decoded",
            "confidence",
        ),
        trace_stage: str = "full",
    ):

        super(HTTP, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.cropper = feature_cropper
        self.backbone_extra = backbone_extra
        self.state_encoder = state_encoder
        self.heatmap_head = heatmap_head
        self.sampler = sampler
        self.endpoint_encoder = endpoint_encoder
        self.traj_head = trajectory_head

        self.traj_decoder = trajectory_decoder
        if self.traj_decoder is None:
            self.traj_decoder = self._traj_decoder

        self.need_backward = need_backward
        self.heatmap_loss = heatmap_loss
        self.heatmap_offset_loss = heatmap_offset_loss
        self.traj_reg_loss = traj_reg_loss
        self.traj_confidence_loss = traj_confidence_loss

        self.teacher_num = teacher_num
        self.teacher_loss_weight = teacher_loss_weight
        self.output_keys = output_keys

        if self.backbone_extra is not None:
            self.cat_op = nn.quantized.FloatFunctional()
            self.quant_extra = QuantStub(scale=None)
        if isinstance(self.cropper, CropperQAT):
            self.quant_crop_feature = QuantStub(scale=None)
        self.quant_agent_feature = QuantStub(scale=None)
        self.quant_state_feature = QuantStub(scale=None)
        self.quant = QuantStub(scale=None)
        self.dequant = DeQuantStub()

        self.trace_stage = trace_stage

    def forward(
        self,
        sample: dict,
        rois: Optional[List[List[torch.Tensor]]] = None,
        endpoints: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        # =============== whether to trace ===============
        if rois is not None or endpoints is not None:
            logger.info("forward trace ...")
            return self._forward_trace(sample, rois=rois, endpoints=endpoints)

        # =============== stage 1st ===============
        history_state = sample["history_state"].movedim(1, 2)[:, :, None]
        heatmap, offset, agent_feature_map, state_feature = self._stage_1st(
            sample["raster_map"],
            sample["rasterized_dynamic_data"],
            history_state,
            sample["history"][:, 0],
            sample["batch_index"],
        )
        agent_feature_map = self.quant_agent_feature(
            self.dequant(agent_feature_map)
        )
        state_feature = self.quant_state_feature(self.dequant(state_feature))

        # =============== sample endpoint ===============
        heatmap = torch.sigmoid(self.dequant(heatmap)[:, 0])
        if offset is not None:
            offset = self.dequant(offset)
        endpoints, confs_1st = self._get_endpoint(
            heatmap,
            offset,
            sample.get("future_traj"),
            sample.get("passable_mask"),
        )

        # =============== stage 2nd ===============
        endpoints_tmp = [x[..., None, None] for x in endpoints]
        trajectory_pred, conf_2nd = self._stage_2nd(
            endpoints_tmp, agent_feature_map, state_feature
        )

        # =============== postprocess ===============
        trajectory_preds = self.dequant(trajectory_pred).split(
            endpoints[0].shape[0]
        )
        if conf_2nd is not None:
            confs_2nd = self.dequant(conf_2nd).split(endpoints[0].shape[0])
        else:
            confs_2nd = None
        (
            heatmap,
            offset,
            endpoint,
            conf_1st,
            trajectory_pred,
            conf_2nd,
            traj_pred_decoded,
            confidence,
        ) = self._decoder(
            heatmap, offset, endpoints, confs_1st, trajectory_preds, confs_2nd
        )

        # =============== loss ===============
        model_outs = {}
        if self.need_backward:
            loss = self._backward(
                heatmap=heatmap,
                offset=offset,
                endpoint=endpoint,
                trajectory_pred=trajectory_pred,
                conf_2nd=conf_2nd,
                traj_pred_decoded=traj_pred_decoded,
                future_mask=sample.get("future_mask"),
                heatmap_gt=sample.get("heatmap"),
                offset_gt=sample.get("offset"),
                trajectory_gt=sample.get("future_traj"),
                passable_mask=sample.get("passable_mask"),
            )
            model_outs.update(loss)

        if self.teacher_num > 0:
            endpoint = endpoint[:, : -self.teacher_num].contiguous()
            traj_pred_decoded = traj_pred_decoded[
                :, : -self.teacher_num
            ].contiguous()
            confidence = confidence[:, : -self.teacher_num].contiguous()

        for key in self.output_keys:
            model_outs[key] = locals().get(key, None)

        return model_outs

    def _forward_trace(
        self,
        test_inputs: dict,
        rois: List[List[torch.Tensor]],
        endpoints: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        # =============== trace stage 1st ===============
        if self.trace_stage != "stage_2nd":
            logger.info("forward trace stage 1st ...")
            (
                heatmap,
                offset,
                agent_feature_map,
                state_feature,
            ) = self._stage_1st(
                test_inputs["raster_map"],
                test_inputs["rasterized_dynamic_data"],
                test_inputs["history_state"],
                rois=rois,
            )
            if self.trace_stage == "stage_1st":
                return heatmap, offset, agent_feature_map, state_feature
        else:
            agent_feature_map = self.quant_agent_feature(
                test_inputs["agent_feature_map"]
            )
            state_feature = self.quant_state_feature(
                test_inputs["state_feature"]
            )
        # =============== trace stage 2nd ===============
        logger.info("forward trace stage 2nd ...")
        trajectory_pred, confs_2nd = self._stage_2nd(
            endpoints, agent_feature_map, state_feature
        )
        if self.trace_stage == "stage_2nd":
            return trajectory_pred

        return heatmap, offset, trajectory_pred

    def _stage_1st(
        self,
        raster_map: torch.Tensor,
        rasterized_dynamic_data: torch.Tensor,
        history_state: torch.Tensor,
        agent_location: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
        rois: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward function of the first stage.

        Shape:
            - Input:
            * raster_map: batch_size x 3 x H x W.
            * rasterized_dynamic_data: batch_size x C_h x H x w.
            * history_state: agent_num x C_h x 1 x time_stamp.
            * agent_location: agent_num x 2.
            * batch_index: agent_num.
            * rois: [[agent_num_i x 4] x batch_size] x scale_num.
            - Output:
            * heatmap: agent_num x 1 x H' x W'.
            * offset: agent_num x 2 x H' x W'.
            * agent_feature_map:
                agent_num x (sum(C_i) + state_feature_channel) x size x size.
            * state_feature: agent_num x state_featuer_channel x 1 x 1.
            where C_i denotes the channel of i_th feature_map from neck,
                size is of cropped feature.
        """
        feature_maps = self._backbone(raster_map, rasterized_dynamic_data)
        feature_maps = self.neck(feature_maps)
        if isinstance(self.cropper, CropperQAT):
            agent_feature_map = self.cropper(
                feature_maps, agent_location, batch_index, rois
            )
        else:
            tmp = [self.dequant(x) for x in feature_maps]
            agent_feature_map = self.quant_crop_feature(
                self.cropper(tmp, agent_location, batch_index)
            )

        #  Concat state_feature and agent_feature_map along channel dimension,
        #  with spatial size (1 x 1) and (size x size) respectively, so the
        #  state_feature need to be tiled in both spatial dimensions.
        state_feature = self.state_encoder(history_state)
        state_feature_tile = self.cat_op.cat(
            [state_feature] * agent_feature_map.shape[2], dim=2
        )
        state_feature_tile = self.cat_op.cat(
            [state_feature_tile] * agent_feature_map.shape[3], dim=3
        )
        agent_feature_map = self.cat_op.cat(
            [agent_feature_map, state_feature_tile], dim=1
        )

        heatmap, offset = (
            self.heatmap_head(agent_feature_map)
            if self.heatmap_head is not None
            else (None, None)
        )

        return heatmap, offset, agent_feature_map, state_feature

    def _stage_2nd(
        self,
        endpoints: torch.Tensor,
        agent_feature_map: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function of the second stage.

        Shape:
            - Input:
            * endpoints: [agent_num x 2 x 1 x 1] x endpoint_num.
            * agent_feature_map:
                agent_num x (sum(C_i) + state_feature_channel) x size x size.
            * state_feature: agent_num x state_featuer_channel x 1 x 1.
            - Output:
            * trajectory_pred:
                (agent_num x endpoint_num) x traj_length x 1 x 1.
            * confidence: (agent_num x endpoint_num) x 1 x 1 x 1.
            where C_i denotes the channel of i_th feature_map from neck,
                size is of cropped feature.
        """
        if self.endpoint_encoder is None or self.traj_head is None:
            return None, None
        endpoint_features = self.endpoint_encoder(endpoints)
        trajectory_pred, confidence = self.traj_head(
            agent_feature_map, endpoint_features, state_feature
        )

        return trajectory_pred, confidence

    def _get_endpoint(
        self,
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        feature_traj_gt: Optional[torch.Tensor] = None,
        passable_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample endpoints from heatmap.

        Shape:
            - Input:
            * heatmap: agent_num x H' x W'.
            * offset: agent_num x 2 x H' x W'.
            - Output:
            * endpoints: [agent_num x 2] x endpoint_num.
            * confidences: [agent_num] x endpoint_num.
        """
        if feature_traj_gt is not None:
            endpoint_gt = feature_traj_gt[:, -1]
        else:
            endpoint_gt = None

        if self.sampler is not None and heatmap is not None:
            endpoints, confidences = self.sampler(
                heatmap, passable_mask, offset=offset
            )

        elif endpoint_gt is not None:
            endpoints = [endpoint_gt]
            confidences = [
                torch.ones(
                    endpoint_gt.shape[0],
                    device=endpoint_gt.device,
                    dtype=endpoint_gt.dtype,
                )
            ]

        if self.teacher_num > 0 and endpoint_gt is not None:
            teacher = endpoint_gt[None] + endpoint_gt.new_tensor(
                np.random.uniform(
                    low=-2,
                    high=2,
                    size=[self.teacher_num, endpoint_gt.shape[0], 2],
                )
            )
            teachers = [x[0] for x in torch.split(teacher, 1, dim=0)]
            endpoints.extend(teachers)
            confidences.extend(
                [
                    torch.ones(
                        endpoint_gt.shape[0],
                        device=endpoint_gt.device,
                        dtype=endpoint_gt.dtype,
                    )
                ]
                * self.teacher_num
            )

        return endpoints, confidences

    def _backbone(
        self, raster_map: torch.Tensor, rasterized_dynamic_data: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Backbone.

        Shape:
            - Input:
            * raster_map: batch_size x 3 x H x W.
            * rasterized_dynamic_data: batch_size x C_h x H x w.
            - Output: [batch_size x C_i x H_i x W_i] x scale_num.
        """
        if self.backbone_extra is None:
            rasterized_input = torch.cat(
                [raster_map, rasterized_dynamic_data], dim=1
            )
            rasterized_input = self.quant(rasterized_input)
            feature_maps = self.backbone(rasterized_input)

        else:
            feature_maps_1 = self.backbone(self.quant(raster_map))
            feature_maps_2 = self.backbone_extra(
                self.quant_extra(rasterized_dynamic_data)
            )
            feature_maps = []
            for i in range(len(feature_maps_1)):
                feature_maps.append(
                    self.cat_op.cat(
                        [feature_maps_1[i], feature_maps_2[i]], dim=1
                    )
                )
        return feature_maps

    def _backward(
        self,
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        endpoint: torch.Tensor,
        trajectory_pred: torch.Tensor,
        conf_2nd: torch.Tensor,
        traj_pred_decoded: torch.Tensor,
        future_mask: Optional[torch.Tensor] = None,
        heatmap_gt: Optional[torch.Tensor] = None,
        offset_gt: Optional[torch.Tensor] = None,
        trajectory_gt: Optional[torch.Tensor] = None,
        passable_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Calculate losses.

        Shape:
            - Input:
            * heatmap: agent_num x H' x W'.
            * offset: agent_num x 2 x H' x W'.
            * endpoint: agent_num x endpoint_num x 2.
            * trajectory_pred:
                agent_num x endpoint_num x predict_time_stamp x 2.
            * conf_2nd: agent_num x endpoint_num.
            * traj_pred_decoded:
                agent_num x endpoint_num x predict_time_stamp x 2.
            * future_mask: agent_num x predict_time_stamp.
            * heatmap_gt: agent_num x H' x W'.
            * offset_gt: agent_num x 2 x H' x W'.
            * trajectory_gt: agent_num x predict_time_stamp x 2.
            * passable_mask: agent_num x H' x W'.
        """
        loss = {}

        if (
            heatmap is not None
            and heatmap_gt is not None
            and self.heatmap_loss is not None
        ):
            heatmap_loss = self.heatmap_loss(
                heatmap, heatmap_gt, passable_mask
            )
            loss["heatmap_loss"] = heatmap_loss

        if (
            offset is not None
            and offset_gt is not None
            and self.heatmap_offset_loss is not None
        ):
            heatmap_offset_loss = self.heatmap_offset_loss(
                offset, offset_gt, heatmap_gt
            )
            loss["heatmap_offset_loss"] = heatmap_offset_loss

        if (
            trajectory_pred is not None
            and trajectory_gt is not None
            and self.traj_reg_loss is not None
        ):
            loss_stage_2nd = self._backward_stage_2nd(
                trajectory_pred,
                endpoint,
                trajectory_gt,
                traj_pred_decoded,
                conf_2nd,
                passable_mask=passable_mask,
            )
            loss.update(loss_stage_2nd)

        return loss

    def _backward_stage_2nd(
        self,
        trajectory_pred: torch.Tensor,
        endpoint: torch.Tensor,
        trajectory_gt: torch.Tensor,
        traj_pred_decoded: Optional[torch.Tensor] = None,
        conf_2nd: Optional[torch.Tensor] = None,
        future_mask: Optional[torch.Tensor] = None,
        passable_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Calculate losses for stage_2nd.

        Shape:
            - Input:
            * trajectory_pred:
                agent_num x endpoint_num x predict_time_stamp x 2.
            * endpoint: agent_num x endpoint_num x 2.
            * trajectory_gt: agent_num x predict_time_stamp x 2.
            * traj_pred_decoded:
                agent_num x endpoint_num x predict_time_stamp x 2.
            * conf_2nd: agent_num x endpoint_num.
            * future_mask: agent_num x predict_time_stamp.
            * passable_mask: agent_num x H' x W'.
        """
        loss = {}
        traj_reg_loss = self.traj_reg_loss(
            trajectory_pred,
            trajectory_gt,
            endpoint,
            future_mask,
            passable_mask,
        )
        loss["traj_reg_loss"] = traj_reg_loss
        if conf_2nd is not None:
            traj_confidence_loss = self.traj_confidence_loss(
                conf_2nd, traj_pred_decoded, trajectory_gt, future_mask
            )
            loss["traj_confidence_loss"] = traj_confidence_loss
        return loss

    def _decoder(
        self,
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        endpoints: List[torch.Tensor],
        confs_1st: List[torch.Tensor],
        trajectory_preds: List[torch.Tensor],
        confs_2nd: List[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        """
        Decode model outputs, default decode function.

        Shape:
            - Input:
            * heatmap: agent_num x H' x W'.
            * offset: agent_num x 2 x H' x W'.
            * endpoints: [agent_num x 2] x endpoint_num.
            * confs_1st: [agent_num] x endpoint_num.
            * trajectory_preds:
                [agent_num x traj_length x 1 x 1] x endpoint_num.
            * confs_2nd: [agent_num x 1 x 1 x 1] x endpoint_num.
            - Ouput:
            * heatmap: agent_num x H' x W'.
            * offset: agent_num x 2 x H' x W'.
            * endpoint: agent_num x endpoint_num x 2.
            * conf_1st: agent_num x endpoint_num.
            * trajectory_pred:
                agent_num x endpoint_num x predict_time_stamp x 2.
            * conf_2nd: agent_num x endpoint_num.
            * traj_pred_decoded:
                agent_num x endpoint_num x predict_time_stamp x 2.
            * confidence: agent_num x endpoint_num.
        """
        endpoint, conf_1st, trajectory_pred, conf_2nd = [
            torch.stack(x, dim=1) if x is not None else x
            for x in [endpoints, confs_1st, trajectory_preds, confs_2nd]
        ]
        agent_num = endpoints[0].shape[0]
        traj_num = len(endpoints)
        trajectory_pred = trajectory_pred.reshape(agent_num, traj_num, -1, 2)
        if conf_2nd is not None:
            conf_2nd = conf_2nd.reshape(agent_num, traj_num)

        traj_pred_decoded = self._traj_decoder(trajectory_pred, endpoint)
        confidence = conf_1st if conf_2nd is None else conf_1st * conf_2nd

        ret = [
            heatmap,
            offset,
            endpoint,
            conf_1st,
            trajectory_pred,
            conf_2nd,
            traj_pred_decoded,
            confidence,
        ]
        return ret

    def _traj_decoder(
        self, trajectory_pred: torch.Tensor, endpoint: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode trajectory_pred.

        Shape:
            - Input:
            * trajectory_pred:
                agent_num x endpoint_num x predict_time_stamp x 2.
            * endpoint: agent_num x endpoint_num x 2.
            - Output: agent_num x endpoint_num x predict_time_stamp x 2.
        """
        T = trajectory_pred.shape[-2]
        anchor = (
            endpoint[..., None, :]
            / endpoint.new_tensor(T)
            * torch.arange(1, T + 1, device=endpoint.device).reshape(
                1, 1, T, 1
            )
        )
        decoded = trajectory_pred + anchor
        return decoded

    def fuse_model(self):
        for module in [
            self.backbone,
            self.neck,
            self.backbone_extra,
            self.state_encoder,
            self.endpoint_encoder,
            self.heatmap_head,
            self.traj_head,
            self.cropper,
        ]:
            if module is None or not hasattr(module, "fuse_model"):
                continue
            module.fuse_model()

    def set_qconfig(self):
        self.qconfig = get_default_qat_qconfig()

        for module in [
            self.backbone,
            self.neck,
            self.backbone_extra,
            self.state_encoder,
            self.endpoint_encoder,
            self.heatmap_head,
            self.traj_head,
            self.cropper,
        ]:
            if module is None or not hasattr(module, "set_qconfig"):
                continue
            module.set_qconfig()

        for loss in [
            self.heatmap_loss,
            self.heatmap_offset_loss,
            self.traj_reg_loss,
            self.traj_confidence_loss,
        ]:
            if loss is None:
                continue
            elif hasattr(loss, "set_qconfig"):
                loss.set_qconfig()
            else:
                loss.qconfig = None

    def set_calibration_qconfig(self):
        self.qconfig = get_default_calib_qconfig()

        for module in [
            self.backbone,
            self.neck,
            self.backbone_extra,
            self.state_encoder,
            self.endpoint_encoder,
            self.heatmap_head,
            self.traj_head,
            self.cropper,
        ]:
            if module is None or not hasattr(
                module, "set_calibration_qconfig"
            ):
                continue
            module.set_calibration_qconfig()

        for loss in [
            self.heatmap_loss,
            self.heatmap_offset_loss,
            self.traj_reg_loss,
            self.traj_confidence_loss,
        ]:
            if loss is None:
                continue
            elif hasattr(loss, "set_calibration_qconfig"):
                loss.set_calibration_qconfig()
            else:
                loss.qconfig = None
