# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Tuple

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.utils import _take_features
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager

__all__ = ["AnchorEncodeNeck", "MultiPathNeck", "TemporalFeatNeck"]


@OBJECT_REGISTRY.register
class MultiPathNeck(nn.Module):
    """MultiPath neck implementation.

    Description of Multipath.
    The input feature map of the head is cropped and fed into a fully-connected
    network to generate trajectories. The trajectories is represented with a
    multi-modal mixture of precomputed anchors, with modal probability and
    Gaussian offset to anchor centers.

    This neck module only performs feature extracting. The trajectory
    generation should use anchor-based decoder.
    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
        stride2channels: Dict,
        roi_input_patch: Tuple[int] = (16, 16),
        roi_output_patch: Tuple[int] = (11, 11),
        conv_channels: Tuple[int] = (16, 16, 16, 16),
        use_depthwise_as_avg: bool = True,
        is_int_infer_model: bool = False,
    ):
        """Initialize method.

        Args:
            in_strides: a list contains the strides of feature maps from
                backbone or neck.
            out_strides: a list contains the strides of this head will
                output.
            stride2channels: a stride to channel dict.
            roi_input_patch: height and weight of cropping roi input
                patches.
            roi_output_patch: height and weight of cropping roi patches.
            conv_channels: a list contains the channels of the head crop
                convs.
            use_depthwise_as_avg: whether to use depthwise instead of the
                normal avg pooling. Defaults to False.
            is_int_infer_model: whether the model is for int inference.
        """
        super(MultiPathNeck, self).__init__()
        self.in_strides = in_strides
        self.out_strides = out_strides
        self.stride2channels = stride2channels
        self.roi_input_patch = roi_input_patch
        self.roi_output_patch = roi_output_patch
        self.conv_channels = conv_channels
        self.use_depthwise_as_avg = use_depthwise_as_avg
        self.is_int_infer_model = is_int_infer_model

        # Build graph.
        self.quant = QuantStub(scale=None)
        grid_quant_scale = self.cal_roi_quanti_scale(
            self.roi_input_patch, self.roi_output_patch
        )
        self.roi_quant = QuantStub(scale=grid_quant_scale)
        self.rroi_layer = horizon.nn.GridSample(padding_mode="zeros")

        # Crop convolution module.
        out_channels = self.stride2channels[self.out_strides[0]]
        conv_channels = [out_channels] + self.conv_channels
        if conv_channels:
            self.crop_conv = [
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    bias=False,
                    dilation=1,
                )
                for in_channel, out_channel in zip(
                    conv_channels[:-1], conv_channels[1:]
                )
            ]
            self.crop_conv = nn.Sequential(*self.crop_conv)
        else:
            self.crop_conv = torch.nn.Sequential()

        # Pooling.
        if self.use_depthwise_as_avg:
            self.avg_pool = nn.Conv2d(
                conv_channels[-1],
                conv_channels[-1],
                kernel_size=self.roi_output_patch[-1],
                stride=1,
                padding=0,
                groups=conv_channels[-1],
                bias=False,
            )
            self.bn = nn.BatchNorm2d(conv_channels[-1])
            self.avg_pool = nn.Sequential(*[self.avg_pool, self.bn])
        else:
            self.avg_pool = nn.AvgPool2d(self.roi_output_patch[-1], stride=1)

        self.cat_op = nn.quantized.FloatFunctional()

    def forward(self, data: Dict):
        """Forward.

        Args:
            data: the model input dictionary with the following keys:
            feats (List): the features from backbone.
            valid_img_coords (List[Tuple]): list of image coordinates
                tuples of prediction objects.
            state_vectors (torch.Tensor, [num_obj, num_states, 1, 1]):
                state vectors of the batch data.
            img_homographys (torch.Tensor, [num_obj, 3, 3]): the
                homography matrix.
            history_img_homographys (torch.Tensor, [num_obj, 3, 3]): the
                historical homography matrix.

        Returns:
            results (Dict): features including:
            rroi_feat (torch.Tensor, [num_obj, num_c_0, 1, 1]): ROI feat \
                extracted by `img_homographys`.
            history_rroi_feat (torch.Tensor, [num_obj, num_c_1, 1, 1]): \
                ROI feat extracted by `history_img_homographys`.
            state_vector_feat (torch.Tensor, [num_obj, num_c_2, 1, 1]): \
                state vector feature.
        """
        feats = data["feats"]
        features = _take_features(feats, self.in_strides, self.out_strides)
        out_feat = features[0]
        device = out_feat.device

        # Extract data
        enable_his_rroi_feature = False
        if self.is_int_infer_model:
            state_vectors = data["state_vectors"]
            batch_homographys = data["img_homographys"]
            batch_img_coords = None

            if "history_img_homographys" in data:
                history_img_homography = data["history_img_homographys"]
                history_valid_img_coords = None
                enable_his_rroi_feature = True
        else:
            state_vectors = data["state_vectors"].to(device)
            batch_homographys = data["img_homographys"].to(device)
            batch_img_coords = data["valid_img_coords"]

            if (
                "history_img_homographys" in data
                and "history_valid_img_coords" in data
            ):
                history_img_homography = data["history_img_homographys"].to(
                    device
                )
                history_valid_img_coords = data["history_valid_img_coords"]
                enable_his_rroi_feature = True

        # Quant
        state_vectors = self.quant(state_vectors)
        batch_homographys = self.roi_quant(batch_homographys)
        rroi_feat = self.rroi_align_forward(
            out_feat, batch_homographys, batch_img_coords
        )

        neck_feats = {
            "rroi_feat": rroi_feat,
            "state_vector_feat": state_vectors,
        }

        if enable_his_rroi_feature:
            history_img_homography = self.roi_quant(history_img_homography)
            history_rroi_feat = self.rroi_align_forward(
                out_feat, history_img_homography, history_valid_img_coords
            )
            neck_feats["history_rroi_feat"] = history_rroi_feat

        return neck_feats

    @staticmethod
    def cal_roi_quanti_scale(input_patch: List, output_path: List):
        """Calculate the quanti scale of roi layer.

        Why calculate here? please refer to:
        http://wiki.hobot.cc/pages/viewpage.action?spaceKey=~wenming.meng&title=grid_sample+op+in+plugin

        Args:
            input_patch: the input roi patch size.
            output_patch: the output roi patch size.

        Return:
            quanti_scale: the quantization scale.
        """
        max_coord = np.max(input_patch + output_path)
        coord_bit_num = int(np.ceil(np.log2(max_coord + 1)))
        coord_shift = 15 - coord_bit_num
        coord_shift = max(min(coord_shift, 8), 0)
        quanti_scale = 1.0 / (1 << coord_shift)
        return quanti_scale

    def rroi_align_forward(
        self, out_feat, all_homographys, all_img_coords=None
    ):
        """Perform forward and get output of valid obs head.

        Args:
            out_feat (torch.Tensor, [B, C, H, W]): backbone output features.
            all_img_coords (List[Tuple]): list of image coordinates tuples of
                prediction objects.
            all_img_homographys (torch.Tensor, [N, 3, 3]): a transform matrix
                from out_feat to centric_patches based on region on interests.

        Returns:
            avg_pooled_features (torch.Tensor, [num_obj, conv_out_c, 1, 1]):
                the feature from rroi align layer.
        """
        roi_feats = self._extract_roi(
            out_feat, all_homographys, all_img_coords
        )

        # Crop conv: [num_obj, conv_out_c, crop_h, crop_w].
        crop_conv_output = self.crop_conv(roi_feats)

        # Pooling: [num_obj, conv_out_c, 1, 1].
        avg_pooled_features = self.avg_pool(crop_conv_output)
        return avg_pooled_features

    def _extract_roi(
        self,
        out_feat: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_img_coords: List = None,
    ):
        """Extract region of interests.

        Args:
            out_feat ([B, C, H, W]): backbone output features.
            batch_offsets ([num_obj, H, W, 2]): the offsets from `out_feat`
                to `centric_patches` based on region on interests.
            batch_img_coords: list of image coordinates tuples of targets.

        Returns:
            centric_patches (torch.Tensor): extracted features based on region
                of interests.
        """
        if self.is_int_infer_model:
            num_obj = batch_offsets.shape[0]
            batch_feats = [out_feat for _ in range(num_obj)]
        else:
            assert batch_img_coords is not None, (
                "The current model is for float or qat, thus the "
                "batch_img_coords cannot be Nonetype."
            )
            out_feat = out_feat.split(1)
            batch_feats = []
            for batch_idx in range(len(batch_img_coords)):
                img_coords = batch_img_coords[batch_idx]
                for _ in img_coords:
                    batch_feats.append(out_feat[batch_idx])

        batch_feats = self.cat_op.cat(batch_feats)
        centric_patches = self.rroi_layer(batch_feats, batch_offsets)
        return centric_patches

    def fuse_model(self):
        if self.use_depthwise_as_avg:
            torch.quantization.fuse_modules(
                self.avg_pool,
                [["0", "1"]],
                inplace=True,
                fuser_func=horizon.quantization.fuse_known_modules,
            )

    def set_qconfig(self):
        self.roi_quant.qconfig = torch.quantization.QConfig(
            activation=horizon.quantization.fake_quantize.default_16bit_fake_quant,  # noqa
            weight=None,
        )


@OBJECT_REGISTRY.register
class AnchorEncodeNeck(nn.Module):
    """Neck to encode anchor features.

    This module is designed in `MultipathLite`. To integrate the old
    `MultipathLiteHead` class to the new trajectory prediction structure,
    we split the customized feature extracting structure here.

    How to start an MultipathLite model in the new structure? The user
    should configure `MultipathNeck` + `AnchorEncodeNeck` +
    `BasicAnchorBasedDecoder`.
    """

    def __init__(
        self,
        anc_conv_channels: Tuple[int] = (1024, 128, 16),
        num_anchors: int = 14,
        traj_len: int = 12,
        anchor_mode: str = "selected",
        is_int_infer_model: bool = False,
    ):
        """Initialize method.

        Args:
            anc_conv_channels: a list contains the channels of the head
                crop convs for anchor feature. Defaults to (1024, 128, 16).
            num_anchors: in fact, it is the number of anchor types in this
                head.
            traj_len: the length of the trajectories.
            anchor_mode: the mode to get the anchors. It will be used as
                the prefix when getting anchor info from input dict.
            is_int_infer_model: whether the model is for int inference.
        """
        super(AnchorEncodeNeck, self).__init__()
        self.num_anchors = num_anchors
        self.traj_len = traj_len
        self.anc_conv_channels = anc_conv_channels
        self.anchor_mode = anchor_mode
        self.is_int_infer_model = is_int_infer_model

        # Build graph.
        anc_conv_in = self.num_anchors * self.traj_len * 2
        anc_conv_channels = [anc_conv_in] + self.anc_conv_channels
        nn_linear = [
            nn.Conv2d(in_hidden, out_hidden, kernel_size=1)
            for in_hidden, out_hidden in zip(
                anc_conv_channels[:-1], anc_conv_channels[1:]
            )
        ]
        activation = [
            nn.ReLU() for i in range(len(self.anc_conv_channels) - 1)
        ] + [nn.Sigmoid()]
        anc_feat = []
        for linear, act, out_c in zip(
            nn_linear, activation, anc_conv_channels[1:]
        ):
            anc_feat.append(linear)
            anc_feat.append(nn.BatchNorm2d(out_c))
            anc_feat.append(act)

        self.anc_feat = nn.Sequential(*anc_feat)
        self.anc_quant_x = QuantStub(scale=None)
        self.anc_quant_y = QuantStub(scale=None)
        self.cat_op = nn.quantized.FloatFunctional()

    def forward(self, data: Dict):  # noqa: D208
        """Forward.

        Args:
            data: the model input dictionary with the following keys: \
            feats (List): the features from backbone. \
            {anc_mode}_anchors_diff_x (torch.Tensor, [1, num_anchors, \
                traj_len, 1]): the difference of anchors in x-coordinates. \
            {anc_mode}_anchors_diff_y (torch.Tensor, [1, num_anchors, \
                traj_len, 1]): the difference of anchors in y-coordinates. \
            {anc_mode}_anchors (torch.Tensor, [1, num_anchors, traj_len, \
                2]): the manually assigned anchors. \

        Returns:
            results (Dict): features including: \
            anchor_encode_feat (torch.Tensor, [num_obj, num_c, 1, 1]): \
                features that encodes the anchor information. \
        """
        feats = data["feats"]
        out_feat = feats[0]
        device = out_feat.device

        anc_x_key = f"{self.anchor_mode}_anchors_diff_x"
        anc_y_key = f"{self.anchor_mode}_anchors_diff_y"
        anc_key = f"{self.anchor_mode}_anchors"
        if self.is_int_infer_model:
            anchors_xy = None
            anchors_x = self.anc_quant_x(data[anc_x_key])
            anchors_y = self.anc_quant_y(data[anc_y_key])
        else:
            anchors_xy = data[anc_key].to(device)
            anchors_x = data[anc_x_key].to(device)
            anchors_y = data[anc_y_key].to(device)
            anchors_x = anchors_x.reshape(
                (-1, self.num_anchors * self.traj_len, 1, 1)
            )
            anchors_y = anchors_y.reshape(
                (-1, self.num_anchors * self.traj_len, 1, 1)
            )
            anchors_x = self.anc_quant_x(anchors_x)
            anchors_y = self.anc_quant_y(anchors_y)

        anchors = self.cat_op.cat([anchors_x, anchors_y], dim=1)
        anc_feats = self.anc_feat(anchors)
        neck_feats = {"anchor_encode_feat": anc_feats}
        if anchors_xy is not None:
            neck_feats[anc_key] = anchors_xy.reshape(
                (-1, self.num_anchors, self.traj_len, 2)
            )
        return neck_feats

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.anc_feat,
            [["0", "1", "2"], ["3", "4", "5"], ["6", "7"]],
            inplace=True,
            fuser_func=horizon.quantization.fuse_known_modules,
        )

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class TemporalFeatNeck(nn.Module):
    """Neck to extract temporal features by TCN.

    The temporal neck needs extra trajectories input, tcn_ctx_trajectories.
    The input is fed into a dilated CNN to capture temporal pattern of the
    obstacles.
    """

    def __init__(
        self,
        tcn_channels: Tuple[int] = (5, 5, 5, 5),
        tcn_output_channels: int = 10,
        is_int_infer_model: bool = False,
    ):
        """Initialize method.

        Args:
            tcn_channels: a list contains the channels of the temporal conv.
            tcn_output_channels: output channel of temporal conv.
            is_int_infer_model: whether the model is for int inference.
        """
        super(TemporalFeatNeck, self).__init__()
        self.tcn_channels = tcn_channels
        self.tcn_output_channels = tcn_output_channels
        self.is_int_infer_model = is_int_infer_model

        # Build graph.
        self.temporal_conv = [
            nn.Conv2d(2, self.tcn_channels[0], kernel_size=(1, 3), bias=False),
            nn.BatchNorm2d(self.tcn_channels[0]),
            nn.ReLU(),
        ]
        for lx in range(len(self.tcn_channels) - 1):
            self.temporal_conv += [
                nn.Conv2d(
                    self.tcn_channels[lx],
                    self.tcn_channels[lx + 1],
                    kernel_size=(1, 3),
                    bias=False,
                    dilation=2,
                ),
                nn.BatchNorm2d(self.tcn_channels[lx + 1]),
            ]
        self.temporal_conv = nn.Sequential(*self.temporal_conv)
        self.traj_quant = QuantStub(scale=None)

    def forward(self, data: Dict):
        """Forward.

        Args:
            data: the model input dictionary with the following keys:
            feats (List): the features from backbone.
            tcn_ctx_trajectories (torch.Tensor, [num_obj, 2, 1, ctx_frames]):
                the context trajectory for tcn calculation.

        Returns:
            results (Dict): features including:
            tcn_feat (torch.Tensor, [num_obj, num_c, 1, 1]): tcn features.
        """
        feats = data["feats"]
        out_feat = feats[0]
        device = out_feat.device

        if self.is_int_infer_model:
            obs_ctx_trajs = data["tcn_ctx_trajectories"]
        else:
            obs_ctx_trajs = data["tcn_ctx_trajectories"].to(device)

        obs_ctx_trajs = self.traj_quant(obs_ctx_trajs)
        tcn_feat = self.temporal_conv(obs_ctx_trajs)
        tcn_feat = tcn_feat.reshape((-1, self.tcn_output_channels, 1, 1))
        neck_feats = {"tcn_feat": tcn_feat}
        return neck_feats

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.temporal_conv,
            [["0", "1", "2"], ["3", "4"], ["5", "6"], ["7", "8"]],
            inplace=True,
            fuser_func=horizon.quantization.fuse_known_modules,
        )

    def set_qconfig(self):
        self.traj_quant.qconfig = torch.quantization.QConfig(
            activation=horizon.quantization.fake_quantize.default_16bit_fake_quant,  # noqa
            weight=None,
        )
