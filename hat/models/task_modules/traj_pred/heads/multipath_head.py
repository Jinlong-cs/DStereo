# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional, Tuple

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.task_modules.traj_pred.heads.basic_heads import (  # noqa: E501
    BasicAnchorBasedDecoder,
)
from hat.models.utils import _take_features
from hat.registry import OBJECT_REGISTRY

__all__ = ["MultiPathHead"]


@OBJECT_REGISTRY.register
class MultiPathHead(BasicAnchorBasedDecoder):
    """MultiPath head implementation.

    The input feature map of the head is cropped and fed into a fully-connected
    network to generate trajectories. The trajectories is represented with a
    multi-modal mixture of precomputed anchors, with modal probability and
    Gaussian offset to anchor centers.

    Except for some basic network structure parameters, the detailed structure
    of Multipath depends on the following options:

    1. `use_state_vectors`: whether to use some extra state vectors of the
        objects. The state vectors include some acceleration information, etc.
        The use of state vectors could improve the model metrics, but for real
        road scenarios, these kinds of information is not easy to obtain. Thus
        if the user is training model for real road scenarios, we recommend
        turning off this option.

    Args:
        anchor_cfg (Dict): the anchor configuration dictionary with the
            following keys:
            1. 'anchor_file': the file path of the pickle anchors.
            2. 'anchor_method': the method to obtain the anchors, it
                should be one of [kmeans, uniform].
            3. 'anchor_num': the number of anchors.
        in_strides (List): a list contains the strides of feature maps from
            backbone or neck.
        out_strides (List): a list contains the strides of this head will
            output.
        stride2channels (Dict): a stride to channel dict.
        input_shape (Tuple): the shape of input context maps.
        roi_input_patch (Tuple): height and weight of cropping roi input
            patches. Defaults to (16, 16).
        roi_output_patch (Tuple): height and weight of cropping roi patches.
            Defaults to (11, 11).
        conv_channels (Tuple): a list contains the channels of the head crop
            convs. Defaults to (16, 16, 8, 8).
        n_hidden_layers (list): a list contains the channels of the head
            fully-connected layers. Defaults to (1024, 1024).
        use_state_vectors (bool): whether to use state vectors. Defaults to
            True.
        num_state_vectors (int): the number of state vectors. Defaults to 3.
        use_depthwise_as_avg (bool): whether to use depthwise instead of the
            normal avg pooling. Defaults to False.
        use_momentum (bool): whether to predict momentum instead of xy offsets.
            Defaults to False.
        cor_clamp (float): a value to clamp gussian correlation coefficient
            output.
        log_std_clamp_min (int): a min value to clamp gaussian std output.
        log_std_clamp_max (int): a max value to clamp gaussian std output.
        is_int_infer_model (bool): whether the model is for int inference.
    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
        stride2channels: Dict,
        input_shape: Tuple[int],
        anchor_cfg: Optional[Dict] = None,
        roi_input_patch: Tuple[int] = (16, 16),
        roi_output_patch: Tuple[int] = (11, 11),
        conv_channels: Tuple[int] = (16, 16, 16, 16),
        n_hidden_layers: Tuple[int] = (1024, 1024),
        use_state_vectors: bool = True,
        num_state_vectors: int = 3,
        use_depthwise_as_avg: bool = True,
        use_momentum: bool = False,
        cor_clamp: float = 0.9,
        log_std_clamp_min: float = -2,
        log_std_clamp_max: float = 2,
        is_int_infer_model: bool = False,
    ):
        """Initialize method."""

        if conv_channels and not isinstance(conv_channels, list):
            raise ValueError(
                "Param conv_channels must be a list, but received"
                f"{type(conv_channels)}"
            )
        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(
                "Param n_hidden_layers must be a list, but received"
                f"{type(n_hidden_layers)}"
            )
        self.in_strides = in_strides
        self.out_strides = out_strides
        self.stride2channels = stride2channels
        self.input_shape = input_shape
        self.roi_input_patch = roi_input_patch
        self.roi_output_patch = roi_output_patch
        self.conv_channels = conv_channels
        self.use_state_vectors = use_state_vectors
        self.num_state_vectors = num_state_vectors
        self.use_depthwise_as_avg = use_depthwise_as_avg

        # State vector size.
        self.num_used_state_vectors = 0
        if self.use_state_vectors:
            self.num_used_state_vectors += num_state_vectors

        flatten_hidden = conv_channels[-1] + self.num_used_state_vectors
        super(MultiPathHead, self).__init__(
            anchor_cfg=anchor_cfg,
            in_channels=flatten_hidden,
            n_hidden_layers=n_hidden_layers,
            use_momentum=use_momentum,
            cor_clamp=cor_clamp,
            log_std_clamp_min=log_std_clamp_min,
            log_std_clamp_max=log_std_clamp_max,
            is_int_infer_model=is_int_infer_model,
        )
        self.build_custom_graph()

    def build_custom_graph(self):
        """Build the custom graph of Multipath."""
        # ROI.
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

        # FC layers 2.
        n_hidden_layers = [conv_channels[-1]] + self.n_hidden_layers
        nn_linear = [
            nn.Conv2d(in_hidden, out_hidden, kernel_size=1)
            for in_hidden, out_hidden in zip(
                n_hidden_layers[:-1], n_hidden_layers[1:]
            )
        ]
        activation = [nn.ReLU() for i in range(len(n_hidden_layers[1:]))]
        self.fc_2 = []
        for linear, act in zip(nn_linear, activation):
            self.fc_2.append(linear)
            self.fc_2.append(act)
        self.fc_2 = nn.Sequential(*self.fc_2)

        # 2-class classification
        self.valid_prob = nn.Conv2d(n_hidden_layers[-1], 1, kernel_size=1)
        self.valid_prob_sig = nn.Sigmoid()

        # Quantization and dequantization.
        self.cat_op = nn.quantized.FloatFunctional()
        self.quant = QuantStub(scale=None)

    def forward(self, data: Dict):
        """Forward.

        Args:
            data (Dict): the model input dictionary with the following keys:
                'feats' (List): the features from backbone.
                'valid_img_coords' (List[Tuple]): list of image coordinates
                    tuples of prediction objects.
                'state_vectors' (torch.Tensor, [num_obj, num_states, 1, 1]):
                    state vectors of the batch data.
                'future_trajectories' (torch.Tensor, [num_obj, traj_len, 2]):
                    the ground-truth trajectories.
                'img_homographys' (torch.Tensor, [num_obj, 3, 3]): the
                    homography matrix.

        Returns:
            results (Dict): the model output dictionary with the following
                keys:
                'gts', 'anchors', 'probabilities', 'log_anchors_probs',
                'means', 'scale_trils', 'track_ids', 'masks', 'resize_ratio',
                'cur_head_mask'.
        """
        results = {}
        if self.is_int_infer_model:
            out_feat, state_vectors, batch_homographys = self.process_data(
                data
            )
            history_img_homography = self.process_data_extend(data)
            anchor_probs, anchor_mean_var = self.get_head_output(
                out_feat, state_vectors, batch_homographys
            )
            track_valid_cls = self.get_head_output_extend(
                out_feat, history_img_homography
            )
            return anchor_probs, anchor_mean_var, track_valid_cls
        else:
            (
                out_feat,
                state_vectors,
                batch_homographys,
                gts,
                batch_img_coords,
                anchors,
            ) = self.process_data(data)
            (
                history_img_homography,
                history_track_id_gt,
                history_valid_img_coords,
            ) = self.process_data_extend(data)
            anchor_probs, anchor_mean_var = self.get_head_output(
                out_feat, state_vectors, batch_homographys, batch_img_coords
            )
            track_valid_cls = self.get_head_output_extend(
                out_feat, history_img_homography, history_valid_img_coords
            )
            means, scale_trils = self._extract_gaussian(
                anchor_mean_var, anchors
            )
            results["anchor_probs"] = anchor_probs
            results["anchor_mean_var"] = anchor_mean_var
            anchor_probs = anchor_probs.reshape((-1, self.num_anchors))
            results["gts"] = gts
            results["anchors"] = anchors
            results["probabilities"] = anchor_probs
            results["log_anchors_probs"] = self.log_softmax(
                results["probabilities"]
            )
            results["means"] = means
            results["scale_trils"] = scale_trils
            results["track_ids"] = data["valid_track_ids"]
            results["masks"] = data["valid_masks"]
            results["resize_ratio"] = data["resize_ratio"]
            results["track_valid_cls"] = track_valid_cls
            results["track_id_gt"] = history_track_id_gt
            if "cur_head_mask" in data:
                results["cur_head_mask"] = data["cur_head_mask"]
        return results

    @staticmethod
    def cal_roi_quanti_scale(input_patch: List, output_path: List):
        """Calculate the quanti scale of roi layer.

        Why calculate here? please refer to:
        http://wiki.hobot.cc/pages/viewpage.action?spaceKey=~wenming.meng&title=grid_sample+op+in+plugin

        Args:
            input_patch (List): the input roi patch size.
            output_patch (List): the output roi patch size.

        Return:
            quanti_scale (int): the quantization scale.
        """
        max_coord = np.max(input_patch + output_path)
        coord_bit_num = int(np.ceil(np.log2(max_coord + 1)))
        coord_shift = 15 - coord_bit_num
        coord_shift = max(min(coord_shift, 8), 0)
        quanti_scale = 1.0 / (1 << coord_shift)
        return quanti_scale

    def process_data(self, data):
        """Process the input data.

        Args:
            data (Dict): the model input dictionary with the following keys:
                'feats' (List): the features output from the backbone.
                'valid_img_coords' (List[Tuple]): list of image coordinates
                    tuples of prediction objects.
                'state_vectors' (torch.Tensor, [num_obj, num_states, 1, 1]):
                    state vectors of MultiPath.
                'future_trajectories' (torch.Tensor, [num_obj, traj_len, 2]):
                    the ground-truth trajectories.
                'img_homographys' (torch.Tensor, [num_obj, 3, 3]): the
                    homography matrix.
            feats (List): the features from backbone.
        """
        feats = data["feats"]
        features = _take_features(feats, self.in_strides, self.out_strides)
        out_feat = features[0]

        if self.is_int_infer_model:
            state_vectors = self.quant(data["state_vectors"])
            batch_homographys = data["img_homographys"]
            return out_feat, state_vectors, batch_homographys
        else:
            state_vectors = self.quant(data["state_vectors"].cuda())
            batch_homographys = data["img_homographys"].cuda()
            gts = data["future_trajectories"]
            batch_img_coords = data["valid_img_coords"]
            anchors = self.anchors.cuda()[None, :, :, :]
            return (
                out_feat,
                state_vectors,
                batch_homographys,
                gts,
                batch_img_coords,
                anchors,
            )

    def process_data_extend(self, data):
        """Process the input data.

        Args:
            data (Dict): the model input dictionary with the following keys:
                'history_img_homographys' (List[Tuple]): list of homography
                    matrix of valid head prediction objects.
                'history_valid_img_coords' (List[Tuple]): list of image
                    coordinates tuples of valid head prediction objects.
        """
        if self.is_int_infer_model:
            history_img_homography = data["history_img_homographys"]
            return history_img_homography
        else:
            history_img_homography = data["history_img_homographys"].cuda()
            history_track_id_gt = data["history_valid_track_id_mask"]
            history_valid_img_coords = data["history_valid_img_coords"]
            return (
                history_img_homography,
                history_track_id_gt,
                history_valid_img_coords,
            )

    def get_head_output_extend(
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
            track_valid_cls (torch.Tensor, [num_obj]): the valid probabilities
                of input obstacles.
        """
        if self.is_int_infer_model:
            roi_feats = self._extract_test_model_roi(out_feat, all_homographys)
        else:
            assert all_img_coords is not None, (
                "The current model is for training, thus the batch_img_coords "
                "cannot be Nonetype."
            )
            roi_feats = self._extract_roi(
                out_feat, all_img_coords, all_homographys
            )

        # Crop conv: [num_obj, conv_out_c, crop_h, crop_w].
        crop_conv_output = self.crop_conv(roi_feats)

        # Pooling: [num_obj, conv_out_c, 1, 1].
        avg_pooled_features = self.avg_pool(crop_conv_output)

        # FC: [num_obj, n_hidden, 1, 1]
        fc_2 = self.fc_2(avg_pooled_features)

        # 2-class classification
        track_valid_cls = self.valid_prob_sig(self.valid_prob(fc_2))

        # Quantization.
        track_valid_cls = self.dequant(track_valid_cls)

        return track_valid_cls

    def get_head_output(
        self, out_feat, state_vectors, batch_homographys, batch_img_coords=None
    ):
        """Perform forward and get head output.

        Args:
            out_feat (torch.Tensor, [B, C, H, W]): backbone output features.
            batch_img_coords (List[Tuple]): list of image coordinates tuples of
                prediction objects.
            state_vectors (torch.Tensor, [num_obj, num_states, 1, 1]): state
                vectors of the batch data.
            batch_homographys (torch.Tensor, [N, 3, 3]): a transform matrix
                from out_feat to centric_patches based on region on interests.

        Returns:
            anchor_probs (torch.Tensor, [num_obj, num_anchors]): the anchor
                probabilities.
            anchor_mean_var (torch.Tensor): [num_obj, num_anchors, traj_len, 5]
                ): `5` means (mean_x, mean_y, log_std_x, log_std_y,
                correlation).
        """
        if self.is_int_infer_model:
            roi_feats = self._extract_test_model_roi(
                out_feat, batch_homographys
            )
        else:
            assert batch_img_coords is not None, (
                "The current model is for training, thus the batch_img_coords "
                "cannot be Nonetype."
            )
            roi_feats = self._extract_roi(
                out_feat, batch_img_coords, batch_homographys
            )

        # Crop conv: [num_obj, conv_out_c, crop_h, crop_w].
        crop_conv_output = self.crop_conv(roi_feats)

        # Pooling: [num_obj, conv_out_c, 1, 1].
        avg_pooled_features = self.avg_pool(crop_conv_output)

        # Concate with state vectors.
        if self.use_state_vectors:
            concatenated_vectors = self.cat_op.cat(
                [avg_pooled_features, state_vectors], dim=1
            )
        else:
            concatenated_vectors = avg_pooled_features

        # FC: [num_obj, n_hidden, 1, 1]
        fc = self.fc(concatenated_vectors)

        # Classification: [num_obj, num_anchors, 1, 1]
        anchor_probs = self.anchor_prob(fc)

        # Regression: [num_obj, num_anchors, traj_len, 5]
        anchor_mean_var = self.anchor_mean_var(fc)

        # Quantization.
        anchor_probs = self.dequant(anchor_probs)
        anchor_mean_var = self.dequant(anchor_mean_var)
        return anchor_probs, anchor_mean_var

    def _extract_roi(self, out_feat, batch_img_coords, batch_offsets):
        """Extract region of interests.

        Args:
            out_feat (torch.Tensor, [B, C, H, W]): backbone output features.
            batch_img_coords (List[Tuple]): list of image coordinates tuples of
                prediction objects.
            batch_offsets (torch.Tensor, [num_obj, H, W, 2]): the offsets
                from out_feat to centric_patches based on region on interests.

        Returns:
            centric_patches (torch.Tensor): extracted features based on region
                of interests.
        """
        out_feat = out_feat.split(1)
        batch_feats = []
        for batch_idx in range(len(batch_img_coords)):
            img_coords = batch_img_coords[batch_idx]
            for _ in img_coords:
                batch_feats.append(out_feat[batch_idx])
        batch_feats = self.cat_op.cat(batch_feats)
        centric_patches = self.rroi_layer(batch_feats, batch_offsets)
        return centric_patches

    def _extract_test_model_roi(self, out_feat, batch_offsets):
        """Extract region of interests for the test model.

        Args:
            out_feat (torch.Tensor, [1, C, H, W]): backbone output features.
            batch_offsets (torch.Tensor, [num_obj, H, W, 2]): the offsets
                from out_feat to centric_patches based on region on interests.

        Returns:
            centric_patches (torch.Tensor): extracted features based on region
                of interests.
        """
        num_obj = batch_offsets.shape[0]
        batch_feats = [out_feat for _ in range(num_obj)]
        batch_feats = self.cat_op.cat(batch_feats)
        batch_offsets = self.roi_quant(batch_offsets)
        centric_patches = self.rroi_layer(batch_feats, batch_offsets)
        return centric_patches

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.fc,
            [["0", "1"], ["2", "3"]],
            inplace=True,
            fuser_func=horizon.quantization.fuse_known_modules,
        )
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
