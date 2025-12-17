# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Callable, Dict, Optional, Tuple

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.core.traj_pred_viz_utils import load_anchors
from hat.models.task_modules.traj_pred.backbones.vectornet_backbone import (
    SubGraphLayer,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager

__all__ = [
    "BasicMlpDecoder",
    "BasicAnchorBasedDecoder",
    "BasicTrackValidDecoder",
    "BasicBehavDecoder",
]


@OBJECT_REGISTRY.register
class BasicMlpDecoder(nn.Module):
    """A simple MLP decoder for trajectory prediction."""

    def __init__(
        self,
        in_channels: int,
        traj_len: int,
        hidden_size: int,
        num_hidden_layers: int = 3,
        enable_scale_trils: bool = False,
        cor_clamp: float = 0.9,
        log_std_clamp_min: float = -2,
        log_std_clamp_max: float = 2,
        is_int_infer_model: bool = False,
        loss: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels of the features.
            traj_len: the length of the trajectory prediction results.
            hidden_size: the hidden size.
            num_hidden_layers: the number of hidden layers.
            enable_scale_trils: whether to output covariance matrices.
            cor_clamp: a value to clamp gussian correlation oefficient output.
            log_std_clamp_min: a min value to clamp gaussian std output.
            log_std_clamp_max: a max value to clamp gaussian std output.
            is_int_infer_model: whether the model is for int inference.
            loss: the customized loss function of this head.
            post_process: the customized post processing function of this
                head.
        """
        super(BasicMlpDecoder, self).__init__()
        assert num_hidden_layers > 0, "The number of hidden layers should > 0."
        self.in_channels = in_channels
        self.traj_len = traj_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.enable_scale_trils = enable_scale_trils
        self.cor_clamp = cor_clamp
        self.log_std_clamp_max = log_std_clamp_max
        self.log_std_clamp_min = log_std_clamp_min
        self.is_int_infer_model = is_int_infer_model
        self.loss = loss
        self.post_process = post_process

        if enable_scale_trils:
            self.out_channels = traj_len * 5
            self.num_out_feat = 5
        else:
            self.out_channels = traj_len * 2
            self.num_out_feat = 2

        mlp_layers = []
        use_relu_list = [True for _ in range(num_hidden_layers - 1)] + [False]
        layer_channels = [hidden_size for _ in range(num_hidden_layers - 1)]
        in_c = [in_channels] + layer_channels
        out_c = layer_channels + [self.out_channels]
        for use_relu, ic, oc in zip(use_relu_list, in_c, out_c):
            mlp_layers.append(
                SubGraphLayer(
                    in_channels=ic,
                    out_channels=oc,
                    num_vec=1,
                    use_relu=use_relu,
                    use_pool=False,
                )
            )
        self.mlp = nn.Sequential(*mlp_layers)
        self.dequant = DeQuantStub()

    def _process_pred_result(self, data: torch.Tensor):
        """Process the model output to generate trajectories.

        Args:
            data ([num_obs, 1, 1, traj_len * 5]): the original output of
                the mlp predictor. Here, `5` means (mean_x, mean_y,
                log_std_x, log_std_y, correlation)

        Returns:
            means (torch.Tensor, [num_obj, 1, traj_len, 2]): means of the
                trajectories (represents by two-dimensional gaussian
                distributions).
            scale_trils (torch.Tensor, [num_obj, 1, traj_len, 2, 2]):
                lower triangular matrices of two dimensional gaussian
                distributions. It is just enabled when `enable_scale_trils`
                is True.
        """
        num_obs = data.shape[0]
        mean_var = data.reshape([-1, self.num_out_feat])
        # means
        delta_xy = mean_var[:, 0:2]
        delta_xy = torch.reshape(delta_xy, [num_obs, -1, self.traj_len, 2])
        means = torch.cumsum(delta_xy, dim=2)

        # scale trils
        if self.enable_scale_trils:
            log_std_x = mean_var[:, 2:3].clamp(
                min=self.log_std_clamp_min, max=self.log_std_clamp_max
            )
            log_std_y = mean_var[:, 4:5].clamp(
                min=self.log_std_clamp_min, max=self.log_std_clamp_max
            )
            cor = self.cor_clamp * (2 * torch.sigmoid(mean_var[:, 3:4]) - 1)

            std_x = torch.exp(log_std_x)
            std_y = torch.exp(log_std_y)
            st_00 = std_x
            st_10 = cor * std_y
            st_11 = torch.sqrt(1 - cor * cor) * std_y
            scale_trils = torch.cat(
                [st_00, torch.zeros_like(st_00), st_10, st_11], -1
            )
            scale_trils = torch.reshape(
                scale_trils, [num_obs, -1, self.traj_len, 2, 2]
            )
            return means, scale_trils
        else:
            return means

    def forward(self, data: torch.Tensor):
        """Forward.

        Args:
            data ([num_obs, in_channels, 1, 1]): the features.

        Returns:
            results (Dict): prediction results that contain keys "means" and
                "scale_trils".
        """
        model_out = self.dequant(self.mlp(data))
        ret = self._process_pred_result(model_out)

        results = {}
        if self.enable_scale_trils:
            results["mean_var"] = ret[0]
            results["probabilities"] = ret[1]
        else:
            results["mean_var"] = ret
            results["probabilities"] = []

        return results


@OBJECT_REGISTRY.register
class BasicAnchorBasedDecoder(nn.Module):
    """The basic anchor-based trajectory prediction decoder."""

    def __init__(
        self,
        in_channels: int = 128,
        anchor_cfg: Optional[Dict] = None,
        n_hidden_layers: Tuple[int] = (1024, 1024),
        use_momentum: bool = False,
        cor_clamp: float = 0.9,
        log_std_clamp_min: float = -2,
        log_std_clamp_max: float = 2,
        is_int_infer_model: bool = False,
        loss: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels. This parameter can be
                int or Dict: 1) If the input of the head model is
                a single tensor, please set this parameter as int.
                2) If the input of the head (`data`) is a dict and
                the "real" input needs to concatenate different
                features in `data`, please set this parametr as a
                dict. The keys are the features in `data` and the
                values are the corresponding number of channels.
            anchor_cfg: the anchor configuration dictionary with the
                following keys:
                1. 'anchor_file': the file path of the pickle anchors.
                2. 'anchor_method': the method to obtain the anchors, it
                    should be one of [kmeans, uniform].
                3. 'anchor_num': the number of anchors.
            n_hidden_layers: a list contains the channels of the head
                fully-connected layers.
            use_momentum: whether to predict momentum instead of xy offsets.
            cor_clamp: a value to clamp gussian correlation coefficient
                output.
            log_std_clamp_min: a min value to clamp gaussian std output.
            log_std_clamp_max: a max value to clamp gaussian std output.
            is_int_infer_model: whether the model is for int inference.
            loss: the customized loss function of this head.
            post_process: the customized post processing function of this
                head.
        """
        super(BasicAnchorBasedDecoder, self).__init__()
        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(
                "Param n_hidden_layers must be a list, but received"
                f"{type(n_hidden_layers)}"
            )

        if type(in_channels) is int:
            self.in_channels = in_channels
            self.need_cat = False
        elif type(in_channels) is dict:
            self.in_channels = 0
            self.keys_to_cat = []
            for key, in_c in in_channels.items():
                self.in_channels += in_c
                self.keys_to_cat.append(key)

            self.need_cat = True
            self.feat_cat_op = nn.quantized.FloatFunctional()
        else:
            raise TypeError(
                "The param in_channels should be either int or dict."
            )

        self.n_hidden_layers = n_hidden_layers
        self.use_momentum = use_momentum
        self.cor_clamp = cor_clamp
        self.log_std_clamp_max = log_std_clamp_max
        self.log_std_clamp_min = log_std_clamp_min
        self.is_int_infer_model = is_int_infer_model
        self.loss = loss
        self.post_process = post_process

        # Load anchor trajectories.
        self.anchors = load_anchors(anchor_cfg)
        if self.anchors is not None:
            if type(self.anchors) is dict:
                for k, v in self.anchors.items():
                    setattr(self, k, v)
            else:
                self.num_anchors = self.anchors.shape[0]
                self.traj_len = self.anchors.shape[1]
        else:
            raise ValueError("The input anchor config is wrong!")

        self.build_basic_decoder_graph(self.in_channels)

    def build_basic_decoder_graph(self, input_channels: int):
        """Build the basic graph of the MLP decoder.

        Args:
            input_channels (int): the input channels.
        """
        # FC layers.
        n_hidden_layers = [input_channels] + self.n_hidden_layers
        nn_linear = [
            nn.Conv2d(in_hidden, out_hidden, kernel_size=1)
            for in_hidden, out_hidden in zip(
                n_hidden_layers[:-1], n_hidden_layers[1:]
            )
        ]
        activation = [nn.ReLU6() for _ in range(len(n_hidden_layers[1:]))]
        self.fc = []
        self.fc_fuse_list = []
        for idx, (linear, act) in enumerate(zip(nn_linear, activation)):
            self.fc.append(linear)
            self.fc.append(act)
            self.fc_fuse_list.append([str(2 * idx), str(2 * idx + 1)])
        self.fc = nn.Sequential(*self.fc)

        # Classification head.
        self.anchor_prob = nn.Conv2d(
            n_hidden_layers[-1], self.num_anchors, kernel_size=1
        )
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        # Gaussian regression head.
        self.anchor_mean_var = nn.Conv2d(
            n_hidden_layers[-1],
            self.num_anchors * self.traj_len * 5,
            kernel_size=1,
        )

        # Quantization and dequantization.
        self.dequant = DeQuantStub()

    def forward(self, data):
        """Forward.

        Args:
            data (torch.Tensor, [num_obs, in_channels, 1, 1]): the features.
                If the input `in_channels` of __init__ is dict, this param
                should be dict too (should include all keys in `in_channels`).

        Returns:
            results (Dict): prediction results that contain keys "means" and
                "scale_trils", "probabilities", "log_anchors_probs", "anchors".
        """
        if type(data) is dict and self.need_cat:
            used_data = [data[k] for k in self.keys_to_cat]
            fc_feats = self.feat_cat_op.cat(used_data, dim=1)
        else:
            fc_feats = data

        # FC: [num_obj, n_hidden, 1, 1]
        fc = self.fc(fc_feats)

        # Classification: [num_obj, num_anchors, 1, 1]
        anchor_probs = self.anchor_prob(fc)

        # Regression: [num_obj, num_anchors, traj_len, 5]
        anchor_mean_var = self.anchor_mean_var(fc)

        # Quantization.
        anchor_probs = self.dequant(anchor_probs)
        anchor_mean_var = self.dequant(anchor_mean_var)

        if self.is_int_infer_model:
            # TODO (shengzhe.dai): should change the head output
            # in other structures that support this decoder.
            results = {
                "probabilities": anchor_probs,
                "mean_var": anchor_mean_var,
            }
        else:
            anchor_probs = anchor_probs.requires_grad_(True)
            anchor_mean_var = anchor_mean_var.requires_grad_(True)
            if hasattr(self, "anchor_key"):
                anchors = data[self.anchor_key].cuda()
            else:
                anchors = self.anchors.cuda()[None, :, :, :]
            means, scale_trils = self._extract_gaussian(
                anchor_mean_var, anchors
            )
            results = {}
            results["anchor_probs"] = anchor_probs
            results["anchor_mean_var"] = anchor_mean_var
            anchor_probs = anchor_probs.reshape((-1, self.num_anchors))
            results["means"] = means
            results["scale_trils"] = scale_trils
            results["anchors"] = anchors
            results["probabilities"] = anchor_probs
            results["log_anchors_probs"] = self.log_softmax(
                results["probabilities"]
            )
        return results

    def _extract_gaussian(self, mean_var: torch.Tensor, anchors: torch.Tensor):
        """Convert a tensor to means and lower triangular matrices.

        Args:
            mean_var ([num_obj, num_anchors, traj_len, 5]): `5` means
                (mean_x, mean_y, log_std_x, log_std_y, correlation).
            anchors ([-1, n_anchors, traj_len, 2]): the anchors.

        Returns:
            means (torch.Tensor, [num_obj, num_anchors, traj_len, 2]): means
                of the two dimensional gaussian distributions.
            scale_trils (torch.Tensor, [num_obj, num_anchors, traj_len, 2, 2]):
                lower triangular matrices of two dimensional gaussian
                distributions.
        """
        self.num_anchors = anchors.size()[1]
        mean_var = torch.reshape(
            mean_var, (-1, self.num_anchors, self.traj_len, 5)
        )

        # Construct means and scale_trils.
        mean_var = torch.reshape(mean_var, (-1, 5))
        means = mean_var[:, 0:2]
        log_std_x = mean_var[:, 2:3].clamp(
            min=self.log_std_clamp_min, max=self.log_std_clamp_max
        )
        log_std_y = mean_var[:, 4:5].clamp(
            min=self.log_std_clamp_min, max=self.log_std_clamp_max
        )
        cor = self.cor_clamp * (2 * torch.sigmoid(mean_var[:, 3:4]) - 1)

        std_x = torch.exp(log_std_x)
        std_y = torch.exp(log_std_y)
        st_00 = std_x
        st_10 = cor * std_y
        st_11 = torch.sqrt(1 - cor * cor) * std_y
        scale_trils = torch.cat(
            [st_00, torch.zeros_like(st_00), st_10, st_11], -1
        )

        means = torch.reshape(means, (-1, self.num_anchors, self.traj_len, 2))
        scale_trils = torch.reshape(
            scale_trils, (-1, self.num_anchors, self.traj_len, 2, 2)
        )

        # Add anchors.
        if self.use_momentum:
            means = torch.cumsum(means, dim=2)
        means = means + anchors
        return means, scale_trils

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.fc,
            self.fc_fuse_list,
            inplace=True,
            fuser_func=horizon.quantization.fuse_known_modules,
        )

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        self.anchor_prob.qconfig = (
            horizon.quantization.get_default_qat_out_qconfig()
        )
        self.anchor_mean_var.qconfig = (
            horizon.quantization.get_default_qat_out_qconfig()
        )


@OBJECT_REGISTRY.register
class BasicTrackValidDecoder(nn.Module):
    """The basic track-valid-head decoder.

    This head aims to filter out some invalid prediction targets
    that are hard to find by rule-based filter functions.
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_hidden_layers: Tuple[int] = (1024, 1024),
        is_int_infer_model: bool = False,
        loss: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels. This parameter can be
                int or Dict: 1) If the input of the head model is
                a single tensor, please set this parameter as int.
                2) If the input of the head (`data`) is a dict and
                the "real" input needs to concatenate different
                features in `data`, please set this parametr as a
                dict. The keys are the features in `data` and the
                values are the corresponding number of channels.
            n_hidden_layers: a list contains the channels of the head
                fully-connected layers. Defaults to (1024, 1024).
            is_int_infer_model: whether the model is for int inference.
            loss: the customized loss function of this head.
            post_process: the customized post processing function of this
                head.
        """
        super(BasicTrackValidDecoder, self).__init__()
        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(
                "Param n_hidden_layers must be a list, but received"
                f"{type(n_hidden_layers)}"
            )
        self.n_hidden_layers = n_hidden_layers
        self.is_int_infer_model = is_int_infer_model
        self.loss = loss
        self.post_process = post_process

        if type(in_channels) is int:
            self.in_channels = in_channels
            self.need_cat = False
        elif type(in_channels) is dict:
            self.in_channels = 0
            self.keys_to_cat = []
            for key, in_c in in_channels.items():
                self.in_channels += in_c
                self.keys_to_cat.append(key)

            self.need_cat = True
            self.feat_cat_op = nn.quantized.FloatFunctional()
        else:
            raise TypeError(
                "The param in_channels should be either int or dict."
            )

        # FC layers.
        n_hidden_layers = [self.in_channels] + self.n_hidden_layers
        nn_linear = [
            nn.Conv2d(in_hidden, out_hidden, kernel_size=1)
            for in_hidden, out_hidden in zip(
                n_hidden_layers[:-1], n_hidden_layers[1:]
            )
        ]
        activation = [nn.ReLU6() for _ in range(len(n_hidden_layers[1:]))]
        self.fc = []
        self.fc_fuse_list = []
        for idx, (linear, act) in enumerate(zip(nn_linear, activation)):
            self.fc.append(linear)
            self.fc.append(act)
            self.fc_fuse_list.append([str(2 * idx), str(2 * idx + 1)])
        self.fc = nn.Sequential(*self.fc)

        # 2-class classification
        self.valid_prob = nn.Conv2d(n_hidden_layers[-1], 3, kernel_size=1)
        self.valid_prob_sig = nn.Sigmoid()

        # Quantization and dequantization.
        self.dequant = DeQuantStub()

    def forward(self, data: torch.Tensor):
        """Forward.

        Args:
            data (torch.Tensor, [num_obs, in_channels, 1, 1]): the features.
                If the input `in_channels` of __init__ is dict, this param
                should be dict too (should include all keys in `in_channels`).

        Returns:
            results (Dict): prediction results that contain keys
                "track_valid_cls".
        """
        if type(data) is dict and self.need_cat:
            used_data = [data[k] for k in self.keys_to_cat]
            data = self.feat_cat_op.cat(used_data, dim=1)

        # FC: [num_obj, n_hidden, 1, 1]
        fc = self.fc(data)

        # 2-class classification
        track_valid_cls = self.valid_prob_sig(self.valid_prob(fc))

        # Dequantization.
        track_valid_cls = self.dequant(track_valid_cls)

        results = {
            "track_valid_cls": track_valid_cls,
        }
        return results

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.fc,
            self.fc_fuse_list,
            inplace=True,
            fuser_func=horizon.quantization.fuse_known_modules,
        )

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class BasicBehavDecoder(nn.Module):
    """The basic decoupled lateral and longitudinal behavior decoder.

    This head is designed for viehcles' lateral and longitudinal traffic
    behaviors prediction. The lateral behaviors includes lane keep,
    left lane change and right lane change. The longitudinal behaviors
    includes keep velocity, speed up and slow down.
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_hidden_layers: Tuple[int] = (1024, 1024),
        is_int_infer_model: bool = False,
        loss: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels. This parameter can be
                int or Dict: 1) If the input of the head model is
                a single tensor, please set this parameter as int.
                2) If the input of the head (`data`) is a dict and
                the "real" input needs to concatenate different
                features in `data`, please set this parametr as a
                dict. The keys are the features in `data` and the
                values are the corresponding number of channels.
            n_hidden_layers: a list contains the channels of the head
                fully-connected layers. Defaults to (1024, 1024).
            is_int_infer_model: whether the model is for int inference.
            loss: the customized loss function of this head.
            post_process: the customized post processing function of this
                head.
        """
        super(BasicBehavDecoder, self).__init__()
        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(
                "Param n_hidden_layers must be a list, but received"
                f"{type(n_hidden_layers)}"
            )
        self.n_hidden_layers = n_hidden_layers
        self.is_int_infer_model = is_int_infer_model
        self.loss = loss
        self.post_process = post_process

        if type(in_channels) is int:
            self.in_channels = in_channels
            self.need_cat = False
        elif type(in_channels) is dict:
            self.in_channels = 0
            self.keys_to_cat = []
            for key, in_c in in_channels.items():
                self.in_channels += in_c
                self.keys_to_cat.append(key)

            self.need_cat = True
            self.feat_cat_op = nn.quantized.FloatFunctional()
        else:
            raise TypeError(
                "The param in_channels should be ether int or list."
            )

        if self.n_hidden_layers:
            # FC layers.
            n_hidden_layers = [self.in_channels] + self.n_hidden_layers
            nn_linear = [
                nn.Conv2d(in_hidden, out_hidden, kernel_size=1)
                for in_hidden, out_hidden in zip(
                    n_hidden_layers[:-1], n_hidden_layers[1:]
                )
            ]
            activation = [nn.ReLU() for _ in range(len(n_hidden_layers[1:]))]
            self.fc = []
            self.fc_fuse_list = []
            for idx, (linear, act) in enumerate(zip(nn_linear, activation)):
                self.fc.append(linear)
                self.fc.append(act)
                self.fc_fuse_list.append([str(2 * idx), str(2 * idx + 1)])
            self.fc = nn.Sequential(*self.fc)
            cls_pre_channels = n_hidden_layers[-1]
        else:
            cls_pre_channels = self.in_channels
        # lateral and longitudinal prediction
        self.lat_cls = nn.Conv2d(cls_pre_channels, 3, kernel_size=1)
        self.lon_cls = nn.Conv2d(cls_pre_channels, 3, kernel_size=1)

        # Quantization and dequantization.
        self.dequant = DeQuantStub()

    def forward(self, data: torch.Tensor):
        """Forward.

        Args:
            data (torch.Tensor, [num_obs, in_channels, 1, 1]): the features.
                If the input `in_channels` of __init__ is dict, this param
                should be dict too (should include all keys in `in_channels`).

        Returns:
            results (Dict): prediction results that contain keys
                "track_valid_cls".
        """
        if type(data) is dict and self.need_cat:
            used_data = [data[k] for k in self.keys_to_cat]
            data = self.feat_cat_op.cat(used_data, dim=1)

        if self.n_hidden_layers:
            # FC: [num_obj, n_hidden, 1, 1]
            fc = self.fc(data)
        else:
            fc = data

        # classification task
        lat_behav_probs = self.lat_cls(fc)
        lon_behav_probs = self.lon_cls(fc)

        # Dequantization.
        lat_behav_probs = self.dequant(lat_behav_probs)
        lon_behav_probs = self.dequant(lon_behav_probs)

        results = {
            "lat_behav_probs": lat_behav_probs,
            "lon_behav_probs": lon_behav_probs,
        }
        return results

    def fuse_model(self):
        if hasattr(self, "fc"):
            torch.quantization.fuse_modules(
                self.fc,
                self.fc_fuse_list,
                inplace=True,
                fuser_func=horizon.quantization.fuse_known_modules,
            )

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.lat_cls.qconfig = qconfig_manager.get_default_qat_out_qconfig()
