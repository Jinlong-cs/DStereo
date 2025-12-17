# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.task_modules.traj_pred.backbones.vectornet_backbone import (
    BasicGlobalGraph as SelfAttn,
)
from hat.models.task_modules.traj_pred.base_modules import LinearLN
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager

__all__ = ["SGNetEncoder", "SGNetCvaeDecoder"]


@OBJECT_REGISTRY.register
class SGNetEncoder(nn.Module):
    """Encoder of SGNet.

    This model is overchanged and has many differences from the origin \
    paper. The main differences refer to README.md.
    """

    def __init__(
        self,
        in_channels: int,
        enc_steps: int = 15,
        dec_steps: int = 45,
        hidden_size: int = 512,
        return_all_step: bool = False,
        is_calibration_step: bool = False,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels.
            enc_steps: encoder steps.
            dec_steps: decoder steps.
            hidden_size: the hidden size.
            return_all_step: whether to return results of all enc
                steps, if the value is False, the encoder will only
                return the result of the last enc step.
            is_calibration_step: whether the current stage is
                calibration. If True, the initialize function of
                LSTM will be changed as randn.
        """
        super(SGNetEncoder, self).__init__()
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps
        self.hidden_size = hidden_size
        self.return_all_step = return_all_step
        self.is_calibration_step = is_calibration_step
        self.goal_hidden_size = hidden_size // 4
        self.lstm_hidden_size = hidden_size + self.goal_hidden_size

        # For fuse model and set qconfig.
        model_fuse_map = {}
        model_list_fuse_map = {}
        model_list_list_fuse_map = {}
        model_int16_qconfig = []
        model_list_int16_qconfig = []

        # Define state init function.
        # -- If the current stage is calibration, it must be randn. Otherwise,
        # the calibration will get divide zero error.
        if is_calibration_step:
            self.lstm_state_init_func = torch.randn
        else:
            self.lstm_state_init_func = torch.zeros

        # Input features.
        self.enc_feat_layer = LinearLN(
            in_channels=in_channels,
            out_channels=hidden_size,
            use_relu=True,
            in_dim=4,
            out_dim=4,
        )

        # Encode basic lstm.
        self.enc_lstm_cell = nn.LSTMCell(self.lstm_hidden_size, hidden_size)
        self.enc_cat_ops = nn.ModuleList([FF() for _ in range(self.enc_steps)])
        model_list_int16_qconfig.append("enc_cat_ops")

        self.enc_hidden_2_goal_hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(
                            in_features=hidden_size,
                            out_features=self.goal_hidden_size,
                        ),
                        nn.ReLU(),
                    ]
                )
                for _ in range(self.enc_steps)
            ]
        )
        model_list_fuse_map["enc_hidden_2_goal_hidden_layers"] = [["0", "1"]]

        # Goal sub model lstm.
        self.goal_lstm_cell = nn.LSTMCell(
            self.goal_hidden_size, self.goal_hidden_size
        )
        self.goal_traj_cat_op = FF()
        self.goal_hidden_cat_op = nn.ModuleList(
            [FF() for _ in range(self.enc_steps)]
        )
        model_list_int16_qconfig.append("goal_hidden_cat_op")

        if self.return_all_step:
            self.all_goal_hidden_cat_op = FF()

        self.goal_hidden_2_goal_input_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            *[
                                nn.Linear(
                                    in_features=self.goal_hidden_size,
                                    out_features=self.goal_hidden_size,
                                ),
                                nn.ReLU(),
                            ]
                        )
                        for _ in range(self.dec_steps)
                    ]
                )
                for _ in range(self.enc_steps)
            ]
        )
        model_list_list_fuse_map["goal_hidden_2_goal_input_layers"] = [
            ["0", "1"]
        ]

        self.goal_hidden_2_traj_layer = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=self.goal_hidden_size,
                    out_channels=hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hidden_size,
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.Tanh(),
            ]
        )
        model_fuse_map["goal_hidden_2_traj_layer"] = [["0", "1"]]
        model_int16_qconfig.append("goal_hidden_2_traj_layer")

        self.goal_attns = nn.ModuleList(
            [
                SelfAttn(
                    in_channels=self.goal_hidden_size,
                    hidden_units=self.goal_hidden_size,
                    num_poly=dec_steps,
                    num_attention_heads=1,
                    use_layernorm=False,
                )
                for _ in range(self.enc_steps)
            ]
        )

        self.input_quant = QuantStub(scale=None)
        self.enc_lstm_input_quant = QuantStub(scale=None)
        self.enc_lstm_hid_quant = QuantStub(scale=None)
        self.enc_lstm_cell_quant = QuantStub(scale=None)
        self.goal_lstm_hid_quant = QuantStub(scale=None)
        self.goal_lstm_cell_quant = QuantStub(scale=None)

        # For fuse model.
        self.model_fuse_map = model_fuse_map
        self.model_list_fuse_map = model_list_fuse_map
        self.model_list_list_fuse_map = model_list_list_fuse_map

        # For set qconfig.
        self.model_int16_qconfig = model_int16_qconfig
        self.model_list_int16_qconfig = model_list_int16_qconfig

    def forward(self, data: torch.Tensor):  # noqa: D208
        """Forward.

        Args:
            data ([batch_size, in_channels, 1, enc_steps]): the input \
                features. Usually, the `in_channels` dimension take \
                historical perception results as input, but it can also \
                contain other features. \

        Returns:
            enc_ret ([batch_size, out_channels, H, W]): the results. \
            if self.enc_with_all_steps, the following tensors have \
            results in all encoder steps. \
                all_goal_trajs: tensor with shape \
                    [batch_size, hidden_size, 1, fut_steps]. \
                all_goal_hiddens: tensor with shape \
                    [batch_size, hidden_size, enc_steps, fut_steps]. \
                all_enc_hiddens: list of tensor (shape \
                    [batch_size, hidden_size]). \

            else, only contains the results in the last encoder step. \
                all_goal_trajs: tensor with shape \
                    [batch_size, hidden_size, 1, fut_steps]. \
                all_goal_hiddens: tensor with shape \
                    [batch_size, hidden_size, 1, fut_steps]. \
                all_enc_hiddens: [batch_size, hidden_size]. \
        """
        data = self.input_quant(data)
        batch_size = data.shape[0]
        device = data.device

        # Extract input feature.
        # shape [batch_size, num_hidden, 1, his_steps]
        enc_feat = self.enc_feat_layer(data)

        # Traverse all history steps and run encoder LSTM.
        goal2enc_input = self.enc_lstm_input_quant(
            torch.zeros(
                [batch_size, self.goal_hidden_size, 1, 1], device=device
            )
        )
        enc_h_shape = [batch_size, self.hidden_size]
        enc_hidden_tuple = [
            self.enc_lstm_hid_quant(
                self.lstm_state_init_func(enc_h_shape, device=device)
            ),
            self.enc_lstm_cell_quant(
                self.lstm_state_init_func(enc_h_shape, device=device)
            ),
        ]

        if self.return_all_step:
            all_goal_hiddens = []
            all_enc_hiddens = []
        else:
            last_goal_hidden = None
            last_enc_hidden = None

        for step in range(0, self.enc_steps):
            tmp_step_feat = enc_feat[:, :, :, step : (step + 1)]
            step_cat_feat = (
                self.enc_cat_ops[step]
                .cat([tmp_step_feat, goal2enc_input], dim=1)
                .reshape([batch_size, -1])
            )
            enc_hidden_tuple = self.enc_lstm_cell(
                step_cat_feat, enc_hidden_tuple
            )
            # enc_h: [batch_size, hidden_size]
            enc_h = enc_hidden_tuple[0]
            # Traverse fut_steps to get goal LSTM feature.
            # goal2enc_input: [batch_size, num_hidden, 1, 1]
            # goal_h: [batch_size, num_hidden, 1, fut_steps])
            goal2enc_input, goal_h = self.SGE_forward(enc_h, batch_size, step)

            if self.return_all_step:
                all_goal_hiddens.append(goal_h)
                all_enc_hiddens.append(enc_h)
            elif step == self.enc_steps - 1:
                last_goal_hidden = goal_h
                last_enc_hidden = enc_h

        if self.return_all_step:
            # all_goal_hiddens: [batch_size, num_hidden, enc_steps, fut_steps]
            all_goal_hiddens = self.all_goal_hidden_cat_op.cat(
                all_goal_hiddens, dim=2
            )
            # all_goal_trajs: [batch_size, num_hidden, 1, fut_steps]
            all_goal_trajs = self.goal_hidden_2_traj_layer(all_goal_hiddens)
            all_goal_trajs = all_goal_trajs[:, :, -1:, :]

            enc_ret = {
                "all_goal_trajs": all_goal_trajs,
                "all_goal_hiddens": all_goal_hiddens,
                "all_enc_hiddens": all_enc_hiddens,
            }
        else:
            goal_trajs = self.goal_hidden_2_traj_layer(last_goal_hidden)
            enc_ret = {
                "all_goal_trajs": goal_trajs,
                "all_goal_hiddens": last_goal_hidden,
                "all_enc_hiddens": last_enc_hidden,
            }

        return enc_ret

    def SGE_forward(
        self, enc_h: torch.Tensor, batch_size: int, step: int = 0
    ):  # noqa: D208
        """SGE forward.

        Args:
            enc_h ([batch_size, hidden_size]): the encoder hidden. \
            batch_size: the batch_size. \
            step: the current step of encoding iteration. \

        Returns:
            goal_attn ([batch_size, num_hidden, 1, 1]): goal attention \
            goal_hidden ([batch_size, num_hidden, 1, fut_steps]): the \
                goal hidden of all steps. \
        """
        tmp_goal_attn_layer = self.goal_attns[step]
        tmp_enc_h_2_goal_h = self.enc_hidden_2_goal_hidden_layers[step]
        goal_h_shape = [batch_size, self.goal_hidden_size]
        device = enc_h.device
        goal_hidden_tuple = [
            self.goal_lstm_hid_quant(
                self.lstm_state_init_func(goal_h_shape, device=device)
            ),
            self.goal_lstm_cell_quant(
                self.lstm_state_init_func(goal_h_shape, device=device)
            ),
        ]
        goal_input = tmp_enc_h_2_goal_h(enc_h)
        goal_h_list = []
        goal_hiddens = []
        for i in range(self.dec_steps):
            tmp_goal_h_2_goal_i = self.goal_hidden_2_goal_input_layers[step][i]
            goal_hidden_tuple = self.goal_lstm_cell(
                goal_input, goal_hidden_tuple
            )
            goal_h = goal_hidden_tuple[0]
            goal_h_list.append(goal_h)
            # goal_input: [batch_size, goal_hidden_size]
            # goal_hiddens: [batch_size, goal_hidden_size, 1, 1]
            goal_input = tmp_goal_h_2_goal_i(goal_h)
            goal_hiddens.append(goal_h.reshape([batch_size, -1, 1, 1]))

        # Here we change the net as self attention to integrate goal hiddens in
        # different time stamps.
        # [batch_size, hidden, 1, fut_steps] -> [batch_size, num_hidden, 1, 1]
        goal_hiddens = self.goal_hidden_cat_op[step].cat(goal_hiddens, dim=3)
        goal_attn = tmp_goal_attn_layer(goal_hiddens)
        return goal_attn, goal_hiddens

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.input_quant.qconfig = torch.quantization.QConfig(
            activation=horizon.quantization.fake_quantize.default_16bit_fake_quant,  # noqa
            weight=None,
        )

        for module in self.goal_attns:
            module.set_qconfig()

        for key in self.model_int16_qconfig:
            tmp_layer = getattr(self, key)
            tmp_layer.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )

        for key in self.model_list_int16_qconfig:
            tmp_mod_list = getattr(self, key)
            for module in tmp_mod_list:
                module.qconfig = horizon.quantization.get_default_qat_qconfig(
                    dtype="qint16"
                )

    def fuse_model(self):
        for module in [self.enc_feat_layer]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

        for key, item in self.model_fuse_map.items():
            tmp_layer = getattr(self, key)
            torch.quantization.fuse_modules(
                tmp_layer,
                item,
                inplace=True,
                fuser_func=horizon.quantization.fuse_known_modules,
            )

        for key, item in self.model_list_fuse_map.items():
            tmp_mod_list = getattr(self, key)
            for module in tmp_mod_list:
                torch.quantization.fuse_modules(
                    module,
                    item,
                    inplace=True,
                    fuser_func=horizon.quantization.fuse_known_modules,
                )

        for key, item in self.model_list_list_fuse_map.items():
            tmp_mod_list = getattr(self, key)
            for mod_list in tmp_mod_list:
                for module in mod_list:
                    torch.quantization.fuse_modules(
                        module,
                        item,
                        inplace=True,
                        fuser_func=horizon.quantization.fuse_known_modules,
                    )


@OBJECT_REGISTRY.register
class SGNetCvaeDecoder(nn.Module):
    """CVAE decoder of SGNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        enc_steps: int = 15,
        dec_steps: int = 45,
        latent_dim: int = 32,
        k_value: int = 20,
        enc_with_all_steps: bool = False,
        is_calibration_step: bool = False,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels.
            out_channles: the output channels.
            hidden_size: the hidden size.
            enc_steps: encoder steps.
            dec_steps: decoder steps.
            latent_dim: the latent dimension of cvae features.
            k_value: the num of output trajectories for one target.
            enc_with_all_steps: whether the encoder module outputs
                hidden tensor for all enc steps.
            is_calibration_step: whether the current stage is
                calibration. If True, the initialize function of
                LSTM will be changed as randn.
        """
        super(SGNetCvaeDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps
        self.latent_dim = latent_dim
        self.k_value = k_value
        self.enc_with_all_steps = enc_with_all_steps
        self.is_calibration_step = is_calibration_step
        self.esp = 1e-4

        model_int16_qconfig = []
        model_fuse_map = {}

        # Define state init function.
        # -- If the current stage is calibration, it must be randn. Otherwise,
        # the calibration will get divide zero error.
        if is_calibration_step:
            self.lstm_state_init_func = torch.randn
        else:
            self.lstm_state_init_func = torch.zeros

        # CVAE module
        self.cvae_feat_layer = LinearLN(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            use_relu=True,
            in_dim=2,
            out_dim=4,
        )
        # DIFF: here we use a linear instead of (GRU cell in SGNet) to
        # reduce computation.
        self.cvae_tar_feat_layer = LinearLN(
            in_channels=out_channels,
            out_channels=self.hidden_size,
            use_relu=True,
            in_dim=4,
            out_dim=4,
        )

        # If the k_value > 1, there is no need to output probabilities.
        if self.k_value > 1:
            self.kprobs_layer = LinearLN(
                in_channels=self.hidden_size,
                out_channels=1,
                use_relu=True,
                in_dim=4,
                out_dim=4,
            )

        channels = [128, 64, latent_dim * 2]
        p_fc_channels = [self.hidden_size] + channels
        p_fc_use_relu = [True] * (len(channels) - 1) + [False]
        p_net = []
        p_net_fuse_map = []
        for p_idx, (in_c, out_c, use_relu) in enumerate(
            zip(p_fc_channels[:-1], p_fc_channels[1:], p_fc_use_relu)
        ):
            p_net.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            if use_relu:
                p_net.append(nn.ReLU())
                p_net_fuse_map.append([str(p_idx * 2), str(p_idx * 2 + 1)])
        self.p_net = nn.Sequential(*p_net)
        model_fuse_map["p_net"] = p_net_fuse_map

        q_fc_channels = [self.hidden_size * 2] + channels
        q_fc_use_relu = [True] * (len(channels) - 1) + [False]
        q_net = []
        q_net_fuse_map = []
        for q_idx, (in_c, out_c, use_relu) in enumerate(
            zip(q_fc_channels[:-1], q_fc_channels[1:], q_fc_use_relu)
        ):
            q_net.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            if use_relu:
                q_net.append(nn.ReLU())
                q_net_fuse_map.append([str(q_idx * 2), str(q_idx * 2 + 1)])
        self.q_net = nn.Sequential(*q_net)
        model_fuse_map["q_net"] = q_net_fuse_map

        self.cvae_2_decoder_layer = LinearLN(
            in_channels=self.hidden_size + self.latent_dim,
            out_channels=self.hidden_size,
            use_relu=True,
            in_dim=4,
            out_dim=4,
        )
        self.cvae_seed_quant = QuantStub(scale=None)
        self.target_quant = QuantStub(scale=None)
        q_float_ops = [
            "q_cat_op",
            "cvae_feat_mul_op",
            "cvae_feat_add_op",
            "cvae_feat_cat_op",
        ]
        for key in q_float_ops:
            setattr(self, key, FF())

        # Decoder module.
        self.goal_hidden_size = self.hidden_size // 4
        self.dec_lstm_hidden_size = self.hidden_size + self.goal_hidden_size
        self.cvae_goal_feat_layer = LinearLN(
            in_channels=self.goal_hidden_size,
            out_channels=self.goal_hidden_size,
            use_relu=True,
            in_dim=4,
            out_dim=4,
        )
        self.dec_hidden_tuple_2_traj_layer = nn.Sequential(
            *[
                LinearLN(
                    in_channels=hidden_size,
                    out_channels=out_channels,
                    use_relu=False,
                    in_dim=4,
                    out_dim=4,
                ),
                nn.Tanh(),
            ]
        )
        model_int16_qconfig.append("dec_hidden_tuple_2_traj_layer")

        self.dec_lstm_cell = nn.LSTMCell(
            self.dec_lstm_hidden_size, self.hidden_size
        )
        self.dec_goal_attn = nn.ModuleList(
            [
                SelfAttn(
                    in_channels=self.goal_hidden_size,
                    hidden_units=self.goal_hidden_size,
                    num_poly=dec_steps,
                    num_attention_heads=1,
                    use_layernorm=False,
                )
                for _ in range(self.dec_steps)
            ]
        )
        self.dec_goal_cat_op = nn.ModuleList(
            [FF() for _ in range(self.dec_steps)]
        )
        self.dec_attn_cat_op = nn.ModuleList(
            [FF() for _ in range(self.dec_steps)]
        )
        self.dec_h_cat_op = nn.ModuleList(
            [FF() for _ in range(self.dec_steps)]
        )
        self.dec_lstm_input_quant = QuantStub(scale=None)
        self.dec_lstm_hid_quant = QuantStub(scale=None)
        self.dec_lstm_cell_quant = QuantStub(scale=None)
        self.dequant = DeQuantStub()
        self.traj_dequant = DeQuantStub()
        if self.k_value > 1:
            self.prob_dequant = DeQuantStub()

        # For fuse and set qconfig.
        self.model_fuse_map = model_fuse_map
        self.model_int16_qconfig = model_int16_qconfig

    def forward(self, data: Dict):  # noqa: D208
        """Forward.

        Args:
            data: includes the following keys. \
            if self.enc_with_all_steps, the following tensors have \
            results in all encoder steps. \
                all_goal_trajs: tensor with shape \
                    [batch_size, hidden_size, 1, fut_steps]. \
                all_goal_hiddens: tensor with shape \
                    [batch_size, hidden_size, enc_steps, fut_steps]. \
                all_enc_hiddens: list of tensor (shape \
                    [batch_size, hidden_size]). \

            else, only contains the results in the last encoder step. \
                all_goal_trajs: tensor with shape \
                    [batch_size, hidden_size, 1, fut_steps]. \
                all_goal_hiddens: tensor with shape \
                    [batch_size, hidden_size, 1, fut_steps]. \
                all_enc_hiddens: [batch_size, hidden_size]. \

        Returns:
            dec_trajs ([batch_size, k_values, out_channels, dec_steps]): the \
                predicted trajectories. \
            KLD: the KL-Divergence between the distribution of p and q. \
            probabilities ([batch_size, 1, 1, k_values]): probabilities. \
        """
        rand_cvae_seed = data["rand_cvae_seed"]
        tar_goal = data["target_traj"]
        if self.enc_with_all_steps:
            enc_h = data["all_enc_hiddens"][-1]
            goal_h = data["all_goal_hiddens"][
                :, :, (self.enc_steps - 1) : self.enc_steps, :
            ]
        else:
            enc_h = data["all_enc_hiddens"]
            goal_h = data["all_goal_hiddens"]
        batch_size = enc_h.shape[0]

        rand_cvae_seed = self.cvae_seed_quant(rand_cvae_seed)
        if tar_goal is not None:
            tar_goal = self.target_quant(tar_goal)

        cvae_feat = self.cvae_feat_layer(enc_h)

        if tar_goal is not None:
            tar_goal = self.cvae_tar_feat_layer(tar_goal)
        (
            cvae_mu,
            cvae_std,
            cvae_feat,
            cvae_mu_p,
            cvae_logvar_p,
        ) = self.cal_cvae_feat(cvae_feat, rand_cvae_seed, tar_goal)

        enc_h = enc_h.reshape([batch_size, -1, 1, 1])
        if self.k_value > 1:
            enc_h = enc_h.repeat(1, 1, 1, self.k_value)

        enc_h = self.cvae_feat_cat_op.cat([enc_h, cvae_feat], dim=1)

        cvae_dec_hidden = self.cvae_2_decoder_layer(enc_h)
        dec_trajs = self.cvae_decoder_forward(cvae_dec_hidden, goal_h)
        dec_trajs = self.traj_dequant(dec_trajs)
        cvae_mu = self.dequant(cvae_mu)
        cvae_std = self.dequant(cvae_std)
        cvae_mu_p = self.dequant(cvae_mu_p)
        cvae_logvar_p = self.dequant(cvae_logvar_p)

        if tar_goal is not None:
            KLD = 0.5 * (
                (cvae_std.exp() / (cvae_logvar_p.exp() + self.esp))
                + (cvae_mu_p - cvae_mu).pow(2)
                / (cvae_logvar_p.exp() + self.esp)
                - 1
                + (cvae_logvar_p - cvae_std)
            )
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            KLD = torch.as_tensor(0.0, device=cvae_logvar_p.device)

        if self.k_value > 1:
            probabilities = self.kprobs_layer(cvae_dec_hidden)
            probabilities = self.prob_dequant(probabilities)
            return dec_trajs, KLD, probabilities
        else:
            return dec_trajs, KLD

    def cvae_decoder_forward(self, cvae_dec_hidden, goal_hidden):
        """Forward function of cvae decoder.

        Args:
            cvae_dec_hidden ([batch_size, num_feats, 1, k_values]): cvae
                decoder hidden tensor.
            goal_hidden ([batch_size, num_feats_2, enc_steps, dec_steps]): the
                goal hidden tensor.

        Returns:
            all_dec_trajs ([batch_size, k_values, out_channels, dec_steps]):
                the trajectory results of the decoder.
        """
        goal_for_dec = self.cvae_goal_feat_layer(goal_hidden)
        batch_size = cvae_dec_hidden.shape[0]
        device = cvae_dec_hidden.device
        fake_batch_size = batch_size * self.k_value
        dec_h_shape = [fake_batch_size, self.hidden_size]
        dec_hidden_tuple = [
            self.dec_lstm_hid_quant(
                self.lstm_state_init_func(dec_h_shape, device=device)
            ),
            self.dec_lstm_cell_quant(
                self.lstm_state_init_func(dec_h_shape, device=device)
            ),
        ]

        zero_goal_h = self.dec_lstm_input_quant(
            torch.zeros(goal_for_dec.shape, device=device)
        )
        all_dec_trajs = []
        all_dec_hiddens = []
        for step in range(self.dec_steps):
            cat_feat = []
            if step > 0:
                cat_feat.append(zero_goal_h[:, :, :, :step])
            if step < self.dec_steps:
                cat_feat.append(goal_for_dec[:, :, :, step:])
            tmp_goal_h = self.dec_goal_cat_op[step].cat(cat_feat, dim=-1)
            # [batch_size, global_hidden_size, 1, 1]
            goal_attn = self.dec_goal_attn[step](tmp_goal_h)
            # [batch_size, global_hidden_size, 1, k_values]
            if self.k_value > 1:
                goal_attn = goal_attn.repeat(1, 1, 1, self.k_value)
            # [batch_size, lstm_hidden_size, 1, k_values]
            dec_lstm_input = (
                self.dec_attn_cat_op[step]
                .cat((cvae_dec_hidden, goal_attn), dim=1)
                .permute(0, 3, 1, 2)
            )
            # [batch_size * k_values, lstm_hidden_size]
            dec_lstm_input = dec_lstm_input.reshape([fake_batch_size, -1])
            dec_hidden_tuple = self.dec_lstm_cell(
                dec_lstm_input, dec_hidden_tuple
            )
            all_dec_hiddens.append(
                dec_hidden_tuple[0].reshape([fake_batch_size, -1, 1, 1])
            )

        # [batch_size * k_values, lstm_hidden_size, 1, dec_steps]
        all_dec_hiddens = self.dec_h_cat_op[step].cat(all_dec_hiddens, dim=3)
        # [batch_size * k_values, out_channels, 1, dec_steps]
        all_dec_trajs = self.dec_hidden_tuple_2_traj_layer(all_dec_hiddens)
        # [batch_size, k_values, out_channels, dec_steps]
        all_dec_trajs = all_dec_trajs.reshape(
            [batch_size, self.k_value, self.out_channels, self.dec_steps]
        )
        return all_dec_trajs

    def cal_cvae_feat(self, feat, rand_cvae_seed, target_feat=None):
        """Calculate cvae features.

        Args:
            feat ([batch_size, num_feats, 1, 1]): the feature to generate
                cvae features.
            rand_cvae_seed (batch_size, 1, 1, k_values): random seeds to
                perturb the original features and generate cvae features.
            target_feat ([batch_size, num_feats_2, 1, 1]): the feature
                extracted from target goal tensor (usually be something
                like ground-truth). This input is only used during
                training.

        Returns:
            z_mu ([batch_size, latent_dim, 1, k_value]): the mean of the
                cvae distribution.
            z_std ([batch_size, latent_dim, 1, k_value]): the std of the
                cvae distribution.
            z_ts ([batch_size, latent_dim, 1, k_value]): the output feats
                used cvae module.
            z_mu_p ([batch_size, latent_dim, 1, k_value]): the mean of the
                cvae distribution of the `p_net`.
            z_logvar_p ([batch_size, latent_dim, 1, k_value]): the std of the
                cvae distribution of the `p_net`.
        """
        # [batch_size, num_feats, 1, 1]
        z_mu_logvar_p = self.p_net(feat)
        z_mu_p = z_mu_logvar_p[:, : self.latent_dim, :, :]
        z_logvar_p = z_mu_logvar_p[:, self.latent_dim :, :, :]
        if target_feat is None:
            z_mu = z_mu_p
            z_std = z_logvar_p
        else:
            q_feat = self.q_cat_op.cat(
                [feat, target_feat[:, :, :, -1:]], dim=1
            )
            z_mu_logvar_q = self.q_net(q_feat)
            z_mu = z_mu_logvar_q[:, : self.latent_dim, :, :]
            z_std = z_mu_logvar_q[:, self.latent_dim :, :, :]

        if self.k_value > 1:
            z_mu_ts = z_mu.repeat(1, 1, 1, self.k_value)
            z_std_ts = z_std.repeat(1, 1, 1, self.k_value)
        else:
            z_mu_ts = z_mu
            z_std_ts = z_std
        z_ts = self.cvae_feat_mul_op.mul(rand_cvae_seed, z_std_ts)
        z_ts = self.cvae_feat_add_op.add(z_mu_ts, z_ts)

        return z_mu, z_std, z_ts, z_mu_p, z_logvar_p

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.target_quant.qconfig = torch.quantization.QConfig(
            activation=horizon.quantization.fake_quantize.default_16bit_fake_quant,  # noqa
            weight=None,
        )

        for key in self.model_int16_qconfig:
            tmp_layer = getattr(self, key)
            tmp_layer.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )

        for module in self.dec_goal_attn:
            module.set_qconfig()

    def fuse_model(self):
        fuse_layers = [
            self.cvae_feat_layer,
            self.cvae_tar_feat_layer,
            self.cvae_2_decoder_layer,
            self.cvae_goal_feat_layer,
        ]
        if self.k_value > 1:
            fuse_layers.append(self.kprobs_layer)
        for module in fuse_layers:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

        for key, item in self.model_fuse_map.items():
            tmp_layer = getattr(self, key)
            torch.quantization.fuse_modules(
                tmp_layer,
                item,
                inplace=True,
                fuser_func=horizon.quantization.fuse_known_modules,
            )
