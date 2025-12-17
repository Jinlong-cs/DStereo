# Copyright (c) Horizon Robotics. All rights reserved.
from __future__ import division
from typing import Callable, List, Optional

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["MDPValueDecoder"]


@OBJECT_REGISTRY.register
class MDPValueDecoder(nn.Module):
    """
    Decoder goal-conditioned reward map into a fixed horizon policy. \
    This method is inspired by VIN(Value iteration network) and there \
    are no learnable parameters in this head.

    Args:
        action_len : dim of grid actions which affects channel of the policy.
        mdp_horizon : horizon of grid action sequence.
        initial_state : initial ego position on grid map.
        grid_dim : height/width of the grid feature map.
        value : initial value number.
        loss: head losses.
        post_process : the post-processing model.
    """

    def __init__(
        self,
        action_len: int,
        mdp_horizon: int,
        initial_state: List[int],
        grid_dim: List[int],
        value: int = -100,
        loss: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        super(MDPValueDecoder, self).__init__()

        self.action_len = action_len
        self.mdp_horizon = mdp_horizon
        self.initial_state = initial_state
        self.grid_dim = grid_dim
        self.value = value
        self.goal_value = -2 * value
        self.loss = loss
        self.post_process = post_process

        # initialize grid action conv parameters
        self._init_backward_transition()
        self._init_forward_transition()

        # Non-linearities:
        # self.exp_op = horizon.nn.SegmentLUT(
        #   lambda x: -torch.exp(-x),
        #   is_centrosymmetric=True, auto_divide_strategy="curvature")  # noqa
        self.exp_op = nn.ModuleList(
            [
                horizon.nn.SegmentLUT(
                    torch.exp,
                    auto_divide_strategy="evenly",
                    input_range=[-12, 0],
                )
                for _ in range(self.mdp_horizon)
            ]
        )
        self.exp_op1 = nn.ModuleList(
            [
                horizon.nn.SegmentLUT(
                    torch.exp,
                    auto_divide_strategy="evenly",
                    input_range=[-10, 0],
                )
                for _ in range(self.mdp_horizon)
            ]
        )

        self.goal_quant = QuantStub()
        self.v_quant = QuantStub()
        self.svf_quant = QuantStub()

    def forward(self, data):
        """Forward.

        Args:
            data (dict): the model input with the following required keys:
                reward: reward feature generated from backbone.
                plan_svf_goal: binary ego navigation goal map.

        Return:
            results (Dict): features including:
            pi : fix horizon behavior policy.
            svf: state visited frequency by policy.
            svf_seg: state visited frequency of each step
        """

        reward = data["reward"]
        goal_r = data["plan_svf_goal"]
        goal_r = self.goal_quant(goal_r)

        pi = self.mdp_backward(reward, goal_r)
        svf_seg = self.mdp_forward(pi)

        svf = self.fw_sum_op.sum(svf_seg, dim=1, keepdim=True)

        results = {"pi": pi, "svf": svf, "svf_seg": svf_seg}

        return results

    def mdp_backward(self, path_r, goal_r):
        """Bellman backward value iteration process to generate policy.

        Args:
            path_r: reward feature map from backbone.
            goal_r: binary ego navigation goal map.

        Return:
            pi ([batch_size, horizon*action, h, w]): entire grid policy
        """
        pi_list = []
        v = self.v_quant(self.ini_v)
        v = v.repeat(goal_r.size(0), 1, 1, 1)

        goal_r = goal_r.mul(self.goal_value)
        goal_r = self.add_op_goal.add(goal_r, v)

        for k in range(self.mdp_horizon):
            v = self.cat_op[k].cat((v, goal_r), dim=1)
            q = self.add_op[k].add(path_r, self.backward(self.pad(v)))
            v, pi = self._log_sum_exp(k, q)

            pi_list.append(pi)

        pi = self.cat_op2.cat(pi_list[::-1], dim=1)

        return pi

    def _log_sum_exp(self, k, q):
        """Qat version of torch.logsumexp.

        Args:
            k: current iteration.
            q: Q function generated from feature map.

        Return:
            v: single step value map.
            pi: ([batch_size, action, h, w]): single step policy.
        """
        q_max, _ = torch.max(q, dim=1, keepdim=True)

        q_sub = self.sub_op[k].sub(q, q_max)
        q_sub = torch.clamp(q_sub, min=-12, max=0)

        q_exp = self.sum_op[k].sum(self.exp_op[k](q_sub), dim=1, keepdim=True)
        v = self.add_op2[k].add(self.log_op[k](q_exp), q_max)
        pi_log = self.sub_op2[k].sub(q, v)

        # clamp to increase the LUT acc
        pi_log = torch.clamp(pi_log, min=-10, max=0)
        pi = self.exp_op1[k](pi_log)

        return v, pi

    def mdp_forward(self, pi_list):
        """Forward mdp diffuse process to generate state visited frequency \
            in a raster to be used as planning path.

        Args:
            pi_list: entire gird policy generated from Bellman
                     iteration process.

        Return:
            svf_seg ([batch_size, horizon, h, w]): state visited
                      frequency map.
        """
        svf_t = self.svf_quant(self.svf_start)
        batch_size = pi_list.size(0)
        svf_t = svf_t.repeat(batch_size, 1, 1, 1)

        svft = []
        for ind in range(self.mdp_horizon):

            pi = pi_list[
                :,
                (ind * self.action_len) : ((ind + 1) * self.action_len),
                :,
                :,
            ]  # noqa

            # TODO: abs is to prevent overflow in int infer model
            pi = horizon.abs(pi)
            pi_t_next = self.forward_pi(pi)
            svf_t_next = self.forward_svf[ind](svf_t)

            d_next = self.fw_mul[ind].mul(pi_t_next, svf_t_next)
            svf_t = self.fw_sum[ind].sum(d_next, dim=1, keepdim=True)

            svft.append(svf_t)

        svf_seg = self.fw_cat_op.cat(svft, dim=1)

        return svf_seg

    def _init_backward_transition(self):
        """Initialize bellman iteration process conv as in VIN."""
        self.backward = nn.Conv2d(2, self.action_len, 3, stride=1, bias=False)
        self.backward.weight.requires_grad = False
        self.backward.weight.data.zero_()
        # Actions: [D, R, U, L]
        self.backward.weight[0, 0, 2, 1] = 1.0
        self.backward.weight[1, 0, 1, 2] = 1.0
        self.backward.weight[2, 0, 0, 1] = 1.0
        self.backward.weight[3, 0, 1, 0] = 1.0
        # Actions: [DR, UR, DL, UL, end]
        self.backward.weight[4, 0, 2, 2] = 1.0
        self.backward.weight[5, 0, 0, 2] = 1.0
        self.backward.weight[6, 0, 2, 0] = 1.0
        self.backward.weight[7, 0, 0, 0] = 1.0
        self.backward.weight[8, 1, 1, 1] = 1.0

        self.pad = nn.ConstantPad2d(1, self.value)
        self.ini_v = nn.Parameter(
            torch.zeros(1, 1, self.grid_dim[0], self.grid_dim[1])
        )
        self.ini_v.requires_grad = False
        self.ini_v.data.fill_(self.value)

        self.add_op_goal = nn.quantized.FloatFunctional()
        self.cat_op = nn.ModuleList(
            [nn.quantized.FloatFunctional() for _ in range(self.mdp_horizon)]
        )
        self.add_op = nn.ModuleList(
            [nn.quantized.FloatFunctional() for _ in range(self.mdp_horizon)]
        )
        self.add_op2 = nn.ModuleList(
            [nn.quantized.FloatFunctional() for _ in range(self.mdp_horizon)]
        )
        self.sub_op = nn.ModuleList(
            [
                horizon.nn.quantized.FloatFunctional()
                for _ in range(self.mdp_horizon)
            ]
        )
        self.sub_op2 = nn.ModuleList(
            [
                horizon.nn.quantized.FloatFunctional()
                for _ in range(self.mdp_horizon)
            ]
        )
        self.sum_op = nn.ModuleList(
            [
                horizon.nn.quantized.FloatFunctional()
                for _ in range(self.mdp_horizon)
            ]
        )
        self.log_op = nn.ModuleList(
            [horizon.nn.HardLog() for _ in range(self.mdp_horizon)]
        )
        self.cat_op2 = nn.quantized.FloatFunctional()

        return

    def _init_forward_transition(self):
        """Initialize forward diffuse process conv as in VIN."""
        self.forward_pi = nn.Conv2d(
            self.action_len,
            self.action_len,
            3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.forward_pi.weight.requires_grad = False
        self.forward_pi.weight.data.zero_()
        self.forward_pi.weight[2, 2, 2, 1] = 1.0
        self.forward_pi.weight[3, 3, 1, 2] = 1.0
        self.forward_pi.weight[0, 0, 0, 1] = 1.0
        self.forward_pi.weight[1, 1, 1, 0] = 1.0

        self.forward_pi.weight[7, 7, 2, 2] = 1.0
        self.forward_pi.weight[6, 6, 0, 2] = 1.0
        self.forward_pi.weight[5, 5, 2, 0] = 1.0
        self.forward_pi.weight[4, 4, 0, 0] = 1.0
        # self.forward_pi.weight[8, 8, 1, 1] = 1.0

        self.forward_svf = nn.ModuleList(
            [
                nn.Conv2d(
                    1, self.action_len, 3, stride=1, padding=1, bias=False
                )
                for _ in range(self.mdp_horizon)
            ]
        )  # noqa
        for forward_svf in self.forward_svf:
            forward_svf.weight.requires_grad = False
            forward_svf.weight.data.zero_()
            forward_svf.weight[2, 0, 2, 1] = 1.0
            forward_svf.weight[3, 0, 1, 2] = 1.0
            forward_svf.weight[0, 0, 0, 1] = 1.0
            forward_svf.weight[1, 0, 1, 0] = 1.0

            forward_svf.weight[7, 0, 2, 2] = 1.0
            forward_svf.weight[6, 0, 0, 2] = 1.0
            forward_svf.weight[5, 0, 2, 0] = 1.0
            forward_svf.weight[4, 0, 0, 0] = 1.0
            # self.forward_svf.weight[8, 0, 1, 1] = 1.0

        self.svf_start = nn.Parameter(
            torch.zeros(1, 1, self.grid_dim[0], self.grid_dim[1])
        )
        self.svf_start.requires_grad = False
        self.svf_start[:, :, self.initial_state[0], self.initial_state[1]] = 1

        self.fw_mul = nn.ModuleList(
            [
                horizon.nn.quantized.FloatFunctional()
                for _ in range(self.mdp_horizon)
            ]
        )
        self.fw_sum = nn.ModuleList(
            [
                horizon.nn.quantized.FloatFunctional()
                for _ in range(self.mdp_horizon)
            ]
        )

        self.fw_cat_op = horizon.nn.quantized.FloatFunctional()
        self.fw_sum_op = horizon.nn.quantized.FloatFunctional()

        return

    def set_backward_config(self):
        self.add_op_goal.qconfig = (
            horizon.quantization.get_default_qat_qconfig("qint16")  # noqa
        )
        self.cat_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.pad.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.backward.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.add_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )
        self.sub_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )
        self.exp_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )
        self.exp_op1.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.sum_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )
        self.log_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )
        self.add_op2.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.sub_op2.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.cat_op2.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa

    def set_forward_config(self):
        self.forward_pi.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.forward_svf.qconfig = (
            horizon.quantization.get_default_qat_qconfig("qint16")  # noqa
        )
        self.fw_mul.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.fw_sum.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.fw_cat_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.fw_sum_op.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa
        self.svf_quant.qconfig = horizon.quantization.get_default_qat_qconfig(
            "qint16"
        )  # noqa

    def set_qconfig(self):
        self.set_backward_config()
        self.set_forward_config()

    def fuse_model(self):
        pass
