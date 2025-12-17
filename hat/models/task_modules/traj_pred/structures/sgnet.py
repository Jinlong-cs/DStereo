# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict, namedtuple

from torch import nn
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["SGNet"]


@OBJECT_REGISTRY.register
class SGNet(nn.Module):
    """SGNet structure.

    A prediction model for first-person view (FPV) scenes.
    """

    def __init__(
        self,
        backbone,
        head,
        neck=None,
        post_process=None,
        losses=None,
        k_value: int = 10,
        is_int_infer_model: bool = False,
    ):
        """Initialize method.

        Args:
            backbone: a dict for building backbone.
            head: a dict for building head.
            neck: a dict for building neck.
            post_process: the post-processing model.
            losses: the losses class.
            k_value: the number of predicted trajectories of one target. If
                the value = 1, the structure will perform single-modal
                prediction, and the output will not include probabilities.
            is_int_infer_model: whether the model is for int inference.
        """
        super(SGNet, self).__init__()
        self.is_int_infer_model = is_int_infer_model
        # Model structure.
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.post_process = post_process
        self.k_value = k_value
        self.multi_modal = self.k_value > 1

        # Loss.
        self.losses = None
        if losses is not None:
            self.losses = nn.ModuleList(_as_list(losses))

        # Quantization.
        self.dequant = DeQuantStub()

    def forward(self, data):  # noqa: D208
        """Forward.

        Args:
            data: (Dict): the model input dictionary with the following keys: \
                "input_x" (torch.Tensor, [batch_size, pred_dim, 1, \
                    enc_steps]): the input history trajectories. \
                "rand_cvae_seed" (torch.Tensor, [batch_size, latent_dim, 1,
                    k_value]): only enabled in multi-modal prediction. \
                "target_y" (torch.Tensor, [batch_size, pred_dim, 1,
                    dec_steps]): the gt trajectories. It can be None during
                    inferring. \

        Returns:
            The returns include the following main outputs: \
                "pred_traj" (torch.Tensor, [batch_size, k_values, \
                    out_channels, dec_steps]): the predicted trajectories. \
                "probabilities" (torch.Tensor, [batch_size, 1, 1, k_values]): \
                    the probabilities of different modals. If single-modal pred
                    is enabled, the output will not include this tensor. \
            If self.is_int_infer_model is False, the output will also include \
            some values for loss and metric calculation. \
        """
        if not self.is_int_infer_model:
            ctx_traj = data["input_traj"].cuda()
            rand_cvae_seed = data["rand_cvae_seed"].cuda()
            all_target_goal = data["target_traj"].cuda()
        else:
            ctx_traj = data["input_x"]
            rand_cvae_seed = data["rand_cvae_seed"]
            all_target_goal = None

        # Backbone.
        enc_ret = self.backbone(ctx_traj)
        enc_ret["rand_cvae_seed"] = rand_cvae_seed
        if "stage" in data and data["stage"][0] == "train":
            enc_ret["target_traj"] = all_target_goal
        else:
            enc_ret["target_traj"] = None

        # Neck.
        if self.neck is not None:
            enc_ret = self.neck(enc_ret)

        # Head.
        model_result = OrderedDict()
        if self.multi_modal:
            pred_traj, kld, prob = self.head(enc_ret)
            model_result["probabilities"] = prob
        else:
            pred_traj, kld = self.head(enc_ret)
        model_result["pred_traj"] = pred_traj

        if self.is_int_infer_model:
            if self.post_process is not None:
                model_result.update(self.post_process(model_result))
            SGNetOutput = namedtuple("SGNetOutput", model_result.keys())
            model_result = SGNetOutput(**model_result)

        # comment out this code when model veriy.
        if not self.is_int_infer_model:
            all_goal_trajs = self.dequant(enc_ret["all_goal_trajs"])
            cur_result = {
                "input_traj": ctx_traj,
                "all_goal_trajs": all_goal_trajs,
                "target_traj": all_target_goal,
                "KLD": kld,
                "stamp": data["stamp"],
                "id": data["id"],
                "date_token": data["date_token"],
                "enable_relative": data["enable_relative"],
                "stage": data["stage"],
                "position": data["position"],
                "height": data["height"],
                "vcs_vel": data["vcs_vel"],
                "global_vel": data["global_vel"],
            }
            model_result.update(cur_result)
            if "images" in data:
                model_result["images"] = data["images"]
            if "pack" in data:
                model_result["pack"] = data["pack"]

            if self.post_process is not None:
                model_result.update(self.post_process(model_result))

            if self.losses is not None and data["stage"][0] == "train":
                for loss in self.losses:
                    model_result.update(loss(model_result))

        return model_result

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.head]:
            if module is None:
                continue
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

        if self.losses is not None:
            for module in self.losses:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None
