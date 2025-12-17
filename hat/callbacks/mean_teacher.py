# Copyright (c) Horizon Robotics. All rights reserved.

import itertools
import logging
import math

import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import get_binding_module
from .callbacks import CallbackMixin

__all__ = ["MeanTeacher"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class MeanTeacher(CallbackMixin):  # noqa: D400
    def __init__(
        self,
        decay: float = 0.9999,
        decay_base: float = 2000,
    ):
        """Callbacks of MeanTeacher.

        Some training algorithms, such as GradientDescent and Momentum
        often benefit from maintaining a moving average of variables
        during optimization. Using the moving averages for evaluations
        often improve results significantly.
        https://arxiv.org/abs/1703.01780

        Args:
            decay: Decay ratio for ema.
            decay_base: Decay base for ema.

        """
        self.decay = lambda x: decay * (1 - math.exp(-x / decay_base))

    def update(self, global_step_id, model):
        with torch.no_grad():
            d = self.decay(global_step_id + 1)
            biding_model = get_binding_module(model)
            msd = biding_model.state_dict()

            for k, v in itertools.chain(
                biding_model.named_parameters(), biding_model.named_buffers()
            ):  # TODO(kris.ji)：是否要使用EMA更新teacher BN的mean，var
                if k.startswith("teacher_"):
                    if "num_batches_tracked" in k:  # 跳过num_batches_tracked
                        continue
                    v *= d
                    stu_k = k.replace("teacher", "student")
                    v += (1.0 - d) * msd[stu_k].detach()

    def on_step_end(
        self,
        global_step_id,
        model=None,
        ema_model=None,
        **kwargs,
    ):
        self.update(global_step_id, model)

    def on_loop_begin(
        self,
        loop,
        model,
        **kwargs,
    ):
        # if not hasattr(loop, "ema_model"):
        #     logger.warning("loop has no ema_model, ema is disabled.")
        #     return

        biding_model = get_binding_module(model)
        for k, v in itertools.chain(
            biding_model.named_parameters(), biding_model.named_buffers()
        ):
            if k.startswith("teacher"):
                v.requires_grad_(False)  # teacher model不使用BP更新weights
                # v.eval()  #TODO(kris.ji): 是否固定BN running-mean，running-var
