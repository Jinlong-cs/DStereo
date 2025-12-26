# Copyright (c) Horizon Robotics. All rights reserved.

import logging

from torch.nn.modules.batchnorm import _BatchNorm

from hat.registry import OBJECT_REGISTRY
from .callbacks import CallbackMixin

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FreezeBN(CallbackMixin):
    def __init__(
        self,
        freeze_affine: bool = False,
        unfreeze_step: int = -1,
        log_once: bool = True,
    ):
        self.freeze_affine = bool(freeze_affine)
        self.unfreeze_step = int(unfreeze_step)
        self.log_once = bool(log_once)
        self._applied = False
        self._disabled = False

    def _freeze(self, model):
        if model is None:
            return
        count = 0
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()
                if self.freeze_affine:
                    if module.weight is not None:
                        module.weight.requires_grad_(False)
                    if module.bias is not None:
                        module.bias.requires_grad_(False)
                count += 1
        if self.log_once and not self._applied:
            logger.info(
                "FreezeBN enabled on %d BN layers (freeze_affine=%s).",
                count,
                self.freeze_affine,
            )
        self._applied = True

    def _unfreeze(self, model):
        if model is None:
            return
        count = 0
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.train()
                if self.freeze_affine:
                    if module.weight is not None:
                        module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)
                count += 1
        if self.log_once:
            logger.info(
                "FreezeBN disabled after step %d on %d BN layers.",
                self.unfreeze_step,
                count,
            )
        self._disabled = True

    def _maybe_toggle(self, model, global_step_id=None):
        if self._disabled:
            return
        if self.unfreeze_step >= 0 and global_step_id is not None:
            if global_step_id >= self.unfreeze_step:
                self._unfreeze(model)
                return
        self._freeze(model)

    def on_loop_begin(self, model=None, **kwargs):
        self._maybe_toggle(model, kwargs.get("global_step_id"))

    def on_epoch_begin(self, model=None, global_step_id=None, **kwargs):
        self._maybe_toggle(model, global_step_id)

    def on_step_begin(self, model=None, global_step_id=None, **kwargs):
        if self._disabled:
            return
        if self.unfreeze_step >= 0 and global_step_id is not None:
            if global_step_id >= self.unfreeze_step:
                self._unfreeze(model)
