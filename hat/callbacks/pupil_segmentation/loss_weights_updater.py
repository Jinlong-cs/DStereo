import logging

import numpy as np

from hat.registry import OBJECT_REGISTRY
from ..callbacks import CallbackMixin

__all__ = ["LossWeightUpdater"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class LossWeightUpdater(CallbackMixin):
    r"""Update loss weight.

    weight = cur_epoch / num_epochs
    """

    def _update_loss_weight(self, x, xlims, ylims, offset):
        r"""Update loss weight.

        Given xlims (x_min, x_max) and ylims (y_min, y_max),
        i.e, start and end, compute the value of y = f(x).
        Offset contains the x0 such that for all x < x0,
        y is clipped to y_min.
        """

        if x < offset:
            return ylims[0]
        elif x > xlims[1]:
            return ylims[1]
        else:
            y = (np.diff(ylims) / np.diff(xlims)) * (x - offset)
            return y.item()

    def on_epoch_begin(self, model, epoch_id, **kwargs):
        num_epochs = model.module.losses.loss_weights["num_epochs"]
        cur_w = self._update_loss_weight(epoch_id, (0, num_epochs), (0, 1), 0)
        model.module.losses.loss_weights["surface_loss_ratio"] = cur_w
