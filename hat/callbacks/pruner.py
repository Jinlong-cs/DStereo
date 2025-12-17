# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import types
from typing import Dict

try:
    from horizon_plugin_pytorch.prune import (
        SemistructedPruner,
        UnstructedPruner,
    )
except ImportError:
    UnstructedPruner = None
    SemistructedPruner = None

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import get_binding_module
from hat.utils.package_helper import require_packages
from .callbacks import CallbackMixin

__all__ = ["Pruner"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Pruner(CallbackMixin):
    """
    Pruner Callback is used for sparse training to imporve model performance.

    Args:
        pruner_type : Pruner type. only support "unstructed" or "semistructed".
        pconfig_dict : pruner config dict.

            .. code-block:: python

                # for semistructed pruner
                {
                    "allowed_module_names": [],  # only prune these modules.
                    "disallowed_module_names": [], # won't prune these modules.
                }
                # for unstructed pruner
                {
                    "": 0.5,  # global pruning rate.
                    "$module_name$": 0.5, # specific module pruning rate.
                    "gmp_step": 1, # gradual magnitude pruning step.
                                   # suggest to be 70% of total training step.
                    "gmp_update_times": 100, # gradual magnitude pruning update
                                             # times. suggest to be 100.
                }

        eval_func: Function to evaluate model precision after pruning.
            Won't do sensitivity analysis if it is None.
        sensitivity_output_path: Path to store the result of sensitivity.
    """

    @require_packages("horizon_plugin_pytorch>=1.9.1")
    def __init__(
        self,
        pruner_type: str,
        pconfig_dict: Dict = None,
        eval_func: types.FunctionType = None,
        sensitivity_output_path: str = "sparse_sensitivity",
    ):
        assert pruner_type in (
            "unstructed",
            "semistructed",
        ), "'pruner_type' must be 'unstructed' or 'semistructed'!"
        self.pruner_type = pruner_type
        self.pconfig_dict = pconfig_dict
        self.pruner = (
            UnstructedPruner()
            if pruner_type == "unstructed"
            else SemistructedPruner()
        )
        self.eval_func = eval_func
        self.sensitivity_output_path = sensitivity_output_path

    def on_loop_begin(self, model, optimizer, **kwargs):
        if self.eval_func is not None:
            self.pruner.sensitivity(
                get_binding_module(model),
                self.eval_func,
            )

        self.pruner.prune(
            get_binding_module(model),
            optimizer,
            self.pconfig_dict,
        )

    def on_epoch_begin(self, **kwargs):
        if self.pruner_type == "semistructed":
            # multi stage sparse
            self.pruner.update_mask()
