import logging
import os
from inspect import isfunction
from typing import Any

import torch
import torch.nn as nn
from horizon_plugin_pytorch import nn as hnn
from torch import Tensor

from hat.utils.apply_func import pytree_convert, to_device
from hat.utils.distributed import get_dist_info
from hat.utils.logger import MSGColor, format_msg

logger = logging.getLogger(__name__)

__all__ = [
    "set_deterministic_level",
    "deterministic_level",
    "maybe_cast_batch_for_deterministic",
    "get_hooked_modules",
    "get_all_modules",
    "get_hooked_ops",
    "get_all_ops",
    "deterministic_summary",
]

# see non-deterministic module and ops:
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html  # noqa E501
non_deterministic_modules = [
    # non-deterministic module in torch
    nn.AvgPool3d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.MaxPool3d,
    nn.AdaptiveMaxPool2d,
    nn.FractionalMaxPool2d,
    nn.FractionalMaxPool3d,
    nn.MaxUnpool1d,
    nn.MaxUnpool2d,
    nn.MaxUnpool3d,
    nn.ReflectionPad1d,
    nn.ReflectionPad2d,
    nn.ReflectionPad3d,
    nn.ReplicationPad1d,
    nn.ReplicationPad2d,
    nn.ReplicationPad3d,
    nn.NLLLoss,
    nn.CTCLoss,
    # non-deterministic module in horizon-plugin-pytorch
    hnn.MultiScaleRoIAlign,
    hnn.qat.MultiScaleRoIAlign,
    hnn.Interpolate,
    hnn.qat.Interpolate,
]

non_deterministic_ops = [
    # non-deterministic op in torch
    "torch.nn.functional.grid_sample",
    "torch.histc",
    "torch.nn.functional.interpolate",
    "torch.bincount",
    "torch.cumsum",
    "torch.Tensor.scatter_reduce",
    "torch.Tensor.resize_",
    "torch.Tensor.put_",
    # non-deterministic op in horizon-plugin-pytorch
    "autocasted_interpolate_outer",
    "horizon_plugin_pytorch.nn.interpolate.autocasted_interpolate_outer",
]


def deterministic_level():
    return int(os.getenv("HAT_DETERMINISTIC_LEVEL", "0"))


def set_deterministic_level(level: int = None):
    """Set deterministic level.

    Args:
        level: Deterministic level. Defaults to None.
    """
    if level:
        assert level in [0, 1, 2], (
            f"`deterministic_level` should be one of [0, 1, 2], " f"but get {level}."
        )
        os.environ["HAT_DETERMINISTIC_LEVEL"] = str(level)
    else:
        level = deterministic_level()

    if level > 0:
        torch.use_deterministic_algorithms(True)

    if level == 2:
        # set CUBLAS_WORKSPACE_CONFIG for `torch.mm`, `torch.mv`, `torch.bmm`
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html  # noqa E501
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def reset_deterministic_level():
    if os.getenv("HAT_DETERMINISTIC_LEVEL", None):
        del os.environ["HAT_DETERMINISTIC_LEVEL"]


hooked_modules_list = set()
modules_list = set()


def register_hooks_to_non_deterministic_modules(model: nn.Module):
    """Add hooks to non-deterministic module.

    Note:
        The hook will put non-deterministic modules running on cpu.

    Args:
        model: Model to add hooks.

    """
    global hooked_modules_list
    global modules_list

    def _forward_pre_device_hook(module, input):
        # move module and input to cpu
        module.cpu()
        input = to_device(input, device="cpu")
        return input

    def _forward_device_hook(module, input, output):
        # move output to cuda
        rank, _ = get_dist_info()
        output = to_device(output, device=rank)
        return output

    def _forward_pre_hook(module, input):
        NonDeterministicOpsTensor.disable_converting()
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            if module.inplace:
                return pytree_convert(
                    input,
                    NonDeterministicOpsTensor,
                    func=lambda x: x.clone(),
                )

        return input

    def _forward_hook(module, input, output):
        NonDeterministicOpsTensor.enable_converting()
        return output

    for name, module in model.named_modules():
        module_name = "{}.{}".format(module.__module__, module.__class__.__name__)

        if not isinstance(module, (nn.Sequential, nn.ModuleDict, nn.ModuleList)):
            modules_list.add(module_name)
            module.register_forward_pre_hook(_forward_pre_hook)
            module.register_forward_hook(_forward_hook)

        if isinstance(module, tuple(non_deterministic_modules)):
            module.register_forward_pre_hook(_forward_pre_device_hook)
            module.register_forward_hook(_forward_device_hook)
            hooked_modules_list.add((name, module_name))


def format_op_name(op):
    if isinstance(op, str):
        return op
    if isinstance(op, torch.nn.Module):
        return "{}.{}".format(op.__module__, op.__class__.__name__)
    if getattr(torch.Tensor, op.__name__, None) is op:
        return "torch.Tensor.{}".format(op.__name__)
    elif getattr(torch, op.__name__, None) is op:
        return "torch.{}".format(op.__name__)
    elif isfunction(op):
        return "{}.{}".format(op.__module__, op.__name__)
    else:
        return str(op)


class NonDeterministicOpsTensor(Tensor):
    _enabled = True
    _op_list = set()
    _hooked_op_list = set()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        func_name = format_op_name(func)
        if "getset_descriptor" not in str(func):
            cls._op_list.add(func_name)

        if cls._enabled and func_name in non_deterministic_ops:
            cls._hooked_op_list.add(func_name)
            args = pytree_convert(
                args,
                NonDeterministicOpsTensor,
                lambda x: x.as_subclass(torch.Tensor),
            )
            kwargs = pytree_convert(
                kwargs,
                NonDeterministicOpsTensor,
                lambda x: x.as_subclass(torch.Tensor),
            )
            # move to cpu
            args = to_device(args, device="cpu")
            kwargs = to_device(kwargs, device="cpu")

            types = (
                torch.Tensor if t is NonDeterministicOpsTensor else t for t in types
            )

            ret = Tensor.__torch_function__(func, types, args, kwargs)
            ret = pytree_convert(
                ret,
                torch.Tensor,
                lambda x: x.as_subclass(NonDeterministicOpsTensor),
            )

            # move back to cuda
            rank, _ = get_dist_info()
            ret = to_device(ret, device=rank)

            return ret
        else:
            return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def enable_converting(cls):
        cls._enabled = True

    @classmethod
    def disable_converting(cls):
        cls._enabled = False

    @classmethod
    def get_op_list(cls):
        return cls._op_list

    @classmethod
    def get_hooked_op_list(cls):
        return cls._hooked_op_list

    @classmethod
    def reset(cls):
        cls._op_list.clear()
        cls._hooked_op_list.clear()


def maybe_cast_batch_for_deterministic(batch: Any) -> Any:
    """Cast data to `NonDeterministicOpsTensor` if `deterministic_level==2`.

    Args:
        batch: Batch data.
    """
    if deterministic_level() == 2:
        batch = pytree_convert(
            batch,
            torch.Tensor,
            lambda x: x.as_subclass(NonDeterministicOpsTensor),  # noqa E501
        )

    return batch


def cast_model_to_deterministic(model: nn.Module):
    """Cast non-deterministic op and module in model to deterministic.

    Note:
        Here add hooks to non-deterministic op and non-deterministic module.
        In case, the hooked op and module will run on cpu.

    Args:
        model: Model to add hooks.
    """
    register_hooks_to_non_deterministic_modules(model)


def get_hooked_modules():
    """Get hooked modules in model."""
    global hooked_modules_list
    return hooked_modules_list


def get_all_modules():
    """Get all modules in model."""
    global modules_list
    return modules_list


def get_hooked_ops():
    """Get hooked ops in model."""
    return NonDeterministicOpsTensor.get_hooked_op_list()


def get_all_ops():
    """Get all ops in model."""
    return NonDeterministicOpsTensor.get_op_list()


def deterministic_summary():
    """Summary deterministic hook information."""
    all_ops, hooked_ops = get_all_ops(), get_hooked_ops()
    all_modules, hooked_modules = get_all_modules(), get_hooked_modules()

    msg = "\n" + "=" * 50 + "DETERMINISTIC HOOK SUMMARY" + "=" * 50 + "\n"
    msg += (
        f"\nAll ops in model: \n{all_ops}\n"
        + f"\nHooked the ops to deterministic: \n{hooked_ops}\n"
    )

    msg += (
        f"\nAll modules in model: \n{all_modules}\n"
        + f"\nHooked the modules to deterministic: \n{hooked_modules}\n"
    )

    logger.info(
        format_msg(
            msg=msg,
            color=MSGColor.GREEN,
        )
    )
