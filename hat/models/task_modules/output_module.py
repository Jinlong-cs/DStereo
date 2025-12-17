# Copyright (c) Horizon Robotics. All rights reserved.
"""output module structure, mainly used in GraphModel."""
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from hatbc.utils import _as_list
    from hatbc.workflow import make_traceable
except ImportError:
    _as_list = None
    make_traceable = property

from hat.registry import OBJECT_REGISTRY


@make_traceable
def convert_to_dict_with_name(
    data: Any,
    suffix: str,
    prefix: str,
    keep_name: bool = False,
) -> Dict[str, Any]:
    """Convert data to dict format."""
    # TODO(min.du, 0.5): fix below bug #
    # if torch._C._get_tracing_state() and this is the most outer
    #  Module, then return dict is forbidden.
    if isinstance(data, dict):
        if keep_name:
            return data
        else:
            return {f"{prefix}_{suffix}_{k}": v for k, v in data.items()}

    elif isinstance(data, (torch.Tensor, list, tuple)):
        name = prefix + "_" + suffix
        return {name: data}
    else:
        raise ValueError(f"Not support return type: {type(data)}")


@OBJECT_REGISTRY.register
class OutputModule(nn.Module):  # noqa: D205,D400
    r"""
    OutputModule construct output module in a network with head, target, loss
    and etc.

    Data flow is showed below:

        input -> head() ->「head_parser()」-> pred -> postprocess() -> pred_result
                                               |
                                               |
                                              \|/
        label -------->「target()」--------> loss() -> loss

    Args:
        head: head layers config.
        loss: loss config.
        target: target computing module config.
        postprocess: post processing module config.
        head_parser: parse head output, optional.
        prefix: prefix of current module, mainly used in multitask model.
            You can ignore it if you only have one output module. Or if you
            set, prefix will appear in returning dict's key name.
        keep_name: whether add prefix and suffix to key
            when call func "convert_to_dict_with_name" with a dict input.
        convert_to_dict: whether convert pred into dict.
        do_fuse: whether to fuse module in qat conversion, when you use the
            module more than once in a model, the flag should be false
            except the first time.
    """  # noqa: E501

    def __init__(
        self,
        head: torch.nn.Module,
        loss: Optional[torch.nn.Module] = None,
        target: Optional[torch.nn.Module] = None,
        postprocess: Optional[torch.nn.Module] = None,
        head_parser: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = None,
        keep_name: Optional[bool] = False,
        do_fuse: Optional[bool] = True,
        convert_to_dict: Optional[bool] = True,
        trace_convert: Optional[bool] = False,
    ):
        super(OutputModule, self).__init__()
        self.head = head
        self.do_fuse = do_fuse
        self.loss = loss
        self.target = target
        if isinstance(postprocess, (list, tuple)):
            postprocess = nn.ModuleList(postprocess)
        self.postprocess = postprocess
        self.head_parser = head_parser

        self.prefix = prefix if prefix else self.__class__.__name__
        self.keep_name = keep_name
        self.convert_to_dict = convert_to_dict
        self.trace_convert = trace_convert

    @property
    def with_postprocess(self) -> bool:
        return self.postprocess is not None

    @property
    def has_target(self) -> bool:
        return self.target is not None

    @property
    def with_loss(self) -> bool:
        return self.loss is not None

    def forward(self, x, label: Optional[Any] = None) -> Dict:
        result = OrderedDict()
        pred = self.head(x)

        if self.head_parser is not None:
            pred = self.head_parser(pred)

        if label is not None and self.target is not None:
            assert not torch.jit.is_scripting()
            target = self.target(label, pred)
        else:
            target = label

        if self.loss is not None:
            assert not torch.jit.is_scripting()
            loss = self.loss(pred, target)
            loss_suffix = ""
            if isinstance(loss, torch.Tensor):
                loss_suffix = self.loss.__class__.__name__
            if not self.trace_convert:
                loss = self.convert_to_dict_with_name(loss, loss_suffix)
            else:
                loss = convert_to_dict_with_name(
                    loss, loss_suffix, self.prefix, self.keep_name
                )
            result.update(loss)

        if self.postprocess is not None:
            if type(self.postprocess) == nn.ModuleList:
                for _idx, each_postprocess in enumerate(self.postprocess):
                    pred = each_postprocess(pred, label)
            else:
                pred = self.postprocess(pred, label)

        if not self.convert_to_dict:
            return pred
        if not self.trace_convert:
            pred = self.convert_to_dict_with_name(pred, "predict")
        else:
            pred = convert_to_dict_with_name(
                pred, "predict", self.prefix, self.keep_name
            )
        result.update(pred)
        if target is not None:
            if not self.trace_convert:
                target = self.convert_to_dict_with_name(target, "target")
            else:
                target = convert_to_dict_with_name(
                    target, "target", self.prefix, self.keep_name
                )
            result.update(target)
        return result

    @torch.jit.unused
    def convert_to_dict_with_name(
        self,
        data: Any,
        suffix: str,
    ) -> Dict[str, Any]:
        """Convert data to dict format."""
        # TODO(min.du, 0.5): fix below bug #
        # if torch._C._get_tracing_state() and this is the most outer
        #  Module, then return dict is forbidden.
        if isinstance(data, dict):
            if self.keep_name:
                return data
            else:
                return {
                    f"{self.prefix}_{suffix}_{k}": v for k, v in data.items()
                }

        elif isinstance(data, (torch.Tensor, list, tuple)):
            name = self.prefix + "_" + suffix
            return {name: data}
        else:
            raise ValueError(f"Not support return type: {type(data)}")

    def fuse_model(self):
        if self.do_fuse:
            for module in [self.head, self.postprocess, self.head_parser]:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.head, self.postprocess, self.head_parser]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        if self.target is not None:
            self.target.qconfig = None
        if self.loss is not None:
            self.loss.qconfig = None


@OBJECT_REGISTRY.register
class MultiHeadOutputModule(nn.Module):
    """
    OutputModule construct that support multiple heads.

    Args:
        heads: Multiple stage2 heads modules.
    """

    def __init__(
        self,
        heads: List[nn.Module],
    ):
        super(MultiHeadOutputModule, self).__init__()

        self.heads = _as_list(heads)

    def forward(self, feats, inputs=None):

        total_bev_preds = OrderedDict()

        for head in self.heads:

            bev_pred = head(
                feats,
                inputs,
            )
            total_bev_preds.update(bev_pred)

        return total_bev_preds
