"""output module structure, mainly used in GraphModel."""
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from .output_module import OutputModule


@OBJECT_REGISTRY.register
class FreezeOutputAttibuteGrad(nn.Module):  # noqa: D205,D400
    """Freeze Output head's Attibute Grad for cyclt train.

    Args:
        target_attribute_flag: Target class property, which defined task.
        head_attribute_name: wait to be freezed, should in range of
            head_channels.
    """

    def __init__(
        self,
        target_attribute_flag: str = None,
        head_attribute_name: list = None,
    ):
        super(FreezeOutputAttibuteGrad, self).__init__()

        self.target_attribute_flag = target_attribute_flag
        self.head_attribute_name = head_attribute_name
        self.set_flag = False

    def forward(
        self,
        head: torch.nn.Module,
        target: object,
    ):
        if self.head_attribute_name is None:
            return

        assert hasattr(
            target, self.target_attribute_flag
        ), f"{target.__class__.__name__} has no attribute:\
                {self.target_attribute_flag}"
        assert isinstance(self.head_attribute_name, (list, set))
        if self.set_flag:
            return

        self.set_flag = getattr(target, self.target_attribute_flag)

        for head_out in self.head_attribute_name:
            if hasattr(head, head_out):
                head_module = getattr(head, head_out)
                if isinstance(head_module, nn.Module):
                    for param in head_module.parameters():
                        param.requires_grad = self.set_flag
                else:
                    raise NotImplementedError(
                        f"{head_out} excepted nn.Module, \
                            but got {type(head_module)}"
                    )
            else:
                raise NotImplementedError(
                    f"{head.__class__.__name__} has no property: \
                            {head_out}"
                )


@OBJECT_REGISTRY.register
class MTFCOS3DOutputModule(OutputModule):  # noqa: D205,D400
    """MTFCOS3DOutputModule construct mtfcos3d output module in a network
    with head, target, loss and etc.

    Args:
        head: head layers config.
        loss: loss config.
        target: target computing module config.
        postprocess: post processing module config.
        head_freeze: module freeze head's attibute grad
        head_parser: parse head output, optional.
        prefix: prefix of current module, mainly used in multitask model.
            You can ignore it if you only have one output module. Or if you
            set, prefix will appear in returning dict's key name.
        keep_name: whether add prefix and suffix to key
            when call func "convert_to_dict_with_name" with a dict input.
        roi_decoder: decoder for roi task.
        roi_task_key: do roi decode if roi_task_key in input label else not.
        output_fpn_feats: output fpn feats  if set True.
        fpn_dequant: fpn feats dequant module when tracing module.
        fpn_desc: fpn feats desc module.

    """

    def __init__(
        self,
        head: torch.nn.Module,
        loss: Optional[torch.nn.Module] = None,
        target: Optional[torch.nn.Module] = None,
        head_freeze: Optional[torch.nn.Module] = None,
        postprocess: Optional[torch.nn.Module] = None,
        head_parser: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = None,
        keep_name: Optional[bool] = False,
        roi_decoder: Optional[torch.nn.Module] = None,
        roi_task_key: Optional[str] = None,
        output_fpn_feats: Optional[bool] = False,
        fpn_dequant: Optional[torch.nn.Module] = None,
        fpn_desc: Optional[torch.nn.Module] = None,
    ):
        super(MTFCOS3DOutputModule, self).__init__(
            head, loss, target, postprocess, head_parser, prefix, keep_name
        )

        self.roi_decoder = roi_decoder
        self.head_freeze = head_freeze
        self.aux_seg_head = None
        self.roi_task_key = roi_task_key
        self.output_fpn_feats = output_fpn_feats
        self.fpn_desc = fpn_desc
        self.fpn_dequant = fpn_dequant

    @property
    def with_postprocess(self) -> bool:
        return self.postprocess is not None

    @property
    def has_target(self) -> bool:
        return self.target is not None

    @property
    def with_loss(self) -> bool:
        return self.loss is not None

    def forward(
        self, x, label: Optional[Any] = None, aux_head: Optional[Any] = None
    ) -> Dict:
        result = OrderedDict()
        pred = self.head(x)
        if self.head_parser is not None:
            pred = self.head_parser(pred)

        # roi task
        if (
            self.roi_task_key is not None
            and label is not None
            and self.roi_task_key in label
        ):
            return self.roi_decoder(pred)

        if label is not None and self.has_target:
            assert (
                not torch.jit.is_scripting()
            ), "this codepath is not supported"
            if aux_head is not None:
                if self.aux_seg_head is None:
                    self.aux_seg_head = deepcopy(aux_head)
                aux_pred: torch.Tensor = aux_head.head(x)
                if aux_head.head_parser is not None:
                    aux_pred = aux_head.head_parser(aux_pred)
                    aux_pred = F.softmax(aux_pred, dim=1).max(dim=1)[1]
                label["gt_seg"] = aux_pred.clone().detach()

            target = self.target(label, pred)
            if self.head_freeze is not None:
                self.head_freeze(self.head, self.target)
        else:
            target = label

        if self.with_loss:
            assert (
                not torch.jit.is_scripting()
            ), "this codepath is not supported"
            loss = self.loss(pred, target)
            result.update(loss)
            if self.aux_seg_head is not None and aux_head is not None:
                aux_seg_loss = self.aux_seg_head(x, label)
                result.update(aux_seg_loss)

        if self.with_postprocess:
            if type(self.postprocess) == nn.ModuleList:
                for _, each_postprocess in enumerate(self.postprocess):
                    pred = each_postprocess(pred, label)
            else:
                pred = self.postprocess(pred, label)

        if self.output_fpn_feats:
            fpn_feats = self.fpn_dequant(x)
            if self.fpn_desc is not None:
                fpn_feats = self.fpn_desc(fpn_feats)
            result.update(fpn_desc_out=fpn_feats)

        result.update(pred)

        return result
