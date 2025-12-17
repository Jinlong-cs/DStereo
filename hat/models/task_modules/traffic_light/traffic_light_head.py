# Copyright (c) Horizon Robotics. All rights reserved.

import collections
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.rpn import RPNVarGNetHead
from hat.registry import OBJECT_REGISTRY
from ..anchor_module import AnchorModule

__all__ = ["TinyVarGNetV2LenAttrHead", "AnchorDetAttrModule"]


@OBJECT_REGISTRY.register
class AnchorDetAttrModule(AnchorModule):
    """The container of Anchor-based detector module.

    This class serves as the container of anchor-based detector,
    which takes feature maps as input and outputs predictions which
    applies onto self-generated anchor boxes.

    All the actaul calculations are implemented in component modules.

    Args:
        anchor_generator: Anchor generator module, generates anchors
            to which the offsets are applied.
        head: Anchor head network, transfroms input feature maps
            into predictions (like regression map and classification
            score in RPN).
        ext_feat: Extra feature module that processes input feature maps
            before head.
        target: Target generator module, generates training target given
            ground truth labels and anchor boxes.
        loss: Loss module, calculates training loss by comparing head
            predictions with training targets.
        postprocess: Postprocess module, applies predictions generated
            by head module onto anchors to get final prediction.
        desc: Desc module, adds user-defined description to prediction.
        output_head_out: Whether to output raw prediction of head module.
            Mostly used for the purpose of visualization.
        output_target: Wheter to output training target. This argument works
            only when loss is presented. Mostly used to
            calculate metrics.
        target_keys: Keys used to get ground truths from input data. When
            target generator is presented, these data are inputs to target
            generator. Otherwise, the data are directly fed into loss module
            as ground truths.
        target_opt_keys: Keys used to get optional ground truths from input
            data. Only works when target generator is presented.
    """

    def __init__(
        self,
        anchor_generator: nn.Module,
        head: nn.Module,
        ext_feat: Optional[nn.Module] = None,
        target: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        output_head_out: bool = False,
        output_target: bool = False,
        target_keys: Tuple[str] = ("gt_boxes", "gt_boxes_num"),
        target_opt_keys: Tuple[str] = (
            "ig_regions",
            "ig_regions_num",
        ),
    ):
        super().__init__(
            anchor_generator,
            head=head,
            ext_feat=ext_feat,
            target=target,
            loss=loss,
            postprocess=postprocess,
            desc=desc,
            output_head_out=output_head_out,
            output_target=output_target,
            target_keys=target_keys,
            target_opt_keys=target_opt_keys,
        )

    def forward(
        self,
        feat_maps: List[torch.Tensor],
        y: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        # optionally process input features with extra layers
        if self.with_ext_feat:
            feat_maps = self.ext_feat(feat_maps)

        # generate anchors
        mlvl_anchors = self.anchor_generator(feat_maps)

        # get prediction scores
        head_out = self.head(feat_maps)

        out_dict = OrderedDict()

        if self._output_head_out:
            out_dict.update(head_out)

        # apply prediction scores to anchors to get final predictions
        if self.with_postprocess:
            pred = self.postprocess(
                mlvl_anchors, head_out, y.get("im_hw", None)
            )
            if self.desc is not None:
                pred = self.desc(pred)
            out_dict.update(pred)

        # calculate loss between predictions and ground truths
        if self.with_loss:
            # generate targets on-the-fly
            if self.has_target:
                _, targets = self.target(
                    mlvl_anchors,
                    *[y[k] for k in self._target_keys],
                    **{k: y.get(k, None) for k in self._target_opt_keys},
                )
            # load targets direclty
            else:
                assert len(self._target_keys) == 1
                targets = y[self._target_keys[0]]

            if self._output_target:
                out_dict.update(targets)

            loss = self.loss(head_out, targets)
            out_dict.update(loss)

        return out_dict


@OBJECT_REGISTRY.register
class TinyVarGNetV2LenAttrHead(RPNVarGNetHead):
    def __init__(
        self,
        in_channels: Union[List[int], int],
        num_channels: List[int],
        num_anchors: List[int],
        feat_strides: List[int],
        num_classes: int,
        is_dim_match: bool,
        bn_kwargs: Dict,
        factor: float,
        group_base: int,
        output_shift: int = 4,
        attr_list: List = None,
        dequant_output: bool = True,
        mode: bool = "train",
    ):
        super(TinyVarGNetV2LenAttrHead, self).__init__(
            in_channels,
            num_channels,
            num_anchors,
            feat_strides,
            num_classes,
            is_dim_match,
            bn_kwargs,
            factor,
            group_base,
            output_shift,
        )
        self.attr_list = attr_list
        self.dequant_output = dequant_output
        for i, stride in enumerate(self.feat_strides):
            # attr classification
            for _, attr_type_ch in enumerate(attr_list):
                attr_type, attr_ch = attr_type_ch
                self.rpn_fc[f"stride_{stride}_{attr_type}"] = ConvModule2d(
                    in_channels=num_channels[i],
                    out_channels=attr_ch * num_anchors[i],
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(
                        attr_ch * num_anchors[i], **bn_kwargs
                    ),
                )
        self.mode = mode

    def forward(
        self, x: List[torch.TensorType]
    ) -> Dict[str, List[torch.Tensor]]:
        cls_pred, reg_pred, outputs = [], [], []
        attr_pred_dict = collections.defaultdict(list)

        for i, (stride, num_anchor) in enumerate(
            zip(self.feat_strides, self.num_anchors)
        ):

            rpn_feature = self.rpn_conv[f"stride_{stride}"](x[i])

            det_feature = self.rpn_fc[f"stride_{stride}"](rpn_feature)
            attr_feature_list = []
            for attr in self.attr_list:
                attr_feature_list.append(
                    self.rpn_fc[f"stride_{stride}_{attr[0]}"](rpn_feature)
                )

            # head output without slicing
            outputs.append(
                {
                    "det_feature": det_feature,
                    "attr_feature_list": attr_feature_list,
                }
            )

            # dequant to output to loss module
            if self.dequant_output:
                det_feature = self.dequant(det_feature)

            # slice output for loss computation, the order matters
            det_feature = det_feature.view(
                -1,
                num_anchor,
                det_feature.shape[1] // num_anchor,
                *det_feature.shape[2:],
            )
            _cls = det_feature[:, :, 4:].flatten(1, 2)
            _reg = det_feature[:, :, :4].flatten(1, 2)

            cls_pred.append(_cls)
            reg_pred.append(_reg)

            # attr classification
            attr_feature_list = [
                self.dequant(attr_feature)
                if self.dequant_output
                else attr_feature
                for attr_feature in attr_feature_list
            ]

            for attr, attr_feature in zip(self.attr_list, attr_feature_list):
                if self.mode != "test":
                    attr_feature = attr_feature.view(
                        -1,
                        num_anchor,
                        attr_feature.shape[1] // num_anchor,
                        *attr_feature.shape[2:],
                    ).flatten(1, 2)
                attr_pred_dict[attr[0]].append(attr_feature)

        return OrderedDict(
            rpn_head_out=outputs,
            rpn_cls_pred=cls_pred,
            rpn_reg_pred=reg_pred,
            rpn_attr_pred_dict=attr_pred_dict,
        )
