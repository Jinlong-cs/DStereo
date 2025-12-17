from collections import OrderedDict
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hat.models.base_modules.postprocess import AnchorPostProcess, FilterModule
from hat.registry import OBJECT_REGISTRY

__all__ = ["PersonDetAnchorPostProcess"]


@OBJECT_REGISTRY.register
class PersonDetAnchorPostProcess(AnchorPostProcess):
    """Post process for anchor-based object detection models.

    The difference between PersonDetAnchorPostProcess and AnchorPostProcess
    is that the output of PersonDetAnchorPostProcess are dequantized.

    Args:
        input_key:
            Hashable object used to query detection output from input.
        num_classes: Class number. Should be the number of foreground
            classes.
        class_offsets: Offset to be added to output class index
            for each branch.
        use_clippings:
            Whether clip box to image size. If input is padded,
            you can clip box to real content by providing image size.
        image_hw: Fixed image size in (h, w),
            set to None if input have different sizes.
        nms_iou_threshold(float): IoU threshold for nms.
        pre_nms_top_k:
            Maximum number of bounding boxes in each image before nms.
        post_nms_top_k: Maximum number of output bounding boxes in each
            image.
        nms_margin: Only supress box2 when
            box1.score - box2.score > nms_margin
        box_filter_threshold:
            Default threshold to filter box by max score.
        nms_padding_mode: The way to pad bbox
            to match the number of output bounding bouxes to post_nms_top_k,
            can be None, "pad_zero" or "rollover".
        bbox_min_hw:
            Minimum height and width of selected bounding boxes.
        input_shift: Shift value for setting qconfig.
    """

    def __init__(
        self,
        input_key: Hashable,
        num_classes: int,
        class_offsets: List[int],
        use_clippings: bool,
        image_hw: Tuple[int, int],
        nms_iou_threshold: float,
        pre_nms_top_k: int,
        post_nms_top_k: int,
        nms_margin: float = 0.0,
        box_filter_threshold: float = 0.0,
        nms_padding_mode: Optional[str] = None,
        bbox_min_hw: Tuple[float, float] = (0, 0),
        input_shift: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_key,
            num_classes,
            class_offsets,
            use_clippings,
            image_hw,
            nms_iou_threshold,
            pre_nms_top_k,
            post_nms_top_k,
            nms_margin,
            box_filter_threshold,
            nms_padding_mode,
            bbox_min_hw,
        )
        self.input_shift = input_shift

    def forward(
        self,
        anchors: List[torch.Tensor],
        head_out: Dict[str, List[torch.Tensor]],
        im_hw: Optional[Tuple[int, int]] = None,
    ) -> List[List[Tuple[torch.Tensor]]]:
        """Forward method.

        The output keyed by "pred_boxes_out" is the float version of
        "pred_boxes", which is used in qat&pt inference.
        """
        device = anchors[0].device
        if self.input_shift is not None and hasattr(self._dpp, "in_scale"):
            self._dpp.register_buffer(
                "in_scale",
                torch.ones(1, dtype=torch.float32).to(device)
                / (1 << self.input_shift),
            )
        head_out = [t.detach() for t in head_out[self.input_key]]
        result_lst = self._dpp(head_out, anchors, image_sizes=im_hw)
        return OrderedDict(
            pred_boxes_out=[self.dequant(r[0]) for r in result_lst],
            pred_boxes=[self.dequant(r[0]) for r in result_lst],
            pred_scores=[self.dequant(r[1]) for r in result_lst],
            pred_cls=[self.dequant(r[2]) for r in result_lst],
        )


@OBJECT_REGISTRY.register
class PersonDetFilterV2(nn.Module):
    """Filter for person detection.

    Args:
        input_key:
            Hashable object used to query detection output from input.
        num_anchor: Anchor number.
        threshold: Default threshold to filter box by max score.
        idx_range: The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.
    """

    def __init__(
        self,
        input_key: Hashable,
        num_anchor: int,
        threshold: float,
        strides: Optional[Sequence[int]] = None,
        idx_range: Optional[Tuple[int, int]] = None,
    ):
        super(PersonDetFilterV2, self).__init__()
        assert strides == [1] or strides is None
        self.input_key = input_key
        self.filter_module = FilterModule(
            threshold=threshold,
            idx_range=idx_range,
        )
        self.num_anchor = num_anchor

    def forward(
        self,
        head_out: Dict[str, List[torch.Tensor]],
        im_hw: Optional[Tuple[int, int]] = None,
    ) -> Sequence[torch.Tensor]:

        rpn_feature = head_out[self.input_key][0]

        dim = rpn_feature.shape[1] // self.num_anchor
        rpn_feature = rpn_feature.view(
            -1, self.num_anchor, dim, *rpn_feature.shape[2:]
        )
        cls_pred = rpn_feature[:, :, 4:].flatten(1, 2)
        reg_pred = rpn_feature[:, :, :4].flatten(1, 2)

        filter_output = self.filter_module(*[cls_pred, reg_pred])

        return {
            "filter_coord": filter_output[0][-3],  # coor
            "filter_scores": filter_output[0][-2],  # score
            "filter_bboxes": filter_output[0][-1],  # box
        }
