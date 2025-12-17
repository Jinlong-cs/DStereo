from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

try:
    from horizon_plugin_pytorch.functional import batched_nms_with_padding
except ImportError:
    pass

from hat.core.box_utils import bbox_clamp, bbox_filter_by_hw
from hat.core.data_struct.app_struct import (
    DetObject,
    DetObjects,
    build_task_struct,
)
from hat.core.data_struct.base_struct import (
    ClsLabels,
    DetBoxes2D,
    DetBoxes3D,
    Lines2D,
    MultipleBoxes2D,
    Points2D_2,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = ["RoIDecoder"]

CLASS_MAP = {
    "detection": DetBoxes2D,
    "classification": ClsLabels,
    "line": Lines2D,
    "detection_3d": DetBoxes3D,
    "kps": Points2D_2,
    "roi_detection": MultipleBoxes2D,
}


@OBJECT_REGISTRY.register
class RoIDecoder(nn.Module):
    """RoI Module Decoder.

    This is a multi-task decoder, collecting predictions from
    multiple outputs which provide various descriptions of
    aligned objects.

    Args:
        task_descs: Task desc.
        input_padding: Input padding,
            in (w_left, w_right, h_top, h_bottom) format.
        nms_threshold: Nms threshold.
        score_threshold: Score threshold.
        transforms: Transforms used in validation dataset.
    """

    def __init__(
        self,
        task_descs: List[Dict[str, str]],
        input_padding: Sequence[int] = (0, 0, 0, 0),
        nms_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
        legacy_bbox: Optional[bool] = True,
        im_hw: Optional[tuple] = None,
        clip_by_im_hw: Optional[bool] = False,
        min_filter_hw: Optional[tuple] = None,
    ):
        super().__init__()
        det_key = None
        for t_name, (_, t_type) in task_descs.items():
            if t_type == "detection":
                assert det_key is None, "1 detection only"
                det_key = t_name
        assert det_key is not None
        self.det_key = det_key
        self.task_descs = task_descs
        assert len(input_padding) == 4
        self.input_padding = input_padding
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.legacy_bbox = legacy_bbox
        self.im_hw = im_hw
        self.clip_by_im_hw = clip_by_im_hw
        self.min_filter_hw = min_filter_hw
        _, self.TASK_STRUCT = build_task_struct(
            "DetObject",
            "DetObjects",
            [
                (t_name, CLASS_MAP[t_type])
                for t_name, (_, t_type) in task_descs.items()
            ],
            bases=(DetObject, DetObjects),
        )

    @require_packages("horizon_plugin_pytorch>=0.16.1")
    def forward(self, *results: Dict[str, torch.Tensor]):
        assert len(results) == len(self.task_descs)
        pick_results = {
            task: res[desc[0]]
            for res, (task, desc) in zip(results, self.task_descs.items())
        }

        lengths = [len(res) for res in pick_results.values()]
        assert max(lengths) == min(lengths)

        batch_tmp = [{} for _ in range(max(lengths))]
        for t_name, res in pick_results.items():
            for i, r in enumerate(res):
                batch_tmp[i][t_name] = r

        # NMS on detection results in loop
        batch_results = []
        for res in batch_tmp:
            res_struct = self.TASK_STRUCT(**res)
            res_struct.inv_pad(*self.input_padding)
            boxes: DetBoxes2D = getattr(res_struct, self.det_key)
            if self.clip_by_im_hw:
                boxes.boxes = bbox_clamp(
                    boxes.boxes, self.im_hw, self.legacy_bbox
                )
            if self.min_filter_hw is not None:
                boxes_mask = bbox_filter_by_hw(
                    boxes.boxes, self.min_filter_hw, self.legacy_bbox
                )
                boxes.boxes = boxes.boxes * boxes_mask
                boxes.scores = boxes.scores * boxes_mask.squeeze(-1)

            if self.nms_threshold is not None:
                boxes: DetBoxes2D = getattr(res_struct, self.det_key)
                if len(boxes.boxes) > 0:
                    num_boxes = boxes.boxes.shape[0]
                    keep_idx = batched_nms_with_padding(
                        boxes.boxes.unsqueeze(0),
                        boxes.scores.unsqueeze(0),
                        boxes.cls_idxs.long().unsqueeze(0),
                        self.nms_threshold,
                        num_boxes,
                        num_boxes,
                        self.legacy_bbox,
                    )
                    if len(keep_idx) > 0:
                        keep_idx = keep_idx[keep_idx != -1]
                        res_struct = res_struct[keep_idx]
            if self.score_threshold is not None:
                res_struct = res_struct.filter_by_lambda(
                    lambda x: getattr(x, self.det_key).scores
                    >= self.score_threshold
                )

            batch_results.append(res_struct)

        return batch_results
