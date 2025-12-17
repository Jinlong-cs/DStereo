import torch
import torch.nn as nn

from hat.core.data_struct.app_struct import (
    DetObject,
    DetObjects,
    build_task_struct,
)
from hat.core.data_struct.base_struct import DetBoxes2D
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FCOSConverter(nn.Module):
    def __init__(
        self,
        task_name: str,
        cls_name_mapping: dict,
    ):
        """Convert the output from fcos decoder to hat detection structure.

        This helps to adapt to HDFlow easily and is generally used in
        hat/models/structures/detectors/bm_fcos.py.

        Args:
            task_name: task name.
            cls_name_mapping: maps category ID to class name, e.g.,
                {0: "cyclist"}.
        """
        super(FCOSConverter, self).__init__()
        self.cls_name_mapping = cls_name_mapping
        assert (
            "_detection" in task_name
        ), "task_name should have a suffix of _detection."
        self.task_name = task_name

        _, self.hat_det_struct = build_task_struct(
            "DetObject",
            "DetObjects",
            [(self.task_name, DetBoxes2D)],
            bases=(DetObject, DetObjects),
        )

    def forward(self, preds):
        assert "pred_bboxes" in preds
        det_results = preds["pred_bboxes"]

        results = []

        for det_bboxes in det_results:
            assert det_bboxes.size(-1) == 6, (
                "det_bboxes should be a nx6 tensor where the first 4 dims "
                "represent the bbox, the 5th dim detection score, the last dim"
                "bbox label id."
            )
            bbox_struct = DetBoxes2D(
                boxes=det_bboxes[:, :4],
                scores=det_bboxes[:, -2],
                cls_idxs=det_bboxes[:, -1],
                cls_name_mapping=self.cls_name_mapping,
            )
            res = self.hat_det_struct(**{self.task_name: bbox_struct})
            results.append(res)

        return results


@OBJECT_REGISTRY.register
class VehicleSideFCOSConverter(nn.Module):
    def __init__(
        self,
        task_name: str,
        cls_name_mapping: dict,
    ):
        """Convert the output from fcos decoder to hat detection structure.

        This helps to adapt to HDFlow easily and is generally used in
        hat/models/structures/detectors/bm_fcos.py.

        Args:
            task_name: task name.
            cls_name_mapping: maps category ID to class name, e.g.,
                {0: "cyclist"}.
        """
        super(VehicleSideFCOSConverter, self).__init__()
        self.cls_name_mapping = cls_name_mapping
        assert (
            "_detection" in task_name
        ), "task_name should have a suffix of _detection."
        self.task_name = task_name

    def forward(self, preds):
        assert "pred_polygons" in preds
        det_results = preds["pred_polygons"]

        results = []

        for det_polygons in det_results:
            assert det_polygons.size(-1) == 10, (
                "det_polygons should be a nx10 tensor where the first 8 dims "
                "represent the polygon, the 9th dim detection score, "
                "the last dim bbox label id."
            )
            x1, y1, x2 = (
                det_polygons[:, 0],
                det_polygons[:, 1],
                det_polygons[:, 2],
            )
            y2 = (det_polygons[:, 5] + det_polygons[:, 7]) / 2.0
            det_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            pred = {}
            pred["boxes"] = det_bboxes
            pred["scores"] = det_polygons[:, -2]
            pred["cls_idxs"] = det_polygons[:, -1]
            pred["polygons"] = det_polygons[:, :-2]
            pred["cls_name"] = self.cls_name_mapping[0]
            results.append(pred)

        return results
