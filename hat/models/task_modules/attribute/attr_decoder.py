from typing import Dict, List, Optional

import torch
import torch.nn.functional as ff
from torch import nn

from hat.core.data_struct.app_struct import (
    DetObject,
    DetObjects,
    build_task_struct,
)
from hat.core.data_struct.base_struct import ClsLabels
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class SoftmaxAttrDecoder(nn.Module):
    """Attribute Module Decoder.

    This is an attribute classification decoder,
    collecting predictions from multiple outputs.

    Args:
        task_descs: Task desc.
        name_dict: Name to convert with the task name.
        model_name: Record the model version.
        merge_result: Whether to merge multi-branch results to one list.
    """

    def __init__(
        self,
        task_descs: List[Dict[str, str]],
        name_dict: Optional[Dict] = None,
        model_name: str = "attr_model_v1",
        merge_result=False,
    ):
        super().__init__()

        self.task_descs = task_descs
        self.name_dict = name_dict
        self.model_name = model_name
        self.merge_result = merge_result

        self.type_len = [len(v) for _, v in self.name_dict.items()]
        self.name_list = list(self.name_dict.keys())
        self.name_list = [
            item.replace("category", "type") if item == "category" else item
            for item in self.name_list
        ]
        if "category" in self.name_dict:
            self.name_dict["type"] = self.name_dict.pop("category")

        _, self.TASK_STRUCT = build_task_struct(
            "DetObject",
            "DetObjects",
            [(t_name, ClsLabels) for t_name in self.name_dict.keys()],
            bases=(DetObject, DetObjects),
        )

    def forward(self, *results: Dict[str, torch.Tensor]):
        assert len(results) == len(self.task_descs)
        pick_results = {
            task: res[desc[0]]
            for res, (task, desc) in zip(results, self.task_descs.items())
        }
        if self.merge_result:
            pick_results = {
                self.model_name: torch.cat(list(pick_results.values()), axis=1)
            }

        all_rois_res = {}
        for _, pred_res in pick_results.items():
            batch_attribute_split = torch.split(pred_res, self.type_len, dim=1)
            assert len(self.type_len) == len(batch_attribute_split)
            for attribute_split, t_name in zip(
                batch_attribute_split, self.name_list
            ):
                cls_name_mapping = {
                    k: v for k, v in enumerate(self.name_dict[t_name])
                }
                batch_scores, batch_cls_idxs = ff.softmax(attribute_split).max(
                    -1
                )

                res = ClsLabels(
                    scores=batch_scores,
                    cls_idxs=batch_cls_idxs,
                    cls_name_mapping=cls_name_mapping,
                )
                all_rois_res[t_name] = res

        res_struct = self.TASK_STRUCT(**all_rois_res)

        return res_struct
