from typing import Dict, List, Optional

import torch
import torch.nn.functional as ff
from torch import nn

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "WkDecoder",
    "WkPredDecoder",
    "AttrDecoder",
]


@OBJECT_REGISTRY.register
class WkDecoder(nn.Module):
    """WorkCondition Module Decoder.

    This is a work condition multi-task classification decoder,
    collecting predictions from multiple outputs.

    Args:
        task_descs: Task desc.
        name_dict: Name to convert with the task name.
        transforms: Transforms used in validation dataset.
        model_name: Record the model version.
    """

    def __init__(
        self,
        task_descs: List[Dict[str, str]],
        transforms: Optional[List] = None,
        name_dict: Optional[Dict] = None,
        model_name: str = "wk_classification_v1",
    ):
        super().__init__()

        self.task_descs = task_descs
        self.transforms = transforms if transforms is not None else []
        self.name_dict = name_dict
        self.model_name = model_name

    def postprocess(self, curr_task, cls_res, softmax=True):
        cls_feature_map = cls_res
        res_dict_list = []
        for curr_batch_fm in cls_feature_map:
            curr_res_cls_id = int(torch.argmax(curr_batch_fm))
            curr_res_cls_name = self.name_dict[curr_task][curr_res_cls_id]
            curr_res_dict = {"pred_result": curr_res_cls_name}
            curr_res_dict["network_output"] = {}
            curr_res_dict["pred_id"] = curr_res_cls_id
            if softmax:
                curr_batch_sfm = ff.softmax(curr_batch_fm, dim=0)
                curr_batch_fm = curr_batch_sfm
            for k, v in zip(self.name_dict[curr_task], curr_batch_fm):
                curr_res_dict["network_output"][k] = float(v)
            res_dict_list.append(curr_res_dict)
        return res_dict_list

    def forward(self, *results: Dict[str, torch.Tensor]):
        assert len(results) == len(self.task_descs)
        pick_results = {
            task: res[desc[0]]
            for res, (task, desc) in zip(results, self.task_descs.items())
        }

        postprocess_res = [{}]
        for curr_task, curr_branch_res in pick_results.items():
            curr_branch_res = self.postprocess(curr_task, curr_branch_res)
            for pp_res_curr_img, branch_res_curr_img in zip(
                postprocess_res, curr_branch_res
            ):  # noqa
                pp_res_curr_img.setdefault(curr_task, [])
                pp_res_curr_img[curr_task].append(branch_res_curr_img)

        postprocess_res[0]["model_name"] = self.model_name

        return postprocess_res


@OBJECT_REGISTRY.register
class WkPredDecoder(WkDecoder):
    """WorkCondition Module Decoder.

    This is a work condition multi-task classification decoder,
    collecting predictions from multiple outputs.

    Args:
        task_descs: Task desc.
        name_dict: Name to convert with the task name.
        transforms: Transforms used in validation dataset.
        model_name: Record the model version.
    """

    def __init__(
        self,
        task_descs: List[Dict[str, str]],
        transforms: Optional[List] = None,
        name_dict: Optional[Dict] = None,
        model_name: str = "wk_classification_v1",
    ):
        super().__init__(
            task_descs,
            transforms,
            name_dict,
            model_name,
        )

    def forward(self, *results: Dict[str, torch.Tensor]):
        assert len(results) == len(self.task_descs)
        pick_results = {
            task: res[desc[0]]
            for res, (task, desc) in zip(results, self.task_descs.items())
        }

        postprocess_res = [{}]
        for curr_task, curr_branch_res in pick_results.items():
            curr_branch_res = self.postprocess(curr_task, curr_branch_res)
            pp_res = []
            for res in curr_branch_res:
                pp_res.append(
                    {
                        "obj_id": 0,
                        "prediction": res["pred_id"],
                        "scores": res["network_output"][res["pred_result"]],
                    }
                )
            postprocess_res[0][curr_task] = pp_res

        postprocess_res[0]["model_name"] = self.model_name

        return postprocess_res


@OBJECT_REGISTRY.register
class AttrDecoder(nn.Module):
    """Attribute Module Decoder.

    This is an attribute classification decoder,
    collecting predictions from multiple outputs.

    Args:
        task_descs: Task desc.
        name_dict: Name to convert with the task name.
        model_name: Record the model version.
        merge_result: Whether to concat result.
        topic_mapper: Attribute name to map.
    """

    def __init__(
        self,
        task_descs: List[Dict[str, str]],
        name_dict: Optional[Dict] = None,
        model_name: str = "attr_model_v1",
        merge_result: bool = False,
        topic_mapper: Dict = None,
    ):
        super().__init__()

        self.task_descs = task_descs
        self.name_dict = name_dict
        self.model_name = model_name
        self.merge_result = merge_result

        self.type_len = []
        for type_name in self.name_dict.keys():
            self.type_len.append(len(self.name_dict[type_name]))
        self.topic_mapper = (
            {"category": "type"} if topic_mapper is None else topic_mapper
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

        label_name = self.name_dict
        type_len = self.type_len

        all_rois_res = []
        for _, pred_res in pick_results.items():
            for one_roi_res in pred_res:
                one_roi_info = {"attrs": {}, "cls_score_2pe": {}}
                for jdx, t_name in enumerate(label_name.keys()):
                    if jdx == 0:
                        one_roi_info["attrs"][t_name] = label_name[t_name][
                            torch.argmax(one_roi_res[0 : type_len[0]])
                        ]
                        one_roi_info["cls_score_2pe"][t_name] = ff.softmax(
                            one_roi_res[0 : type_len[0]], dim=0
                        ).tolist()
                    else:
                        one_roi_info["attrs"][t_name] = label_name[t_name][
                            torch.argmax(
                                one_roi_res[
                                    sum(type_len[:jdx]) : sum(
                                        type_len[: jdx + 1]
                                    )
                                ]
                            )
                        ]
                        one_roi_info["cls_score_2pe"][t_name] = ff.softmax(
                            one_roi_res[
                                sum(type_len[:jdx]) : sum(type_len[: jdx + 1])
                            ],
                            dim=0,
                        ).tolist()
                for _, roi_info in one_roi_info.items():
                    for src_k, dst_k in self.topic_mapper.items():
                        if src_k in roi_info:
                            roi_info[dst_k] = roi_info.pop(src_k)

                all_rois_res.append(one_roi_info)

        return all_rois_res
