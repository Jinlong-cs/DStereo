# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as ff
from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.nn import DetectionPostProcessV1
from horizon_plugin_pytorch.quantization import (
    FakeQuantize,
    FixedScaleObserver,
    default_weight_8bit_fake_quant,
)
from horizon_plugin_pytorch.utils import fx_helper
from horizon_plugin_pytorch.utils.script_quantized_fn import (
    script_quantized_fn,
)
from torch import Tensor
from torch.quantization import QConfig

from hat.models.base_modules.postprocess import AnchorPostProcess
from hat.models.task_modules.fcos import FCOSMultiStrideFilter
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "TLClsPredDecoder",
    "TLDetAttrDecoder",
    "TLAttrDetPostProcess",
    "RPNTrafficLightFilter",
]


@OBJECT_REGISTRY.register
class TLClsPredDecoder(nn.Module):
    """Traffic Light Multitask Module Decoder.

    This is a traffic light multi-task classification decoder,
    collecting predictions from multiple outputs.

    Args:
        task_descs: Task desc.
        transforms: Transforms used in validation dataset.
        model_name: Record the model version.
        eval_time_num: How many classes are evaluated in the countdown.
    """

    def __init__(
        self,
        task_descs: List[Dict[str, str]],
        transforms: Optional[List] = None,
        model_name: str = "tl_mtl_cls_v1",
        eval_time_num: int = 100,
    ):
        super().__init__()

        self.task_descs = task_descs
        self.transforms = transforms if transforms is not None else []
        self.model_name = model_name
        self.eval_time_num = eval_time_num

    def norm_cls_postprocess(self, cls_res, softmax=True):
        cls_feature_map = cls_res
        res_dict_list = []
        for curr_batch_fm in cls_feature_map:
            curr_res_cls_id = int(torch.argmax(curr_batch_fm))
            if softmax:
                curr_batch_sfm = ff.softmax(curr_batch_fm, dim=0)
                curr_batch_fm = curr_batch_sfm
            obj = {
                "obj_id": 0,
                "prediction": curr_res_cls_id,
                "scores": curr_batch_fm.tolist(),
            }
            res_dict_list.append(obj)
        return res_dict_list

    def time_cls_postprocess(self, cls_res):
        cls_feature_map = cls_res
        res_dict_list = []
        for curr_batch_fm in cls_feature_map:
            pred_nums_digit = int(torch.argmax(curr_batch_fm[:3]))
            pred_hundreds_place = int(torch.argmax(curr_batch_fm[3:13]))
            pred_tens_place = int(torch.argmax(curr_batch_fm[13:23]))
            pred_ones_place = int(torch.argmax(curr_batch_fm[23:33]))
            curr_res_cls_id = (
                (pred_hundreds_place * 100 if pred_nums_digit >= 2 else 0)
                + (pred_tens_place * 10 if pred_nums_digit >= 1 else 0)
                + (pred_ones_place * 1 if pred_nums_digit >= 0 else 0)
            )

            obj = {
                "obj_id": 0,
                "prediction": curr_res_cls_id,
                # TODO: The current scheme cannot directly calculate the score
                # of each class, in order to meet the requirements of platform
                # evaluation, use an all-one, shape-compliant tensor for
                # filling.
                "scores": torch.ones(self.eval_time_num).tolist(),
            }
            res_dict_list.append(obj)
        return res_dict_list

    def forward(self, *results: Dict[str, torch.Tensor]):
        assert len(results) == len(self.task_descs)
        pick_results = {
            task: res[desc[0]]
            for res, (task, desc) in zip(results, self.task_descs.items())
        }

        postprocess_res = [{}]
        for curr_task, curr_branch_res in pick_results.items():
            curr_branch_res = curr_branch_res.flatten(1)
            if curr_task == "traffic_light_time_classification":
                curr_branch_res = self.time_cls_postprocess(curr_branch_res)
            else:
                curr_branch_res = self.norm_cls_postprocess(
                    curr_branch_res, softmax=False
                )

            postprocess_res[0][curr_task] = curr_branch_res

        postprocess_res[0]["model_name"] = self.model_name

        return postprocess_res


@OBJECT_REGISTRY.register
class TLDetAttrDecoder(nn.Module):
    """Traffic Light Multitask Module Decoder.

    This is a traffic light multi-task detection decoder,
    collecting predictions from multiple outputs.

    Args:
        num_classes:
        score_threshold_per_class: The threshold for the detection
            box filtering.
        task_descs: Task desc.
        transforms: Transforms used in validation dataset.
        model_name: Record the model version.
    """

    def __init__(
        self,
        score_threshold_per_class: List[int],
        task_descs: List[Dict[str, str]],
        transforms: Optional[List] = None,
        model_name: str = "tl_mtl_cls_v1",
    ):
        super().__init__()

        self.score_threshold_per_class = score_threshold_per_class
        self.task_descs = task_descs
        self.transforms = transforms if transforms is not None else []
        self.model_name = model_name

    def forward(self, *results: Dict[str, torch.Tensor]):
        assert len(results) == len(self.task_descs)
        det_info = torch.stack(results[0]["pred_boxes_out"], dim=0)
        cls_info = torch.stack(results[0]["pred_scores"], dim=0).unsqueeze(-1)
        cls_label = torch.stack(results[0]["pred_cls"], dim=0).unsqueeze(-1)
        type_info = torch.stack(
            [attr[0] for attr in results[0]["attr_cls"]], dim=0
        )
        type_label = torch.argmax(type_info, dim=-1).unsqueeze(-1)
        color_info = torch.stack(
            [attr[1] for attr in results[0]["attr_cls"]], dim=0
        )
        color_label = torch.argmax(color_info, dim=-1).unsqueeze(-1)

        results = []
        bboxes = torch.cat(
            (det_info, cls_info, cls_label, type_label, color_label), dim=-1
        )
        for class_id, score_threshold_i in enumerate(
            self.score_threshold_per_class
        ):
            output_i = []
            for bbox in bboxes:
                # det_cls_mask
                cls_mask = bbox[:, 5] == class_id
                # det_score_mask
                det_mask = bbox[:, 4] > score_threshold_i
                bbox = bbox[cls_mask & det_mask]
                output_i.append(bbox)
            results.append(output_i)
        return results


@OBJECT_REGISTRY.register
class TLAttrDetPostProcess(AnchorPostProcess):
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
        attr_list: List = None,
        mode: str = "val",
    ) -> None:

        super(TLAttrDetPostProcess, self).__init__(
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
        if mode == "train" or mode == "val":
            self._dpp = DetectionAttrPostProcess(
                num_classes,
                [attr[1] for attr in attr_list],
                box_filter_threshold,
                class_offsets,
                use_clippings,
                image_hw,
                nms_iou_threshold,
                pre_nms_top_k,
                post_nms_top_k,
                nms_margin=nms_margin,
                nms_padding_mode=nms_padding_mode,
                bbox_min_hw=bbox_min_hw,
            )
            self.attr_list = attr_list
        elif mode == "test":
            self._dpp = DetectionPostProcessV1(
                num_classes,
                box_filter_threshold,
                class_offsets,
                use_clippings,
                image_hw,
                nms_iou_threshold,
                pre_nms_top_k,
                post_nms_top_k,
                use_stable_sort=True,
                nms_margin=nms_margin,
                nms_padding_mode=nms_padding_mode,
                bbox_min_hw=bbox_min_hw,
            )
        self.cat_func = torch.nn.quantized.FloatFunctional()
        self.mode = mode

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
        if self.mode == "test":
            attr_type = head_out["rpn_attr_pred_dict"]["type"][0]
            attr_color = head_out["rpn_attr_pred_dict"]["color"][0]
            head_out = [
                t["det_feature"].detach() for t in head_out[self.input_key]
            ]
            result_lst = self._dpp(head_out, anchors, image_sizes=im_hw)
            return OrderedDict(
                pred_boxes_out=[self.dequant(r[0]) for r in result_lst],
                pred_boxes=[r[0] for r in result_lst],
                pred_scores=[self.dequant(r[1]) for r in result_lst],
                pred_cls=[r[2] for r in result_lst],
                attr_type=self.dequant(attr_type),
                attr_color=self.dequant(attr_color),
            )
        else:
            num_anchor = anchors[0].shape[1] // 4

            det_feature = [
                t["det_feature"].detach() for t in head_out[self.input_key]
            ]
            num_attr = len(self.attr_list)
            attr_feature_list = [
                [attr[i].detach() for i in range(num_attr)]
                for attr in [
                    t["attr_feature_list"] for t in head_out[self.input_key]
                ]
            ]
            det_shape = det_feature[0].shape
            det_feature = [
                det.reshape(
                    det_shape[0],
                    num_anchor,
                    det_shape[1] // num_anchor,
                    det_shape[-2],
                    det_shape[-1],
                )
                for det in det_feature
            ]
            for i, (k, v) in enumerate(self.attr_list):
                setattr(
                    self,
                    f"{k}_feature",
                    [
                        attr_feature[i].reshape(
                            det_shape[0],
                            num_anchor,
                            v,
                            det_shape[-2],
                            det_shape[-1],
                        )
                        for attr_feature in attr_feature_list
                    ],
                )
            det_feature = [
                self.dequant(
                    self.cat_func.cat(
                        [det, type_attr, color_attr], dim=2
                    ).reshape(det_shape[0], -1, det_shape[-2], det_shape[-1])
                )
                for det, type_attr, color_attr in zip(
                    det_feature, self.type_feature, self.color_feature
                )
            ]
            result_lst = self._dpp(det_feature, anchors, image_sizes=im_hw)
            return OrderedDict(
                pred_boxes_out=[self.dequant(r[0]) for r in result_lst],
                pred_boxes=[r[0] for r in result_lst],
                pred_scores=[self.dequant(r[1]) for r in result_lst],
                pred_cls=[r[2] for r in result_lst],
                attr_cls=[r[3] for r in result_lst],
            )

    def set_qconfig(self):
        qcfg = QConfig(
            activation=FakeQuantize.with_args(
                observer=FixedScaleObserver,
                quant_min=qinfo("qint8").min,
                quant_max=qinfo("qint8").max,
                dtype="qint8",
                # scale=1 / 2 ** self.output_shift,
                scale=1 / 2 ** 4,
            ),
            weight=default_weight_8bit_fake_quant,
        )
        self.cat_func.qconfig = qcfg


@fx_helper.wrap
class DetectionAttrPostProcess(DetectionPostProcessV1):
    def __init__(
        self,
        num_classes: int,
        attr_classes: List[int],
        box_filter_threshold: float,
        class_offsets: List[int],
        use_clippings: bool,
        image_size: Tuple[int, int],
        nms_threshold: float,
        pre_nms_top_k: int,
        post_nms_top_k: int,
        nms_padding_mode: Optional[str] = None,
        nms_margin: float = 0.0,
        use_stable_sort: bool = None,
        bbox_min_hw: Tuple[float, float] = (0, 0),
    ):
        super(DetectionAttrPostProcess, self).__init__(
            num_classes,
            box_filter_threshold,
            class_offsets,
            use_clippings,
            image_size,
            nms_threshold,
            pre_nms_top_k,
            post_nms_top_k,
            nms_padding_mode,
            nms_margin,
            use_stable_sort,
            bbox_min_hw,
        )
        self.attr_classes = attr_classes

    @script_quantized_fn
    def forward(
        self, data: List[Tensor], anchors: List[Tensor], image_sizes=None
    ) -> Tensor:
        """Forward pass of ~DetectionPostProcessV1.

        Args:
            data (List[Tensor]): (N, (4 + num_classes) * anchor_num, H, W)
            anchors (List[Tensor]): (N, anchor_num * 4, H, W)
            image_sizes (Tensor[batch_size, (h, w)], optional):
                Defaults to None.

        Returns:
            List[Tuple[Tensor, Tensor, Tensor]]:
                list of (bbox (x1, y1, x2, y2), score, class_idx).
        """
        if self.use_clippings:
            assert (
                self.image_size is not None or image_sizes is not None
            ), "image size must be provided if use_clippings == True"
        branch_num = len(data)
        batch_size = data[0].size(0)

        per_image_rets_list = []

        # rearrange anchor and data
        flatten_anchors, deltas, scores = [], [], []

        # TODO: check if class offsets should work in batch-wise instead
        # of branch-wise
        cls_offsets = []

        for i in range(branch_num):
            flatten_anchors.append(
                anchors[i]
                .detach()
                .permute(0, 2, 3, 1)
                .reshape(batch_size, -1, 4)
            )
            unfolded_data = (
                data[i]
                .detach()
                .permute(0, 2, 3, 1)
                .reshape(
                    batch_size,
                    -1,
                    4 + self.num_classes + sum(self.attr_classes),  # noqa
                )
            )

            deltas.append(unfolded_data[..., :4])
            scores.append(unfolded_data[..., 4:])

            cls_offsets.append(
                unfolded_data.new_ones((1, unfolded_data.shape[1]))
                * self.class_offsets[i]
            )

        # (B, N, C)
        flatten_anchors = torch.cat(flatten_anchors, dim=1)
        deltas = torch.cat(deltas, dim=1)
        scores = torch.cat(scores, dim=1)

        # (1, N)
        cls_offsets = torch.cat(cls_offsets, dim=1)

        # (B, N)
        max_scores, max_score_idxs = torch.max(
            scores[..., : self.num_classes], dim=-1
        )

        # (B, N)
        cls_idxs = max_score_idxs + cls_offsets

        # decode, (B, N, 4)
        pred_boxes = self.coder.decode_single(
            deltas.view(-1, 4),
            flatten_anchors.view(-1, 4),
        ).view(batch_size, -1, 4)

        im_hw = image_sizes if image_sizes is not None else self._image_size
        pred_boxes = self.clip_boxes_to_image(pred_boxes, im_hw)

        combine_boxes = torch.cat(
            [
                pred_boxes,
                scores[..., self.num_classes :],
                cls_idxs[..., None],
                max_scores[..., None],
            ],
            dim=2,
        )
        sorted_boxes = self._sort(combine_boxes)

        for single_boxes in sorted_boxes:

            # filter by score threshold
            score_mask = single_boxes[:, -1] >= self.box_filter_threshold

            # filter invalid
            valid_mask = torch.logical_and(
                single_boxes[:, 0] < single_boxes[:, 2] - self.bbox_min_hw[1],
                single_boxes[:, 1] < single_boxes[:, 3] - self.bbox_min_hw[0],
            )
            single_boxes = single_boxes[
                torch.logical_and(score_mask, valid_mask)
            ]

            # nms
            single_boxes = self._nms(single_boxes)

            # filter
            if single_boxes.shape[0] > self.post_nms_top_k:
                single_boxes = single_boxes[: self.post_nms_top_k]

            # pad
            else:
                if self.nms_padding_mode is not None:
                    single_boxes = self.pad_data(single_boxes)

            pred_box = single_boxes[:, :4]
            cls_idxs = single_boxes[:, -2]
            max_scores = single_boxes[:, -1]
            attr_idx = 4
            attr_scores = []
            for attr_ch in self.attr_classes:
                attr_scores.append(
                    single_boxes[:, attr_idx : attr_idx + attr_ch]
                )
                attr_idx += attr_ch

            per_image_rets_list.append(
                (pred_box, max_scores, cls_idxs, attr_scores)
            )

        return per_image_rets_list


@OBJECT_REGISTRY.register
class RPNTrafficLightFilter(FCOSMultiStrideFilter):
    def __init__(
        self,
        threshold: float,
        strides: Sequence[int],
        idx_range: Optional[Tuple[int, int]] = None,
        for_compile: bool = False,
    ):
        super(RPNTrafficLightFilter, self).__init__(
            strides,
            threshold,
            idx_range,
            for_compile,
        )

    def forward(
        self,
        anchors: List[torch.Tensor],
        head_out: Dict[str, List[torch.Tensor]],
        im_hw: Optional[Tuple[int, int]] = None,
    ) -> Sequence[torch.Tensor]:

        cls_pred = head_out["rpn_cls_pred"][0]
        reg_pred = head_out["rpn_reg_pred"][0]

        filter_input_split = [
            cls_pred,
            reg_pred,
        ]

        filter_output = self.filter_module(*filter_input_split)

        return {
            "filter_coord": filter_output[0][-3],
            "filter_scores": filter_output[0][-2],
            "filter_bboxes": filter_output[0][-1],
            "type_pred": head_out["rpn_attr_pred_dict"]["type"][0],
            "color_pred": head_out["rpn_attr_pred_dict"]["color"][0],
        }
