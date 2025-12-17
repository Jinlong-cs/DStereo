# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from easydict import EasyDict

from hat.core.data_struct.app_struct import (
    DetObject,
    DetObjects,
    build_task_struct,
)
from hat.core.data_struct.base_struct import ClsLabels, Mask, Masks
from hat.models.base_modules.postprocess import PostProcessorBase
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy
from .utils import dynamic_upsample, mask_matrix_nms

__all__ = ["SOLOV2Decoder"]


@OBJECT_REGISTRY.register
class SOLOV2Decoder(PostProcessorBase):
    """Decoder for SOLOV2 output.

    Args:
        num_classes: Number of categories excluding the background category.
        strides: Downsample factor of each feature map.
        object_name: The name of the predicted objects.
        kernel_out_channels: Number of output channels of the mask feature map
            branch. This is the channel count of the mask feature map that to
            be dynamically convolved with the predicted kernel. Default: 256.
        mask_stride: Downsample factor of the mask feature map output.
            Default: 4.
        upsample_mask_logit: Upsampling mask logits with the predicted
            sampling factors. Default: False.
        num_grids: Divided image into a uniform grids, each feature map has a
            different grid value. The number of output channels is grid ** 2.
        dynamic_conv_size: Dynamic Conv kernel size. Default: 1.
        test_cfg_updater: NMS configurations.
        combine: Combine instance segmentation output to remove overlap and get
            pred_id_map. Default: True.
        attr_names: Instance attributes to be predicted in parallel with
            classification.
        cls_name_mapping: Dict {category_id: class_name}.
        add_cls_into_attr. If True, classification results are saved as
            attributes, and the first attr_name is used as class name.
            Default: True.
        dict_to_hat_struct: Convert result to hat structure. Default is True.
        mask2polygon_fn: Funtion to convert mask to polygon. Default is None.
    """

    def __init__(
        self,
        num_classes: int,
        strides: Optional[Sequence[int]] = None,
        object_name: str = "instance",
        kernel_out_channels: int = 256,
        mask_stride: int = 4,
        upsample_mask_logit: bool = False,
        num_grids: Sequence[int] = None,
        dynamic_conv_size: int = 1,
        test_cfg_updater: Optional[Dict] = None,
        combine: bool = True,
        attr_names: Optional[List[str]] = None,
        cls_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
        add_cls_into_attr: bool = True,
        dict_to_hat_struct: bool = True,
        mask2polygon_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.strides = strides if strides else [4, 8, 16, 32, 64]
        self.cls_out_channels = num_classes
        self.kernel_out_channels = kernel_out_channels
        self.mask_stride = mask_stride
        self.upsample_mask_logit = upsample_mask_logit
        self.num_grids = num_grids if num_grids else [40, 36, 24, 16, 12]
        self.dynamic_conv_size = dynamic_conv_size
        test_cfg = {
            "nms_pre": 500,
            "score_thr": 0.1,
            "mask_thr": 0.5,
            "filter_thr": 0.05,
            "kernel": "gaussian",  # gaussian/linear
            "sigma": 2.0,
            "max_per_img": 25,
        }
        if test_cfg_updater:
            test_cfg.update(test_cfg_updater)
        self.test_cfg = EasyDict(test_cfg)
        self.num_levels = len(self.strides)
        self.combine = combine
        self.object_name = object_name
        self.attr_names = attr_names if attr_names else []
        self.cls_name_mapping = cls_name_mapping
        self.add_cls_into_attr = add_cls_into_attr
        if self.add_cls_into_attr:
            assert (
                len(self.attr_names) > 0
            ), "the first attr_name is used as class name"

        self.dict_to_hat_struct = dict_to_hat_struct
        if self.dict_to_hat_struct:
            _, self.hat_struct = build_task_struct(
                "DetObject",
                "DetObjects",
                [
                    (f"{self.object_name}_instanceseg", Masks),
                ]
                + [
                    (f"{self.object_name}_{k}", ClsLabels)
                    for k in self.attr_names
                ],
                bases=(DetObject, DetObjects),
            )

        self.mask2polygon_fn = mask2polygon_fn
        self.mask2polygon_bool = mask2polygon_fn is not None

    def forward(self, pred: Sequence[Any], label: dict) -> List[DetObjects]:
        (
            mlvl_kernel_preds,
            mlvl_cls_scores,
            mlvl_attr_preds,
            mask_feats,
            upsample_factors,
        ) = pred

        num_imgs = len(mask_feats)
        num_levels = self.num_levels

        #  parse image meta for decoding
        img_metas = [
            {
                k: label[k][ind]
                for k in ["img_height", "img_width", "img_shape"]
            }
            for ind in range(num_imgs)
        ]
        for img_meta in img_metas:
            img_meta.update(
                {
                    "ori_shape": (
                        int(img_meta.pop("img_height")),
                        int(img_meta.pop("img_width")),
                    )
                }
            )

        num_attrs = len(mlvl_attr_preds[0])
        mlvl_attr_preds = (
            [
                [
                    torch.stack(
                        list(mlvl_attr_preds[lvl][attr].softmax(1).max(1)),
                        dim=3,
                    )
                    for lvl in range(num_levels)
                ]
                for attr in range(num_attrs)
            ]
            if num_attrs > 0
            else [
                [
                    mask_feats.new_zeros(
                        [num_imgs, self.num_grids[lvl], self.num_grids[lvl], 2]
                    )
                    for lvl in range(num_levels)
                ]
            ]
        )

        for lvl in range(num_levels):
            cls_scores = mlvl_cls_scores[lvl]
            cls_scores = cls_scores.sigmoid()
            local_max = F.max_pool2d(cls_scores, 2, stride=1, padding=1)
            keep_mask = local_max[:, :, :-1, :-1] == cls_scores
            cls_scores = cls_scores * keep_mask
            mlvl_cls_scores[lvl] = cls_scores.permute(0, 2, 3, 1)

        results_list = []
        for img_id in range(len(img_metas)):
            img_cls_pred = [
                mlvl_cls_scores[lvl][img_id].view(-1, self.cls_out_channels)
                for lvl in range(num_levels)
            ]
            img_mask_feats = mask_feats[[img_id]]
            img_upsample_factor = upsample_factors[img_id]
            img_kernel_pred = [
                mlvl_kernel_preds[lvl][img_id]
                .permute(1, 2, 0)
                .view(-1, self.kernel_out_channels)
                for lvl in range(num_levels)
            ]
            img_attr_pred = [
                torch.cat(
                    [
                        # (N, attr_num, 2), 2 for [score, label]
                        mlvl_attr_preds_i[lvl][img_id].view(-1, 1, 2)
                        for mlvl_attr_preds_i in mlvl_attr_preds
                    ],
                    1,
                )
                for lvl in range(num_levels)
            ]

            img_cls_pred = torch.cat(img_cls_pred, dim=0)
            img_kernel_pred = torch.cat(img_kernel_pred, dim=0)
            img_attr_pred = torch.cat(img_attr_pred, dim=0)
            results = self._get_results_single(
                img_kernel_pred,
                img_attr_pred,
                img_cls_pred,
                img_mask_feats,
                img_upsample_factor,
                img_meta=img_metas[img_id],
            )

            if self.combine:
                results = self._combine(results)

            if self.mask2polygon_bool:
                results.polygons = self._masks2polygons(results.masks)

            if self.dict_to_hat_struct:
                results = self._dict_to_hat_struct(results)

            results_list.append(results)
        return results_list

    def _get_results_single(
        self,
        kernel_preds: torch.Tensor,
        attr_preds: torch.Tensor,
        cls_scores: torch.Tensor,
        mask_feats: torch.Tensor,
        upsample_factor: Optional[torch.Tensor],
        img_meta: dict,
        cfg=None,
    ) -> DetObjects:
        def empty_results(cls_scores):
            """Generate a empty results."""
            masks = cls_scores.new_zeros(0, *img_meta["ori_shape"][:2]).bool()
            res = EasyDict({"masks": masks, "attributes": []})
            for attr_name in self.attr_names:
                attr = {
                    "name": f"{self.object_name}_{attr_name}",
                    "cls_idxs": cls_scores.new_zeros(0),
                    "scores": cls_scores.new_zeros(0),
                    "cls_name_mapping": self.cls_name_mapping[attr_name],
                }
                res.attributes.append(attr)
            return res

        cfg = self.test_cfg if cfg is None else cfg

        # process.
        score_mask = cls_scores > cfg.score_thr
        cls_scores = cls_scores[score_mask]
        if len(cls_scores) == 0:
            return empty_results(cls_scores)

        # cate_labels & kernel_preds
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1].int()
        kernel_preds = kernel_preds[inds[:, 0]]
        # attribute_preds
        attr_preds = attr_preds[inds[:, 0]]

        # trans vector.
        lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(lvl_interval[-1])

        strides[: lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl - 1] : lvl_interval[lvl]] *= self.strides[
                lvl
            ]
        strides = strides[inds[:, 0]]

        # mask encoding.
        kernel_preds = kernel_preds.view(
            kernel_preds.size(0),
            -1,
            self.dynamic_conv_size,
            self.dynamic_conv_size,
        )
        mask_preds = (
            F.conv2d(mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()
        )

        # mask.
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(cls_scores)
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]
        attr_preds = attr_preds[keep]

        # maskness.
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores

        scores, labels, _, keep_inds = mask_matrix_nms(
            masks,
            cls_labels,
            cls_scores,
            mask_area=sum_masks,
            nms_pre=cfg.nms_pre,
            max_num=cfg.max_per_img,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            filter_thr=cfg.filter_thr,
        )
        if keep_inds.sum() == 0:
            return empty_results(cls_scores)
        mask_preds = mask_preds[keep_inds]
        attr_preds = attr_preds[keep_inds]

        featmap_size = mask_feats.size()[-2:]

        if self.upsample_mask_logit:
            mask_preds = dynamic_upsample(
                mask_preds.unsqueeze(0),
                upsample_factor,
                self.mask_stride,
            )
        else:
            upsampled_size = (
                int(featmap_size[0] * self.mask_stride),
                int(featmap_size[1] * self.mask_stride),
            )
            mask_preds = F.interpolate(
                mask_preds.unsqueeze(0),
                size=upsampled_size,
                mode="bilinear",
                align_corners=False,
            )

        ori_shape = img_meta["ori_shape"]
        img_shape = img_meta["img_shape"]
        mask_preds = F.interpolate(
            mask_preds[..., : img_shape[1], : img_shape[2]],
            size=ori_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        masks = mask_preds > cfg.mask_thr

        # import pudb;pudb.set_trace()
        res = EasyDict({"masks": masks, "attributes": []})
        if self.add_cls_into_attr:
            cls_name = self.attr_names[0]
            cls = {
                "name": f"{self.object_name}_{cls_name}",
                "cls_idxs": labels,
                "scores": scores,
                "cls_name_mapping": self.cls_name_mapping[cls_name],
            }
            res.attributes.append(cls)

        if self.add_cls_into_attr:
            attr_names = self.attr_names[1:]
        else:
            attr_names = self.attr_names
        for i, attr_name in enumerate(attr_names):
            attr = {
                "name": f"{self.object_name}_{attr_name}",
                "cls_idxs": attr_preds[:, i, 1],
                "scores": attr_preds[:, i, 0],
                "cls_name_mapping": self.cls_name_mapping[attr_name],
            }
            res.attributes.append(attr)
        return res

    def _combine(self, results):
        masks = results.masks
        w, h = masks.shape[-2], masks.shape[-1]
        pred_id_map = torch.zeros(
            size=[w, h], device=masks.device, dtype=torch.int64
        )

        obj_num = masks.shape[0]
        if obj_num == 0:
            results.pred_id_map = pred_id_map
            return results

        # start from 1, 0 for background
        index = pred_id_map.new_tensor([1])
        keep = torch.zeros(obj_num).bool()
        for _idx in range(obj_num):
            _mask = masks[_idx]
            mask_area = _mask.sum().item()
            intersect = _mask & (pred_id_map > 0)
            intersect_area = intersect.sum().item()
            # remove if heavily overlapped
            if mask_area == 0 or intersect_area * 1.0 / mask_area > 0.5:
                continue
            if intersect_area > 0:
                _mask = _mask & (pred_id_map == 0)
                masks[_idx] = _mask
            pred_id_map[_mask] = index
            keep[_idx] = True
            index += 1

        results.masks = masks[keep]
        for attr in results.attributes:
            attr["cls_idxs"] = attr["cls_idxs"][keep]
            attr["scores"] = attr["scores"][keep]
        results.pred_id_map = pred_id_map
        return results

    def _masks2polygons(self, masks):
        polygons = []
        for m in masks:
            m = convert_numpy(m)
            polygon = self.mask2polygon_fn(m)
            polygons.append(polygon)
        return polygons

    def _dict_to_hat_struct(self, results):
        res = {}
        key = f"{self.object_name}_instanceseg"
        res.update({key: Masks(results.masks)})
        for attr in results.attributes:
            key = attr.pop("name")
            res.update({key: ClsLabels(**attr)})
        res_hat_struct = self.hat_struct(**res)
        if self.combine:
            res_hat_struct.pred_id_map = Mask(results.pred_id_map)
        return res_hat_struct
