from collections import ChainMap, OrderedDict

import numpy as np
import torch

from hat.data.transforms.affine import LabelAffineTransform
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FrcnnInvTransforms(object):
    def __init__(self, img_transforms, return_meta=False):
        self.img_transforms = img_transforms
        self.return_meta = return_meta

    def _slice_dict_to_batch_list(self, data):
        assert isinstance(data, dict)
        res = []
        for key, batch in data.items():
            if (
                isinstance(batch, (list, tuple))
                and len(batch) > 0
                and isinstance(batch[0], torch.Tensor)
            ):
                res.append([{key: b} for b in zip(*batch)])
            else:
                res.append([{key: b} for b in batch])
        res = [dict(ChainMap(*r)) for r in zip(*res)]
        return res

    def __call__(self, results, data_meta):
        if isinstance(data_meta, tuple):
            transforms_meta = data_meta[0].get("transform_meta", None)
        else:
            transforms_meta = data_meta.get("transform_meta", None)
        if not transforms_meta:
            return results
        for trans in self.img_transforms[::-1]:
            if hasattr(trans, "inverse_transform"):
                inv_results = OrderedDict()
                meta = transforms_meta.pop(-1)
                batch_meta = self._slice_dict_to_batch_list(meta)
                # for each task group
                if "original_calib" in meta.keys():
                    data_meta["calib"] = meta.pop("original_calib")

                for task_key, task_res in results.items():
                    inv_results[task_key] = []
                    # for each batch
                    for res, meta in zip(task_res, batch_meta):
                        if res is None:
                            continue
                        inv_task_res = trans.inverse_transform(res, meta)
                        inv_results[task_key].append(inv_task_res)
        if self.return_meta:
            return inv_results, meta
        else:
            return inv_results


@OBJECT_REGISTRY.register
class RescaleTransforms(object):
    def __init__(self, factor=1, exclude_task=None):
        if isinstance(factor, (int, float)):
            factor = (factor, factor)
        self.factor = factor
        if isinstance(exclude_task, str):
            exclude_task = [exclude_task]
        self.exclude_task = exclude_task

    def __call__(self, results, *args):
        # for each task group
        for task_key, task_res in results.items():
            if task_key in self.exclude_task:
                continue
            # for each batch
            for res in task_res:
                res.rescale(self.factor[0], self.factor[1])
        return results


@OBJECT_REGISTRY.register
class ComposeInverseTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results, *args, **kwargs):
        for transform in self.transforms:
            results = transform(results, *args, **kwargs)
        return results


@OBJECT_REGISTRY.register
class TLLensInverseTransforms(object):
    def __init__(self):
        self._bbox_ts = LabelAffineTransform(label_type="box")

    def _cal_iop(self, bbox1, bbox2):
        # bbox1：shell box, bbox2：common box
        box1_tl_x, box1_tl_y, box1_lr_x, box1_lr_y = bbox1
        box2_tl_x, box2_tl_y, box2_lr_x, box2_lr_y = bbox2
        area2 = (box2_lr_y - box2_tl_y + 1) * (box2_lr_x - box2_tl_x + 1)
        x1 = max(box1_tl_x, box2_tl_x)
        y1 = max(box1_tl_y, box2_tl_y)
        x2 = min(box1_lr_x, box2_lr_x)
        y2 = min(box1_lr_y, box2_lr_y)
        inter = max(0.0, x2 - x1 + 1) * max(0.0, y2 - y1 + 1)
        iop = inter / area2
        return iop

    def _is_common_box_in_shell(self, pred_bbox, crop_roi):
        # 判断是否在roi内
        res = []
        for bbox in pred_bbox:
            iop = self._cal_iop(crop_roi, bbox)
            if iop >= 0.2:
                res.append(True)
            else:
                res.append(False)
        return res

    def __call__(self, results, data_meta):
        inverse_affine_aug_param = data_meta.get(
            "inverse_affine_aug_param", None
        )
        if (
            "detection" not in results.keys()
            or inverse_affine_aug_param is None
        ):  # noqa
            return results
        pred_det_boxes = results["detection"][0]
        inverse_det_boxes = []
        for det_box, inverse_param, crop_roi in zip(
            pred_det_boxes, inverse_affine_aug_param, data_meta["crop_roi"]
        ):
            if det_box.shape[0] == 0:
                inverse_det_boxes.append(det_box)
                continue
            det_box[:, :4] = torch.Tensor(
                self._bbox_ts(
                    np.array(det_box[:, :4].cpu()),
                    np.array(inverse_param.cpu()),
                ),
            )
            filter_flag = self._is_common_box_in_shell(
                det_box[:, :4], crop_roi
            )
            inverse_det_boxes.append(det_box[filter_flag])
            # inverse_det_boxes.append(det_box)
        results["detection"][0] = inverse_det_boxes
        return results
