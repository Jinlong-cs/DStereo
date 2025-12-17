# from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch

try:
    from hatbc.message import Attribute, Instance, Mask2D, Polygon2D
    from hatbc.utils.compress import CompressMethod
except ImportError:
    Attribute, Instance, Mask2D, Polygon2D = None, None, None, None
    CompressMethod = None

from hat.core.data_struct.base_struct import Mask
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.package_helper import require_packages


def default_convert_fn(pred, task_name):
    pred_masks = getattr(pred, task_name)
    foreground, _ = pred_masks.masks.max(0)
    return Mask(foreground.long())


@OBJECT_REGISTRY.register
class InstanceSegToParsing(torch.nn.Module):
    """Convert instance segmentation output to semantic segmentation.

    Args:
        task_name: The name of task, used as instance topic.
        convert_fn: Funtion for converting.
    """

    def __init__(
        self,
        task_name: str = "lane_instanceseg",
        convert_fn: Optional[Callable] = None,
    ):
        super(InstanceSegToParsing, self).__init__()
        self.task_name = task_name

        if convert_fn is not None:
            self.convert_fn = convert_fn
        else:
            self.convert_fn = partial(
                default_convert_fn,
                task_name=task_name,
            )

    def forward(self, preds):
        return [self.convert_fn(pred) for pred in preds]


@OBJECT_REGISTRY.register
class InstanceSegToMsg(torch.nn.Module):
    """Convert instance segmentation output to hatbc message format.

    Args:
        task_name: The name of task, used as instance topic.
        to_cpu: Whether move data to cpu. Default: False.
    """

    @require_packages("hatbc")
    def __init__(
        self,
        task_name: str = "",
        to_cpu: bool = False,
    ):
        super(InstanceSegToMsg, self).__init__()
        self.task_name = task_name
        self.to_cpu = to_cpu

    def forward(self, preds):
        return [self.convert(pred) for pred in preds]

    def convert(self, pred):
        def _get_value(cls_name_mapping, cls_idx):
            if cls_name_mapping is None:
                return cls_idx
            if isinstance(cls_idx, torch.Tensor):
                cls_idx = cls_idx.item()
            return cls_name_mapping[cls_idx]

        masks = pred.masks
        attributes = pred.attributes
        polygons = pred.get("polygons")

        instances = []
        for i, mask in enumerate(masks):
            instance = Instance(topic=self.task_name)
            if self.to_cpu:
                mask = mask.cpu()
            instance.mask2ds.append(
                Mask2D(
                    topic=self.task_name,
                    data=mask,
                    data_compress_method=CompressMethod.IMENCODE_PNG,
                )
            )
            if polygons is not None:
                polygons_i = _as_list(polygons[i])
                for polygon in polygons_i:
                    instance.polygon2ds.append(
                        Polygon2D(topic=self.task_name, data=polygon)
                    )
            for attr in attributes:
                cls_name_mapping = attr["cls_name_mapping"]
                cls_idx = attr["cls_idxs"][i]
                cls_name = _get_value(cls_name_mapping, cls_idx)
                score = attr["scores"][i]
                if self.to_cpu:
                    cls_idx = cls_idx.cpu()
                    score = score.cpu()
                instance.attributes.append(
                    Attribute(topic=attr["name"], value=cls_name, score=score)
                )
            instances.append(instance)
        return instances
