from collections import OrderedDict

import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["DepthHeadParserWithScale"]


def disp_to_depth(pred, scale=None):
    """Scale the depth to abs scale.

    Args:
        out_strides: Output strides of head prediction.
    """
    pred = pred / 128 + 1
    if scale is not None:
        pred_depth = pred[:, :1] * scale
        pred[:, :1] = pred_depth
    return pred


@OBJECT_REGISTRY.register
class DepthHeadParserWithScale(object):
    """Reisze head predictions and rescle the predicted value for depth task.

    Args:
        out_strides: Output strides of head prediction.
        scale: Value for scale the output depth.
        scale_name: Which output value is need to scale.
    """

    def __init__(
        self,
        out_strides: OrderedDict,
        scale: float = None,
        scale_name: str = "pred_depth",
    ):
        self.out_strides = out_strides
        self.scale = scale
        self.scale_name = _as_list(scale_name)

    def __call__(self, preds):
        resize_preds = OrderedDict()
        for name, strides in self.out_strides.items():
            assert (
                name in preds
            ), f"out_strides key {name} not in prediction output dict."
            strides = _as_list(strides)
            resize_preds[name] = []
            preds_name = _as_list(preds[name])
            assert len(preds_name) == len(
                strides
            ), "length of prediction [{}] does not match out_strides."
            for i in range(len(strides)):
                pred = F.interpolate(
                    preds_name[i], scale_factor=strides[i], mode="bilinear"
                )
                if name in self.scale_name:
                    pred = disp_to_depth(pred, self.scale)
                resize_preds[name].append(pred)

        return resize_preds
