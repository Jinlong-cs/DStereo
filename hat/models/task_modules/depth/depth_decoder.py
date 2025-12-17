from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["DepthMultiDecoder", "DepthDecoder"]


@OBJECT_REGISTRY.register
class DepthDecoder(nn.Module):
    """Generate prediction class map for depth task.

    Args:
        in_strides: Strides of input predictions.
        out_stride: Stride of prediction to be upsampled.
        output_name: The key corresponding to the prediction class map.
            Default: depth_preds
        gt_scale: Value for gt scale.
            Default: 1.0
        depth_type: Depth coordinate. Default: Cartesian
    """

    def __init__(
        self,
        in_strides: List[int],
        out_stride: int,
        gt_scale: float = 1.0,
        output_name: str = "depth_preds",
        depth_type: str = "Cartesian",
    ) -> Dict:
        super().__init__()

        assert depth_type in [
            "Cartesian",
            "Cylindrical",
        ], "depth type \
            currently only supports Cartesian or Cylindrical."

        self.depth_type = depth_type
        self.in_strides = _as_list(in_strides)
        self.out_stride = out_stride
        self.gt_scale = gt_scale
        self.output_name = output_name
        assert isinstance(
            out_stride, int
        ), f"out stride type should be int, but got {type(out_stride)}"

    def _depth_decoder(self, label, pred_depth):
        if self.depth_type == "Cartesian":
            pred_depth = pred_depth * self.gt_scale
            return pred_depth

        if self.depth_type == "Cylindrical":
            assert (
                "virtual_cam_params" in label
                and label["virtual_cam_params"] is not None
            )
            virtual_cam_params = torch.tensor(label["virtual_cam_params"])
            center_x = virtual_cam_params[:, 0, 2]
            focal_u = virtual_cam_params[:, 0, 0]
            b, _, h, w = pred_depth.shape
            device = pred_depth.device
            dtype = pred_depth.dtype
            x_range = torch.arange(w, dtype=dtype, device=device)
            y_range = torch.arange(h, dtype=dtype, device=device)
            _, x = torch.meshgrid(y_range, x_range)
            x = x.repeat(b, 1, 1)
            center_x = center_x.unsqueeze(dim=-1).unsqueeze(dim=-1)
            focal_u = focal_u.unsqueeze(dim=-1).unsqueeze(dim=-1)
            theta = (x - center_x) / focal_u
            rho, conf = torch.split(pred_depth, 1, dim=1)
            rho = rho * self.gt_scale
            rho[rho < 0] = 0
            real_depth = torch.abs(
                torch.mul(rho, torch.cos(theta).unsqueeze(dim=1))
            )
            pred_depth = torch.cat([real_depth, conf], dim=1)
        return pred_depth

    def forward(self, pred, label):
        output = {}
        pred = _as_list(pred)
        for pred_i, in_stride in zip(pred, self.in_strides):
            if in_stride == self.out_stride:
                pred_i = F.interpolate(
                    pred_i, scale_factor=in_stride, mode="bilinear"
                )
                pred_i = self._depth_decoder(label, pred_i)
                output.update({self.output_name: pred_i})
        return output


@OBJECT_REGISTRY.register
class DepthMultiDecoder(nn.Module):
    """Generate prediction class map for depth task.

    Args:
        pred_names: Names of prediction
        decoder_modules: List of single depth decoder.
    """

    def __init__(
        self,
        pred_names: List[str],
        decoder_modules: List[DepthDecoder],
    ):
        super().__init__()
        self.pred_names = _as_list(pred_names)
        self.decoder_modules = _as_list(decoder_modules)
        assert len(self.pred_names) == len(self.decoder_modules)

    def forward(self, preds, data):
        decoder_output = {}
        for name, decode_module in zip(self.pred_names, self.decoder_modules):
            pred = preds[name]
            decoder_output.update(decode_module(pred, data))
        return decoder_output
