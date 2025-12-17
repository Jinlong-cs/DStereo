from collections import OrderedDict
from typing import List, Optional, Union

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = [
    "DepthMultiTargets",
    "DepthValueTarget",
]


def get_theta_x_for_cylindrical_camera(
    depth: torch.Tensor, intrsic_camera_params: torch.Tensor
) -> torch.Tensor:
    """Get theta x for each grid cell position of depth tensor.

    Args:
        depth: [B, C, H, W] The source tensor where we calculate the theta x.
        intrsic_camera_params: [N, 3, 3] Multiple intrinsic matrix tensor.

    Returns:
        The theta x tensor of shape [B, H, W].
    """
    assert intrsic_camera_params is not None
    center_x = intrsic_camera_params[:, 0, 2]
    focal_u = intrsic_camera_params[:, 0, 0]
    b, _, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype
    x_range = torch.arange(w, dtype=dtype, device=device)
    y_range = torch.arange(h, dtype=dtype, device=device)
    _, x = torch.meshgrid(y_range, x_range)
    x = x.repeat(b, 1, 1)
    center_x = center_x.unsqueeze(dim=-1).unsqueeze(dim=-1)
    focal_u = focal_u.unsqueeze(dim=-1).unsqueeze(dim=-1)
    theta_x = (x - center_x) / focal_u
    return theta_x


@OBJECT_REGISTRY.register
class DepthMultiTargets(object):
    """Generate training targets for depth task.

    Args:
        low_depth: Minimal depth value.
        hight_depth: Maximum depth value.
        downsample_scale: Downsampling factor used for target.
            Default: 2
        label_name: The key corresponding to the gt seg in
            label. Default: gt_depth
        depth_type: Depth target is in Cartesian or Cylindrical
            coordinate.
        target_modules: Target modules for loss computation.
        pred_names: Predctions' key.
        valid_hfov: Only points in hfov range [-hfov, hfov]
            will be used for train. Default: None
        gt_scale: Value for gt scale.
            Default: 1.0
    """

    def __init__(
        self,
        low_depth: float = 0.01,
        high_depth: float = 150.0,
        downsample_scale: int = 2,
        label_name: str = "gt_depth",
        depth_type: str = "Cartesian",
        target_modules: Optional[List] = None,
        pred_names: Union[List, str] = "pred_depth",
        valid_hfov: Optional[float] = None,
        gt_scale: float = 1.0,
    ):
        assert (
            downsample_scale >= 0
            and isinstance(downsample_scale, int)
            and (downsample_scale & (downsample_scale - 1)) == 0
        ), "downsample_scale is not valid!"
        assert depth_type in [
            "Cartesian",
            "Cylindrical",
        ], "depth type \
            currently only supports Cartesian or Cylindrical."

        self.low_depth = low_depth
        self.high_depth = high_depth

        self.max_pooling = nn.MaxPool2d(
            kernel_size=downsample_scale, stride=downsample_scale
        )
        pad_value = int(0.5 * (downsample_scale - 1))
        self.zero_pad = nn.ZeroPad2d((pad_value, 0, pad_value, 0))
        self.downsample = True if downsample_scale > 1 else False
        self.label_name = label_name
        self.depth_type = depth_type

        self.target_modules = _as_list(target_modules)
        self.pred_names = _as_list(pred_names)
        assert len(self.target_modules) == len(self.pred_names)
        self.valid_hfov = valid_hfov
        self.gt_scale = gt_scale
        self._theta_x = None

    def _get_theta_x(self, label, depth):
        assert (
            "virtual_cam_params" in label
            and label["virtual_cam_params"] is not None
        )
        virtual_cam_params = label["virtual_cam_params"]
        self._theta_x = get_theta_x_for_cylindrical_camera(
            depth, virtual_cam_params
        )
        return self._theta_x

    def _depth_decoder(self, label, depth, eps=1e-6):
        b, _, h, w = depth.shape
        if self.downsample:
            pad_depth = self.zero_pad(depth)
            depth = pad_depth[:, :, 0:h, 0:w]
            max_pool_depth = self.max_pooling(depth)
            depth *= -1
            depth[depth == 0] = float("-inf")
            min_pool_depth = self.max_pooling(depth) * -1
            depth = torch.min(min_pool_depth, max_pool_depth)

        mask = (depth > self.low_depth) * (depth <= self.high_depth)

        if self.depth_type == "Cylindrical":
            if self._theta_x is None:
                theta = self._get_theta_x(label, depth)
            else:
                theta = self._theta_x
            depth = torch.abs(
                torch.div(depth, torch.cos(theta[:b].unsqueeze(dim=1)))
            )

        if self.valid_hfov is not None:
            if self._theta_x is None:
                self._get_theta_x(label, depth)

            negative = torch.full(
                self._theta_x[:1].shape, False, device=self._theta_x.device
            )
            negative[torch.abs(self._theta_x[:1]) > self.valid_hfov / 2] = True
            negative = negative[:, None]
            negative = negative.repeat(b, 1, 1, 1)
            mask[negative] = False

            depth[negative] = 0.0

        depth = depth / self.gt_scale

        return depth.detach(), mask

    def _get_depth_target(self, label):
        depth = label[self.label_name]
        depth, mask = self._depth_decoder(label, depth)
        return depth, mask

    def __call__(self, label, preds):
        loss_input_list = []
        depth, mask = self._get_depth_target(label)
        label[self.label_name] = depth
        label["mask"] = mask
        for i in range(len(self.target_modules)):
            loss_input_list.extend(
                self.target_modules[i](label, preds[self.pred_names[i]])
            )
        return loss_input_list


@OBJECT_REGISTRY.register
class DepthValueTarget(object):
    """Get targets that directly supervise depth value.

    Args:
        with_smooth: Whether use origin-img as input of regularizatoin
            loss of textureless area.
        with_virtual_normal: Whether use virtual normal loss.
        label_name: The key corresponding to the gt seg in
            label. Default: gt_depth
    """

    def __init__(
        self,
        with_smooth: bool = False,
        with_virtual_normal: bool = False,
        label_name: str = "gt_depth",
    ):
        self.with_smooth = with_smooth
        self.label_name = label_name
        self.with_virtual_normal = with_virtual_normal

    def __call__(self, label, preds):
        loss_input = OrderedDict()
        loss_input["pred"] = []
        loss_input["target"] = []

        depth = label[self.label_name]

        target_dict = {}
        target_dict["mask"] = label.get("mask", None)

        targets = []
        pred_list = _as_list(preds)

        if self.with_smooth:
            pred_list.extend(preds)

        if self.with_virtual_normal:
            pred_list.extend(preds)

        target_dict["gt_depth"] = depth
        for _ in range(len(pred_list)):
            if self.with_smooth:
                assert "origin_img" in label
                target_dict["origin_img"] = label["origin_img"]
            targets.append(target_dict)

        loss_input["pred"].extend(pred_list)
        loss_input["target"].extend(targets)

        if len(pred_list) == 1:
            targets = loss_input["target"][0]
            loss_input["pred"] = pred_list[0]
            loss_input["target"] = targets
        return [loss_input]
