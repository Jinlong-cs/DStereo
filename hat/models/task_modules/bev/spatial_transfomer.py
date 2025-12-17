# Copyright (c) Horizon Robotics. All rights reserved.
import random
import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
)
from horizon_plugin_pytorch.quantization.stubs import QuantStub as HQuantStub
from torch.quantization import QConfig

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

grid_qconfig = QConfig(
    activation=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=qinfo("qint16").min,
        quant_max=qinfo("qint16").max,
        dtype="qint16",
        saturate=True,
    ),
    weight=None,
)

__all__ = [
    "SpatialTransfomer",
    "SpatialTransfomerFixedOffset",
    "SpatialTransfomerWithOffset",
    "RandomRotation",
]


def get_random_idx(prob: float, n: int) -> Optional[int]:
    """
    Generate a random index in [0,1,...n-1] with a probability.

    Args:
        prob: probability.
        n: max index.
    Returns:
        a randoms index.
    """
    flag = np.random.binomial(1, prob, 1).item()
    idx = random.randint(0, n - 1) if flag else None
    return idx


def get_random_idxs(prob: float, idxs: Sequence) -> Sequence:
    """
    Return idxs with a probability.

    Args:
        prob: probability.
        idxs: idx sequence.
    Returns:
        None if flag is False, else idxs.
    """
    flag = np.random.binomial(1, prob, 1).item()
    return idxs if flag else None


class SpatialTransfomer(nn.Module):  # noqa: D205,D400
    """
    Layer which transform feature from one view
    to other view with homography matrix.
    NOTE:
    when block_warp_padding is None, this module will do grid_sample directly.
    and block_warp_padding padding is not None, workflow will as below:
    1) calculate homography offset with homography matrix.
    2) crop homography offset with block_warp_padding parameter.
    3) do grid_sample.
    4) concatenate features if multi_warp.
    5) pad the warp result with block_warp_padding parameter.

    Args:
        height: height of grid.
        width: width of grid.
        grid_quant_scale: quant scale of grid.
        mode: mode for grid_sample.
        padding_mode: padding_mode for grid_sample.
        eps: a small value to avoid overflow.
        use_horizon_grid_sample: whether use grid_sample op
            in horizon plugin.
        block_warp_padding: order is (left,right,up,bottom).
        multi_warp_nums: number of multiple warps. A feature can be warped
            multiple times then fused, which requires the offset to be a list.

    """

    def __init__(
        self,
        height: int,
        width: int,
        grid_quant_scale: Optional[float] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        eps: float = 1e-7,
        use_horizon_grid_sample: bool = True,
        block_warp_padding: Tuple[int, int, int, int] = None,
        multi_warp_nums: int = 1,
    ):
        super(SpatialTransfomer, self).__init__()

        self.height = height
        self.width = width
        self.mode = mode
        self.padding_mode = padding_mode
        self.eps = eps
        self.use_horizon_grid_sample = use_horizon_grid_sample

        meshgrid = np.meshgrid(range(width), range(height), indexing="xy")
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)  # (2,h,w)

        ones = np.ones((1, height, width), dtype="float32")
        pix_coords = np.concatenate([id_coords, ones], axis=0).reshape(
            (1, 3, -1)
        )

        self.pix_coords = nn.Parameter(
            torch.from_numpy(pix_coords), requires_grad=False
        )

        self.block_warp_padding = block_warp_padding
        if block_warp_padding:
            assert len(block_warp_padding) == 4
            assert all([_ >= 0 for _ in block_warp_padding])
            assert (
                block_warp_padding[0] < width and block_warp_padding[1] < width
            )
            assert (
                block_warp_padding[2] < height
                and block_warp_padding[3] < height
            )
            self.zero_pad = nn.ZeroPad2d(padding=block_warp_padding)

        if use_horizon_grid_sample:
            self.grid_samples = nn.ModuleList(
                [
                    hnn.GridSample(mode=mode, padding_mode=padding_mode)
                    for _ in range(multi_warp_nums)
                ]
            )
        else:
            self.grid_sample = F.grid_sample
        self.grid_quant_stubs = nn.ModuleList(
            [
                HQuantStub(scale=grid_quant_scale)
                for _ in range(multi_warp_nums)
            ]
        )

        self.multi_warp_nums = multi_warp_nums
        if multi_warp_nums > 1:
            # Temporarily only support one fusion method
            self.feat_cat = nn.quantized.FloatFunctional()
            self.valid_point_cat = nn.quantized.FloatFunctional()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load checkpoints of previous version."""
        # the key is different in early versions
        # for example, 'grid_sample' become 'grid_samples' now
        st_keys = [key for key in state_dict if key.startswith(prefix)]
        new_st_state_dict = {}
        for key in st_keys:
            if prefix + "grid_sample." in key:
                warnings.warn(
                    "Deprecated warning: the old parameters in "
                    "SpatialTransfomer in qat/int_infer "
                    "stage will be deprecated, now update to the "
                    "new parameters automatically."
                )
                new_key = key.replace(
                    prefix + "grid_sample.", prefix + "grid_samples.0."
                )
                new_st_state_dict[new_key] = state_dict[key]
                state_dict.pop(key)
            if prefix + "grid_quant_stub." in key:
                new_key = key.replace(
                    prefix + "grid_quant_stub.", prefix + "grid_quant_stubs.0."
                )
                new_st_state_dict[new_key] = state_dict[key]
                state_dict.pop(key)
        state_dict.update(new_st_state_dict)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_qconfig(self) -> None:
        for module in self.grid_quant_stubs:
            module.qconfig = grid_qconfig

    def forward(
        self,
        feature: Union[torch.Tensor, QTensor],
        transformation: Union[torch.Tensor, Sequence],
    ) -> Tuple:
        """
        Forward method.

        Args:
            feature : a tensor,shape is (n,c,h,w)
            transformation : a tensor with shape of (n,3,3), or a list with
                shape num_warps*(n,3,3).
        """
        _, _, h, w = feature.shape

        transformation = _as_list(transformation)
        assert len(transformation) == self.multi_warp_nums, (
            "the `multi_warp_nums` must be equal to the homo number,"
            f" got {self.multi_warp_nums} vs {len(transformation)}"
        )
        multi_feats = []
        multi_valid_points = []
        for i in range(self.multi_warp_nums):
            cam_points = torch.matmul(transformation[i], self.pix_coords)

            # convert to float32 in case of float16
            # without below will cause bug in amp mode
            cam_points = cam_points.to(self.pix_coords.dtype)
            # donot convert to fp32 in qat training
            if type(feature) == torch.Tensor:
                feature = feature.to(self.pix_coords.dtype)

            new_pix_coords = cam_points[:, :2, :] / (
                cam_points[:, 2, :].unsqueeze(1) + self.eps
            )
            valid_points_x = (new_pix_coords[:, 0] - w / 2).abs() < (w / 2)
            valid_points_y = (new_pix_coords[:, 1] - h / 2).abs() < (h / 2)
            valid_points = (valid_points_x * valid_points_y).view(
                -1, 1, self.height, self.width
            )

            new_pix_coords = new_pix_coords - self.pix_coords[:, :2, :]
            new_pix_coords = new_pix_coords.view(
                -1, 2, self.height, self.width
            )
            new_pix_coords = new_pix_coords.permute(0, 2, 3, 1)

            if self.use_horizon_grid_sample:
                if self.block_warp_padding:
                    pad_l, pad_r, pad_u, pad_b = self.block_warp_padding
                    new_pix_coords = new_pix_coords[
                        :,
                        pad_u : self.height - pad_b,
                        pad_l : self.width - pad_r,
                        :,
                    ]
                    new_pix_coords[:, :, :, 0] += pad_l
                    new_pix_coords[:, :, :, 1] += pad_u
                multi_feats.append(
                    self.grid_samples[i](
                        feature, self.grid_quant_stubs[i](new_pix_coords)
                    )
                )

            else:
                new_pix_coords[..., 0] /= w - 1
                new_pix_coords[..., 1] /= h - 1
                new_pix_coords = (new_pix_coords - 0.5) * 2
                valid_points = (
                    new_pix_coords.abs().max(dim=-1)[0].unsqueeze(1) <= 1
                )
                multi_feats.append(
                    self.grid_sample(
                        feature,
                        new_pix_coords,
                        mode=self.mode,
                        padding_mode=self.padding_mode,
                        align_corners=False,
                    )
                )
            multi_valid_points.append(valid_points.float())
        if self.multi_warp_nums > 1:
            homography_feat = self.feat_cat.cat(multi_feats, dim=1)
            valid_points = self.valid_point_cat.cat(multi_valid_points, dim=1)
        else:
            homography_feat = multi_feats[0]
            valid_points = multi_valid_points[0]

        if self.use_horizon_grid_sample and self.block_warp_padding:
            homography_feat = self.zero_pad(homography_feat)

        return homography_feat, valid_points


class SpatialTransfomerFixedOffset(nn.Module):  # noqa: D205,D400
    """
    Layer which transform feature from one view
    to other view with fixed homography offset.

    Args:
        height: height of grid.
        width: width of grid.
        grid_quant_scale: quant scale of grid.
        homography: homography matrix.
        homo_offset: homo_offset matrix.
        mode: mode for grid_sample.
        padding_mode: padding_mode for grid_sample.
        multi_warp_nums: number of multiple warps. A feature can be warped
            multiple times then fused, which requires the homography or
            homo_offset to be a list.

    """

    def __init__(
        self,
        height: int,
        width: int,
        homography: Optional[Union[torch.Tensor, list]] = None,
        homo_offset: Optional[Union[torch.Tensor, list]] = None,
        grid_quant_scale: Optional[float] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        eps: float = 1e-7,
        multi_warp_nums: int = 1,
    ):
        super(SpatialTransfomerFixedOffset, self).__init__()

        self.height = height
        self.width = width
        self.mode = mode
        self.padding_mode = padding_mode
        self.eps = eps
        self.offset = nn.ParameterList()
        if homo_offset is not None:
            homo_offset = _as_list(homo_offset)
            assert len(homo_offset) == multi_warp_nums, (
                "the `multi_warp_nums` must be equal to the homo number,"
                f" got {multi_warp_nums} vs {len(homo_offset)}"
            )
            for each_homo in homo_offset:
                self.offset.append(
                    nn.Parameter(each_homo, requires_grad=False)
                )
        elif homography is not None:
            homography = _as_list(homography)
            assert len(homography) == multi_warp_nums, (
                "the `multi_warp_nums` must be equal to the homo number,"
                f" got {multi_warp_nums} vs {len(homography)}"
            )
            for each_homo in homography:
                self.offset.append(
                    nn.Parameter(
                        self.generate_homo_offset(each_homo),
                        requires_grad=False,
                    )
                )
        else:
            raise ValueError(
                "only one of (homo_offset,homography) should be None"
            )
        self.grid_samples = nn.ModuleList()
        self.grid_quant_stubs = nn.ModuleList()
        for _ in range(multi_warp_nums):
            self.grid_samples.append(
                hnn.GridSample(mode=mode, padding_mode=padding_mode)
            )
            self.grid_quant_stubs.append(HQuantStub(scale=grid_quant_scale))
        self.multi_warp_nums = multi_warp_nums
        if multi_warp_nums > 1:
            self.cat = nn.quantized.FloatFunctional()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load checkpoints of previous version."""
        # the key is different in early versions
        # for example, 'grid_sample' become 'grid_samples' now
        st_keys = [key for key in state_dict if key.startswith(prefix)]
        new_st_state_dict = {}
        for key in st_keys:
            if prefix + "grid_sample." in key:
                warnings.warn(
                    "Deprecated warning: the old parameters in "
                    "SpatialTransfomerFixedOffset in qat/int_infer "
                    "stage will be deprecated, now update to the "
                    "new parameters automatically."
                )
                new_key = key.replace(
                    prefix + "grid_sample.", prefix + "grid_samples.0."
                )
                new_st_state_dict[new_key] = state_dict[key]
                state_dict.pop(key)
            if prefix + "grid_quant_stub." in key:
                new_key = key.replace(
                    prefix + "grid_quant_stub.", prefix + "grid_quant_stubs.0."
                )
                new_st_state_dict[new_key] = state_dict[key]
                state_dict.pop(key)
        state_dict.update(new_st_state_dict)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def generate_homo_offset(self, homography: torch.Tensor) -> torch.Tensor:
        # NOTE: Calculating homo_offset on cpu here but gpu durning training.
        # The calculation results of the two methods are slightly different,
        # but basically do not affect the final result.
        y, x = torch.meshgrid(
            torch.arange(self.height), torch.arange(self.width)
        )
        id_coords = torch.stack([x, y], dim=0).float()
        ones = torch.ones((1, self.height, self.width)).float()
        pix_coords = torch.cat([id_coords, ones], dim=0).reshape((1, 3, -1))

        cam_points = torch.matmul(homography, pix_coords)

        new_pix_coords = cam_points[:, :2, :] / (
            cam_points[:, 2, :].unsqueeze(1) + self.eps
        )
        homo_offset = new_pix_coords - pix_coords[:, :2, :]
        homo_offset = homo_offset.view(-1, 2, self.height, self.width)
        homo_offset = homo_offset.permute(0, 2, 3, 1)
        return homo_offset

    def forward(
        self, feature: Union[torch.Tensor, QTensor]
    ) -> Union[torch.Tensor, QTensor]:
        """
        Foward method.

        Args:
            feature : a tensor,shape is (n,c,h,w)
        """
        multi_feats = []
        for i in range(self.multi_warp_nums):
            offset_each_plane = self.offset[i]
            multi_feats.append(
                self.grid_samples[i](
                    feature, self.grid_quant_stubs[i](offset_each_plane)
                )
            )
        if self.multi_warp_nums > 1:
            homography_feat = self.cat.cat(multi_feats, dim=1)
        else:
            homography_feat = multi_feats[0]
        return homography_feat

    def set_qconfig(self) -> None:
        for module in self.grid_quant_stubs:
            module.qconfig = grid_qconfig


class SpatialTransfomerWithOffset(nn.Module):  # noqa: D205,D400
    """
    Layer which transform feature from one view
    to other view with homography offset.
    NOTE:
    when block_warp_padding is None, this module will do grid_sample directly.
    and when block_warp_padding is not None, workflow will as below:
    1) crop homography offset with block_warp_padding parameter.
    2) do grid_sample.
    3) concatenate features if multi_warp.
    4) pad the warp result with block_warp_padding parameter.

    Args:
        height: height of grid.
        width: width of grid.
        grid_quant_scale: quant scale of grid.
        mode: mode for grid_sample.
        padding_mode: padding_mode for grid_sample.
        block_warp_padding: order is (left,right,up,bottom).
        compile_model: compile model or not. only for int_infer step.
        multi_warp_nums: number of multiple warps. A feature can be warped
            multiple times then fused, which requires the offset to be a list.
    """

    def __init__(
        self,
        height: int,
        width: int,
        grid_quant_scale: Optional[float] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        block_warp_padding: Tuple[int, int, int, int] = None,
        compile_model: bool = False,
        multi_warp_nums: int = 1,
    ):
        super(SpatialTransfomerWithOffset, self).__init__()
        self.height = height
        self.width = width
        self.grid_sample = hnn.GridSample(mode=mode, padding_mode=padding_mode)
        self.grid_quant_stub = HQuantStub(scale=grid_quant_scale)
        self.compile_model = compile_model
        self.block_warp_padding = block_warp_padding
        if block_warp_padding:
            assert len(block_warp_padding) == 4
            assert all([_ >= 0 for _ in block_warp_padding])
            assert (
                block_warp_padding[0] < width and block_warp_padding[1] < width
            )
            assert (
                block_warp_padding[2] < height
                and block_warp_padding[3] < height
            )
            self.zero_pad = nn.ZeroPad2d(padding=block_warp_padding)
        self.multi_warp_nums = multi_warp_nums
        if multi_warp_nums > 1:
            # Temporarily only support one fusion method
            self.cat = nn.quantized.FloatFunctional()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load checkpoints of previous version."""
        # the key is different in early versions
        # for example, 'grid_sample' become 'grid_samples' now
        st_keys = [key for key in state_dict if key.startswith(prefix)]
        new_st_state_dict = {}
        for key in st_keys:
            if prefix + "grid_samples.0." in key:
                warnings.warn(
                    "Deprecated warning: the old parameters in "
                    "SpatialTransfomerWithOffset in qat/int_infer "
                    "stage will be deprecated, now update to the "
                    "new parameters automatically."
                )
                new_key = key.replace(
                    prefix + "grid_samples.0.", prefix + "grid_sample."
                )
                new_st_state_dict[new_key] = state_dict[key]
                state_dict.pop(key)
            if prefix + "grid_quant_stubs.0." in key:
                new_key = key.replace(
                    prefix + "grid_quant_stubs.0.", prefix + "grid_quant_stub."
                )
                new_st_state_dict[new_key] = state_dict[key]
                state_dict.pop(key)
        state_dict.update(new_st_state_dict)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        feature: Union[torch.Tensor, QTensor],
        offset: Union[torch.Tensor, Sequence],
    ) -> Union[torch.Tensor, QTensor]:
        """
        Foward method.

        Args:
            feature : a tensor,shape is (n,c,h,w)
            offset : a tensor with shape (n,h,w,2) or a list with
                shape num_warps*(n,h,w,2).
        """
        offset = _as_list(offset)
        assert len(offset) == self.multi_warp_nums, (
            "the `multi_warp_nums` must be equal to the homo number,"
            f" got {self.multi_warp_nums} vs {len(offset)}"
        )

        if not self.compile_model:

            bs = int(offset[0].shape[0])
            # offset：[(bs,h,w,2),(bs,h,w,2),(bs,h,w,2)，(bs,h,w,2)..]
            #         -> (warp_nums*bs,h,w,2)
            offset = torch.cat(offset, dim=0)

            # feature: (bs,c,h,w) -> (warp_nums*bs,c,h,w)
            feature = feature.repeat(self.multi_warp_nums, 1, 1, 1)
            if self.block_warp_padding:
                pad_l, pad_r, pad_u, pad_b = self.block_warp_padding
                block_warp_offset = torch.clone(
                    offset[
                        :,
                        pad_u : self.height - pad_b,
                        pad_l : self.width - pad_r,
                        :,
                    ]
                )
                block_warp_offset[:, :, :, 0] += pad_l
                block_warp_offset[:, :, :, 1] += pad_u
            else:
                block_warp_offset = offset
            multi_feats = self.grid_sample(
                feature,
                self.grid_quant_stub(block_warp_offset),
            )
            # multi_feats: (warp_nums*bs,c,h,w)
            #           -> [(bs,c,h,w),(bs,c,h,w)..]
            multi_feats = torch.split(multi_feats, bs, dim=0)
        else:
            multi_feats = []
            for i in range(self.multi_warp_nums):
                block_warp_offset = offset[i]
                multi_feats.append(
                    self.grid_sample(
                        feature,
                        self.grid_quant_stub(block_warp_offset),
                    )
                )
        if self.multi_warp_nums > 1:
            homography_feat = self.cat.cat(multi_feats, dim=1)
        else:
            homography_feat = multi_feats[0]
        if self.block_warp_padding:
            homography_feat = self.zero_pad(homography_feat)

        return homography_feat

    def set_qconfig(self) -> None:
        self.grid_quant_stub.qconfig = grid_qconfig


@OBJECT_REGISTRY.register
class RandomRotation(nn.Module):
    """
    Random rotate data durning BEV training.

    Args:
        height: height of grid.
        grid_quant_scale: quanti scale of grid.
        width: width of grid.
        angles: rotation angle
        mode: mode for grid_sample.
        padding_mode: padding_mode for grid_sample.
        use_horizon_grid_sample: whether use grid_sample op
            in horizon plugin.

    """

    def __init__(
        self,
        height: int,
        width: int,
        grid_quant_scale: float,
        angles: Sequence[int] = (0,),
        mode: str = "bilinear",
        padding_mode: str = "border",
        eps: float = 1e-7,
        use_horizon_grid_sample: bool = True,
    ):
        super(RandomRotation, self).__init__()
        self.angle_num = len(angles)

        rot_mats = []
        trans_mat1 = (
            np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
            .reshape(3, 3)
            .astype("float32")
        )
        trans_mat2 = (
            np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]])
            .reshape(3, 3)
            .astype("float32")
        )
        for angle in angles:
            assert angle in [
                0,
                90,
                180,
                270,
            ], "rotation angle must in [0, 90, 180, 270]"
            r = (
                np.array(
                    [
                        [
                            np.cos(np.deg2rad(angle)),
                            -np.sin(np.deg2rad(angle)),
                            0,
                        ],
                        [
                            np.sin(np.deg2rad(angle)),
                            np.cos(np.deg2rad(angle)),
                            0,
                        ],
                        [0, 0, 1],
                    ]
                )
                .reshape(3, 3)
                .astype("float32")
            )
            rot_mats.append(trans_mat2 @ r @ trans_mat1)

        rot_mats = np.stack(rot_mats)
        self.rotation_mat = nn.Parameter(
            torch.from_numpy(rot_mats), requires_grad=False
        )

        self.st = SpatialTransfomer(
            height,
            width,
            grid_quant_scale=grid_quant_scale,
            mode=mode,
            padding_mode=padding_mode,
            eps=eps,
            use_horizon_grid_sample=use_horizon_grid_sample,
        )

    def get_rot_mat(self, nums: int) -> torch.Tensor:
        idx = np.random.randint(0, self.angle_num, size=nums)
        return self.rotation_mat[idx]

    def forward(self, datas: Sequence) -> Union[Sequence[Any], torch.Tensor]:
        """
        Forward method.

        Args:
            datas (list[tensor]): a list tensor to rotation.
        Returns:
            result: (list[tensor]): a list tensor after rotation.
            rot_mat: (tensor): a rotation matrix.

        """
        bs = datas[0].shape[0]
        rot_mat = self.get_rot_mat(bs)  # (b,3,3)
        result = []
        for data in datas:
            result.append(self.st(data, rot_mat)[0])
        return result, rot_mat

    def set_qconfig(self) -> None:
        self.st.set_qconfig()
