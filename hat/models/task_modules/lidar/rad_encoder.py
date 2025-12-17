import logging
import math
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__file__)

__all__ = [
    "RadFeatureExtractor",
    "RadScatter",
    "RadGridMaker",
    "RadFeatureConcat",
]


def masked_max(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    keepdim: bool = True,
):
    """Find max value in dim of tensor with valid mask.

    Args:
        dim: max value in which dim.
        keepdim : finaly return tensor with origin dim.

    Returns:
        the max value tensor.
    """
    tensorc = tensor.clone()
    tensorc[mask] = float("-inf")
    return torch.max(tensorc, dim=dim, keepdim=keepdim)[0]


def masked_min(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    keepdim: bool = True,
):
    """Find min value in dim of tensor with valid mask.

    Args:
        dim: min value in which dim.
        keepdim : finaly return tensor with origin dim.

    Returns:
        the min value tensor.
    """
    tensorc = tensor.clone()
    tensorc[mask] = float("inf")
    return torch.min(tensorc, dim=dim, keepdim=keepdim)[0]


def get_normalize_kb(prev_range: List[int], new_range: List[int]):
    """Get normalize paramter k(scale) and b(offset).

    Args:
        prev_range: pre value range.
        new_range: use new range put value.

    Returns:
        k:scale translate form pre_range to new range.
        b:quantization offset.
    """
    if not new_range:
        k = 1
        b = 0
    else:
        k = (new_range[0] - new_range[1]) / (prev_range[0] - prev_range[1])
        b = new_range[0] - prev_range[0] * k
    return k, b


@OBJECT_REGISTRY.register_module
class RadFeatureExtractor(nn.Module):
    """Rad's Feature Extractor.

    Extractor points feature with some op.
    You can read paper <Realtime and Accurate 3D Object Detection>.

    Args:
    num_input_features:  Number of input features.

    """

    def __init__(
        self,
        num_input_features: int = 4,
    ):
        super(RadFeatureExtractor, self).__init__()
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        # TODO by shijie.sun
        # points_mean = features[:, :, : self.num_input_features].sum(
        #     dim=1, keepdim=False
        # ) / num_voxels.type_as(features).view(-1, 1)
        return features.contiguous()


@OBJECT_REGISTRY.register_module
class RadScatter(nn.Module):
    def __init__(
        self,
        intensity_range: List[int],
        density_range: List[int],
        add_intensity=True,
        add_density: bool = False,
        add_location=False,
        occupancy_range: List[float] = None,
        add_pp_features: bool = False,
        add_occupancy: bool = True,
        use_occupancy_density: bool = False,
        num_input_features: int = 64,
        max_points_in_voxel: int = 5,
        max_points_in_pillar: int = 30,
        add_maxz: bool = False,
        maxz_range: List[float] = None,
        add_timestamp: bool = False,
        is_vcs: bool = False,
    ):
        """Rad's Scatter.

        Converts learned features from dense tensor to sparse pseudo image.
        This replaces SECOND's second.pytorch.voxelnet.SparseMiddleExtractor.
        You can read paper <Realtime and Accurate 3D Object Detection>.

        Args:
        num_input_features:  Number of input features.
        add_intensity: add points intensity feature into input tensor.
        add_density:bool: add points density feature into input tensor.
        add_location=False: add location map into input tensor.
        add_pp_features: add point pillars feature into input tensor.
        add_occupancy:add point pillars feature into input tensor.
        add_maxz:add point max z value feature into input tensor.
        add_timestamp:add timestamp feature into input tensor.
        intensity_range: range of intensity.
        density_range: range of density.
        occupancy_range:range of occupancy.
        maxz_range:List[float]=None.
        is_vcs: use vertical coordinate systems.
        """

        super().__init__()
        self.nchannels = num_input_features

        # occupancy
        self.add_occupancy = add_occupancy
        self.use_occupancy_density = use_occupancy_density
        if self.add_occupancy:
            if self.use_occupancy_density:
                max_points_in_voxel = max_points_in_voxel
                normalize_upper_bound = (
                    math.ceil(math.log(max_points_in_voxel + 1) * 100) / 100
                )
                occupancy_origin_range = [0, normalize_upper_bound]
            else:
                occupancy_origin_range = [0, 1]
            self.occupancy_range = occupancy_range
            self.ko, self.bo = get_normalize_kb(
                occupancy_origin_range, self.occupancy_range
            )
        # pp_features
        self.add_pp_features = add_pp_features
        # intensity
        self.add_intensity = add_intensity
        if self.add_intensity:
            self.intensity_range = intensity_range
            self.ki, self.bi = get_normalize_kb([0, 255], self.intensity_range)
        # density
        self.add_density = add_density
        if self.add_density:
            self.density_range = density_range
            max_points_in_pillar = max_points_in_pillar
            normalize_upper_bound = (
                math.ceil(math.log(max_points_in_pillar + 1) * 100) / 100
            )
            self.kd, self.bd = get_normalize_kb(
                [0, normalize_upper_bound], self.density_range
            )
        # maxz
        self.add_maxz = add_maxz
        if self.add_maxz:
            self.maxz_range = maxz_range
            self.kz, self.bz = get_normalize_kb([-4, 2], self.maxz_range)

        self.add_location = add_location
        self.add_timestamp = add_timestamp
        self.is_vcs = is_vcs

    def forward(
        self,
        batch_size: int,
        input_shape: np.ndarray,
        voxel_features: torch.Tensor = None,
        coords: np.ndarray = None,
        num_in_voxels: np.ndarray = None,
        voxel_features_pillars: torch.Tensor = None,
        coords_pillars: np.ndarray = None,
        num_in_pillars: np.ndarray = None,
    ):

        output_list = []

        if self.add_occupancy:
            # Get a four-dimensional background tensor convas.
            # The tensor element of the background plate is [BS, Z, y, x],
            # which is equivalent to voxel operation

            batch_occupancy = self.forward_occupancy(
                voxel_features,
                coords,
                num_in_voxels,
                batch_size,
                input_shape,
                use_density=self.use_occupancy_density,
            )
            batch_occupancy = self.ko * batch_occupancy + self.bo
            output_list.append(batch_occupancy)
        if self.add_pp_features:
            batch_pp_features = self.forward_features(
                voxel_features_pillars,
                coords_pillars,
                num_in_pillars,
                batch_size,
                input_shape,
                dimension=[4, 9],
                mode="mean",
            )
            output_list.append(batch_pp_features)
        if self.add_intensity:
            # normalize before scatter will make pillars with 0 points stay
            # with 0 intensity, not physically right but is consistent with
            # deployment code
            voxel_features_pillars = self.ki * voxel_features_pillars + self.bi
            # Get a four-dimensional background tensor convas,
            # and the tensor element of the background plate is
            # [BS, fear_dim * point_num, y, x]
            # This is the pillars way

            batch_intensity = self.forward_features(
                voxel_features_pillars,
                coords_pillars,
                num_in_pillars,
                batch_size,
                input_shape,
                dimension=3,
                mode="max",
            )
            output_list.append(batch_intensity)

        # add max z and density
        if self.add_density:
            batch_density = self.forward_density(
                voxel_features_pillars,
                coords_pillars,
                num_in_pillars,
                batch_size,
                input_shape,
            )
            batch_density = self.kd * batch_density + self.bd
            output_list.append(batch_density)
        if self.add_timestamp:
            batch_timestamp = self.forward_features(
                voxel_features_pillars,
                coords_pillars,
                num_in_pillars,
                batch_size,
                input_shape,
                dimension=4,
                mode="max",
            )
            batch_timestamp = batch_timestamp.to(batch_occupancy.device)
            output_list.append(batch_timestamp)
        if self.add_maxz:
            batch_maxz = self.forward_features(
                voxel_features_pillars,
                coords_pillars,
                num_in_pillars,
                batch_size,
                input_shape,
                dimension=2,
                mode="max",
            )
            batch_maxz = self.kz * batch_maxz + self.bz
            output_list.append(batch_maxz)
        if self.add_location:
            batch_location = self.forward_location(
                voxel_features_pillars,
                coords_pillars,
                num_in_pillars,
                batch_size,
                input_shape,
            )
            output_list.append(batch_location)

        ret_tensor = torch.cat(output_list, dim=1)  # Splicing on channel
        return ret_tensor

    def forward_occupancy(
        self,
        voxel_features: torch.Tensor,
        coords: np.ndarray,
        num_in_voxels: np.ndarray,
        batch_size: int,
        input_shape: np.ndarray,
        use_density: bool = False,
    ):
        """Make occupancy tensor."""

        self.nx = input_shape[0]
        self.ny = input_shape[1]
        self.nz = input_shape[2]

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nx * self.ny * self.nz,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            if self.is_vcs:
                indices = (
                    this_coords[:, 1] * self.nx * self.ny
                    + (self.nx - this_coords[:, 3] - 1) * self.ny
                    + (self.ny - this_coords[:, 2] - 1)
                )
            else:
                indices = (
                    this_coords[:, 1] * self.nx * self.ny
                    + this_coords[:, 2] * self.nx
                    + this_coords[:, 3]
                )
            indices = indices.type(torch.long)

            # Now scatter the blob back to the canvas.
            if use_density:
                batch_num_in_voxels = num_in_voxels.to(
                    voxel_features.device
                ).to(voxel_features.dtype)[batch_mask]
                batch_num_in_voxels = torch.log(batch_num_in_voxels + 1.0)
                canvas[indices] = batch_num_in_voxels
            else:
                canvas[indices] = 1

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels*nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        if self.is_vcs:
            batch_canvas = batch_canvas.view(
                batch_size, self.nz, self.nx, self.ny
            )
        else:
            batch_canvas = batch_canvas.view(
                batch_size, self.nz, self.ny, self.nx
            )
        return batch_canvas

    def forward_features(
        self,
        voxel_features: torch.Tensor,
        coords: np.ndarray,
        num_in_pillars: np.ndarray,
        batch_size: int,
        input_shape: np.ndarray,
        dimension: Any,
        mode: str = "max",
    ):
        """Make feature tensor.

        Gather feature to concat tensor in channel.

        """
        if not isinstance(dimension, (list, tuple)):
            dimension = (dimension, dimension + 1)
        nchannels = dimension[1] - dimension[0]
        # voxel_features = voxel_features[:, :, dimension[0]:dimension[1]]

        if mode == "max":
            mask = torch.all(voxel_features == 0, dim=2, keepdim=True)
            mask = mask.repeat([1, 1, voxel_features.shape[2]])
            voxel_features = masked_max(
                voxel_features, mask, dim=1, keepdim=False
            )
        elif mode == "min":
            mask = torch.all(voxel_features == 0, dim=2, keepdim=True)
            mask = mask.repeat([1, 1, voxel_features.shape[2]])
            voxel_features = masked_min(
                voxel_features, mask, dim=1, keepdim=False
            )
        elif mode == "mean":
            voxel_features = torch.sum(
                voxel_features, dim=1, keepdim=False
            ) / num_in_pillars.type_as(voxel_features).view(-1, 1)
        elif mode == "sum":
            voxel_features = torch.sum(voxel_features, dim=1, keepdim=False)
        self.nx = input_shape[0]
        self.ny = input_shape[1]
        self.nz = input_shape[2]

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )  # [1,x*y]

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            if self.is_vcs:
                indices = (self.nx - this_coords[:, 3] - 1) * self.ny + (
                    self.ny - this_coords[:, 2] - 1
                )
            else:
                indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[
                batch_mask, :
            ]  # [point_max_num,point_info_dim]
            voxels = voxels.t()  # [point_info_dim,point_max_num]

            # Now scatter the blob back to the canvas.
            # canvas[indices] = 1
            canvas[:, indices] = voxels[dimension[0] : dimension[1], :]

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        if self.is_vcs:
            batch_canvas = batch_canvas.view(
                batch_size, nchannels, self.nx, self.ny
            )
        else:
            batch_canvas = batch_canvas.view(
                batch_size, nchannels, self.ny, self.nx
            )  # [bs,1*point_max_num,y,x]
        return batch_canvas

    def forward_density(
        self,
        voxel_features: torch.Tensor,
        coords: np.ndarray,
        num_in_pillars: np.ndarray,
        batch_size: int,
        input_shape: np.ndarray,
    ):
        """Make density tensor."""
        self.nx = input_shape[0]
        self.ny = input_shape[1]
        self.nz = input_shape[2]

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            if self.is_vcs:
                indices = (self.nx - this_coords[:, 3] - 1) * self.ny + (
                    self.ny - this_coords[:, 2] - 1
                )
            else:
                indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            batch_num_in_pillars = num_in_pillars.to(voxel_features.device).to(
                voxel_features.dtype
            )[batch_mask]
            batch_num_in_pillars = torch.log(batch_num_in_pillars + 1.0)
            canvas[indices] = batch_num_in_pillars

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        if self.is_vcs:
            batch_canvas = batch_canvas.view(batch_size, 1, self.nx, self.ny)
        else:
            batch_canvas = batch_canvas.view(batch_size, 1, self.ny, self.nx)
        return batch_canvas

    def forward_location(
        self,
        voxel_features: torch.Tensor,
        coords: np.ndarray,
        num_in_pillars: np.ndarray,
        batch_size: int,
        input_shape: np.ndarray,
    ):
        """Make location tensor."""
        self.nx = input_shape[0]
        self.ny = input_shape[1]
        self.nz = input_shape[2]

        x_location = torch.arange(-self.nx // 2, self.nx // 2) * 2 / self.nx
        y_location = torch.flip(torch.arange(0, self.ny) / self.ny, dims=[0])
        y_grid, x_gird = torch.meshgrid(y_location, x_location)

        mesh_grid = (
            torch.stack((y_grid, x_gird), 2)
            .view((1, self.ny, self.nx, 2))
            .repeat(batch_size, 1, 1, 1)
            .permute(0, 3, 1, 2)
        )

        mesh_grid = mesh_grid.type_as(voxel_features).to(voxel_features.device)

        return mesh_grid


@OBJECT_REGISTRY.register_module
class RadGridMaker(nn.Module):
    def __init__(self, down_scale: int):
        """Rad's bev grid maker.

        Args:
        down_scale:down sampling scale.

        """

        super().__init__()
        self.grid_resizer = nn.MaxPool2d(
            kernel_size=down_scale, stride=down_scale
        )
        self.grid_relu = nn.ReLU()

        self.floatmod_sum = FloatFunctional()
        self.floatmod_addscalar = FloatFunctional()
        self.floatmod_sub = FloatFunctional()

        self.num = 0

    def make_bin_grid(self, tensor: torch.Tensor):

        max_tensor = self.floatmod_sum.sum(tensor, dim=1, keepdim=True)

        minus = self.floatmod_addscalar.add_scalar(max_tensor, -1)
        minus_relu = self.grid_relu(minus)
        grid = self.floatmod_sub.sub(max_tensor, minus_relu)

        return grid

    def make_high_map(
        self, tensor: torch.Tensor, start_channel: int, end_channel: int
    ):

        max_tensor = self.floatmod_sum.sum(
            tensor[:, start_channel:end_channel], dim=1, keepdim=True
        )

        minus = self.floatmod_addscalar.add_scalar(max_tensor, -1)
        minus_relu = self.grid_relu(minus)
        grid = self.floatmod_sub.sub(max_tensor, minus_relu)

        return grid

    def forward(
        self,
        x: torch.Tensor,
        start_channel: int,
        end_channel: int,
        with_bin: bool = False,
    ):
        x_high = self.make_high_map(
            x, start_channel=start_channel, end_channel=end_channel
        )
        if with_bin:
            x_bin = self.make_bin_grid(x)
            x = self.floatmod.add(x_bin, x_high)
        else:
            x = x_high
        x = self.grid_resizer(x)
        return x

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register_module
class RadFeatureConcat(nn.Module):
    """Rad's feature concat."""

    def forward(self, feature_list):
        features_cat = torch.stack(feature_list, dim=1)  # [frames, B, C, H, W]
        _, _, C, H, W = features_cat.shape
        features = features_cat.reshape(-1, C, H, W)
        return features.contiguous()
