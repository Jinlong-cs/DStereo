# Copyright (c) Horizon Robotics. All rights reserved.

import horizon_plugin_pytorch.nn as hnn
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

from hat.models.task_modules.bev.spatial_transfomer import SpatialTransfomer
from hat.models.task_modules.traj_pred.necks import MultiPathNeck
from hat.registry import OBJECT_REGISTRY

__all__ = ["MultipathPreProcess"]


@OBJECT_REGISTRY.register
class MultipathPreProcess(nn.Module):
    """The class is used for multipath preprocess.

    Since we use map categories as the model input, the model need a
    'LookUpTabel (LUT)' operator to convert map categories to RGB colors.
    In addition, some operations must be done between the 'LUT' and the
    backbone model are added in this class, such as affine transform.

    Args:
        data_shape (List, [C, H, W]): the input image shape of the backbone.
        table (dict): the color table. The keys are road elements and the
            values are the corresponding colors.
        road_map_colored (bool, optional): whether the road map is colored.
            Default to False. If this value is False, the parameter `table`
            cannot be NoneType.
        map_augmentation (bool, optional): whether to perform map augmentation.
        drivable_area_table (dict, optional): the drivable_area color table.
        use_drivable_area (bool, optional): whether use drivable area as input.
            If this value is True, the parameter 'drivable_area_table' cannot
            be NoneType. Default to False.
        is_int_infer_model (bool, optional): whether the model is for int
            inference. Default to False.
    """

    def __init__(
        self,
        data_shape,
        table=None,
        road_map_colored=False,
        map_augmentation=False,
        drivable_area_table=None,
        use_drivable_area=False,
        is_int_infer_model=False,
    ):
        super().__init__()
        self.is_int_infer_model = is_int_infer_model
        self.map_augmentation = map_augmentation
        if is_int_infer_model:
            self.map_augmentation = False
        self.road_map_colored = road_map_colored
        if not road_map_colored:
            assert table is not None, (
                "The road maps are not colored, but the input color table"
                "is Nonetype."
            )
        if table is not None:
            self.r_tab = hnn.LookUpTable(table["r_val"])
            self.g_tab = hnn.LookUpTable(table["g_val"])
            self.b_tab = hnn.LookUpTable(table["b_val"])

        self.use_drivable_area = use_drivable_area and (
            drivable_area_table is not None
        )
        if self.use_drivable_area:
            self.drivable_r_tab = hnn.LookUpTable(drivable_area_table["r_val"])
            self.drivable_g_tab = hnn.LookUpTable(drivable_area_table["g_val"])
            self.drivable_b_tab = hnn.LookUpTable(drivable_area_table["b_val"])

        # Quantization.
        self.quant = QuantStub(scale=None)
        self.road_quant = QuantStub(scale=1)
        self.cat_op = nn.quantized.FloatFunctional()
        if self.use_drivable_area:
            self.drivable_road_quant = QuantStub(scale=1)

        # Road map data augmentation.
        grid_quant_scale = MultiPathNeck.cal_roi_quanti_scale(
            data_shape[1:], data_shape[1:]
        )
        self.map_affine = SpatialTransfomer(
            height=data_shape[1],
            width=data_shape[2],
            padding_mode="border",
            grid_quant_scale=grid_quant_scale,
            eps=0,
            use_horizon_grid_sample=True,
        )

    def color_road_map(self, batch_road_maps):
        """Color the road map by look up table (LUT).

        Args:
            batch_road_maps (torch.tensor, [batch_size, 1, h, w]): the road
                maps. Each value is the type of road.

        Return:
            colored_maps (torch.tensor, [batch_size, 3, h, w]): the colored
                road map.
        """
        red_frames = self.r_tab(batch_road_maps)
        green_frames = self.g_tab(batch_road_maps)
        blue_frames = self.b_tab(batch_road_maps)
        colored_maps = self.cat_op.cat(
            [red_frames, green_frames, blue_frames], 1
        )
        return colored_maps

    def color_drivable_road_map(self, batch_road_maps):
        """Color the road map by look up table (LUT).

        Args:
            batch_road_maps (torch.tensor, [batch_size, 1, h, w]): the road
                maps. Each value is the type of road.

        Return:
            colored_maps (torch.tensor, [batch_size, 3, h, w]): the colored
                road map.
        """
        red_frames = self.drivable_r_tab(batch_road_maps)
        green_frames = self.drivable_g_tab(batch_road_maps)
        blue_frames = self.drivable_b_tab(batch_road_maps)
        colored_maps = self.cat_op.cat(
            [red_frames, green_frames, blue_frames], 1
        )
        return colored_maps

    def forward(self, data):
        """Forward.

        Args:
            data (Dict): the model input dictionary with the following keys:
                'road_map' (torch.Tensor, [batch_size, 1, h, w]): the road
                    maps. Each value is the type of road.
                'rendered_obs' (torch.Tensor, [batch_size, 3, h, w]): the
                    rendered obstacle occupancy map.

        Returns:
            batch_rendered_frames (torch.tensor, [batch_size, n, h, w]): the
                rendered result of current frame, road map, obstacles and
                drivable areas (optional) are concatenated together.
        """
        if self.is_int_infer_model:
            batch_road_maps = data["road_map"]
            batch_rendered_obs = data["rendered_obs"]
            if self.use_drivable_area:
                batch_drivable_road_maps = data["drivable_road_map"]
        else:
            batch_road_maps = data["road_map"].cuda()
            batch_rendered_obs = data["rendered_obs"].cuda()
            batch_affine_transforms = data["affine_transforms"].cuda()
            if self.use_drivable_area:
                batch_drivable_road_maps = data["drivable_road_map"].cuda()

        # Color the road map and perform augmentation.
        batch_road_maps = self.road_quant(batch_road_maps)
        if not self.road_map_colored:
            batch_road_maps = self.color_road_map(batch_road_maps)
        if self.map_augmentation:
            batch_road_maps = self.map_affine(
                batch_road_maps, batch_affine_transforms
            )[0]

        # If need, color the drivable map and perform augmentation.
        if self.use_drivable_area:
            batch_drivable_road_maps = self.drivable_road_quant(
                batch_drivable_road_maps
            )
            if not self.road_map_colored:
                batch_drivable_road_maps = self.color_drivable_road_map(
                    batch_drivable_road_maps
                )
            if self.map_augmentation:
                batch_drivable_road_maps = self.map_affine(
                    batch_drivable_road_maps, batch_affine_transforms
                )[0]
            batch_road_maps = self.cat_op.cat(
                (batch_road_maps, batch_drivable_road_maps), 1
            )

        batch_rendered_obs = self.quant(batch_rendered_obs)
        # Concatenate the road map and rendered maps.
        batch_rendered_frames = self.cat_op.cat(
            (batch_road_maps, batch_rendered_obs), 1
        )
        return batch_rendered_frames

    def fuse_model(self):
        pass

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
