# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Callable, Dict, List, Optional

import horizon_plugin_pytorch.nn as hnn
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

from hat.models.task_modules.bev.spatial_transfomer import SpatialTransfomer
from hat.models.task_modules.traj_pred.necks import MultiPathNeck
from hat.models.task_modules.traj_pred.structures import BasicTrajPredStructure
from hat.registry import OBJECT_REGISTRY

__all__ = ["MultipathV2"]


@OBJECT_REGISTRY.register
class MultipathV2(BasicTrajPredStructure):
    """PyTorch-based Multipath implementation (version 2).

    Compared to version 1. we decouple the feature extractions from original
    MultipathHead (generate MultipathNeck). All the specific head generator
    are also decoupled (E.g., the trajectories will be generated in
    BasicAnchorBasedDecoder, the valid head classification are inferred in
    BasicTrackValidDecoder, etc.). These changes make the `multi-head`
    structure useful (not a fake structure any more)

    Also, we enable different heads have their own loss function and post-
    processing function, instead of using those defined in structure. This
    makes loss and post-processing func design more flexible.

    """

    def __init__(
        self,
        data_shape: List,
        backbone: Callable,
        necks: Optional[Dict] = None,
        heads: Optional[Dict] = None,
        post_process: Optional[Callable] = None,
        losses: Optional[Dict] = None,
        is_int_infer_model: bool = False,
        # Belows are customized parameters.
        table: Optional[Dict] = None,
        road_map_colored: bool = False,
        map_augmentation: bool = False,
        drivable_area_table: Optional[Dict] = None,
        use_drivable_area: bool = False,
    ):
        """Initialize method.

        Args:
            data_shape: the input image shape of the backbone.
            backbone: a dict for building backbone or callable instance.
            necks: the OrderedDict to build necks. Each element is a dict
                for building neck.
            heads: the OrderedDict to build heads. Each element is a dict
                for building head.
            post_process: the post-processing model.
            losses: the losses.
            is_int_infer_model: whether the model is for int inference.
            table: the color table. The keys are road elements and the values
                are the corresponding colors.
            road_map_colored: whether the road map is colored. If this value
                is False, the parameter `table` cannot be NoneType.
            map_augmentation: whether to perform map augmentation.
            drivable_area_table: the color table for drivable area map.
            use_drivable_area: whether to use drivable area map.
        """
        self.data_shape = data_shape
        self.table = table
        self.road_map_colored = road_map_colored
        self.map_augmentation = map_augmentation
        self.drivable_area_table = drivable_area_table
        self.use_drivable_area = use_drivable_area
        if is_int_infer_model:
            self.map_augmentation = False

        kwargs = {
            "backbone": backbone,
            "necks": necks,
            "heads": heads,
            "post_process": post_process,
            "losses": losses,
            "is_int_infer_model": is_int_infer_model,
        }
        super(MultipathV2, self).__init__(**kwargs)
        self.build_custom_structure()

    def build_custom_structure(self):
        # Build color map.
        if not self.road_map_colored:
            assert self.table is not None, (
                "The road maps are not colored, but the input color table"
                "is Nonetype."
            )
        if self.table is not None:
            self.r_tab = hnn.LookUpTable(self.table["r_val"])
            self.g_tab = hnn.LookUpTable(self.table["g_val"])
            self.b_tab = hnn.LookUpTable(self.table["b_val"])

        self.use_drivable_area = self.use_drivable_area and (
            self.drivable_area_table is not None
        )
        if self.use_drivable_area:
            self.drivable_r_tab = hnn.LookUpTable(
                self.drivable_area_table["r_val"]
            )
            self.drivable_g_tab = hnn.LookUpTable(
                self.drivable_area_table["g_val"]
            )
            self.drivable_b_tab = hnn.LookUpTable(
                self.drivable_area_table["b_val"]
            )

        # Quantization.
        self.quant = QuantStub(scale=None)
        self.road_quant = QuantStub(scale=1)
        self.cat_op = nn.quantized.FloatFunctional()
        if self.use_drivable_area:
            self.drivable_road_quant = QuantStub(scale=1)

        # Road map data augmentation.
        grid_quant_scale = MultiPathNeck.cal_roi_quanti_scale(
            self.data_shape[1:], self.data_shape[1:]
        )
        self.map_affine = SpatialTransfomer(
            height=self.data_shape[1],
            width=self.data_shape[2],
            padding_mode="border",
            grid_quant_scale=grid_quant_scale,
            eps=0,
            use_horizon_grid_sample=True,
        )

    def color_road_map(self, batch_road_maps, prefix=None):
        """Color the road map by look up table (LUT).

        Args:
            batch_road_maps (torch.tensor, [batch_size, 1, h, w]): the road
                maps. Each value is the type of road.
            prefix: the prefix to get tab layers.

        Return:
            colored_maps (torch.tensor, [batch_size, 3, h, w]): the colored
                road map.
        """
        keys = ["r_tab", "g_tab", "b_tab"]
        if prefix is not None:
            keys = [f"{prefix}_{k}" for k in keys]

        all_frames = []
        for k in keys:
            tmp_tab = getattr(self, k)
            all_frames.append(tmp_tab(batch_road_maps))

        colored_maps = self.cat_op.cat(all_frames, 1)
        return colored_maps

    def custom_data_preprocess(self, data: Dict):  # noqa: D401
        """Customized data preprocess function.

        Args:
            data (Dict): the model input dictionary with the following keys:
            road_map (torch.Tensor, [batch_size, 1, h, w]): the road \
                maps. Each value is the type of road. \
            rendered_obs (torch.Tensor, [batch_size, 3, h, w]): the \
                rendered obstacle occupancy map. \
            valid_img_coords (List[Tuple]): list of image coordinates \
                tuples of prediction objects. \
            state_vectors (torch.Tensor, [num_obj, num_states, 1, 1]): \
                state vectors of the batch data. \
            future_trajectories (torch.Tensor, [num_obj, traj_len, 2]): the
                ground-truth trajectories.
            img_homographys (torch.Tensor, [num_obj, 3, 3]): the homography
                matrix.

        Returns:
            data: the updated data dict.
            backbone_data: data for backbone input.
            gt_data: gt data for metric and postprocess.
        """
        gt_data = {}
        if self.is_int_infer_model:
            batch_road_maps = data["road_map"]
            batch_rendered_obs = data["rendered_obs"]
            if self.use_drivable_area:
                batch_drivable_road_maps = data["drivable_road_map"]
            if self.map_augmentation:
                batch_affine_transforms = data["affine_transforms"]
        else:
            batch_road_maps = data["road_map"].cuda()
            batch_rendered_obs = data["rendered_obs"].cuda()
            if self.use_drivable_area:
                batch_drivable_road_maps = data["drivable_road_map"].cuda()
            if self.map_augmentation:
                batch_affine_transforms = data["affine_transforms"].cuda()

            # Extract ground truths.
            gt_data = {
                "gts": data["future_trajectories"],
                "track_ids": data["valid_track_ids"],
                "masks": data["valid_masks"],
                "resize_ratio": data["resize_ratio"],
            }
            if "lat_behaviors" in data:
                gt_data["lat_behav_gts"] = data["lat_behaviors"]
            if "lon_behaviors" in data:
                gt_data["lon_behav_gts"] = data["lon_behaviors"]
            if "valid_behav_track_ids" in data:
                gt_data["behav_track_ids"] = data["valid_behav_track_ids"]
            if "cur_head_mask" in data:
                gt_data["cur_head_mask"] = data["cur_head_mask"]

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
                batch_drivable_road_maps = self.color_road_map(
                    batch_drivable_road_maps, prefix="drivable"
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
        data["rendered_frames"] = batch_rendered_frames

        return data, batch_rendered_frames, gt_data
