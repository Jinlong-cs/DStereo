# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict, namedtuple

import horizon_plugin_pytorch.nn as hnn
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

from hat.models.task_modules.bev.spatial_transfomer import SpatialTransfomer
from hat.models.task_modules.traj_pred.necks import MultiPathNeck
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["Multipath"]


@OBJECT_REGISTRY.register
class Multipath(nn.Module):
    """PyTorch-based Multipath implementation.

    The Backbone of Multipath is a convolutional network, the user can specify
    difference variations to the standard. The output feature map of the
    backbone is cropped (with rotation) out and sent to the head.

    Args:
        data_shape (List, [C, H, W]): the input image shape of the backbone.
        backbone (dict): a dict for building backbone.
        neck (dict): a dict for building neck.
        heads (dict): the OrderedDict to build heads. Each element is a dict
            for building head.
        post_process (dict): the post-processing model.
        table (dict): the color table. The keys are road elements and the
            values are the corresponding colors.
        road_map_colored (bool, optional): whether the road map is colored.
            Default to False. If this value is False, the parameter `table`
            cannot be NoneType.
        losses (dict): the losses.
        is_int_infer_model (bool, optional): whether the model is for int
            inference. Default to False.
        map_augmentation (bool, optional): whether to perform map augmentation.
    """

    def __init__(
        self,
        data_shape,
        backbone,
        neck=None,
        heads=None,
        post_process=None,
        table=None,
        road_map_colored=False,
        losses=None,
        is_int_infer_model=False,
        map_augmentation=False,
        drivable_area_table=None,
        use_drivable_area=False,
    ):
        super(Multipath, self).__init__()
        self.is_int_infer_model = is_int_infer_model
        self.map_augmentation = map_augmentation
        if is_int_infer_model:
            self.map_augmentation = False

        # Model structure.
        self.backbone = backbone
        self.neck = neck
        self.head_name_list = []
        if heads is not None:
            for head_name, head in heads.items():
                if head is None:
                    continue
                setattr(self, head_name, head)
                self.head_name_list.append(head_name)
        self.post_process = post_process

        # Color map.
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

        # Loss.
        self.losses = None
        if losses is not None:
            self.losses = nn.ModuleList(_as_list(losses))

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

    @staticmethod
    def _build(obj, builder):
        if obj is None:
            return None
        return builder(obj) if isinstance(obj, dict) else obj

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
                'valid_img_coords' (List[Tuple]): list of image coordinates
                    tuples of prediction objects.
                'state_vectors' (torch.Tensor, [num_obj, num_states, 1, 1]):
                    state vectors of the batch data.
                'future_trajectories' (torch.Tensor, [num_obj, traj_len, 2]):
                    the ground-truth trajectories.
                'img_homographys' (torch.Tensor, [num_obj, 3, 3]): the
                    homography matrix.

        Returns:
            results (Dict): the model output dictionary with the following
                keys:
                -- if the current mode is not testing:
                1. keys start with all the valid head (in self.head_name_list)
                    'gts', 'anchors', 'probabilities', 'log_anchors_probs',
                    'means', 'scale_trils', 'track_ids', 'masks',
                    'resize_ratio', 'cur_head_mask'.
                2. all keys in data.
                3. 'rendered_frames'.
                -- if the current mode is testing:
                1. 'probabilities', 'mean_var'
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
        data["rendered_frames"] = batch_rendered_frames

        feats = self.backbone(batch_rendered_frames)
        feats = self.neck(feats) if self.neck else feats
        data["feats"] = feats

        model_result = OrderedDict()
        if self.is_int_infer_model:
            for head_name in self.head_name_list:
                cur_head = getattr(self, head_name)
                anchor_probs, anchor_mean_var, track_valid_cls = cur_head(data)
                cur_result = {
                    "probabilities": anchor_probs,
                    "mean_var": anchor_mean_var,
                    "track_valid_cls": track_valid_cls,
                }
                cur_result = {
                    head_name + "_" + key: value
                    for key, value in cur_result.items()
                }
                model_result.update(cur_result)
            if self.post_process is not None:
                model_result.update(self.post_process(model_result, data))
            MultipathOutput = namedtuple(
                "MultipathOutput", model_result.keys()
            )
            model_result = MultipathOutput(**model_result)
        else:
            model_result.update(data)
            for i, head_name in enumerate(self.head_name_list):
                head_outs = OrderedDict()
                cur_head = getattr(self, head_name)
                if ("head_mask" in data) and len(data["head_mask"]):
                    data["cur_head_mask"] = data["head_mask"][i]
                out = cur_head(data)
                head_outs.update(out)

                if self.losses is not None:
                    for loss in self.losses:
                        head_outs.update(loss(head_outs))

                if self.post_process is not None:
                    head_outs.update(self.post_process(head_outs, data))

                head_results = {
                    head_name + "_" + key: value
                    for key, value in head_outs.items()
                }
                model_result.update(head_results)
        return model_result

    def fuse_model(self):
        for module in [self.backbone, self.neck]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        for head_name in self.head_name_list:
            head_cur = getattr(self, head_name)
            head_cur.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        heads = [getattr(self, name) for name in self.head_name_list]
        for module in [self.backbone, self.neck] + heads:
            if module is None:
                continue
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

        if self.losses is not None:
            for module in self.losses:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None
