# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict, namedtuple

import horizon_plugin_pytorch.nn as hnn
import torch
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["Uniformpath"]


@OBJECT_REGISTRY.register
class Uniformpath(nn.Module):
    """PyTorch-based Uniformpath implementation.

    This model exists to assign a uniform or static trajectory to the
    filtered obstacles.

    Args:
        anchor_num (int): the number of anchors.
        traj_len (int): the trajectory length to be predicted.
        heads (dict): the OrderedDict to build heads. Each element is a
            dict for building head.
        post_process (dict): the post-processing model.
        table (dict): the color table. The keys are road elements and the
            values are the corresponding colors.
        road_map_colored (bool, optional): whether the road map is colored.
            Default to False. If this value is False, the parameter `table`
            cannot be NoneType.
        losses (dict): the losses. Default to None.
        is_int_infer_model (bool, optional): whether the model is for int
            inference. Default to False.
    """

    def __init__(
        self,
        anchor_num: int = 124,
        traj_len: int = 12,
        heads=None,
        post_process=None,
        table=None,
        road_map_colored=False,
        losses=None,
        is_int_infer_model: bool = False,
    ):
        super(Uniformpath, self).__init__()

        self.traj_len = traj_len
        self.anchor_num = anchor_num
        self.is_int_infer_model = is_int_infer_model

        # Model structure.
        self.heads = heads
        self.head_name_list = []
        if heads is not None:
            for head_name in heads.keys():
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

        # Quantization.
        self.quant = QuantStub(scale=None)
        self.road_quant = QuantStub(scale=1)
        self.cat_op = nn.quantized.FloatFunctional()

        # Loss.
        self.losses = None
        if losses is not None:
            self.losses = nn.ModuleList(_as_list(losses))

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

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

    @torch.no_grad()
    def forward(self, data):
        """Forward.

        Args:
            data (Dict): the model input dictionary with the following keys: \
                'road_map' (torch.Tensor, [batch_size, 1, h, w]): the road \
                    maps. Each value is the type of road.
                'rendered_obs' (torch.Tensor, [batch_size, 3, h, w]): the \
                    rendered obstacle occupancy map.
                'filtered_obs_ids' (List, [num_obj]): ids of filtered \
                    obstacles.
                'filtered_obs_trajs' (torch.Tensor, \
                    [num_obj, num_anchor, traj_len, 2]): rule trajectories \
                     of filtered obstacles.
                'filtered_obs_gt' (torch.Tensor, [num_obj, traj_len, 2]): \
                    the ground-truth trajectories of filtered obstacles.
                'filtered_obs_gt_masks' (torch.Tensor, [num_obj, traj_len]): \
                    the ground-truth trajectories mask of filtered obstacles.

        Returns:
            results (Dict): the model output dictionary with the following \
                keys:
                -- if the current mode is not testing:
                1. keys start with all valid head (in self.head_name_list) \
                    'gts', 'probabilities', 'log_anchors_probs', 'means', \
                    'masks', 'cur_head_mask'.
                2. all keys in data.
                3. 'rendered_frames'.
                -- if the current mode is testing:
                1. 'probabilities', 'mean_var'
        """
        batch_road_maps = data["road_map"]
        batch_rendered_obs = data["rendered_obs"]

        # Color the road map and perform augmentation.
        batch_road_maps = self.road_quant(batch_road_maps)
        if not self.road_map_colored:
            batch_road_maps = self.color_road_map(batch_road_maps)

        batch_rendered_obs = self.quant(batch_rendered_obs)
        # Concatenate the road map and rendered maps.
        batch_rendered_frames = self.cat_op.cat(
            (batch_road_maps, batch_rendered_obs), 1
        )
        data["rendered_frames"] = batch_rendered_frames

        num_objs = data["filtered_obs_trajs"].shape[0]
        probabilities = torch.ones([num_objs, self.anchor_num])
        data["probabilities"] = probabilities
        data["log_anchors_probs"] = self.log_softmax(probabilities)
        model_result = OrderedDict()
        if self.is_int_infer_model:
            for head_name in self.head_name_list:
                model_result["probabilities"] = torch.ones(
                    [num_objs, 1, 1, self.anchor_num]
                )
                model_result["mean_var"] = torch.ones(
                    [num_objs, 60, 1, self.anchor_num]
                )
                cur_result = {
                    head_name + "_" + key: value
                    for key, value in model_result.items()
                }
                model_result.update(cur_result)
            if self.post_process is not None:
                model_result.update(self.post_process(model_result))
            UniformpathOutput = namedtuple(
                "UniformpathOutput", model_result.keys()
            )
            model_result = UniformpathOutput(**model_result)
        else:
            model_result.update(data)
            for head_name in self.head_name_list:
                if self.post_process is not None:
                    model_result.update(
                        self.post_process(model_result, self.anchor_num)
                    )
                head_results = {
                    head_name + "_" + key: value
                    for key, value in model_result.items()
                }
                model_result.update(head_results)
        return model_result

    def set_qconfig(self):
        pass

    def fuse_model(self):
        pass
