# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict, namedtuple

import horizon_plugin_pytorch.nn as hnn
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["ValuePlanNet"]


@OBJECT_REGISTRY.register
class ValuePlanNet(nn.Module):
    """PyTorch-based planner implementation.

    The basic nn planner structures following the HAT standard.

    Args:
        backbone (dict): a dict for building backbone.
        neck (dict): a dict for building neck.
        heads (dict): the OrderedDict to build heads. Each element is a dict
            for building head.
        table (dict): the color table. The keys are road elements and the
            values are the corresponding colors.
        road_map_colored (dict): whether the road map is colored. Default True.
             If this value is True, the parameter `table` cannot be NoneType.
        is_int_infer_model (bool, optional): whether the model is for int
            inference. Default to False.
    """

    def __init__(
        self,
        backbone,
        neck=None,
        heads=None,
        table=None,
        road_map_colored=True,
        is_int_infer_model=False,
    ):
        super(ValuePlanNet, self).__init__()

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
        self.is_int_infer_model = is_int_infer_model

        # Color map.
        self.road_map_colored = road_map_colored
        if road_map_colored:
            assert table is not None, (
                "The road maps are not colored, but the input color table"
                "is Nonetype."
            )
        if table is not None:
            self.r_tab = hnn.LookUpTable(table["r_val"])
            self.g_tab = hnn.LookUpTable(table["g_val"])
            self.b_tab = hnn.LookUpTable(table["b_val"])

        # Quantization.
        self.cat_op = nn.quantized.FloatFunctional()
        self.quant = QuantStub(scale=1 / 128.0)
        self.road_quant = QuantStub(scale=1)
        self.dequant = DeQuantStub()

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

    def forward(self, data):
        """Forward.

        Args:
            data (Dict): the model input dictionary with the following keys:
            road_map: raster road map with a fixed ego position.
            rendered_obs: occupancy raster frames of agents obstacles.

        Returns:
            results (Dict): the model output dictionary including
            backbones outputs,neck outputs,heads outputs,
            losses outputs and postprocess outputs.

        """

        batch_road_maps = data["road_map"]
        batch_road_maps = self.road_quant(batch_road_maps)
        if self.road_map_colored:
            batch_road_maps = self.color_road_map(batch_road_maps)
            data["rendered_map"] = batch_road_maps

        batch_rendered_obs = data["rendered_obs"]
        batch_rendered_obs = self.quant(batch_rendered_obs)

        # Concatenate the road map and rendered maps.
        batch_inputs = self.cat_op.cat(
            (batch_road_maps, batch_rendered_obs), 1
        )

        data["feats"] = self.backbone(batch_inputs)
        neck_out = self.neck(data)
        data.update(neck_out)

        neck_results = {
            key: self.dequant(value) for key, value in neck_out.items()
        }

        model_result = OrderedDict()
        if self.is_int_infer_model:
            for head_name in self.head_name_list:
                cur_head = getattr(self, head_name)
                head_outs = cur_head(data)
                data.update(head_outs)

                head_results = {
                    key: self.dequant(value)
                    for key, value in head_outs.items()
                }
                model_result.update(head_results)
            model_result.update(neck_results)

            ImitationOutput = namedtuple(
                "ImitationValueOutput", model_result.keys()
            )
            model_result = ImitationOutput(**model_result)
        else:
            model_result.update(data)
            for _, head_name in enumerate(self.head_name_list):
                cur_head = getattr(self, head_name)
                head_outs = cur_head(data)
                data.update(head_outs)

                head_results = {
                    head_name + "_" + key: self.dequant(value)
                    for key, value in head_outs.items()
                }

                if cur_head.loss is not None:
                    head_results.update(cur_head.loss(head_results, data))

                if cur_head.post_process is not None:
                    head_results.update(
                        cur_head.post_process(head_results, data)
                    )

                model_result.update(head_results)
            model_result.update(neck_results)

        return model_result

    def fuse_model(self):
        heads = [getattr(self, name) for name in self.head_name_list]
        for module in [self.backbone, self.neck] + heads:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        heads = [getattr(self, name) for name in self.head_name_list]
        for module in [self.backbone, self.neck] + heads:
            if module is None:
                continue
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
