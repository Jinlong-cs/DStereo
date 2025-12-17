# Copyright (c) Horizon Robotics. All rights reserved.

from copy import deepcopy
from typing import List

from torch import nn

from hat.registry import OBJECT_REGISTRY
from .h3d_loss_hub import (
    loss_mesh_chamfer_distance,
    loss_mesh_edge,
    loss_mesh_laplacian_smoothing,
    loss_mesh_normal_consistency,
    loss_ms_ssim_l1,
    loss_smoothl1,
)


@OBJECT_REGISTRY.register
class H3DLossStucture(nn.Module):
    """Loss structure for hand pose estimation.

    Args:
        loss_forward_list: List of loss forward func and params.
            loss forward list data structure:
                [
                    {
                        loss_name: loss name for show.
                        forward_func: can be func or str,
                            str support loss_func_map.
                        weight: weight of loss, default 1.
                        forward_params=dict(): var str will be automatically
                        escaped as a variable.
                            Such as "output_decoder['pred_intrinsic3x3']".
                    }
                    funcxx_dict,
                ]
            Defaults to None.
        loss_class_init_list: List of class init func and params.
            loss_class_init_list data structure:
                [
                    class_init: func for class init.
                    forward_func_name: class forward func name,
                        used for call func.
                    param=dict(): class init params.
                ]
            Defaults to None.
        including_outputs: List of var name want to return.
            Defaults to None.
    """

    def __init__(
        self,
        loss_forward_list: List = None,
        loss_class_init_list: List = None,
        including_outputs: List = None,
    ):
        super(H3DLossStucture, self).__init__()
        self.including_outputs = including_outputs

        self.loss_func_map = {
            "smoothl1": loss_smoothl1,
            "mesh_edge": loss_mesh_edge,
            "mesh_normal_consistency": loss_mesh_normal_consistency,
            "mesh_laplacian_smoothing": loss_mesh_laplacian_smoothing,
            "mesh_chamfer_distance": loss_mesh_chamfer_distance,
            "ms_ssim_l1": loss_ms_ssim_l1,
        }

        if loss_class_init_list is not None:
            for class_init_dict in loss_class_init_list:
                _class_init_func = class_init_dict["class_init"]
                _param = class_init_dict["param"]

                loss_forward_func = _class_init_func(**_param)
                self.loss_func_map.update(
                    {class_init_dict["forward_func_name"]: loss_forward_func}
                )

        self.loss_forward_list = loss_forward_list
        self._assert()
        self._get_parse_list()

    def _assert(self):
        """Assert data structure.

        loss_list: list(
            dict{
                loss_name: for log
                forward_func: string or func
                weight: int, default: 1.0
                forward_params:{
                    xxx
                }
            }
        )
        """
        for loss_forward_func in self.loss_forward_list:
            if isinstance(loss_forward_func["forward_func"], str):
                loss_forward_func["loss_name"] = loss_forward_func.get(
                    "loss_name", loss_forward_func["forward_func"]
                )
                assert (
                    loss_forward_func["forward_func"]
                    in self.loss_func_map.keys()
                ), "{} is not vaild forward func".format(
                    loss_forward_func["forward_func"]
                )
            else:
                loss_forward_func_name = loss_forward_func[
                    "forward_func"
                ].__name__
                loss_forward_func["loss_name"] = loss_forward_func.get(
                    "loss_name", loss_forward_func_name
                )
                # if forward_func is func, add this func to self.loss_func_map
                self.loss_func_map[loss_forward_func_name] = loss_forward_func[
                    "forward_func"
                ]
            loss_forward_func["forward_params"] = loss_forward_func.get(
                "forward_params", {}
            )
            loss_forward_func["weight"] = loss_forward_func.get("weight", 1.0)

    def _get_parse_list(self):
        self._parse_list = []
        for loss_forward_func in self.loss_forward_list:
            _params = loss_forward_func["forward_params"]
            _parse_per_func = [
                [param_name, param_v]
                for param_name, param_v in _params.items()
                if isinstance(param_v, str)
                and (
                    param_v.startswith("output_decoder")
                    or param_v.startswith("data")
                )
            ]
            self._parse_list.append(tuple(_parse_per_func))
        self._parse_list = tuple(self._parse_list)

    def _parse(self, _params, parse_list, output_decoder, data):
        params = deepcopy(_params)
        for parse_name, parse_v in parse_list:
            params[parse_name] = eval(parse_v)
        return params

    def forward(self, output_decoder, data):
        if self.including_outputs is not None:
            pass

        losses = {}
        for loss_forward_func, parse_list in zip(
            self.loss_forward_list, self._parse_list
        ):
            loss_name = loss_forward_func["loss_name"]
            forward_params = self._parse(
                loss_forward_func["forward_params"],
                parse_list,
                output_decoder,
                data,
            )
            loss_v = self.loss_func_map[loss_forward_func["forward_func"]](
                **forward_params
            )
            if loss_name == "update":
                losses.update(loss_v)
            else:
                losses[loss_name] = loss_forward_func["weight"] * loss_v
        return losses
