import logging
from typing import Sequence, cast

import numpy as np
import tensorrt as trt
import torch
from torch import nn
from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters.converter_utils import (
    add_reduce_layer,
    get_trt_plugin,
    get_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.fx.tracer.acc_tracer.acc_normalizer import (
    _normalization_dict,
    register_acc_op,
    register_acc_op_mapping,
)

logger = logging.getLogger(__name__)


EXTENSIONS = {}


def remove_acc_ops(op_and_target):
    if op_and_target in _normalization_dict:
        _normalization_dict.pop(op_and_target)


def _c_interpolate():
    from torch_tensorrt.fx.converters.acc_ops_converters import TRTTensor

    op_and_target = ("call_function", torch._C._nn.upsample_bilinear2d)
    remove_acc_ops(op_and_target)

    @register_acc_op_mapping(
        op_and_target=op_and_target,
        arg_replacement_tuples=[
            ("input", "input"),
            ("scale_factors", "scale_factors", True),
            ("output_size", "output_size", True),
            ("align_corners", "align_corners", True),
            ("antialias", "antialias", True),
        ],
    )
    @register_acc_op
    def c_interpolate(
        *,
        input,
        scale_factors=None,
        output_size=None,
        antialias=False,
        align_corners=None,
    ):
        if antialias:
            return torch._C._nn.upsample_bilinear2d_aa(
                input, output_size, align_corners, scale_factors
            )
        return torch._C._nn.upsample_bilinear2d(
            input,
            output_size,
            align_corners,
            scale_factors,
        )

    @tensorrt_converter(c_interpolate)
    def acc_ops_c_interpolate(
        network,
        target,
        args,
        kwargs,
        name,
    ):
        input_val = kwargs["input"]
        output_size = kwargs["output_size"]
        scale_factors = kwargs["scale_factors"]
        align_corners = kwargs["align_corners"]

        if not isinstance(input_val, TRTTensor):
            raise RuntimeError(
                f"c_interpolate received input {input_val} that is not part "
                "of the TensorRT region!"
            )

        dim = input_val.shape
        ranks = len(input_val.shape)
        if network.has_implicit_batch_dimension:
            assert (
                ranks >= 2 and ranks <= 4
            ), "Interpolate expects inputs are 3D,4D,5D in shape"
            ranks = ranks - 1
        else:
            assert (
                ranks >= 3 and ranks <= 5
            ), "Interpolate expects inputs are 3D,4D,5D in shape"
            ranks = ranks - 2

        layer = network.add_resize(input_val)
        if network.has_implicit_batch_dimension:
            if output_size is not None:
                layer.shape = [dim[0]] + list(output_size)
            if scale_factors is not None:
                layer.scales = [1] + list(scale_factors)
        else:
            if output_size is not None:
                layer.shape = [dim[0], dim[1]] + list(output_size)
            if scale_factors is not None:
                layer.scales = [1, 1] + list(scale_factors)

        layer.resize_mode = trt.ResizeMode.LINEAR

        if align_corners is not None:
            layer.coordinate_transformation = (
                trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            )

        set_layer_name(layer, target, name)
        return layer.get_output(0)


def _mean_replace_adp_avg_pool2d():
    from torch_tensorrt.fx.converters.acc_ops_converters import (
        acc_ops_adaptive_avg_poolnd,
    )

    op_and_target = ("call_function", nn.functional.adaptive_avg_pool2d)
    remove_acc_ops(op_and_target)

    @register_acc_op_mapping(op_and_target=op_and_target)
    @register_acc_op
    def adaptive_avg_pool2d(*, input, output_size):
        if output_size == 1:
            return torch.mean(input, dim=[2, 3], keepdim=True)
        else:
            return nn.functional.adaptive_avg_pool2d(
                input=input, output_size=output_size
            )

    @tensorrt_converter(adaptive_avg_pool2d)
    def acc_ops_adaptive_avg_pool2d(network, target, args, kwargs, name):
        if kwargs["output_size"] == 1:
            _kwargs = {}
            _kwargs["input"] = kwargs["input"]
            _kwargs["keepdim"] = True
            _kwargs["dim"] = [2, 3]
            return add_reduce_layer(
                network, target, args, _kwargs, trt.ReduceOperation.AVG, name
            )
        else:
            return acc_ops_adaptive_avg_poolnd(
                network, target, args, kwargs, name
            )


def _group_norm():
    op_and_target = ("call_function", nn.functional.group_norm)
    remove_acc_ops(op_and_target)

    @register_acc_op_mapping(op_and_target=op_and_target)
    @register_acc_op
    def group_norm(*, input, num_groups, weight, bias, eps):
        return nn.functional.group_norm(
            input=input,
            num_groups=num_groups,
            weight=weight,
            bias=bias,
            eps=eps,
        )

    @tensorrt_converter(group_norm)
    def acc_ops_group_norm(network, target, args, kwargs, layer_name):
        # args/kwargs should have already been normalized to kwargs
        assert len(args) == 0
        input_val = kwargs["input"]
        num_group = kwargs["num_groups"]
        weight = get_trt_tensor(
            network, kwargs["weight"], f"{layer_name}_weight"
        )
        bias = get_trt_tensor(network, kwargs["bias"], f"{layer_name}_bias")
        eps = kwargs["eps"]

        if not isinstance(input_val, trt.tensorrt.ITensor):
            raise RuntimeError(
                f"GroupNorm2d received input {input_val} that is not part "
                "of the TensorRT region!"
            )
        plugin_name = "GroupNormalizationPlugin"
        num_groups_field = trt.PluginField(
            "num_groups",
            np.array(num_group, dtype=np.int32),
            trt.PluginFieldType.INT32,
        )
        eps_field = trt.PluginField(
            "eps", np.array(eps, dtype=np.float32), trt.PluginFieldType.FLOAT32
        )
        field_collection = trt.PluginFieldCollection(
            [num_groups_field, eps_field]
        )
        try:
            plugin = get_trt_plugin(plugin_name, field_collection, "1")
        except AssertionError:
            logger.info(
                "Unable to find group norm plugin, fall back to TensorRT implementation."  # noqa
            )
        layer = network.add_plugin_v2([input_val, weight, bias], plugin)
        layer.name = layer_name
        return layer.get_output(0)


def _dynamic_shape_pad():
    """torch_tensorrt provide pad, but seem some bug for dynamic shape."""
    from torch_tensorrt.fx.tracer.acc_tracer import acc_ops
    from torch_tensorrt.fx.types import TRTTensor

    @tensorrt_converter(acc_ops.pad)
    def acc_ops_pad_with_padding_layer(
        network, target, args, kwargs, layer_name
    ):
        input_val = kwargs["input"]
        pad = cast(Sequence[int], kwargs["pad"])
        mode = kwargs["mode"]
        value = kwargs["value"] if kwargs["value"] is not None else 0
        rank = len(input_val.shape)  # type: ignore[union-attr]

        if not isinstance(input_val, TRTTensor):
            raise RuntimeError(
                f"pad received input {input_val} that is not part "
                "of the TensorRT region!"
            )

        if mode != "constant":
            raise RuntimeError(
                f"Currently we only support constant mode for pad, got {mode}."  # noqa
            )

        if len(pad) / 2 > rank:
            raise RuntimeError(
                f"Trying to pad last {len(pad) / 2} dimension but the input only has {rank} dimension."  # noqa
            )

        if value != 0:
            raise RuntimeError(
                f"Currently we only support padding value of 0, got {value}."
            )

        if len(pad) > 4:
            raise RuntimeError(
                "Currently we only support padding last two dimensions."
            )  # noqa

        pre_padding = tuple(
            pad[len(pad) - i - 2] for i in range(0, len(pad), 2)
        )  # noqa
        post_padding = tuple(
            pad[len(pad) - i - 1] for i in range(0, len(pad), 2)
        )  # noqa

        layer = network.add_padding(
            input_val,
            pre_padding if len(pre_padding) == 2 else (0,) + pre_padding,
            post_padding if len(post_padding) == 2 else (0,) + post_padding,
        )
        set_layer_name(layer, target, layer_name)
        return layer.get_output(0)


EXTENSIONS["group_norm"] = _group_norm
EXTENSIONS["mean_replace_adp_avg_pool2d"] = _mean_replace_adp_avg_pool2d
EXTENSIONS["c_interpolate"] = _c_interpolate
EXTENSIONS["dynamic_shape_pad"] = _dynamic_shape_pad


def list_extension():
    for k, v in EXTENSIONS.items():
        logger.info(
            f"Extension name: {k}, file location: {v.__globals__['__file__']}"
        )


def load_extension(name):
    assert (
        name in EXTENSIONS
    ), f"Not found extension for {name}! Please check file {__file__} EXTENSIONS."  # noqa
    EXTENSIONS[name]()
    logger.info(f"Load extension: {name} successfully!")
