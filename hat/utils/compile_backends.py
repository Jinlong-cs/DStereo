import logging

import torch
import torchdynamo
from torch_tensorrt.fx.fx2trt import InputTensorSpec, TRTInterpreter
from torch_tensorrt.fx.passes.lower_basic_pass import transform_setitem
from torch_tensorrt.fx.tools.trt_splitter import (
    TRTSplitter,
    TRTSplitterSetting,
)
from torch_tensorrt.fx.tracer.acc_tracer import acc_tracer
from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt.fx.utils import LowerPrecision
from torchdynamo.optimizations.backends import BACKENDS, create_backend
from torchdynamo.optimizations.normalize import normalize_ir

logger = logging.getLogger(__name__)


@create_backend
def fx2trt_plus(
    subgraph: torch.fx.GraphModule,
    max_fx_piece: int = 8,
    min_fx_block: int = 4,
    dynamic_batch_size: bool = False,
    fp16_mode: bool = False,
    force_fp32_output: bool = False,
    sparse_weights: bool = False,
    timing_cache: bool = False,
    max_batch_size: int = None,
):
    """
    Backend for torchdynamo tensorrt optimizer.

    The custom backend has the following differences from the official
    implementation `fx2trt`.
    1. More flexible setting, like force_fp32_output, sparse_weights
        timing_cache and configurable max_fx_piece, min_fx_block.
    2. Add compile skip when the graph module include less than min_fx_block,
        we consider its perf is not good (even error accumulation when fp16)
        and fall back to non-TRT.

    Args:
        subgraph: fx graph model.
        max_fx_piece: The FX-split subgraph with maximum tolerance,
            skip compile if more.
        min_fx_block: The FX obtain block with minimum tolerance,
            skip compile if less.
        dynamic_batch_size: Turn on dynamic batch size for trt compile.
            Support batch size range (1, cur_batch_size * 2).
        fp16_mode: TRT optimizer as fp16.
        force_fp32_output: force output to be fp32.
        sparse_weights: allow the builder to examine weights and
            use optimized functions when weights have suitable sparsity.
        timing_cache: enable timing cache for TensorRT.

    Return:
        TRTInterpreterResult
    """
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    try:
        model = subgraph.model
        inputs = subgraph.example_inputs
        # normalize
        model = normalize_ir(model, inputs)
        # pass rewrite
        model = transform_setitem(model, inputs)
        acc_model = acc_tracer.trace(model, inputs)
        # Split by unsupported ops
        # num_piece represents split graph
        splitter_setting = TRTSplitterSetting()
        splitter_setting.use_implicit_batch_dim = False
        splitter = TRTSplitter(acc_model, inputs, settings=splitter_setting)
        splitter.node_support_preview()
        split_mod = splitter()
        num_piece = 0
        for name, _ in split_mod.named_children():
            logger.info(f"graph is split into {name}")
            num_piece += 1
        # if the graph module is split into pieces larger than 8,
        # we consider its perf is not good and fall back to non-TRT
        if num_piece > max_fx_piece:
            logger.info(
                f"The graph module is split into {num_piece} which is \
                large than the threshold=8. Fall back to non-TRT module."
            )
            return None

        # if the graph module include less than min_fx_block,
        # we consider its perf is not good (even error accumulation when fp16)
        # and fall back to non-TRT
        num_block = len(list(model.named_children()))
        if num_block <= min_fx_block:
            for name, _ in model.named_children():
                logger.info(
                    f"The graph module has {num_block} block "
                    f"which is large than the threshold=4."
                    f"Skip block {name}"
                )
            for name, _ in model.named_parameters():
                logger.info(
                    f"The graph module has {num_block} block "
                    f"which is large than the threshold=4."
                    f"Skip block {name}"
                )
            return None

        if fp16_mode:
            precision = LowerPrecision.FP16
        else:
            precision = LowerPrecision.FP32

        def get_submod_inputs(mod, submod, inputs):
            acc_inputs = None

            def get_input(self, inputs):
                nonlocal acc_inputs
                acc_inputs = inputs

            handle = submod.register_forward_pre_hook(get_input)
            mod(*inputs)
            handle.remove()
            return acc_inputs

        for name, _ in split_mod.named_children():
            if "_run_on_acc" in name:
                submod = getattr(split_mod, name)
                # Get submodule inputs for fx2trt
                acc_inputs = get_submod_inputs(split_mod, submod, inputs)

                # fx2trt replacement
                if dynamic_batch_size:
                    cur_batch_size = acc_inputs[0].shape[0]
                    if max_batch_size is None:
                        max_batch_size = cur_batch_size * 2
                    input_spec = InputTensorSpec.from_tensors_with_dynamic_batch_size(  # noqa
                        acc_inputs,
                        (1, cur_batch_size, max_batch_size),
                        opt_profile_replica=2,
                    )
                else:
                    input_spec = InputTensorSpec.from_tensors(acc_inputs)
                interp = TRTInterpreter(
                    submod,
                    input_spec,
                    explicit_batch_dimension=True,
                )
                r = interp.run(
                    max_workspace_size=20 << 30,
                    lower_precision=precision,
                    force_fp32_output=force_fp32_output,
                    sparse_weights=sparse_weights,
                    timing_cache=timing_cache,
                    # For profile
                    # profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
                )
                trt_mod = TRTModule(*r)

                setattr(split_mod, name, trt_mod)
            else:
                submod = getattr(split_mod, name)
        return subgraph.wrap_returns(split_mod)
    except Exception as e:
        logger.exception(e)
        logger.error("FX2TRT conversion error")
        return None


def default_dynamo_config(
    cache_size_limit=8,
    debug=False,
    dynamic_shape=False,
):
    torchdynamo.config.raise_on_backend_error = False
    # reduce default 64 to 8 prevent too much re-compile from
    # slowing forward.
    # refer to <Excessive Recompilation>:
    # https://pytorch.org/docs/stable/dynamo/troubleshooting.html
    torchdynamo.config.cache_size_limit = cache_size_limit
    torchdynamo.config.dynamic_shapes = dynamic_shape
    if debug:
        torchdynamo.config.verbose = True


def tensorRT_backend(
    debug: bool = False,
    fp16: bool = False,
    max_fx_piece: int = 8,
    min_fx_block: int = 4,
    cache_size_limit: int = 8,
    sparse_weights: bool = False,
    timing_cache: bool = False,
    dynamic_shape: bool = False,
    max_batch_size: int = None,
):
    def _backend(gm: torch.fx.GraphModule, example_inputs):
        fp16_mode = False
        force_fp32_output = False
        if fp16:
            fp16_mode = True
            force_fp32_output = True
        default_dynamo_config(
            cache_size_limit=cache_size_limit,
            debug=debug,
            dynamic_shape=dynamic_shape,
        )
        trt_compiled = BACKENDS["fx2trt_plus"](
            gm,
            example_inputs,
            fp16_mode=fp16_mode,
            force_fp32_output=force_fp32_output,
            max_fx_piece=max_fx_piece,
            min_fx_block=min_fx_block,
            sparse_weights=sparse_weights,
            timing_cache=timing_cache,
            dynamic_batch_size=dynamic_shape,
            max_batch_size=max_batch_size,
        )
        if trt_compiled is not None:
            return trt_compiled
        else:
            logger.warning(
                "FX2TRT conversion failed on the subgraph.\
                    Return GraphModule forward instead"
            )
            return gm.forward

    if dynamic_shape and max_batch_size is None:
        logger.warning(
            "The maximum batch_size supported by TensorRT \
            is not specified, which will be set to twice of \
            the current batch_size Please confirm that this \
            value is sufficient for your usage scenario."
        )

    return _backend
