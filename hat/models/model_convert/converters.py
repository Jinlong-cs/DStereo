# Copyright (c) Horizon Robotics. All rights reserved.

import functools
import logging
import os
from abc import abstractmethod
from itertools import chain
from typing import Callable, Dict, List, Optional, Union

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.utils.quant_profiler import check_qconfig

try:
    from hatbc.workflow.symbol import Node
except ImportError:
    Node = None

from hat.models.structures.multitask_graph_model import MultitaskGraphModel
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from hat.utils.checkpoint import load_checkpoint, load_state_dict
from hat.utils.dynamo import CompileBackendWrapper, disable_compile
from hat.utils.global_var import set_value
from hat.utils.logger import MSGColor, format_msg
from hat.utils.model_helpers import (
    match_children_modules_by_name,
    match_children_modules_by_regex,
)
from hat.utils.package_helper import (
    check_packages_available,
    raise_error_if_import_failed,
    require_packages,
)

try:
    import torchdynamo
except ImportError:
    torchdynamo = None

try:
    import horizon_plugin_profiler
except ImportError:
    horizon_plugin_profiler = None

__all__ = [
    "Float2QAT",
    "Float2Calibration",
    "QATFusePartBN",
    "QAT2Quantize",
    "LoadCheckpoint",
    "RepModel2Deploy",
    "Torch2Compile",
]

logger = logging.getLogger(__name__)


def _preserve_qat_mode(model, preserve_dict):
    if preserve_dict is not None:
        if check_packages_available("horizon_plugin_profiler", raise_exception=False):
            from horizon_plugin_profiler import set_preserve_qat_mode
        elif check_packages_available(
            "horizon_plugin_pytorch>=1.2.0", raise_exception=False
        ):
            from horizon_plugin_pytorch.utils.quant_profiler import (
                set_preserve_qat_mode,
            )
        else:
            raise_error_if_import_failed(
                horizon_plugin_profiler, "horizon_plugin_profiler"
            )

        prefixes = preserve_dict.get("prefixes", ())
        types = preserve_dict.get("types", ())
        set_preserve_qat_mode(model, prefixes, types)


class BaseConverter(object):
    """Base class for defining the process of model convert.

    Args:
        convert_mode: convert mechanism, can be choosen from ("eager", "fx").
    """

    def __init__(self, convert_mode: str = "eager"):
        assert convert_mode in (
            "eager",
            "fx",
        ), "convert_mode must be 'eager' or 'fx', but receive {}".format(convert_mode)
        if convert_mode == "fx":
            check_packages_available("horizon_plugin_pytorch>=1.0.3")

        self.convert_mode = convert_mode

    @abstractmethod
    def __call__(self, model):
        raise NotImplementedError


@OBJECT_REGISTRY.register
class Float2QAT(BaseConverter):
    """Define the process of convert float model to qat model.

    Args:
        convert_mode: convert mechanism, can be choosen from ("eager", "fx").
        qconfig_dict: only used when convert_mode == 'fx', please refer to
            the doc of `horizon.quantization.prepare_qat_fx` for more info
        prepare_custom_config_dict: only used when convert_mode == 'fx',
            please refer to the doc of `horizon.quantization.prepare_qat_fx`
            for more info
        hybrid: only used when convert_mode == 'fx', please refer to the doc
            of `horizon.quantization.prepare_qat_fx` for more info
        hybrid_dict: only used when convert_mode == 'fx', please refer to the
            doc of `horizon.quantization.prepare_qat_fx` for more info
        optimize_graph: whether to do some process on origin model for special
            purpose. Currently only support using torch.fx to fix cat input
            scale(only used on Bernoulli).
        qconfig_setter: set qconfig automatically. Value is an instance of
            horizon_plugin_pytorch.quantization.qconfig_template.QconfigSetter
        example_inputs: example inputs for tracing graph. only used when
            qconfig_setter is set.
    """

    def __init__(
        self,
        convert_mode="eager",
        qconfig_dict=None,
        prepare_custom_config_dict=None,
        hybrid=False,
        hybrid_dict=None,
        optimize_graph=False,
        qconfig_setter=None,
        example_inputs=None,
    ):
        super(Float2QAT, self).__init__(convert_mode)
        self.qconfig_dict = qconfig_dict
        self.prepare_custom_config_dict = prepare_custom_config_dict
        self.hybrid = hybrid
        self.hybrid_dict = hybrid_dict
        self.optimize_graph = optimize_graph
        self.qconfig_setter = qconfig_setter
        self.example_inputs = example_inputs
        self.is_qconfig_template_available = check_packages_available(
            "horizon_plugin_pytorch>=2.1.6",
            raise_exception=False,
        )

    def __call__(self, model):
        if self.convert_mode == "eager":
            # make sure the input model is a float model
            model.fuse_model()

        qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
        if self.qconfig_setter is None or self.example_inputs is None:
            model.qconfig = qconfig_manager.get_default_qconfig()
        if hasattr(model, "set_qconfig"):
            model.set_qconfig()
        elif self.qconfig_setter is None or self.example_inputs is None:
            raise RuntimeError("`model` should implement `set_qconfig()`")
        if self.convert_mode == "eager":
            if self.is_qconfig_template_available:
                horizon.quantization.prepare_qat(
                    model,
                    inplace=True,
                    optimize_graph=self.optimize_graph,
                    example_inputs=self.example_inputs,
                    qconfig_setter=self.qconfig_setter,
                )
            else:
                horizon.quantization.prepare_qat(
                    model,
                    inplace=True,
                    optimize_graph=self.optimize_graph,
                )
        else:
            if self.is_qconfig_template_available:
                model = horizon.quantization.prepare_qat_fx(
                    model,
                    self.qconfig_dict,
                    self.prepare_custom_config_dict,
                    hybrid=self.hybrid,
                    hybrid_dict=self.hybrid_dict,
                    example_inputs=self.example_inputs,
                    qconfig_setter=self.qconfig_setter,
                )
            else:
                model = horizon.quantization.prepare_qat_fx(
                    model,
                    self.qconfig_dict,
                    self.prepare_custom_config_dict,
                    hybrid=self.hybrid,
                    hybrid_dict=self.hybrid_dict,
                )
        deploy_inputs = dict(
            img=torch.randn((2, 3, 480, 640)),
        )
        check_qconfig(
            model,
            deploy_inputs,
            out_dir="/open_explorer/ddk/samples/ai_toolchain/horizon_model_train_sample/scripts",
        )
        logger.info(
            format_msg(
                "Successfully convert float model to qat model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class QATFusePartBN(BaseConverter):
    """Define the process of fusing bn in a QAT model.

    Usually used in step fuse bn. Note that module do fuse bn only when
    block implement block."fuse_method"().

    Args:
        qat_fuse_patterns: Regex, compile by re.
        fuse_method: Fuse bn method that block calls.
        regex: Whether to match by regex. if not, match by module name.
        strict: Whether the regular expression is required to be all matched.
    """

    def __init__(
        self,
        qat_fuse_patterns: List[str],
        fuse_method: str = "fuse_norm",
        regex: bool = True,
        strict: bool = False,
    ):
        self.qat_fuse_patterns = qat_fuse_patterns
        self.fuse_method = fuse_method
        self.regex = regex
        self.strict = strict
        super(QATFusePartBN, self).__init__()

    def _fuse_bn(self, model: nn.Module):
        if hasattr(model, self.fuse_method):
            return getattr(model, self.fuse_method)()
        else:
            names = []
            for n, m in model.named_children():
                names.append(n)
                setattr(model, n, self._fuse_bn(m))
            return model

    @property
    def get_match_method(self):
        if self.regex:
            return match_children_modules_by_regex
        else:
            return match_children_modules_by_name

    def __call__(self, model):
        # check qat mode in with bn.
        assert horizon.qat_mode.get_qat_mode() in [
            "with_bn",
            "with_bn_reverse_fold",
        ], (
            f"QATFusePartBN only support in with bn mode."
            f"But get {horizon.qat_mode.get_qat_mode()}"
        )

        gen = self.get_match_method
        for n, m in gen(model, self.qat_fuse_patterns, strict=self.strict):
            setattr(model, n, self._fuse_bn(m))

        logger.info(
            format_msg(
                "Successfully qat float model to qat fuse bn model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class Float2Calibration(BaseConverter):
    """Define the process of convert float model to calibration model.

    Args:
        convert_mode: convert mechanism, can be choosen from ("eager", "fx").
        qconfig_dict: only used when convert_mode == 'fx', please refer to
            the doc of `horizon.quantization.prepare_calibration_fx`
            for more info
        prepare_custom_config_dict: only used when convert_mode == 'fx',
            please refer to the doc of
            `horizon.quantization.prepare_calibration_fx` for more info
        hybrid: only used when convert_mode == 'fx', please refer to the doc
            of `horizon.quantization.prepare_calibration_fx` for more info
        hybrid_dict: only used when convert_mode == 'fx', please refer to the
            doc of `horizon.quantization.prepare_calibration_fx` for more info
        qconfig_setter: set qconfig automatically. Value is an instance of
            horizon_plugin_pytorch.quantization.qconfig_template.QconfigSetter
        example_inputs: example inputs for tracing graph. only used when
            qconfig_setter is set.
    """

    def __init__(
        self,
        convert_mode="eager",
        qconfig_dict=None,
        prepare_custom_config_dict=None,
        hybrid=False,
        hybrid_dict=None,
        qconfig_setter=None,
        example_inputs=None,
    ):
        super(Float2Calibration, self).__init__(convert_mode)
        self.qconfig_dict = qconfig_dict
        self.prepare_custom_config_dict = prepare_custom_config_dict
        self.hybrid = hybrid
        self.hybrid_dict = hybrid_dict
        self.qconfig_setter = qconfig_setter
        self.example_inputs = example_inputs
        self.is_qconfig_template_available = check_packages_available(
            "horizon_plugin_pytorch>=2.1.6",
            raise_exception=False,
        )

    def __call__(self, model):
        if self.convert_mode == "eager":
            # make sure the input model is a float model
            model.fuse_model()

        qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.CALIBRATION)
        if self.convert_mode == "eager":
            model.qconfig = qconfig_manager.get_default_qconfig()
        if self.convert_mode == "eager":
            if self.is_qconfig_template_available:
                horizon.quantization.prepare_qat(
                    model,
                    inplace=True,
                    example_inputs=self.example_inputs,
                    qconfig_setter=self.qconfig_setter,
                )
            else:
                horizon.quantization.prepare_qat(
                    model,
                    inplace=True,
                )
        else:
            model.eval()
            if self.is_qconfig_template_available:
                model = horizon.quantization.prepare_qat_fx(
                    model,
                    self.qconfig_dict,
                    self.prepare_custom_config_dict,
                    hybrid=self.hybrid,
                    hybrid_dict=self.hybrid_dict,
                    example_inputs=self.example_inputs,
                    qconfig_setter=self.qconfig_setter,
                )
            else:
                model = horizon.quantization.prepare_qat_fx(
                    model,
                    self.qconfig_dict,
                    self.prepare_custom_config_dict,
                    hybrid=self.hybrid,
                    hybrid_dict=self.hybrid_dict,
                )
        logger.info(
            format_msg(
                "Successfully convert float model to calibration model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class QAT2Quantize(BaseConverter):
    """Define the process of convert qat model to quantize model.

    Args:
        convert_mode: convert mechanism, can be choosen from ("eager", "fx").
            if qat model is a hybrid model prepared in "fx" mode, convert_mode
            must be "fx".
        convert_custom_config_dict: only used when convert_mode == 'fx',
            please refer to the doc of `horizon.quantization.convert_fx`
            for more info
        preserve_qat_mode_dict: only used when debugging quantized model
            precision problems. If provided, specified ops in dict preserves
            qat mode in quantized model. This dict format is
                {
                    "prefixes": (tuple, Optional) Set preserve_qat_mode by the
                        prefix of qualified name. Defaults to tuple().
                    "types": (tuple, Optional) Set preserve_qat_mode by module
                        type. Defaults to tuple().
                }
                Eg. preserve_qat_mode_dict={
                    "prefixes": ("model.backbone.conv1", ...)
                    "types": (horizon.nn.qat.Conv2d, ...)
                }
        fast_mode: Whether to accelerate quantized model forward. If set True,
                   quantized model cannot be compiled.
        use_cutlass: Whether to use cutlass accelerate quantized conv.
    """

    def __init__(
        self,
        convert_mode="eager",
        convert_custom_config_dict=None,
        preserve_qat_mode_dict=None,
        fast_mode: bool = False,
        use_cutlass: bool = False,
    ):
        super(QAT2Quantize, self).__init__(convert_mode)
        self.convert_custom_config_dict = convert_custom_config_dict
        self.preserve_qat_mode_dict = preserve_qat_mode_dict
        self.fast_mode = fast_mode
        self.use_cutlass = use_cutlass

    def __call__(self, model):
        kwargs = {}

        if self.fast_mode:
            check_packages_available(
                "horizon_plugin_pytorch>=1.6.3",
                raise_msg="`fast_mode` require `horizon_plugin_pytorch>=1.6.3`",  # noqa E501
            )
            kwargs["fast_mode"] = self.fast_mode

        if self.use_cutlass:
            check_packages_available(
                "horizon_plugin_pytorch>=1.10.1",
                raise_msg="`use_cutlass` require `horizon_plugin_pytorch>=1.10.1`",  # noqa E501
            )
            os.environ["USE_CUTLASS"] = "1"

        _preserve_qat_mode(model, self.preserve_qat_mode_dict)
        # make sure the input model is a qat model
        if self.convert_mode == "eager":
            horizon.quantization.convert(
                model.eval(),
                inplace=True,
                **kwargs,
            )
        else:
            model = horizon.quantization.convert_fx(
                model.eval(),
                **kwargs,
            )
        logger.info(
            format_msg(
                "Successfully convert qat model to quantize model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class LoadCheckpoint(BaseConverter):
    """Load the checkpoint from file to model and return the checkpoint.

    LoadCheckpoint usually happens before or after BaseConverter.It means
    the model needs to load parameters before or after BaseConverter.

    Args:
        checkpoint_path: Path of the checkpoint file.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        ignore_tensor_shape: Whether to ignore matched key name but
            unmatched shape of tensor while loading state dict.
        verbose: Show unexpect_key and miss_key info.
        return_checkpoint: whether return the values of the checkpoint.
        enable_tracking: whether enable tracking checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str,
        state_dict_update_func: Optional[Callable] = None,
        check_hash: bool = True,
        allow_miss: bool = False,
        ignore_extra: bool = False,
        ignore_tensor_shape: bool = False,
        verbose: bool = False,
        enable_tracking: bool = False,
    ):
        super(LoadCheckpoint, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.state_dict_update_func = state_dict_update_func
        self.check_hash = check_hash
        self.allow_miss = allow_miss
        self.ignore_extra = ignore_extra
        self.ignore_tensor_shape = ignore_tensor_shape
        self.verbose = verbose
        self.enable_tracking = enable_tracking

    def __call__(self, model):
        if self.checkpoint_path is None:
            logger.info(
                format_msg(
                    f"Checkpoint path is None, skip.",  # noqa E501
                    MSGColor.RED,
                )
            )
            return model

        model_checkpoint = load_checkpoint(
            path_or_dict=self.checkpoint_path,
            map_location="cpu",
            state_dict_update_func=self.state_dict_update_func,
            check_hash=self.check_hash,
            enable_tracking=self.enable_tracking,
        )
        set_value("model_checkpoint", model_checkpoint)
        model = load_state_dict(
            model,
            path_or_dict=model_checkpoint,
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            ignore_tensor_shape=self.ignore_tensor_shape,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                f"Load the checkpoint successfully from {self.checkpoint_path}",  # noqa E501
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class LoadMeanTeacherCheckpoint(BaseConverter):
    """Load the Mean-teacher model checkpoint.

    student and teacher model have same structure.
    LoadMeanTeacherCheckpoint usually happens before or after BaseConverter.
    It means the model needs to load parameters before or after BaseConverter.

    Args:
        checkpoint_path: Path of the checkpoint file.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        verbose: Show unexpect_key and miss_key info.
        return_checkpoint: whether return the values of the checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str,
        strip_prefix: str = "module.",
        state_dict_update_func: Optional[Callable] = None,
        check_hash: bool = True,
        allow_miss: bool = False,
        ignore_extra: bool = False,
        verbose: bool = False,
    ):
        super(LoadMeanTeacherCheckpoint, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.strip_prefix = strip_prefix
        self.state_dict_update_func = state_dict_update_func
        self.check_hash = check_hash
        self.allow_miss = allow_miss
        self.ignore_extra = ignore_extra
        self.verbose = verbose

    def __call__(self, model):
        if self.checkpoint_path is None:
            logger.info(
                format_msg(
                    f"Checkpoint path is None, skip.",  # noqa E501
                    MSGColor.RED,
                )
            )
            return model

        model_checkpoint = load_checkpoint(
            path_or_dict=self.checkpoint_path,
            map_location="cpu",
            state_dict_update_func=self.state_dict_update_func,
            check_hash=self.check_hash,
        )
        set_value("model_checkpoint", model_checkpoint)
        model = load_state_dict(
            model,
            path_or_dict=model_checkpoint,
            strip_prefix=self.strip_prefix,
            state_dict_update_func=self.state_dict_update_func,
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                f"Load the checkpoint successfully from {self.checkpoint_path}",  # noqa E501
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class TorchCompile(BaseConverter):
    """Convert torch module to compile wrap module.

    NOTE: Compilation occurs at the first model forward!
    Slower is as expected!

    Args:
        compile_backend: TorchDynamo compile optimizer backend.
        load_extensions: Load extension from hat.utils.trt_fx_extension.py.
    """

    @require_packages("torchdynamo")
    def __init__(
        self,
        compile_backend: Optional[Union[Callable, str]] = None,
        load_extensions: Optional[Union[List[str], str]] = None,
    ):
        super(TorchCompile, self).__init__()
        self.compile_backend = compile_backend
        from hat.utils.trt_fx_extension import load_extension

        if load_extensions is not None:
            if isinstance(load_extensions, str):
                load_extensions = [load_extensions]
            for ext in load_extensions:
                load_extension(ext)

    def __call__(self, model):
        logger.info(
            format_msg(
                f"Wrap torchDynamo optimizer by backend "
                f"{self.compile_backend}...\n"
                f"NOTE: the first time forward is slower due to compilation!",
                MSGColor.RED,
            ),
        )
        model = torchdynamo.optimize(self.compile_backend)(model)

        return model


@OBJECT_REGISTRY.register
class Torch2Compile(BaseConverter):
    """Compile model(nn.Module) by `torch.compile()` in torch>=2.0.

    .. Note::
       compile_submodules and skip_modules are mutually exclusive and
       can only be selected for use. If none of them are used,
       the entire model will be compiled.

    Args:
       compile_submodules: Module to compile, support regex or module name.
       skip_modules: Module to skip compile, support regex or module name.
       regex: Whether to match by regex. if not, match by module name.
       strict: Whether regular expression is required to be all matched.
       dynamo_cfg: A dictionary of options to set `torch._dynamo.config`.
       kwargs: Args of `torch.compile` interface, see:
       https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
    """

    @require_packages("torch>=2.0")
    def __init__(
        self,
        compile_submodules: List[str] = None,
        skip_modules: List[str] = None,
        regex: bool = True,
        strict: bool = False,
        dynamo_cfg: Optional[Dict] = None,
        **kwargs,
    ):
        super(Torch2Compile, self).__init__()
        self.compile_submodules = compile_submodules
        self.skip_modules = skip_modules
        self.regex = regex
        self.strict = strict
        self.compile_args = kwargs

        if compile_submodules and skip_modules:
            raise RuntimeError("compile and skip cannot be used simultaneously!!!")

        default_dynamo_cfg = {"log_level": logging.WARNING}

        if dynamo_cfg is None:
            self.dynamo_cfg = default_dynamo_cfg
        else:
            self.dynamo_cfg = dict(
                chain(default_dynamo_cfg.items(), dynamo_cfg.items())
            )

    def __call__(self, model):
        # set dynamo config
        try:
            from torch import _dynamo as torch_dynamo

            torch_dynamo.reset()

            for k, v in self.dynamo_cfg.items():
                if hasattr(torch_dynamo.config, k):
                    setattr(torch_dynamo.config, k, v)
                else:
                    logger.warning(
                        format_msg(
                            msg=f"`torch._dynamo.config` does not have attr `{k}`, skip set `torch._dynamo.config.{k}={v}`",  # noqa E501
                            color=MSGColor.RED,
                        )
                    )
        except Exception as e:
            logger.warning(
                format_msg(
                    msg=f"Failed to set `torch._dynamo.config`, caused by `{e}`",  # noqa E501
                    color=MSGColor.RED,
                )
            )

        # skip module
        if self.skip_modules:
            model = self.skip_compile_modules(
                model=model,
                skip_modules=self.skip_modules,
                regex=self.regex,
                strict=self.strict,
            )

        logger.info(
            format_msg(
                msg="Using `torch.compile`, which may take a few minutes in first iteration step...",  # noqa E501
                color=MSGColor.GREEN,
            )
        )
        backend = self.compile_args.pop("backend", "inductor")
        backend = CompileBackendWrapper(backend=backend, **self.compile_args)
        if self.compile_submodules:
            model = self.compile_modules(
                model=model,
                compile_submodules=self.compile_submodules,
                regex=self.regex,
                strict=self.strict,
                backend=backend,
                **self.compile_args,
            )
        else:
            model = torch.compile(model, backend=backend, **self.compile_args)

        logger.info(
            format_msg(
                msg="Wrapped model with `torch.compile()`.",  # noqa E501
                color=MSGColor.GREEN,
            )
        )
        return model

    @staticmethod
    def compile_modules(
        model: nn.Module,
        compile_submodules: List[str],
        regex: bool = True,
        strict: bool = False,
        **kwargs,
    ):
        """Add a wrap hook to compile submodule.

        Args:
            model: Model to add hook.
            skip_modules: Submodule to compile, support regex or module name.
            regex: Whether to match by regex. if not, match by module name.
            strict: Whether regular expression is required to be all matched.
        """
        if regex:
            gen = match_children_modules_by_regex
        else:
            gen = match_children_modules_by_name

        white_list = []
        for n, m in gen(model, compile_submodules, strict=strict):
            setattr(model, n, torch.compile(m, **kwargs))
            white_list.append(n)

        logger.info(
            format_msg(
                f"Compile submodules:\n {white_list}.\n",
                MSGColor.GREEN,
            )
        )

        return model

    @staticmethod
    def skip_compile_modules(
        model: nn.Module,
        skip_modules: List[str],
        regex: bool = True,
        strict: bool = False,
    ):
        """Add a wrap hook to skip compile.

        Args:
            model: Model to add hook.
            skip_modules: Module to skip compile, support regex or module name.
            regex: Whether to match by regex. if not, match by module name.
            strict: Whether regular expression is required to be all matched.
        """

        def _wrap_forward(forward_func):
            @functools.wraps(forward_func)
            @disable_compile
            def wrapper(*args, **kwargs):
                output = forward_func(*args, **kwargs)
                return output

            return wrapper

        if regex:
            gen = match_children_modules_by_regex
        else:
            gen = match_children_modules_by_name

        disabled_list = []
        for n, m in gen(model, skip_modules, strict=strict):
            # __call__ will keep the hook
            m.__call__ = _wrap_forward(m.__call__)
            disabled_list.append(n)
        logger.info(
            format_msg(
                f"Skip compile modules:\n {disabled_list}.\n",
                MSGColor.GREEN,
            )
        )

        return model


@OBJECT_REGISTRY.register
class RepModel2Deploy(BaseConverter):
    """Convert Reparameterized model to deploy mode."""

    def __init__(self):
        super(RepModel2Deploy, self).__init__()

    def __call__(self, model):
        for m in model.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        return model


@OBJECT_REGISTRY.register
class GraphModelSplit(BaseConverter):
    """Split graph model in deploy mode."""

    def __init__(
        self,
        split_nodes: List[str],
        next_bases: List[str],
        save_models: Optional[List[str]] = None,
        pick_models_index: Optional[int] = None,
    ):
        super(GraphModelSplit, self).__init__()
        assert len(split_nodes) == len(next_bases)
        assert all([n in ["top", "bottom"] for n in next_bases])
        if save_models is not None:
            assert len(split_nodes) == len(save_models)
            assert all(
                [k in [None, "", "top", "bottom", "top,bottom"] for k in save_models]
            )
        else:
            save_models = ["top,bottom"] * len(split_nodes)
        self.split_nodes = split_nodes
        self.next_bases = next_bases
        self.save_models = save_models
        self.pick_models_index = pick_models_index

    def __call__(self, graph_model: MultitaskGraphModel):
        top_models = []
        bottom_models = []
        for s_n, n_b, k_m in zip(self.split_nodes, self.next_bases, self.save_models):
            top_model, bottom_model, _, _ = graph_model.split_module(
                out_names=None, split_node_name=s_n, common_module_flatten=True
            )
            if k_m in ["top", "top,bottom"]:
                top_models.append(top_model)
            if k_m in ["bottom", "top,bottom"]:
                bottom_models.append(bottom_model)
            if n_b == "top":
                graph_model = top_model
            else:
                graph_model = bottom_model
        split_models = top_models + bottom_models
        if self.pick_models_index is not None:
            return split_models[self.pick_models_index]
        else:
            return split_models


@OBJECT_REGISTRY.register
class GraphModelInputKeyMapping(BaseConverter):
    """Mapping input key in graph model for deploy mode."""

    def __init__(
        self,
        input_key_mapping: Dict[str, str],
    ):
        super(GraphModelInputKeyMapping, self).__init__()
        self.input_key_mapping = input_key_mapping

    def __call__(self, graph_model: MultitaskGraphModel):
        graph = graph_model.get_sub_graph(graph_model._output_names)
        graph_replace_key = graph.copy(deepcopy=False)
        _record_node = {}
        created_place_holders = {}

        def _replace_input_key(node):
            for input_i in node.inputs:
                if input_i.op != "PLACEHOLDER":
                    continue
                if input_i.name in self.input_key_mapping:
                    if (
                        self.input_key_mapping[input_i.name]
                        not in created_place_holders
                    ):
                        created_place_holders[self.input_key_mapping[input_i.name]] = (
                            Node.create_placeholder(
                                name=self.input_key_mapping[input_i.name],
                                attr=input_i.attr,
                            )
                        )
                    create_node = created_place_holders[
                        self.input_key_mapping[input_i.name]
                    ]
                    _record_node[node._inputs.pop(input_i).name] = create_node
                    node._inputs[create_node] = create_node

            def _replace_inner(inp):
                if isinstance(inp, (list, tuple)):
                    new_list = []
                    for item in inp:
                        new_item = _replace_inner(item)
                        new_list.append(new_item)
                    return type(inp)(new_list)
                elif isinstance(inp, dict):
                    new_dict = {}
                    for key, value in inp.items():
                        new_value = _replace_inner(value)
                        if key in _record_node:
                            _key = _record_node[key].name
                        else:
                            _key = key
                        new_dict[_key] = new_value
                    return new_dict
                elif isinstance(inp, Node) and inp.name in _record_node:
                    return _record_node[inp.name]
                else:
                    return inp

            node._args = _replace_inner(node._args)
            node._kwargs = _replace_inner(node._kwargs)

        graph_replace_key.post_order_dfs_visit(fvisit=_replace_input_key)
        graph_replace_key = graph_replace_key.copy(deepcopy=False)

        graph_model_relace_key = graph_model.from_existed_graph_and_modules(
            graph_replace_key,
            node2name=graph_model.node2name,
            flatten_outputs=graph_model.flatten_outputs,
            output_names=graph_model._output_names,
            name2inds_fmts=graph_model._name2inds_fmts,
        )
        return graph_model_relace_key


@OBJECT_REGISTRY.register
class FixWeightQScale(BaseConverter):
    """Fix qscale of weight while calibration or qat stage."""

    def __init__(self):
        super(FixWeightQScale, self).__init__()

    def __call__(self, model):
        for m in model.modules():
            if hasattr(m, "fix_weight_qscale"):
                m.fix_weight_qscale()
        return model


@OBJECT_REGISTRY.register
class LoadHbir(BaseConverter):
    """Load hbir module from file.

    Args:
        path: hbir model path
    """

    def __init__(self, path):
        self.path = path
        super(LoadHbir, self).__init__()

    @require_packages("hbdk4")
    def __call__(self, model):
        from hbdk4.compiler import load

        model = load(self.path)
        return model
