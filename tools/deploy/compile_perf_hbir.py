# Copyright (c) Horizon Robotics. All rights reserved.
# This script requires user define key "hbir_compiler" in config file, whose
# type must be "HbirCompiler".
# If "hbir_compiler" is not defined, we will try to gather required info
# from "compile_cfg".
# All arguments of HbirCompiler.__init__ can be modified with command line
# arguments.

import argparse
import logging
import os
import typing
from distutils.version import LooseVersion

from horizon_plugin_pytorch import March

from hat.registry import OBJECT_REGISTRY, RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import require_packages
from hat.utils.setup_env import setup_args_env

require_packages("hbdk4")
from hbdk4.compiler import compile, convert, load  # noqa: E402

require_packages("horizon_plugin_pytorch>=1.10.3")
from horizon_plugin_pytorch.quantization.hbdk4 import (  # noqa: E402
    get_hbir_input_flattener,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        help="Config file path",
    )

    init_args = typing.get_type_hints(HbirCompiler.__init__)
    init_args.pop("return")
    for name, type in init_args.items():
        parser.add_argument("--" + name, type=type, required=False)

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


@OBJECT_REGISTRY.register
class HbirCompiler:
    """Compile mlir.Module and estimate performance.

    Args:
        model_path: QAT hbir module file path.
        out_path: A filesystem path to save hbm. Must ends with ".hbm".
        march: BPU march, options are "bayes", "b25", "b25e", "b30", "b30g".
        opt: Optimization level. Defaults to 2.
        jobs: Number of threads launched during compiler optimization.
            Defaults to 4.
        max_time_per_fc:
            Set maximum time constraint (unit:us) for per funccall.
        debug: Set whether to contain debug info in HBM.
        preserve_quant_dequant:
            Whether preserve the quantize node on model input and
            dequantize node on model output.
        input_source:
            A structure of strings has the same structure with model input.
            Supported options are "ddr" and "pyramid".
            Defaults to None.
    """

    logger = None

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger

    def __init__(
        self,
        model_path: str,
        out_path: str,
        march: str,
        opt: int,
        jobs: int,
        max_time_per_fc: float,
        debug: bool,
        preserve_transpose: bool,
        preserve_quant_dequant: bool,
        input_source=None,
    ) -> None:
        if preserve_quant_dequant and not preserve_transpose:
            raise ValueError(
                "Cannot preserve quant dequant when dropping transpose"
            )

        self.model = load(model_path)
        self.out_path = out_path
        self.march = march
        self.opt = opt
        self.jobs = jobs
        self.debug = debug
        self.max_time_per_fc = max_time_per_fc

        self.preserve_transpose = preserve_transpose
        self.preserve_quant_dequant = preserve_quant_dequant
        if input_source is None:
            self.input_source = input_source
        else:
            if isinstance(input_source, str):
                input_source = input_source.split(",")
            self.input_source = get_hbir_input_flattener(self.model)(
                input_source
            )
            self.input_source = [
                source.strip() for source in self.input_source
            ]

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def set_input_source(self, input_node, source):
        if source == "ddr":
            pass
        elif source == "pyramid":
            # Convert NHWC data from pyramid to NCHW
            input_node.insert_transpose([0, 3, 1, 2])
            input_node.insert_image_convert()
        else:
            msg = "Unsupported input source {}".format(source)
            self.logger.error(msg)
            raise ValueError(msg)

    def remove_cpu_qcast_dcast(self, args):
        for arg in args:
            removable, diagnostic = arg.is_removable
            if removable:
                attached_op = arg.get_attached_op[0]
                if attached_op.type == "hbtl.call":
                    schema = attached_op.schema
                    if schema.namespace == "quant" and schema.signature in [
                        "qcast",
                        "dcast",
                    ]:
                        removed, diagnostic = arg.remove_attached_op()
                        if removed is False:
                            raise RuntimeError(
                                "Remove quant/dequant failed, reason is:"
                                "\n{}".format(diagnostic)
                            )

    def remove_quant_dequant(self, model):
        from hbdk4.compiler import version

        if LooseVersion(version.VERSION) > LooseVersion("4.0.13"):
            self.remove_cpu_qcast_dcast(model[0].inputs)
            self.remove_cpu_qcast_dcast(model[0].outputs)
        else:
            from hbdk4.compiler.toolbox import remove_function_io  # noqa: E402

            remove_function_io(model, "qnt.quantize")
            remove_function_io(model, "qnt.dequantize")

    def __call__(self):
        if not self.preserve_quant_dequant:
            self.remove_quant_dequant(self.model)
        if self.input_source is not None:
            func = self.model.functions[0]
            for node, source in zip(func.inputs, self.input_source):
                self.set_input_source(node, source)

        quantized_model = convert(self.model, self.march)

        compile(
            quantized_model,
            path=self.out_path,
            march=self.march,
            opt=self.opt,
            jobs=self.jobs,
            max_time_per_fc=self.max_time_per_fc,
            debug=self.debug,
        )

        self.logger.info(
            format_msg("Compiled model: %s" % self.out_path, MSGColor.GREEN)
        )


def compile_perf_hbir(
    cfg_file,
    compile_args,
):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger.info("=" * 50 + "BEGIN HBIR COMPILE" + "=" * 50)

    default_config = dict(  # noqa: C408
        type="HbirCompiler",
        march=March.BAYES,
        opt=2,
        jobs=6,
        debug=False,
        max_time_per_fc=0.0,
        preserve_transpose=False,
        preserve_quant_dequant=False,
    )

    if cfg_file is not None:
        cfg = Config.fromfile(cfg_file)
        if "hbir_compiler" in cfg:
            hbir_compiler = cfg.hbir_compiler
        else:
            logger.warning(
                format_msg(
                    "Do not find hbir_compiler in config file, "
                    "trying to collect need info automaticlly.",
                    MSGColor.RED,
                )
            )

            if "compile_cfg" in cfg:
                hbir_compiler = dict(  # noqa: C408
                    march=cfg.compile_cfg["march"],
                    out_path=cfg.compile_cfg["hbm"],
                    input_source=cfg.compile_cfg["input_source"],
                )
                if "ckpt_dir" in cfg:
                    model_path = os.path.join(cfg.ckpt_dir, "qat.bc")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(cfg.ckpt_dir, "qat.mlir")
                    if not os.path.exists(model_path):
                        msg = "Model file {} do not exist.".format(
                            os.path.join(cfg.ckpt_dir, "qat.bc/mlir")
                        )

                    hbir_compiler["model_path"] = model_path
                else:
                    msg = "Do not find ckpt_dir in config file."
                    logger.error(format_msg(msg, MSGColor.RED))
                    raise ValueError(msg)
            else:
                msg = "Do not find compile_cfg in config file."
                logger.error(format_msg(msg, MSGColor.RED))
                raise ValueError(msg)

        default_config.update(hbir_compiler)

    compile_kwargs = vars(compile_args)
    if cfg_file is not None:
        compile_kwargs.pop("config")
    for k, v in compile_kwargs.items():
        if v is not None:
            default_config[k] = v

    logger.info("Compile config:")
    logger.info(default_config)

    HbirCompiler.set_logger(logger)
    with RegistryContext():
        hbir_compiler: HbirCompiler = build_from_registry(default_config)
        hbir_compiler()

    logger.info("=" * 50 + "END HBIR COMPILE" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    compile_perf_hbir(
        cfg_file=args.config,
        compile_args=args,
    )
