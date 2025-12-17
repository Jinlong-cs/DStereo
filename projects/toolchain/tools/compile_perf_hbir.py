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
from collections import defaultdict

from hbdk4.compiler import compile, load, save  # noqa: E402
from hbdk4.compiler.toolbox import remove_function_io  # noqa: E402
from horizon_plugin_pytorch import March
from horizon_plugin_pytorch.quantization.hbdk4 import (  # noqa: E402
    get_hbir_input_flattener,
)

from hat.registry import OBJECT_REGISTRY, RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import require_packages
from hat.utils.setup_env import setup_args_env


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
        model_path: Quantized hbir module file path.
        out_path: A filesystem path to save hbm. Must ends with ".hbm".
        march: BPU march, options are "bayes", "b25", "b25e", "b30", "b30g".
        opt: Optimization level. Defaults to 2.
        jobs: Number of threads launched during compiler optimization.
            Defaults to 4.
        max_time_per_fc:
            Set maximum time constraint (unit:us) for per funccall.
        debug: Set whether to contain debug info in HBM.
        preserve_transpose:
            Whether preserve the transpose node on model input and output.
        preserve_quant_dequant:
            Whether preserve the quantize node on model input and
            dequantize node on model output.
        input_source:
            A structure of strings has the same structure with model input.
            Supported options are "ddr" and "pyramid".
            Defaults to None.
        transpose_dim:
            For transpose inputs or outputs node.
        split_dim:
            For split pyramid inputs.
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
        transpose_dim: dict = None,
        split_dim: dict = None,
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
        if transpose_dim is None:
            self.transpose_dim = defaultdict(dict)
        else:
            self.transpose_dim = transpose_dim
        if split_dim is None:
            self.split_dim = defaultdict(dict)
        else:
            self.split_dim = split_dim

    def set_input_source(
        self, input_node, source, transpose_dim=None, split_dim=None
    ):
        source = "ddr"
        if source == "ddr":
            self.logger.info(
                f"Input node : {input_node.name}, compiled with ddr!!"
                f"transpose_dim: {transpose_dim}"
            )
            # self.logger.info(f"{source}: {transpose_dim}")
            if transpose_dim is not None:
                input_node.insert_transpose(transpose_dim)
            else:
                pass
        elif source == "pyramid":
            self.logger.info(
                f"Input node : {input_node.name}, compiled with pyramid!!"
            )
            input_node.insert_transpose([0, 3, 1, 2])
            input_node.insert_image_convert()
        else:
            msg = "Unsupported input source {}".format(source)
            self.logger.error(msg)
            raise ValueError(msg)

    def set_output_layout(self, output_node, transpose_dim):
        self.logger.info(
            f"Output node : {output_node.name}, transpose_dim: {transpose_dim}"
        )
        output_node.insert_transpose(transpose_dim)

    def set_output_split(self, output_node, split_dim):
        self.logger.info(f"output split_dim: {split_dim}")
        output_node.insert_split(split_dim)

    def remove_quant_dequant(self, model):
        remove_function_io(model, "qnt.quantize")
        remove_function_io(model, "qnt.dequantize")

    def __call__(self):
        if not self.preserve_quant_dequant:
            self.remove_quant_dequant(self.model)

        if self.input_source is not None:
            func = self.model.functions[0]

            inputs_split_dim = self.split_dim.get("inputs", defaultdict(dict))

            for idx, value in inputs_split_dim.items():
                self.logger.info(
                    f"inputs{int(idx)} split_dim: {value[0]} size: {value[1]}"
                )
                node_tmp = func.inputs[int(idx)]
                node_tmp.insert_split(value[0])
                for _ in range(value[1] - 1):
                    self.input_source.insert(int(idx), "pyramid")

            offset = 0

            inputs_transpose_dim = self.transpose_dim.get(
                "inputs", defaultdict(dict)
            )
            global_transpose_dim = inputs_transpose_dim.get("global", None)

            num_inputs = len(func.inputs)

            for idx in range(num_inputs):
                node = func.inputs[idx + offset]
                source = self.input_source[idx]

                if global_transpose_dim is not None:
                    node_transpose_dim = global_transpose_dim
                    # if idx < min_split_idx:
                    #     node_transpose_dim = None
                else:
                    node_transpose_dim = inputs_transpose_dim.get(
                        str(idx - offset + 1), None
                    )
                self.set_input_source(node, source, node_transpose_dim)
                if source == "pyramid":
                    offset = offset + 1

        outputs_transpose_dim = self.transpose_dim.get(
            "outputs", defaultdict(dict)
        )

        if len(outputs_transpose_dim):
            global_transpose_dim = outputs_transpose_dim.get("global", None)
            for idx, node in enumerate(func.outputs):
                if global_transpose_dim is not None:
                    node_transpose_dim = global_transpose_dim
                else:
                    node_transpose_dim = outputs_transpose_dim.get(
                        str(idx), None
                    )

                if node_transpose_dim is not None:
                    self.set_output_layout(node, node_transpose_dim)

        dst_dir = os.path.dirname(self.out_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        save(self.model, os.path.join(dst_dir, "beforcompilequantized.bc"))
        compile(
            self.model,
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
    logger.info("=" * 50 + "BEGIN HBLR COMPILE" + "=" * 50)

    default_config = dict(  # noqa: C408
        type="HbirCompiler",
        march=March.BAYES,
        opt=2,
        jobs=32,
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
                    transpose_dim=cfg.compile_cfg.get(
                        "transpose_dim", defaultdict(dict)
                    ),
                    split_dim=cfg.compile_cfg.get(
                        "split_dim", defaultdict(dict)
                    ),
                )
                if "save_dir" in cfg:
                    model_path = os.path.join(cfg.save_dir, "quantized.bc")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(
                            cfg.ckpt_dir, "quantized.mlir"
                        )
                    if not os.path.exists(model_path):
                        msg = "Model file {} do not exist.".format(
                            os.path.join(cfg.ckpt_dir, "quantized.bc/mlir")
                        )

                    hbir_compiler["model_path"] = model_path
                elif "ckpt_dir" in cfg:
                    model_path = os.path.join(cfg.ckpt_dir, "quantized.bc")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(
                            cfg.ckpt_dir, "quantized.mlir"
                        )
                    if not os.path.exists(model_path):
                        msg = "Model file {} do not exist.".format(
                            os.path.join(cfg.ckpt_dir, "quantized.bc/mlir")
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

    logger.info("=" * 50 + "END HBLR COMPILE" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    compile_perf_hbir(
        cfg_file=args.config,
        compile_args=args,
    )
