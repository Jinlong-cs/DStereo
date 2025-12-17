# Copyright (c) Horizon Robotics. All rights reserved.
# Export hbir from QAT model, and convert QAT hbir to Quantized hbir.
# This script requires user define key "hbir_exporter" in config file, whose
# type must be "HbirExporter".
# If "hbir_exporter" is not defined, we will try to gather required info
# from "deploy_model", "deploy_inputs", "qat/calibration_predictor"
# and "ckpt_dir".

import argparse
import logging
import os

import horizon_plugin_pytorch as horizon
import torch
from torch.utils._pytree import tree_flatten

from hat.registry import OBJECT_REGISTRY, RegistryContext, build_from_registry
from hat.utils.apply_func import _as_list
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import require_packages
from hat.utils.setup_env import setup_args_env

require_packages("hbdk4")
from hbdk4.compiler import Module, convert, save  # noqa: E402

require_packages("horizon_plugin_pytorch>=1.10.3")
from horizon_plugin_pytorch.quantization.hbdk4 import export  # noqa: E402
from horizon_plugin_pytorch.utils.misc import (  # noqa: E402
    pytree_convert,
    tensor_struct_repr,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Config file path.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        required=False,
        help="Where to save exported hbir files.",
    )
    parser.add_argument(
        "--with-check",
        action="store_true",
        help="Whether check the mlir model forward with example input.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set logger level to DEBUG."
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


@OBJECT_REGISTRY.register
class HbirExporter:
    """Export hbir from qat nn.Module and convert hbir to quantized.

    Args:
        model: Input model.
        model_convert_pipeline:
            Model convert pipeline that converts model to qat mode.
        example_inputs: Model input for exporting.
            **NOTE**: Hbir model do not support dynamic shape currently.
        march: BPU march.
        save_path: Where to save exported hbir files.
        with_check: Whether check the mlir model forward with example input.
    """

    logger = None

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger

    def __init__(
        self,
        model: torch.nn.Module,
        model_convert_pipeline,
        example_inputs,
        march,
        save_path,
        with_check,
        model_name: str = "model",
    ) -> None:
        horizon.march.set_march(march)

        self.model = model_convert_pipeline(model)
        self.example_inputs = _as_list(example_inputs)
        self.march = march
        self.save_path = save_path
        self.with_check = with_check
        self.model_name = model_name

    def get_hbir_input(self, example_inputs):
        flat_inputs, _ = tree_flatten(example_inputs)
        flat_inputs = pytree_convert(
            flat_inputs, torch.Tensor, lambda x: x.cpu().numpy()
        )

        return flat_inputs

    def save_hbir(self, model, path: str, debug: bool):
        dir_name = os.path.dirname(path)
        if len(dir_name) > 0:
            os.makedirs(dir_name, exist_ok=True)
        if debug:
            if isinstance(model, Module):
                model = model.module

            if not path.endswith(".mlir"):
                path += ".mlir"
            with open(path, "w") as f:
                f.writelines(
                    str(
                        model.operation.get_asm(
                            enable_debug_info=True, pretty_debug_info=False
                        )
                    )
                )
        else:
            if not path.endswith(".bc"):
                path += ".bc"
            save(model, path)

    def __call__(self, debug) -> None:
        self.logger.info(
            "Exporting hbir with input {}".format(
                tensor_struct_repr(self.example_inputs)
            )
        )

        self.model.eval()
        qat_hbir = export(
            self.model, self.example_inputs, name=self.model_name
        )

        self.logger.info(
            format_msg(
                "Saving qat hbir to {}".format(self.save_path), MSGColor.GREEN
            )
        )
        self.save_hbir(qat_hbir, os.path.join(self.save_path, "qat"), debug)

        if self.with_check:
            self.logger.info("Checking qat hbir with example_inputs.")
            qat_hbir.forward(*self.get_hbir_input(self.example_inputs))
            self.logger.info(
                format_msg("Qat hbir check passed.", MSGColor.GREEN)
            )

        self.logger.info("Converting hbir to quantized.")
        quantized_hbir = convert(qat_hbir, self.march)

        self.logger.info(
            format_msg(
                "Saving quantized hbir to {}".format(self.save_path),
                MSGColor.GREEN,
            )
        )
        self.save_hbir(
            quantized_hbir,
            os.path.join(self.save_path, "quantized"),
            debug,
        )

        if self.with_check:
            self.logger.info("Checking quantized hbir with example_inputs.")
            quantized_hbir.forward(*self.get_hbir_input(self.example_inputs))
            self.logger.info(
                format_msg("Quantized hbir check passed.", MSGColor.GREEN)
            )


def export_hbir(
    config_file_path,
    save_path=None,
    with_check=False,
    debug=False,
):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )

    cfg = Config.fromfile(config_file_path)

    logger.info("=" * 50 + "BEGIN EXPORTING HBIR" + "=" * 50)

    if "hbir_exporter" in cfg:
        hbir_exporter = cfg.hbir_exporter
        if (
            "type" not in hbir_exporter
            or hbir_exporter["type"] != "HbirExporter"
        ):
            msg = "hbir_exporter must have 'type'='HbirExporter'"
            logger.error(msg)
            raise ValueError(msg)
    else:
        logger.warning(
            format_msg(
                "Do not find hbir_exporter in config file, "
                "trying to collect need info automaticlly.",
                MSGColor.RED,
            )
        )

        def raise_key_missing_error(key):
            msg = "Do not find {} in config file.".format(key)
            logger.error(format_msg(msg, MSGColor.RED))
            raise ValueError(msg)

        if "deploy_model" in cfg:
            model = cfg.deploy_model
        else:
            raise_key_missing_error("deploy_model")

        if "deploy_inputs" in cfg:
            example_inputs = cfg.deploy_inputs
        else:
            raise_key_missing_error("deploy_inputs")

        if "qat_predictor" in cfg:
            predictor = cfg.qat_predictor
        elif "calibration_predictor" in cfg:
            predictor = cfg.calibration_predictor
        else:
            msg = (
                "Do not find qat_predictor or calibration_predictor"
                " in config file."
            )
            logger.error(format_msg(msg, MSGColor.RED))
            raise ValueError(msg)
        model_convert_pipeline = predictor["model_convert_pipeline"]

        if "ckpt_dir" in cfg:
            save_path = cfg.ckpt_dir
        else:
            raise_key_missing_error("ckpt_dir")

        hbir_exporter = dict(  # noqa: C408
            type="HbirExporter",
            model=model,
            model_convert_pipeline=model_convert_pipeline,
            example_inputs=example_inputs,
            save_path=save_path,
            model_name=cfg.get("task_name", "model"),
        )

    if "march" not in hbir_exporter:
        if "march" not in cfg:
            logger.warning(
                format_msg(
                    "Please make sure the march is provided in configs. "
                    "Defaultly use {}".format(horizon.march.March.BAYES),
                    MSGColor.RED,
                )
            )

            hbir_exporter["march"] = horizon.march.March.BAYES
        else:
            hbir_exporter["march"] = cfg.march

    if save_path is not None:
        hbir_exporter["save_path"] = save_path

    hbir_exporter["with_check"] = (
        hbir_exporter.get("with_check", False) or with_check
    )

    logger.info("Export config:")
    logger.info(tensor_struct_repr(hbir_exporter))

    HbirExporter.set_logger(logger)
    with RegistryContext():
        hbir_exporter: HbirExporter = build_from_registry(hbir_exporter)
        hbir_exporter(debug)

    logger.info("=" * 50 + "END EXPORTING HBIR" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    export_hbir(args.config, args.save_path, args.with_check, args.debug)
