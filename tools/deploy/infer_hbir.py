# Copyright (c) Horizon Robotics. All rights reserved.
# This script requires user define key "hbir_predictor" in config file, whose
# type must be "HbirPredictor".

import argparse
import logging

from hat.registry import OBJECT_REGISTRY, RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import require_packages
from hat.utils.setup_env import setup_args_env

require_packages("hbdk4")
from hbdk4.compiler import load  # noqa: E402

require_packages("horizon_plugin_pytorch>=1.10.3")
from horizon_plugin_pytorch.quantization.hbdk4 import (  # noqa: E402
    get_hbir_input_flattener,
    get_hbir_output_unflattener,
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
        "--step-num",
        type=int,
        default=-1,
        required=False,
        help="Default to -1, which means run one epoch",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set logger level to DEBUG."
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


@OBJECT_REGISTRY.register
class HbirPredictor:
    """Run hbir model forward with custom processes.

    Args:
        dataloader (Iterable): A iterable object contains data.
        model_path (str, optional): Hbir model path. Defaults to None.
        pre_process (callable, optional):
            Modify batched data. Defaults to None.
        post_process (callable, optional):
            Occupy model output. Defaults to None.
        end_process (callable, optional):
            Conclude predict result. Defaults to None.
    """

    logger = None

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger

    def __init__(
        self,
        dataloader,
        model_path=None,
        pre_process=None,
        post_process=None,
        end_process=None,
    ) -> None:
        if model_path is not None:
            self.model = load(model_path)
        else:
            self.model = None

        self.dataloader = dataloader
        self.pre_process = pre_process
        self.post_process = post_process
        self.end_process = end_process

        self.input_flattener = get_hbir_input_flattener(self.model)
        self.output_unflattener = get_hbir_output_unflattener(self.model)

    def flatten_hbir_input(self, batched_data):
        flat_input = self.input_flattener(batched_data)

        return flat_input

    def __call__(self, step_num, model_path=None) -> None:
        if model_path is not None:
            model = load(model_path)
        else:
            model = self.model

        step_count = 0
        for batched_data in self.dataloader:

            if self.pre_process is not None:
                batched_data = self.pre_process(batched_data)

            hbir_output = model[0](*self.flatten_hbir_input(batched_data))
            hbir_output = self.output_unflattener(hbir_output)

            if self.post_process is not None:
                self.post_process(hbir_output)

            print(".", end="", flush=True)

            step_count += 1
            if step_count == step_num:
                break

        if self.end_process is not None:
            self.end_process()


def predict_hbir(config_file_path, step_num, debug):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )

    cfg = Config.fromfile(config_file_path)

    logger.info("=" * 50 + "BEGIN HBIR PREDICT" + "=" * 50)

    if "hbir_predictor" not in cfg:
        msg = "Do not find hbir_predictor in config file."
        logger.error(format_msg(msg, MSGColor.RED))
        raise ValueError(msg)

    hbir_predict_config = cfg.hbir_predictor
    hbir_predict_config["type"] = "HbirPredictor"

    logger.info("Predict config:")
    logger.info(hbir_predict_config)

    HbirPredictor.set_logger(logger)
    with RegistryContext():
        hbir_predictor: HbirPredictor = build_from_registry(
            hbir_predict_config
        )
        hbir_predictor(step_num)

    logger.info("=" * 50 + "END HBIR PREDICT" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    predict_hbir(args.config, args.step_num, args.debug)
