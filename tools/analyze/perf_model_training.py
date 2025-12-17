# Copyright (c) Horizon Robotics. All rights reserved.
"""Perf model training speed tool.

Usage:

>> python tools/perf_model_training.py --config tests/data/toy_multitask/multitask.py --stage float --allow-all-rank

"""  # noqa
import argparse
import pprint
from typing import Sequence, Union

import torch.backends.cudnn as cudnn

from hat.engine import build_launcher
from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config, filter_configs
from hat.utils.distributed import get_dist_info
from hat.utils.logger import (
    DisableLogger,
    MSGColor,
    format_msg,
    init_rank_logger,
)
from hat.utils.seed import seed_training
from hat.utils.setup_env import setup_args_env, setup_hat_env

LOG_DIR = "work_dirs/hat_logss"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help=(
            "the training stage, you should define "
            "{stage}_trainer in your config"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default=None,
        help="GPU device ids like '0,1,2,3', "
        "will override `device_ids` in config",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["torch", "mpi"],
        default=None,
        help="job launcher for multi machines",
    )
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
        help="export HAT_PIPELINE_TEST=1, which used in config",
    )
    parser.add_argument(
        "--allow-all-rank",
        action="store_true",
        default=False,
        help="allow all rank process to show perf result",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def main(
    device: Union[None, int, Sequence[int]],
    cfg_file: str,
    stage: str,
    allow_all_rank: bool,
):  # noqa: D205,D400
    """
    Args:
        device: run on cpu (if None), or gpu (list of gpu ids)
        cfg_file: Config file used to build log file name.
        stage: Current training stage used to build log file name.
        allow_all_rank: Allow all rank process to show perf result.
    """
    cfg = Config.fromfile(cfg_file)

    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)
    if allow_all_rank:
        disable_logger = False

    # 1. init logger
    logger = init_rank_logger(
        rank, save_dir=LOG_DIR, cfg_file=cfg_file, step=stage, prefix="train-"
    )
    logger.info("=" * 50 + "BEGIN %s STAGE" % stage.upper() + "=" * 50)

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`disable_current_rank_logger` in config if you don't want this.",  # noqa: E501
                MSGColor.GREEN,
            )
        )
    else:
        logger.info(pprint.pformat(filter_configs(cfg)))

    # 2. seed training
    cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        seed_training(cfg.seed)

    # 3. build and run trainer
    with DisableLogger(disable_logger), RegistryContext():
        model_training_perf = cfg["model_training_perf"]
        model_training_perf["trainer"]["device"] = device
        model_training_perf = build_from_registry(model_training_perf)

        msg = "Run in perf model training mode"
        logger.info(format_msg(msg, MSGColor.GREEN))

        model_training_perf.run()

    logger.info("=" * 50 + "END Perf Model Training" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    setup_hat_env(args.stage, args.pipeline_test)

    config = Config.fromfile(args.config)

    if args.device_ids is not None:
        ids = list(map(int, args.device_ids.split(",")))
    else:
        ids = config.device_ids

    trainer_config = getattr(config, f"{args.stage}_trainer")
    launch = build_launcher(trainer_config)
    launch(
        main,
        ids,
        dist_url=args.dist_url,
        dist_launcher=args.launcher,
        args=(
            args.config,
            args.stage,
            args.allow_all_rank,
        ),
    )
