# Copyright (c) Horizon Robotics. All rights reserved.
"""perf dataloader speed tool.

Usage:

>> python tools/perf_dataloader.py --config tests/data/toy_multitask/multitask.py --stage float --allow-all-rank

"""  # noqa
import argparse
import pprint
from typing import Sequence, Union

import torch
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
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        default=False,
        help="skip transform when perf dataloader",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="num_workers of dataloader",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def remove_transform(cfg, logger):
    """Remove transform in cfg."""
    if cfg["type"] == "MultitaskLoader":
        for _task, loader in cfg["loaders"].items():
            assert loader["type"] == torch.utils.data.dataloader.DataLoader
            remove_transform(loader, logger)
    elif cfg["type"] == torch.utils.data.dataloader.DataLoader:
        dataset = cfg["dataset"]
        remove_transform(dataset, logger)
    elif cfg["type"] == "ConcatDataset":
        assert "datasets" in cfg
        for dataset_i in cfg["datasets"]:
            remove_transform(dataset_i, logger)
    elif cfg["type"] == "ComposeDataset":
        assert "datasets" in cfg
        for dataset_i in cfg["datasets"]:
            remove_transform(dataset_i, logger)
    elif "transform" in cfg:
        logger.info(
            format_msg("remove transform: ", MSGColor.GREEN)
            + f"{cfg['transform']}"
        )  # noqa
        cfg.pop("transform")
    elif "transforms" in cfg:
        logger.info(
            format_msg("remove transform: ", MSGColor.GREEN)
            + f"{cfg['transforms']}"
        )  # noqa
        cfg.pop("transforms")
    else:
        print("no transform found", cfg)


def main(
    device: Union[None, int, Sequence[int]],
    cfg_file: str,
    stage: str,
    allow_all_rank: bool,
    skip_transform: bool,
    num_workers: int,
):  # noqa: D205,D400
    """
    Args:
        device: run on cpu (if None), or gpu (list of gpu ids)
        cfg_file: Config file used to build log file name.
        stage: Current training stage used to build log file name.
        allow_all_rank: Allow all rank process to show perf result.
        skip_transform: Skip transform when perf dataloader.
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
        data_loader_perf = cfg["data_loader_perf"]
        if data_loader_perf["dataloader"]["type"] == "MultitaskLoader":
            for task_name in data_loader_perf["dataloader"]["loaders"]:
                data_loader_perf["dataloader"]["loaders"][task_name][
                    "num_workers"
                ] = num_workers
                if num_workers < 1:
                    data_loader_perf["dataloader"]["loaders"][task_name].pop(
                        "persistent_workers", None
                    )
        else:
            data_loader_perf["dataloader"]["num_workers"] = num_workers
        if skip_transform:
            remove_transform(data_loader_perf["data_loader"], logger)
            logger.info(
                format_msg("perf dataloader without transform", MSGColor.GREEN)
            )
        data_loader_perf = build_from_registry(data_loader_perf)

        msg = "Run in perf dataloader speed mode"
        logger.info(format_msg(msg, MSGColor.GREEN))

        data_loader_perf.run()

    logger.info("=" * 50 + "END Perf Dataloader" + "=" * 50)


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
            args.skip_transform,
            args.num_workers,
        ),
    )
