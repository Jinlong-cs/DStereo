"""predict tools."""
import argparse
import ast
import logging
import os
import warnings
from typing import List, Sequence, Union

import horizon_plugin_pytorch as horizon
import torch

import hat
from hat.engine import build_launcher
from hat.registry import RegistryContext, build_from_registry
from hat.utils.checkpoint import load_state_dict
from hat.utils.config import Config, ConfigVersion
from hat.utils.deterministic import set_deterministic_level
from hat.utils.distributed import get_dist_info
from hat.utils.logger import (
    LOG_DIR,
    DisableLogger,
    ExperimentLogger,
    MSGColor,
    format_msg,
    init_rank_logger,
    rank_zero_info,
)
from hat.utils.package_check import package_check
from hat.utils.seed import seed_training
from hat.utils.setup_env import setup_args_env, setup_hat_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        required=True,
        help=(
            "the predict stage, you should define "
            "{stage}_predictor in your config"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="predict config file path",
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
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="NCCL",
        choices=["NCCL", "GLOO"],
        help="dist url for init process",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["mpi", "torch"],
        default=None,
        help="job launcher for multi machines",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint path used to predict",
    )
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
        help="export HAT_PIPELINE_TEST=1, which used in config",
    )
    parser.add_argument(
        "--opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--opts-overwrite",
        type=ast.literal_eval,
        default=True,
        help="True or False, default True, "
        "Weather to overwrite existing (keys, values) in configs",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=False,
        help="export HAT_ENABLE_MODEL_TRACKING=1, which enable aidi tracking",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="project_id.",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def predict_entrance(
    device: Union[None, int, Sequence[int]],
    stage: str,
    cfg_file: str,
    cfg_opts: List,
    cfg_opts_overwrite: bool,
    ckpt: str = None,
):
    if isinstance(cfg_file, Config):
        cfg = cfg_file
        cfg_file = cfg._filename
    else:
        # Avoid repeated fromfile causing different behaviors in same process
        cfg = Config.fromfile(cfg_file)
        if cfg_opts is not None:
            cfg.merge_from_list_or_dict(cfg_opts, overwrite=cfg_opts_overwrite)

    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)

    # 1. init logger
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=cfg_file,
        step=stage,
        prefix="predict-",
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )

    set_deterministic_level(level=cfg.get("deterministic_level", None))
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    if cfg.get("seed", None) is not None:
        seed_training(cfg.seed)

    rank_zero_info("=" * 50 + "BEGIN %s PREDICT" % stage.upper() + "=" * 50)
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    # build model
    assert hasattr(
        cfg, f"{stage}_predictor"
    ), f"you should define {stage}_predictor in the config file"
    predictor = cfg[f"{stage}_predictor"]

    with DisableLogger(disable_logger), RegistryContext():
        predictor["device"] = device
        predictor = build_from_registry(predictor)
        if ckpt is not None:
            logger.warning("Make sure ckpt is consistent with training stage")
            model = predictor.model
            load_pred_ckpt_func = cfg.get(
                "load_pred_ckpt_func", load_state_dict
            )
            load_pred_ckpt_func(
                model,
                path_or_dict=ckpt,
                map_location="cpu",
            )
            predictor.model = model
        predictor.fit()

    rank_zero_info("=" * 50 + "END PREDICT" + "=" * 50)
    rank_zero_info("=" * 50 + "END %s PREDICT" % stage.upper() + "=" * 50)


def predict(
    stage: str,
    config: str,
    device_ids: str = None,
    dist_url: str = "auto",
    backend: str = "NCCL",
    launcher: str = None,
    pipeline_test: bool = False,
    opts: list = None,
    opts_overwrite: bool = None,
    ckpt: str = None,
    args_env: list = None,
    enable_tracking: bool = False,
    project_id: str = None,
):
    """Predict  function.

    Args:
        stage: The training stage, you should define {stage}_predictor
               in your config.
        config: Config file path.
        device_ids: GPU device ids like '0,1,2,3', will override
                    `device_ids` in config.
        dist_url: Dist url for init process, such as tcp://localhost:8000.
        backend: Dist url for init process.
        launcher: Job launcher for multi machines.
        ckpt: Checkpoint path used to predict.
        args_env: The args will be set in env.
        enable_tracking: Whether to enable aidi tracking.
        project_id: Whether set env for project_id
    """
    if args_env:
        setup_args_env(args_env)
    setup_hat_env(stage, pipeline_test, enable_tracking, project_id)
    package_check()

    config_info = Config.fromfile(config)
    # check config version
    config_version = config_info.get("VERSION", None)
    if config_version is not None:
        assert (
            config_version == ConfigVersion.v2
        ), "{} only support config with version 2, not version {}".format(
            os.path.basename(__file__), config_version.value
        )
    else:
        warnings.warn(
            "VERSION will must set in config in the future."
            "You can refer to configs/classification/resnet18.py,"
            "and configs/classification/bernoulli/mobilenetv1.py."
        )

    if opts is not None:
        config_info.merge_from_list_or_dict(opts, overwrite=opts_overwrite)

    if device_ids is not None:
        ids = list(map(int, device_ids.split(",")))
    else:
        ids = config_info.device_ids
    if ids is None or ids == -1:
        assert backend == "GLOO", f"backend should be GLOO, but get {backend}"
    num_processes = config_info.get("num_processes", None)

    predictor_cfg = config_info[f"{stage}_predictor"]
    experiment_logger = ExperimentLogger(enable_tracking=enable_tracking)
    experiment_logger.init_group(f"prediction-group-{stage}")

    launch = build_launcher(predictor_cfg)
    if launch is hat.engine.ddp_trainer.launch and launcher is not None:
        config_info = config_info
    else:
        config_info = config

    launch(
        predict_entrance,
        ids,
        dist_url=dist_url,
        dist_launcher=launcher,
        num_processes=num_processes,
        backend=backend,
        args=(
            stage,
            config_info,
            opts,
            opts_overwrite,
            ckpt,
        ),
    )


if __name__ == "__main__":

    try:
        args, args_env = parse_args()

        predict(
            stage=args.stage,
            config=args.config,
            device_ids=args.device_ids,
            dist_url=args.dist_url,
            backend=args.backend,
            launcher=args.launcher,
            pipeline_test=args.pipeline_test,
            opts=args.opts,
            opts_overwrite=args.opts_overwrite,
            project_id=args.project_id,
            ckpt=args.ckpt,
            args_env=args_env,
            enable_tracking=args.enable_tracking,
        )
    except Exception as e:
        ExperimentLogger.log_exception(
            exception=e,
            prefix="predict failed",
            enable_tracking=args.enable_tracking,
        )
        raise e
