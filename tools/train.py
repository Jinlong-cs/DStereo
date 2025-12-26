"""train tools."""

import argparse
import ast
import logging
import os
import pprint
import warnings
from typing import List, Sequence, Union

import horizon_plugin_pytorch as horizon
import torch.backends.cudnn as cudnn

import hat
from hat.engine import build_launcher
from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config, ConfigVersion, filter_configs
from hat.utils.deterministic import set_deterministic_level
from hat.utils.distributed import get_dist_info, rank_zero_only
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
from hat.utils.thread_init import init_num_threads

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
            "the training stage, you should define " "{stage}_trainer in your config"
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
        help="GPU device ids like '0,1,2,3', " "will override `device_ids` in config",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
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
        "--level",
        type=int,
        default=logging.WARNING,
        help="Set the logging level for other rank except rank0",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=False,
        help="export HAT_ENABLE_MODEL_TRACKING=1, which enable aidi tracking",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="enable W&B logging",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-log-every-steps", type=int, default=None)
    parser.add_argument("--fixed-train-index", type=int, default=None)
    parser.add_argument("--fixed-val-index", type=int, default=None)
    parser.add_argument("--vis-every-steps", type=int, default=None)
    parser.add_argument("--vis-every-epochs", type=int, default=None)
    parser.add_argument("--vis-num-samples", type=int, default=None)
    parser.add_argument(
        "--vis-strategy",
        type=str,
        choices=["fixed", "random"],
        default=None,
    )
    parser.add_argument("--vis-seed", type=int, default=None)
    parser.add_argument("--vis-train-indices", type=int, nargs="+", default=None)
    parser.add_argument("--vis-val-indices", type=int, nargs="+", default=None)
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument(
        "--log-pretrained-baseline",
        type=ast.literal_eval,
        default=True,
        help="True or False, default True, log pretrained baseline in W&B",
    )
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        default=False,
        help="enable TensorBoard logging",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def train_entrance(
    device: Union[None, int, Sequence[int]],
    cfg_file: str,
    cfg_opts: List,
    cfg_opts_overwrite: bool,
    stage: str,
    level: int = logging.WARNING,
):
    """Training entrance function for launcher.

    Args:
        device: run on cpu (if None), or gpu (list of gpu ids)
        cfg_file: Config file used to build log file name.
        cfg_opts: Custom config options from command-line.
        cfg_opts_overwrite: Weather to overwrite existing {k: v} in configs.
        stage: Current training stage used to build log file name.
        level: logging level for other rank except rank0.
    """

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
        prefix="train-",
        filter_warning=cfg.get("filter_warning", False),
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )

    if (
        "redirect_config_logging_path" in cfg
        and cfg["redirect_config_logging_path"]
        and rank == 0
    ):
        with open(cfg["redirect_config_logging_path"], "w") as fid:
            fid.write(pprint.pformat(filter_configs(cfg)))
        rank_zero_info(
            "save config logging output to %s" % cfg["redirect_config_logging_path"]
        )
    else:
        rank_zero_info(pprint.pformat(filter_configs(cfg)))

    rank_zero_info("=" * 50 + "BEGIN %s STAGE" % stage.upper() + "=" * 50)

    # 2. init num threads and gpu affinity
    init_num_threads()
    if int(os.environ.get("USE_DCU", 0)) == 0:
        from hat.utils.gpu_affinity import set_affinity

        set_affinity(device, mode=cfg.get("gpu_affinity", "none"))

    # 3. seed training
    set_deterministic_level(level=cfg.get("deterministic_level", None))
    cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        seed_training(cfg.seed)

    if "march" not in cfg:
        rank_zero_only(
            logger.warning(
                format_msg(
                    f"Please make sure the march is provided in configs. "
                    f"Defaultly use {horizon.march.March.BAYES}",
                    MSGColor.RED,
                )
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    logger.info("build and run trainer")
    # 4. build and run trainer
    with DisableLogger(disable_logger, level), RegistryContext():
        trainer = getattr(cfg, f"{stage}_trainer")
        trainer["device"] = device
        logger.info("prepare for build_from_registry")
        trainer = build_from_registry(trainer)
        logger.info("build_from_registry done")
        trainer.fit()

    rank_zero_info("=" * 50 + "END %s STAGE" % stage.upper() + "=" * 50)


def train(
    stage: str,
    config: str,
    device_ids: str = None,
    dist_url: str = "auto",
    launcher: str = None,
    pipeline_test: bool = False,
    opts: list = None,
    opts_overwrite: bool = None,
    args_env: list = None,
    level: int = logging.WARNING,
    enable_tracking: bool = False,
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_name: str = None,
    wandb_tags: str = None,
    wandb_resume: str = None,
    wandb_run_id: str = None,
    wandb_log_every_steps: int = None,
    fixed_train_index: int = None,
    fixed_val_index: int = None,
    vis_every_steps: int = None,
    vis_every_epochs: int = None,
    vis_num_samples: int = None,
    vis_strategy: str = None,
    vis_seed: int = None,
    vis_train_indices: list = None,
    vis_val_indices: list = None,
    pretrained_ckpt: str = None,
    log_pretrained_baseline: bool = True,
    use_tensorboard: bool = False,
):
    """Training  function.

    Args:
        stage: The training stage, you should define {stage}_trainer
               in your config.
        config: Config file path.
        device_ids: GPU device ids like '0,1,2,3', will override
                    `device_ids` in config.
        dist_url: Dist url for init process, such as tcp://localhost:8000.
        launcher: Job launcher for multi machines.
        pipeline_test: export HAT_PIPELINE_TEST=1, which used in config.
        opts: modify config options.
        opts_overwrite: True or False, default True, weather to overwrite
                        existing (keys, values) in configs.
        args_env: The args will be set in env.
        level: logging level for other rank except rank0.
        enable_tracking: Whether to enable aidi tracking.
    """
    if args_env:
        setup_args_env(args_env)
    os.environ["HAT_USE_TENSORBOARD"] = "1" if use_tensorboard else "0"
    os.environ["HAT_USE_WANDB"] = "1" if use_wandb else "0"
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_name:
        os.environ["WANDB_NAME"] = wandb_name
    if wandb_tags:
        os.environ["WANDB_TAGS"] = wandb_tags
    if wandb_resume:
        os.environ["WANDB_RESUME"] = wandb_resume
    if wandb_run_id:
        os.environ["WANDB_RUN_ID"] = wandb_run_id
    if wandb_log_every_steps is not None:
        os.environ["WANDB_LOG_EVERY_STEPS"] = str(wandb_log_every_steps)
    if fixed_train_index is not None:
        os.environ["WANDB_FIXED_TRAIN_INDEX"] = str(fixed_train_index)
    if fixed_val_index is not None:
        os.environ["WANDB_FIXED_VAL_INDEX"] = str(fixed_val_index)
    if vis_every_steps is not None:
        os.environ["WANDB_VIS_EVERY_STEPS"] = str(vis_every_steps)
    if vis_every_epochs is not None:
        os.environ["WANDB_VIS_EVERY_EPOCHS"] = str(vis_every_epochs)
    if vis_num_samples is not None:
        os.environ["WANDB_VIS_NUM_SAMPLES"] = str(vis_num_samples)
    if vis_strategy:
        os.environ["WANDB_VIS_STRATEGY"] = str(vis_strategy)
    if vis_seed is not None:
        os.environ["WANDB_VIS_SEED"] = str(vis_seed)
    if vis_train_indices is not None:
        os.environ["WANDB_VIS_TRAIN_INDICES"] = ",".join(
            [str(i) for i in vis_train_indices]
        )
    if vis_val_indices is not None:
        os.environ["WANDB_VIS_VAL_INDICES"] = ",".join(
            [str(i) for i in vis_val_indices]
        )
    if pretrained_ckpt:
        os.environ["HAT_PRETRAINED_BASELINE_CKPT"] = pretrained_ckpt
    if log_pretrained_baseline is not None:
        os.environ["HAT_LOG_PRETRAINED_BASELINE"] = (
            "1" if log_pretrained_baseline else "0"
        )
    setup_hat_env(
        stage,
        pipeline_test,
        enable_tracking,
    )

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
            "You can refer to examples/classification/resnet18.py."
        )

    if opts is not None:
        config_info.merge_from_list_or_dict(opts, overwrite=opts_overwrite)
    os.environ["HAT_MONITOR_RANK_ZERO"] = str(
        int(config_info.get("log_rank_zero_only", False))
    )

    if device_ids is not None:
        ids = list(map(int, device_ids.split(",")))
    else:
        ids = config_info.device_ids

    assert hasattr(
        config_info, f"{stage}_trainer"
    ), f"There are not {stage}_trainer in config"
    trainer_config = getattr(config_info, f"{stage}_trainer")

    experiment_logger = ExperimentLogger(enable_tracking=enable_tracking)
    experiment_logger.init_group(f"training-group-{stage}")
    experiment_logger.log_config(config_info)

    launch = build_launcher(trainer_config)
    if launch is hat.engine.ddp_trainer.launch and launcher is not None:
        config_info = config_info
    else:
        config_info = config

    launch(
        train_entrance,
        ids,
        dist_url=dist_url,
        dist_launcher=launcher,
        args=(
            config_info,
            opts,
            opts_overwrite,
            stage,
            level,
        ),
    )


if __name__ == "__main__":

    try:
        args, args_env = parse_args()
        train(
            stage=args.stage,
            config=args.config,
            device_ids=args.device_ids,
            dist_url=args.dist_url,
            launcher=args.launcher,
            pipeline_test=args.pipeline_test,
            opts=args.opts,
            opts_overwrite=args.opts_overwrite,
            level=args.level,
            args_env=args_env,
            enable_tracking=args.enable_tracking,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            wandb_tags=args.wandb_tags,
            wandb_resume=args.wandb_resume,
            wandb_run_id=args.wandb_run_id,
            wandb_log_every_steps=args.wandb_log_every_steps,
            fixed_train_index=args.fixed_train_index,
            fixed_val_index=args.fixed_val_index,
            vis_every_steps=args.vis_every_steps,
            vis_every_epochs=args.vis_every_epochs,
            vis_num_samples=args.vis_num_samples,
            vis_strategy=args.vis_strategy,
            vis_seed=args.vis_seed,
            vis_train_indices=args.vis_train_indices,
            vis_val_indices=args.vis_val_indices,
            pretrained_ckpt=args.pretrained_ckpt,
            log_pretrained_baseline=args.log_pretrained_baseline,
            use_tensorboard=args.use_tensorboard,
        )

    except Exception as e:
        ExperimentLogger.log_exception(
            exception=e,
            prefix="train failed",
            enable_tracking=args.enable_tracking,
        )
        raise e
