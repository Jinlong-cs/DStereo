"""
ISP inference tool.
"""
import inspect
import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional

import horizon_plugin_pytorch as horizon
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from hat.engine.launcher import build_launcher
from hat.registry import RegistryContext
from hat.utils import Config
from hat.utils.distributed import get_dist_info
from hat.utils.logger import LOG_DIR, DisableLogger, init_rank_logger
from hat.utils.seed import seed_everything

# isort: off
# TODO(min.du): move trainer_wrapper and utilities to hat #
from tools.trainer_wrapper import TrainerWrapper, TRAINING_STEPS

# isort: on


def seed_training(seed: int):  # noqa: D205,D400
    """Set seed for pseudo-random number generators in:
    pytorch, numpy, python.random, set cudnn state as well.

    Args:
        seed: the integer value seed for global random state.
    """
    seed_everything(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


class InferBase(ABC):
    """
    Base class for Inference.
    """

    def __init__(self):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class Inference(InferBase):
    r"""
    Simple ISP inference engine.

    Args:
        config: Input training config file path.
        step: Training step, only support {TRAINING_STEPS}.
        gpu_ids: Index of gpus to use.
        val_only: Only to validate.
        val_ckpt: checkpoint for validation.

    """

    def __init__(
        self,
        config: str,
        step: str,
        gpu_ids: Optional[int] = None,
        val_only: bool = False,
        val_ckpt: str = None,
    ):

        super(Inference, self).__init__()

        # create config and logger
        self.cfg = Config.fromfile(config)
        self.step2solver = self.cfg.step2solver
        self.logger = init_rank_logger(
            0, save_dir=LOG_DIR, cfg_file=config, step=step, prefix="infer-"
        )

        # disable logging output for non-zero rank
        rank, _ = get_dist_info()
        disable_logger = rank != 0 and self.cfg.get(
            "log_rank_zero_only", False
        )

        self.logger.info(f"GPU: {gpu_ids}")

        # 3. seed training
        cudnn.benchmark = self.cfg.cudnn_benchmark
        if self.cfg.seed is not None:
            seed_training(self.cfg.seed)

        self.step = step

        horizon.march.set_march(
            self.cfg.get("march", horizon.march.March.BAYES)
        )
        qat_mode = self.cfg.get("qat_mode", "fuse_bn")
        horizon.qat_mode.set_qat_mode(qat_mode)

        # 4. build trainer
        with DisableLogger(disable_logger, logging.WARNING), RegistryContext():
            self.trainer = TrainerWrapper(
                cfg=self.cfg,
                train_step=step,
                logger=self.logger,
                device=gpu_ids,
            )
            self.trainer.prepare_fit(
                export_ckpt_only=False,
                val_only=val_only,
                val_ckpt=val_ckpt,
            )

    @staticmethod
    def run_in_ddp() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--config",
            "-c",
            type=str,
            required=True,
            help="train config file path",
        )
        parser.add_argument(
            "--step",
            type=str,
            help="training step in: %s" % TRAINING_STEPS,
            choices=TRAINING_STEPS,
        )
        parser.add_argument(
            "--gpu-ids",
            type=str,
            default="",
            help="ids of gpus to use, separated by `,`",
        )
        parser.add_argument(
            "--val_only",
            action="store_true",
            help="Whether skip training and do validation only.",
        )
        parser.add_argument(
            "--val_ckpt",
            type=str,
            default="",
            help="validation checkpoint file path",
        )
        return parser

    @staticmethod
    def add_ddp_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
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
        return parser

    def run(self):
        """
        Main loop of inference.
        """
        self.trainer.fit()


def main_inner(local_rank, args):
    parameters = inspect.signature(Inference.__init__).parameters.keys()
    kwargs = {k: getattr(args, k) for k in dir(args) if k in parameters}

    if Inference.run_in_ddp():
        kwargs["gpu_ids"] = local_rank

    engine = Inference(**kwargs)
    engine.run()


def main(args):
    if isinstance(args.gpu_ids, str):
        if "," in args.gpu_ids:
            ids = list(map(int, args.gpu_ids.split(",")))
        else:
            ids = [int(args.gpu_ids)]
    else:
        ids = args.gpu_ids

    launch = build_launcher(config.step2solver[args.step]["trainer"])
    launch(
        main_inner,
        ids,
        dist_url=args.dist_url,
        dist_launcher=args.launcher,
        args=(args,),
    )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser = Inference.add_argparse_args(parser)
    parser = Inference.add_ddp_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)
    if not args.gpu_ids:
        logging.info("load gpu_ids from --config")
        args.gpu_ids = config.device_ids

    main(args)
