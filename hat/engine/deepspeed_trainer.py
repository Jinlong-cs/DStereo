import logging
import os
import signal
from datetime import timedelta
from typing import Callable, Dict, Iterable, Optional, Sequence

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from hat.callbacks import CallbackMixin
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, to_cuda
from hat.utils.distributed import (
    find_free_port,
    get_local_host,
    get_local_process_group,
)
from hat.utils.saved_tensor import (
    checkpoint_with_saved_tensor,
    support_saved_tensor,
)
from .ddp_trainer import convert_sync_bn
from .launcher import register_launcher
from .processors import BatchProcessorMixin
from .trainer import Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import deepspeed
except ImportError:
    deepspeed = None


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("deepspeed_trainer")
class DeepSpeedTrainer(Trainer):
    """DeepSpeedTrainer tool.

    DeepSpeedTrainer is a tool function to new a `Trainer`
    instance, which support distributed training with deepspeed,
    a deep learning optimization library that makes distributed
    training and inference easy, efficient, and effective


    Args:
        model: Model config or a `nn.Module` instance.
        data_loader: Training data loader config or a instantiated data loader.
        optimizer: Optimizer config or a optimizer instance.
        batch_processor: Batch processor config or a `BatchProcessorMixin`
            instance.
        device: GPU id.
        stop_by: Stop training by counting epoch or step.
            If equal to 'epoch', stop training when
            `epoch_id == num_epochs - 1`.
            If equal to 'step', stop training when
            `global_step_id == num_steps - 1`.
            Default 'epoch'.
        num_epochs: Num of training epochs, should be non-negative integer.
            If stop_by != 'epoch', no-op.
            Set 0 to skip training and run `self.on_loop_begin/end` only.
        start_epoch: Training start epoch, should be non-negative integer.
        num_steps: Num of training steps, should be non-negative integer.
            If stop_by != 'step', no-op.
            Set 0 to skip training and run `self.on_loop_begin/end` only.
        start_step: Training start step, should be non-negative integer.
        callbacks: Callback configs or instances.
        sync_bn: Whether to convert bn to sync_bn.
        sync_bn_by_host: Whether sync bn within host node
        train_metrics: Metrics on training data.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during training and
            assist in identifying bottlenecks.
        find_unused_parameters: Args of DistributedDataParallel module.
        compiler: Converter of `torch.compile`.
        resume_optimizer: whether load optimizer dict when resume checkpoint.
        resume_epoch_or_step: whether need to resume epoch_or_step
            when resume checkpoint.
        distributed_config: Distributed training config.
        config_params: Params fo deepspeed config.
        config_json: Path of deepspeed config file, instead of config_params
            you can config deepspeed in a json file
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        batch_processor: BatchProcessorMixin,
        device: int,
        optimizer: torch.optim.Optimizer,
        stop_by: Optional[str] = "epoch",
        num_epochs: Optional[int] = 1,
        start_epoch: Optional[int] = 0,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = 0,
        callbacks: Optional[Sequence[CallbackMixin]] = None,
        sync_bn: Optional[bool] = False,
        sync_bn_by_host: Optional[bool] = False,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        compiler: Optional[Dict] = None,
        resume_optimizer: Optional[bool] = False,
        resume_epoch_or_step: Optional[bool] = False,
        checkpoint_dir: Optional[str] = None,
        custom_load_fn: Optional[Callable] = None,
        config_params: Dict = None,
        config_json: Dict = None,
        **kwargs,
    ):  # noqa: D205,D400
        super(DeepSpeedTrainer, self).__init__(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            batch_processor=batch_processor,
            device=device,
            stop_by=stop_by,
            num_epochs=num_epochs,
            start_epoch=start_epoch,
            num_steps=num_steps,
            start_step=start_step,
            callbacks=callbacks,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            profiler=profiler,
            compiler=None,
            **kwargs,
        )
        assert deepspeed is not None, "Please install deepspeed"
        assert isinstance(
            self.device, int
        ), "%s, run `DistributedDataParallel` model" " only on one gpu" % type(
            self.device
        )
        current_device = torch.cuda.current_device()
        assert current_device == self.device, "%d vs. %d" % (
            current_device,
            self.device,
        )
        self.sync_bn = sync_bn
        self.sync_bn_by_host = sync_bn_by_host

        self.model.cuda(self.device)
        # move optimizer to cuda
        if isinstance(self.optimizer, torch.optim.Optimizer):
            to_cuda(self.optimizer, self.device, inplace=True)
        assert not isinstance(
            self.model, nn.parallel.DistributedDataParallel
        ), "is already a `DistributedDataParallel` instance"
        if sync_bn:
            self.model = convert_sync_bn(
                self.model,
                process_group=get_local_process_group(),
                local_sync=sync_bn_by_host,
            )
        if support_saved_tensor():
            checkpoint_with_saved_tensor(self.model)

        if batch_processor.enable_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            config_params=config_params,
            model=self.model,
            optimizer=self.optimizer,
            model_parameters=self.model.parameters(),
            config=config_json,
        )
        if config_params.get("activation_checkpointing", False):
            deepspeed.checkpointing.configure(
                mpu_=None,
                deepspeed_config=config_params if config_json is None else config_json,
            )

        if checkpoint_dir is not None:
            ckpt_dir, ckpt_tag = os.path.split(checkpoint_dir)
            _, client_states = self.model.load_checkpoint(
                load_dir=ckpt_dir,
                tag=ckpt_tag,
                custom_load_fn=custom_load_fn,
                load_optimizer_states=resume_optimizer,
            )
            self.checkpoint = client_states
            self._resume_from_checkpoint(resume_epoch_or_step, False, False)

        if compiler:
            self.model = compiler(self.model)


def launch(
    main_func,
    device_ids,
    dist_url="auto",
    dist_launcher=None,
    num_processes=None,
    backend="NCCL",
    args=(),
):
    if device_ids is not None:
        device_ids = _as_list(device_ids)
        num_devices = len(device_ids)
        assert num_devices > 0

        num_processes = num_processes if num_processes else num_devices
        assert num_processes > 0
        if num_processes == num_devices and backend != "NCCL":
            logger.warning(
                "NCCL is the best choice in case of single " "process on single gpu."
            )

        # Note: if device_ids=[1, 3], then after setting
        # `CUDA_VISIBLE_DEVICES`, new device_ids=[0, 1].
        str_ids = list(map(str, device_ids))
        if torch.version.cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
        elif torch.version.hip is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str_ids)
    else:
        num_devices = None
        num_processes = num_processes if num_processes else 1

    if dist_url == "auto":
        port = find_free_port()
        dist_url = "tcp://localhost:%s" % port

    if dist_launcher is not None:
        assert dist_launcher == "torch", "Only support torch as launcher for deepspeed"
        master_port = os.getenv("MASTER_PORT", None)
        if master_port:
            logger.warning(f"DDP master port will be set to {master_port}")
        _main_func(
            -1,
            main_func,
            None,
            backend,
            num_devices,
            -1,
            args,
        )
    else:
        try:
            mp.spawn(
                _main_func,
                nprocs=num_processes,
                args=(
                    main_func,
                    dist_url,
                    backend,
                    num_devices,
                    num_processes,
                    args,
                ),
            )
        # when press Ctrl+c, all sub processes will exits too.
        except KeyboardInterrupt as exception:
            logger.exception(str(exception))
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def _main_func(
    local_rank,
    main_func,
    dist_url,
    backend,
    num_devices,
    num_processes,
    args,
):
    host_name = get_local_host()
    logger.info(
        f"Launch with rank: {os.getenv('RANK') if local_rank==-1 else local_rank} "  # noqa E501
        f"world_size: {os.getenv('WORLD_SIZE', None)} "
        f"hostname: {host_name} "
        f"dist_url: {dist_url} "
        f"num_devices: {num_devices} "
        f"num_processes: {num_processes} "
    )

    if os.getenv("LOCAL_RANK", None) is None:
        if local_rank != -1:
            os.environ["LOCAL_RANK"] = str(local_rank)
        elif os.getenv("RANK") is not None:
            os.environ["LOCAL_RANK"] = os.getenv("RANK")
        else:
            logging.error(
                "Please set LOCAL_RANK environment variable "
                "for deepspeed initialization"
            )

    try:
        deepspeed.init_distributed(
            dist_backend=backend,
            init_method=dist_url,
            world_size=num_processes,
            rank=local_rank,
            auto_mpi_discovery=False,
            timeout=timedelta(
                seconds=int(os.environ.get("HAT_PROCESS_GROUP_TIMEOUT", "1800"))
            ),
        )

    except Exception as e:
        logger.error(f"init process group({local_rank}:{dist_url}) error!" + str(e))
        raise e

    if num_devices is not None:
        local_rank = int(os.environ["LOCAL_RANK"]) if local_rank == -1 else local_rank
        torch.cuda.set_device(local_rank % num_devices)
        main_func(local_rank % num_devices, *args)
    else:
        main_func(None, *args)


register_launcher("DeepSpeedTrainer", launch)
register_launcher("deepspeed_trainer", launch)
