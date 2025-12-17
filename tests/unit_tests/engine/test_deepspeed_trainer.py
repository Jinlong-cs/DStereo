import copy
import os
import random

import pytest
import torch
import torchvision

from hat.engine.launcher import build_launcher
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.registry import build_from_registry
from hat.utils.distributed import get_dist_info
from hat.utils.seed import seed_everything
from tests.data.toy_modules import *  # noqa: F403,F401
from tests.utils import init_test_logger

tmp_dir = "tmp_ckpt_dir"

config_params = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 10000,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": False,
        },
    },
    "zero_allow_untested_optimizer": True,
}

trainer_cfg = dict(
    type="deepspeed_trainer",
    config_params=config_params,
    model=dict(
        type="ToyModel",
        backbone=dict(
            type="ToyBackbone",
            strides=(1, 2, 4, 8, 16, 32),
            channels=(3, 8, 8, 16, 32, 64),
        ),
        head=dict(
            type="ToyHead",
            in_channels=64,
            fc_filter=128,
            num_classes=1000,
            with_dequant=True,
        ),
        loss=dict(type="ToyLoss"),
    ),
    data_loader=dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="RandDataset",
            length=50,
            example=(torch.randn((3, 224, 224)), random.randint(0, 1000 - 1)),
            clone=True,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    ),
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.001,
        momentum=0.9,
    ),
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=True,
        batch_transforms=None,
        loss_collector=collect_loss_by_regex("^.*loss.*"),
    ),
    num_epochs=2,
    device=None,
    callbacks=[
        dict(
            type="StatsMonitor",
            log_freq=1,
        ),
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            lr_decay_id=[1, 2],
        ),
        dict(type="Checkpoint", save_dir="/tmp/%s" % tmp_dir),
    ],
    sync_bn=True,
    sync_bn_by_host=True,
)


def _dpspd_main(gpu_id, *args):
    init_test_logger(rank=gpu_id)

    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    trainer = build_from_registry(cfg)
    trainer.fit()


@pytest.mark.serial_task
def test_dpspd_trainer_fit():
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)

    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(_dpspd_main, device_ids, dist_url)


def _amp_main(gpu_id, *args):
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    cfg["batch_processor"]["enable_amp"] = True
    cfg["batch_processor"]["enable_amp_dtype"] = args[0]
    cfg["batch_processor"]["enable_channels_last"] = True

    trainer = build_from_registry(cfg)
    trainer.fit()


@pytest.mark.parametrize(
    ["enable_amp_dtype"],
    [
        pytest.param(torch.float16),
        pytest.param(torch.bfloat16),
    ],
)
@pytest.mark.serial_task
def test_amp_fit(enable_amp_dtype):
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)

    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    if enable_amp_dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            return

    launcher(
        _amp_main,
        device_ids,
        dist_url,
        args=(enable_amp_dtype,),
    )


def _amp_custom_main(gpu_id, *args):
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    cfg["batch_processor"]["grad_scaler"] = dict(
        type=torch.cuda.amp.GradScaler,
        growth_interval=230,
    )

    trainer = build_from_registry(cfg)
    trainer.fit()


@pytest.mark.serial_task
def test_amp_custom_fit():
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)

    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(_amp_custom_main, device_ids, dist_url)


def _part_dpspd_main(gpu_id, *args):
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    # cfg["convert_submodule_list"] = "backbone"

    trainer = build_from_registry(cfg)
    trainer.fit()


@pytest.mark.serial_task
def test_part_dpspd_fit():
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)

    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(_part_dpspd_main, device_ids, dist_url)


class ConvertImageDtype(torchvision.transforms.ConvertImageDtype):
    def __init__(self, **kwargs):
        super(ConvertImageDtype, self).__init__(**kwargs)

    def forward(self, x):
        return super().forward(x[0]), x[1]


def _batch_transforms_main(gpu_id, *args):
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    cfg["batch_processor"]["batch_transforms"] = [
        dict(
            type=ConvertImageDtype,
            dtype=torch.float32,
        ),
    ]
    trainer = build_from_registry(cfg)
    trainer.fit()


@pytest.mark.serial_task
def test_batch_transforms_fit():
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)

    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(
        _batch_transforms_main,
        device_ids,
        dist_url,
    )


def _channels_last_main(gpu_id, *args):
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    cfg["batch_processor"]["enable_channels_last"] = True
    cfg["batch_processor"]["channels_last_keys"] = None
    trainer = build_from_registry(cfg)
    trainer.fit()


@pytest.mark.serial_task
def test_channels_last_fit():
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)

    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(_channels_last_main, device_ids, dist_url)


def _resume_main(
    gpu_id,
    tmpdir,
    resume_optimizer,
    resume_step_or_epoch,
    stop_by,
    enable_amp,
    *args,
):

    # ------------ config ------------
    ckpt_dir = os.path.join(tmpdir, stop_by)
    cfg = copy.deepcopy(trainer_cfg)
    cfg["stop_by"] = stop_by
    cfg["device"] = gpu_id
    cfg["callbacks"] = [
        dict(
            type="StatsMonitor",
            log_freq=1,
        ),
        dict(
            type="StepDecayLrUpdater",
            update_by=stop_by,
            warmup_by=stop_by,
            warmup_len=0,
            lr_decay_id=[1, 2],
        ),
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            interval_by=stop_by,
            save_hash=False,
        ),
    ]
    if stop_by == "step":
        cfg.pop("num_epochs")
        cfg["num_steps"] = 2
    cfg["batch_processor"]["enable_amp"] = enable_amp

    # ------------ training ------------
    trainer = build_from_registry(cfg)
    trainer.fit()

    _, world = get_dist_info()
    if world > 1:
        torch.distributed.barrier()

    # ------------ resume config------------
    cfg_resume = copy.deepcopy(cfg)

    cfg_resume["checkpoint_dir"] = os.path.join(ckpt_dir, "ds-last-ckpt")
    if stop_by == "step":
        cfg_resume["num_steps"] = 5
    else:
        cfg_resume["num_epochs"] = 5
    cfg_resume["resume_optimizer"] = resume_optimizer
    cfg_resume["resume_epoch_or_step"] = resume_step_or_epoch

    # ------------ resume training ------------

    resumed_trainer = build_from_registry(cfg_resume)
    opt_checkpoint_state = torch.load(
        os.path.join(
            cfg_resume["checkpoint_dir"],
            "zero_pp_rank_0_mp_rank_00_optim_states.pt",
        )
    )
    opt_checkpoint_state = opt_checkpoint_state["optimizer_state_dict"][
        "base_optimizer_state"
    ]

    resumed_optimizer = resumed_trainer.optimizer.state_dict()[
        "base_optimizer_state"
    ]
    checkpoint_state = resumed_trainer.checkpoint

    assert resumed_optimizer
    if resume_optimizer:
        # ----------- check optimizer -------------
        assert len(resumed_optimizer["state"]) == len(
            opt_checkpoint_state["state"]
        )
        assert len(resumed_optimizer["param_groups"]) == len(
            opt_checkpoint_state["param_groups"]
        )
        assert (
            resumed_optimizer["param_groups"]
            == opt_checkpoint_state["param_groups"]
        )

        # ----------- check grad scaler -------------
        grad_scaler_state_dict = checkpoint_state["grad_scaler"]
        assert (
            resumed_trainer.batch_processor.grad_scaler.state_dict()
            == grad_scaler_state_dict
        )

    # ----------- check epoch and step -------------
    if resume_step_or_epoch:
        assert (
            resumed_trainer.start_epoch == checkpoint_state["epoch"]
            if stop_by == "step"
            else checkpoint_state["epoch"] + 1
        )
        assert (
            resumed_trainer.start_step == checkpoint_state["step"] + 1
            if stop_by == "step"
            else checkpoint_state["step"]
        )

    else:
        assert resumed_trainer.start_epoch == 0
        assert resumed_trainer.start_step == 0


@pytest.mark.parametrize(
    ["resume_optimizer", "resume_step_or_epoch", "stop_by", "enable_amp"],
    [
        pytest.param(True, True, "epoch", True),
        pytest.param(False, False, "epoch", False),
        pytest.param(True, True, "step", True),
        pytest.param(False, False, "step", False),
    ],
)
@pytest.mark.serial_task
def test_resume(
    tmpdir,
    resume_optimizer,
    resume_step_or_epoch,
    stop_by,
    enable_amp,
):
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)
    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(
        _resume_main,
        device_ids,
        dist_url,
        args=(
            tmpdir,
            resume_optimizer,
            resume_step_or_epoch,
            stop_by,
            enable_amp,
        ),
    )
    pass


if __name__ == "__main__":
    pytest.main(["-s", __file__])
