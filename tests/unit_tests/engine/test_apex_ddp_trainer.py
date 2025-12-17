import copy
import os
import random
import time

import pytest
import torch
import torch.distributed as dist

from hat.engine.launcher import build_launcher
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.registry import build_from_registry
from hat.utils.seed import seed_everything
from tests.data.toy_modules import *  # noqa: F403,F401
from tests.data.toy_multitask.multitask import float_trainer
from tests.utils import init_test_logger

trainer_cfg = dict(
    type="apex_distributed_data_parallel_trainer",
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
        type="MultiBatchProcessor",
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
        dict(type="Checkpoint", save_dir="/tmp/%s" % time.time()),
    ],
    sync_bn=True,
)
trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]


def _apex_ddp_main(gpu_id, *args):
    init_test_logger(rank=gpu_id)

    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    trainer = build_from_registry(cfg)
    trainer.fit()


def _apex_ddp_delay_sync(gpu_id, *args):
    init_test_logger(rank=gpu_id)

    cfg = copy.deepcopy(float_trainer)
    cfg["type"] = "apex_distributed_data_parallel_trainer"
    cfg["device"] = gpu_id
    cfg["batch_processor"]["delay_sync"] = True

    cfg["model"] = build_from_registry(cfg["model"])
    cfg["optimizer"]["model"] = cfg["model"]
    cfg["callbacks"] = trainer_cfg["callbacks"]
    trainer = build_from_registry(cfg)
    trainer.fit()

    # check all modules have same grad because of global sync.
    model = trainer.model.module
    world_size = dist.get_world_size()
    assert world_size == torch.cuda.device_count()
    for params in model.parameters():
        if params.requires_grad:
            grads = params.grad
            all_grads = [torch.zeros_like(grads) for _ in range(world_size)]
            dist.all_gather(all_grads, grads)
            for i in range(world_size):
                if i < world_size - 1:
                    assert all_grads[i].equal(all_grads[i + 1])


try:
    import apex
except ImportError:
    apex = None


@pytest.mark.skipif(apex is None, reason="requiring apex")
@pytest.mark.serial_task
def test_apex_ddp_trainer_fit():
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)

    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(_apex_ddp_main, device_ids, dist_url)


@pytest.mark.skipif(apex is None, reason="requiring apex")
@pytest.mark.serial_task
def test_apex_ddp_trainer_delay_sync():
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)
    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(_apex_ddp_delay_sync, device_ids, dist_url)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
