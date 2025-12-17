import copy
import os
import random
import time

import pytest
import torch

from hat.engine.launcher import build_launcher
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.registry import build_from_registry
from hat.utils.seed import seed_everything
from tests.data.toy_modules import *  # noqa: F403,F401

trainer_cfg = dict(
    type="data_parallel_trainer",
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
        batch_size=2,
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
)
trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]


def _dp_main(device_ids, *args):
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = device_ids
    cfg["data_loader"]["batch_size"] = len(device_ids) * 1

    trainer = build_from_registry(cfg)
    trainer.fit()


def test_dp_trainer_fit():
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)
    device_ids = list(range(torch.cuda.device_count()))
    launcher(_dp_main, device_ids)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
