import copy
import random
import time

import pytest
import torch

from hat.callbacks import CallbackMixin
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.registry import build_from_registry
from tests.data.toy_modules import *  # noqa: F403,F401

trainer_cfg = dict(
    type="Trainer",
    model=dict(
        type="ToyModel",
        backbone=dict(type="ToyBackbone", strides=(1, 2), channels=(3, 8)),
        head=dict(
            type="ToyHead",
            in_channels=8,
            fc_filter=16,
            num_classes=10,
            with_dequant=True,
        ),
        loss=dict(type="ToyLoss"),
    ),
    data_loader=dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="RandDataset",
            length=4,
            example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
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
    device=None,
    stop_by="epoch",
    num_epochs=2,
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
            step_log_interval=10,
        ),
        dict(type="Checkpoint", save_dir="/tmp/%s" % time.time()),
    ],
)
trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]


class TestStateCallback(CallbackMixin):
    def __init__(self):
        self.last_epoch_id = None
        self.last_step_id = None
        self.last_global_step_id = None

    def on_epoch_end(self, epoch_id, **kwargs):
        self.last_epoch_id = epoch_id

    def on_step_end(self, step_id, global_step_id, **kwargs):
        self.last_step_id = step_id
        self.last_global_step_id = global_step_id


def test_trainer_fit_on_gpu():
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = 0
    trainer = build_from_registry(cfg)
    trainer.fit()


def test_trainer_fit_on_cpu():
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = None
    trainer = build_from_registry(cfg)
    trainer.fit()


def test_stop_by_epoch():
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = 0
    cfg["stop_by"] = "epoch"
    cfg["num_epochs"] = 2
    cfg["num_steps"] = 10  # expect to be useless even though is set

    cfg["callbacks"] = []
    test_cb = TestStateCallback()
    cfg["callbacks"].append(test_cb)

    trainer = build_from_registry(cfg)
    trainer.fit()

    loader_len = len(trainer.data_loader)
    assert test_cb.last_epoch_id == cfg["num_epochs"] - 1
    assert test_cb.last_step_id == loader_len - 1
    assert test_cb.last_global_step_id == cfg["num_epochs"] * loader_len - 1


def test_stop_by_step():
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = 0
    cfg["stop_by"] = "step"
    cfg["num_steps"] = 3
    cfg["num_epochs"] = 10  # expect to be useless even though is set

    cfg["callbacks"] = []
    test_cb = TestStateCallback()
    cfg["callbacks"].append(test_cb)

    trainer = build_from_registry(cfg)
    trainer.fit()

    loader_len = len(trainer.data_loader)
    assert test_cb.last_epoch_id == cfg["num_steps"] // loader_len
    assert test_cb.last_step_id == (cfg["num_steps"] - 1) % loader_len
    assert test_cb.last_global_step_id == cfg["num_steps"] - 1


if __name__ == "__main__":
    pytest.main(["-s", __file__])
