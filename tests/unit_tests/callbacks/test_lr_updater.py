# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.


import copy
import random
from math import cos, pi

import pytest
import torch

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.registry import build_from_registry
from tests.data.toy_dataset import ToyGenDataset
from tests.data.toy_modules import *  # noqa: F403,F401

toy_bone = dict(type="ToyBackbone", strides=(1, 2, 4), channels=(3, 8, 8))
toy_optimizer = dict(
    type=torch.optim.SGD,
    params={"weight": dict(weight_decay=4e-5)},
    lr=0.001,
    momentum=0.9,
)


def _check_lr(opt, target_lr):
    for group in opt.param_groups:
        assert (
            abs(group["lr"] - target_lr) < 1e-9
        ), f"{group['lr']} vs. {target_lr}"


def test_poly_lr_updater_by_step_warmup_by_step():
    num_steps = 8
    warmup_steps = 3
    max_update = num_steps - warmup_steps
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="step",
        power=1.0,
        final_lr=final_lr,
        warmup_by="step",
        warmup_mode="linear",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=1.0,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)

    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(opt, target_lr=init_lr * (1 - (3 - warmup_steps) / max_update))

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(opt, target_lr=init_lr * (1 - (4 - warmup_steps) / max_update))

    # 3. when global_step_id >= num_steps
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=num_steps // loader_len,
        step_id=num_steps % loader_len,
        global_step_id=num_steps,
    )
    _check_lr(opt, target_lr=final_lr)


def test_poly_lr_updater_by_step_warmup_by_step_gen_dataset():
    num_steps = 8
    warmup_steps = 3
    max_update = num_steps - warmup_steps
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="step",
        power=1.0,
        final_lr=final_lr,
        warmup_by="step",
        warmup_mode="linear",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=1.0,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    batch_size = 1
    loader_len = 2
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)
    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(opt, target_lr=init_lr * (1 - (3 - warmup_steps) / max_update))

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(opt, target_lr=init_lr * (1 - (4 - warmup_steps) / max_update))

    # 3. when global_step_id >= num_steps
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=num_steps // loader_len,
        step_id=num_steps % loader_len,
        global_step_id=num_steps,
    )
    _check_lr(opt, target_lr=final_lr)


def test_poly_lr_updater_by_step_warmup_by_epoch():
    warmup_epoch = 2
    max_update = 6
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="step",
        power=1.0,
        final_lr=final_lr,
        warmup_by="epoch",
        warmup_mode="linear",
        warmup_len=warmup_epoch,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=1.0,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)

    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(opt, target_lr=init_lr * 3 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(
        opt, target_lr=init_lr * (1 - (4 - updater.warmup_steps) / max_update)
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_lr(
        opt, target_lr=init_lr * (1 - (5 - updater.warmup_steps) / max_update)
    )

    # 3. when global_step_id >= (
    #    max_update + updater.warmup_steps)
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=10 // loader_len,
        step_id=10 % loader_len,
        global_step_id=10,
    )
    _check_lr(opt, target_lr=final_lr)


def test_poly_lr_updater_by_step_warmup_by_epoch_gen_dataset():
    warmup_epoch = 2
    max_update = 6
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="step",
        power=1.0,
        final_lr=final_lr,
        warmup_by="epoch",
        warmup_mode="linear",
        warmup_len=warmup_epoch,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=1.0,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    batch_size = 1
    loader_len = 2
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)
    with pytest.raises(AssertionError) as e:
        updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
        assert (
            str(e.value)
            == "You can't get the length of data_loader, We recommand you set warmup_by == 'step' and update_by == 'step'"  # noqa: 501
        )


def test_poly_lr_updater_constant_warmup():
    num_steps = 8
    warmup_steps = 3
    max_update = num_steps - warmup_steps
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    warmup_lr_ratio = 0.1
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="step",
        power=1.0,
        final_lr=final_lr,
        warmup_by="step",
        warmup_mode="constant",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=warmup_lr_ratio,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)

    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * warmup_lr_ratio)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * warmup_lr_ratio)


def test_poly_lr_updater_constant_warmup_gen_dataset():
    num_steps = 8
    warmup_steps = 3
    max_update = num_steps - warmup_steps
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    warmup_lr_ratio = 0.1
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="step",
        power=1.0,
        final_lr=final_lr,
        warmup_by="step",
        warmup_mode="constant",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=warmup_lr_ratio,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    batch_size = 1
    loader_len = 2
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * warmup_lr_ratio)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * warmup_lr_ratio)


def test_poly_lr_updater_by_epoch_warmup_by_epoch():
    num_epochs = 8
    warmup_epochs = 3
    max_update = num_epochs - warmup_epochs
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    batch_size = 1
    loader_len = 2
    warmup_steps = warmup_epochs * loader_len
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="epoch",
        power=1.0,
        final_lr=final_lr,
        warmup_by="epoch",
        warmup_mode="linear",
        warmup_len=warmup_epochs,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=1.0,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(
        range(loader_len * batch_size), batch_size=batch_size
    )

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / warmup_steps)

    # 2. formal training stage
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=4, global_step_id=4 * loader_len
    )
    _check_lr(opt, target_lr=init_lr * (1 - (4 - warmup_epochs) / max_update))

    updater.on_epoch_begin(
        optimizer=opt, epoch_id=5, global_step_id=5 * loader_len
    )
    _check_lr(opt, target_lr=init_lr * (1 - (5 - warmup_epochs) / max_update))

    # 3. when epoch_id >= num_epochs
    updater.on_epoch_begin(
        optimizer=opt,
        epoch_id=num_epochs,
        global_step_id=num_epochs * loader_len,
    )
    _check_lr(opt, target_lr=final_lr)


def test_poly_lr_updater_by_epoch_warmup_by_epoch_gen_dataset():
    num_epochs = 8
    warmup_epochs = 3
    max_update = num_epochs - warmup_epochs
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    batch_size = 1
    loader_len = 2
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="epoch",
        power=1.0,
        final_lr=final_lr,
        warmup_by="epoch",
        warmup_mode="linear",
        warmup_len=warmup_epochs,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=1.0,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    with pytest.raises(AssertionError) as e:
        updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
        assert (
            str(e.value)
            == "You can't get the length of data_loader, We recommand you set warmup_by == 'step' and update_by == 'step'"  # noqa: 501
        )


def test_poly_lr_updater_by_epoch_warmup_by_step():
    warmup_steps = 3
    max_update = 4
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    batch_size = 1
    loader_len = 2
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="epoch",
        power=1.0,
        final_lr=final_lr,
        warmup_by="step",
        warmup_mode="linear",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=1.0,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(
        range(loader_len * batch_size), batch_size=batch_size
    )

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / warmup_steps)

    # 2. formal training stage
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=2, global_step_id=2 * loader_len
    )
    _check_lr(
        opt, target_lr=init_lr * (1 - (2 - updater.warmup_epochs) / max_update)
    )

    updater.on_epoch_begin(
        optimizer=opt, epoch_id=3, global_step_id=3 * loader_len
    )
    _check_lr(
        opt, target_lr=init_lr * (1 - (3 - updater.warmup_epochs) / max_update)
    )

    # 3. when epoch_id - warmup_epochs >= max_update
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=6, global_step_id=6 * loader_len
    )
    _check_lr(opt, target_lr=final_lr)


def test_poly_lr_updater_by_epoch_warmup_by_step_gen_dataset():
    warmup_steps = 3
    max_update = 4
    final_lr = 0.0
    toy_opt = copy.deepcopy(toy_optimizer)
    batch_size = 1
    loader_len = 2
    updater = dict(
        type="PolyLrUpdater",
        max_update=max_update,
        update_by="epoch",
        power=1.0,
        final_lr=final_lr,
        warmup_by="step",
        warmup_mode="linear",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.0,
        warmup_lr_ratio=1.0,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    with pytest.raises(AssertionError) as e:
        updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
        assert (
            str(e.value)
            == "You can't get the length of data_loader, We recommand you set warmup_by == 'step' and update_by == 'step'"  # noqa: 501
        )


def test_poly_lr_updater_in_trainer():
    num_steps = 4
    warmup_steps = 2
    max_update = num_steps - warmup_steps
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
                length=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
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
            loss_collector=collect_loss_by_regex("^.*loss.*"),
        ),
        device=None,
        stop_by="step",
        num_steps=num_steps,
        callbacks=[
            dict(
                type="PolyLrUpdater",
                max_update=max_update,
                update_by="step",
                power=1.0,
                final_lr=0.0,
                warmup_by="step",
                warmup_mode="linear",
                warmup_len=warmup_steps,
                warmup_begin_lr=0.0,
                warmup_lr_ratio=1.0,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


def test_poly_lr_updater_in_trainer_gen_dataset():
    num_steps = 4
    warmup_steps = 2
    max_update = num_steps - warmup_steps
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
                type="ToyGenDataset",
                batch_size=1,
                loader_len=2,
                example=(
                    torch.randn((1, 3, 14, 14)),
                    torch.randint(0, 10 - 1, (1,)),
                ),
                clone=True,
            ),
            batch_size=None,
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
        stop_by="step",
        num_steps=num_steps,
        callbacks=[
            dict(
                type="PolyLrUpdater",
                max_update=max_update,
                update_by="step",
                power=1.0,
                final_lr=0.0,
                warmup_by="step",
                warmup_mode="linear",
                warmup_len=warmup_steps,
                warmup_begin_lr=0.0,
                warmup_lr_ratio=1.0,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


def test_stepdecay_lr_updater_warmup_by_epoch():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="StepDecayLrUpdater",
        update_by="epoch",
        warmup_by="epoch",
        warmup_len=1,
        lr_decay_id=[2, 4],
        lr_decay_factor=0.1,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=1, global_step_id=1 * loader_len
    )
    _check_lr(opt, target_lr=init_lr)

    # first decay
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=2, global_step_id=2 * loader_len
    )
    _check_lr(opt, target_lr=init_lr * 0.1)

    updater.on_epoch_begin(
        optimizer=opt, epoch_id=3, global_step_id=3 * loader_len
    )
    _check_lr(opt, target_lr=init_lr * 0.1)

    # second decay
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=4, global_step_id=4 * loader_len
    )
    _check_lr(opt, target_lr=init_lr * 0.1 * 0.1)


def test_stepdecay_lr_updater_warmup_by_epoch_gen_dataset():
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="StepDecayLrUpdater",
        update_by="epoch",
        warmup_by="epoch",
        warmup_len=1,
        lr_decay_id=[2, 4],
        lr_decay_factor=0.1,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    batch_size = 1
    loader_len = 2
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    with pytest.raises(AssertionError) as e:
        updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
        assert (
            str(e.value)
            == "You can't get the length of data_loader, We recommand you set warmup_by == 'step' and update_by == 'step'"  # noqa: 501
        )


def test_stepdecay_lr_updater_by_epoch_warmup_by_step():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="StepDecayLrUpdater",
        update_by="epoch",
        warmup_by="step",
        warmup_len=3,
        lr_decay_id=[2, 4],
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(opt, target_lr=init_lr * 2 / updater.warmup_steps)

    # 2. formal training stage
    # first decay
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=2, global_step_id=2 * loader_len
    )
    _check_lr(opt, target_lr=init_lr * 0.1)

    updater.on_epoch_begin(
        optimizer=opt, epoch_id=3, global_step_id=3 * loader_len
    )
    _check_lr(opt, target_lr=init_lr * 0.1)

    # second decay
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=4, global_step_id=4 * loader_len
    )
    _check_lr(opt, target_lr=init_lr * 0.1 * 0.1)


def test_stepdecay_lr_updater_by_epoch_warmup_by_step_gen_dataset():
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="StepDecayLrUpdater",
        update_by="epoch",
        warmup_by="step",
        warmup_len=3,
        lr_decay_id=[2, 4],
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    batch_size = 1
    loader_len = 2
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)
    with pytest.raises(AssertionError) as e:
        updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
        assert (
            str(e.value)
            == "You can't get the length of data_loader, We recommand you set warmup_by == 'step' and update_by == 'step'"  # noqa: 501
        )


def test_stepdecay_lr_updater_by_step():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="StepDecayLrUpdater",
        update_by="step",
        warmup_by="step",
        warmup_len=3,
        lr_decay_id=[4, 6],
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(opt, target_lr=init_lr)

    # first decay
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(opt, target_lr=init_lr * 0.1)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_lr(opt, target_lr=init_lr * 0.1)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=6 // loader_len,
        step_id=6 % loader_len,
        global_step_id=6,
    )
    _check_lr(opt, target_lr=init_lr * 0.1 * 0.1)


def test_stepdecay_lr_updater_by_step_gen_dataset():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="StepDecayLrUpdater",
        update_by="step",
        warmup_by="step",
        warmup_len=3,
        lr_decay_id=[4, 6],
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    batch_size = 1
    loader_len = 2
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)
    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(opt, target_lr=init_lr)

    # first decay
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(opt, target_lr=init_lr * 0.1)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_lr(opt, target_lr=init_lr * 0.1)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=6 // loader_len,
        step_id=6 % loader_len,
        global_step_id=6,
    )
    _check_lr(opt, target_lr=init_lr * 0.1 * 0.1)


def test_stepdecay_lr_updater_in_trainer():
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
                length=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
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
            loss_collector=collect_loss_by_regex("^.*loss.*"),
        ),
        device=None,
        stop_by="step",
        num_steps=8,
        callbacks=[
            dict(
                type="StepDecayLrUpdater",
                warmup_by="epoch",
                warmup_len=1,
                lr_decay_id=[1, 2],
                lr_decay_factor=0.1,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


def test_stepdecay_lr_updater_in_trainer_gen_dataset():
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
                type="ToyGenDataset",
                batch_size=1,
                loader_len=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
            batch_size=None,
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
        stop_by="step",
        num_steps=8,
        callbacks=[
            dict(
                type="StepDecayLrUpdater",
                warmup_by="epoch",
                warmup_len=1,
                lr_decay_id=[1, 2],
                lr_decay_factor=0.1,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    with pytest.raises(AssertionError) as e:
        trainer.fit()
        assert (
            str(e.value)
            == "You can't get the length of data_loader, We recommand you set warmup_by == 'step' and update_by == 'step'"  # noqa: 501
        )


def test_cos_lr_updater_warmup_by_step():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="CosLrUpdater",
        warmup_by="step",
        warmup_len=3,
        stop_lr=0.001,
    )
    stop_lr = updater["stop_lr"]
    num_epochs = 4
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(
        optimizer=opt, data_loader=data_loader, num_epochs=num_epochs
    )
    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(opt, target_lr=init_lr * 2 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(
        opt,
        target_lr=(init_lr - stop_lr)
        * (1 + cos(pi * (3 - updater.warmup_steps) / 5))
        / 2
        + stop_lr,
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(
        opt,
        target_lr=(init_lr - stop_lr)
        * (1 + cos(pi * (4 - updater.warmup_steps) / 5))
        / 2
        + stop_lr,
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_lr(
        opt,
        target_lr=(init_lr - stop_lr)
        * (1 + cos(pi * (5 - updater.warmup_steps) / 5))
        / 2
        + stop_lr,
    )


def test_cos_lr_updater_warmup_by_step_gen_dataset():
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="CosLrUpdater",
        warmup_by="step",
        warmup_len=3,
        stop_lr=0.001,
    )
    num_epochs = 4
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    batch_size = 1
    loader_len = 2
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)
    with pytest.raises(AttributeError) as e:
        updater.on_loop_begin(
            optimizer=opt, data_loader=data_loader, num_epochs=num_epochs
        )
        assert (
            str(e.value)
            == "'CosLrUpdater' object has no attribute 'step_per_epoch'"
        )


def test_cos_lr_updater_warmup_by_epoch():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="CosLrUpdater",
        warmup_by="epoch",
        warmup_len=1,
    )
    num_epochs = 4
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(
        optimizer=opt, data_loader=data_loader, num_epochs=num_epochs
    )
    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=init_lr * 0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(opt, target_lr=init_lr * 1 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(
        opt,
        target_lr=init_lr
        * (1 + cos(pi * (2 - updater.warmup_steps) / updater.max_steps))
        / 2,
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(
        opt,
        target_lr=init_lr
        * (1 + cos(pi * (3 - updater.warmup_steps) / updater.max_steps))
        / 2,
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(
        opt,
        target_lr=init_lr
        * (1 + cos(pi * (4 - updater.warmup_steps) / updater.max_steps))
        / 2,
    )


def test_cos_lr_updater_warmup_by_epoch_gen_dataset():
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="CosLrUpdater",
        warmup_by="epoch",
        warmup_len=1,
    )
    num_epochs = 4
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    batch_size = 1
    loader_len = 2
    dataset = ToyGenDataset(batch_size, loader_len, 1, clone=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    with pytest.raises(AssertionError) as e:
        updater.on_loop_begin(
            optimizer=opt, data_loader=data_loader, num_epochs=num_epochs
        )
        assert (
            str(e.value)
            == "You can't get the length of data_loader, We recommand you set warmup_by == 'step' and update_by == 'step'"  # noqa: 501
        )


def test_cos_lr_updater_warmup_in_train():
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
                length=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
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
            loss_collector=collect_loss_by_regex("^.*loss.*"),
        ),
        device=None,
        stop_by="step",
        num_epochs=4,
        num_steps=8,
        callbacks=[
            dict(
                type="CosLrUpdater",
                warmup_by="epoch",
                warmup_len=1,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


def test_cos_lr_updater_warmup_in_train_gen_dataset():
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
                type="ToyGenDataset",
                batch_size=1,
                loader_len=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
            batch_size=None,
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
        stop_by="step",
        num_epochs=4,
        num_steps=8,
        callbacks=[
            dict(
                type="CosLrUpdater",
                warmup_by="epoch",
                warmup_len=1,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    with pytest.raises(AssertionError) as e:
        trainer.fit()
        assert (
            str(e.value)
            == "You can't get the length of data_loader, We recommand you set warmup_by == 'step' and update_by == 'step'"  # noqa: 501
        )


def test_noam_lr_updater_by_step():
    # 准备 opt 和 model
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    # 准备 updater
    updater_config = dict(
        type="NoamLrUpdater",
        d_model=256,
        warmup_step=4000,
        update_by="step",
    )
    updater = build_from_registry(updater_config)
    # 测试
    test_global_step_id = [
        0,
        1,
        2000,
        2001,
        3998,
        3999,
        4000,
        4001,
        5000,
        7000,
        8000,
    ]
    test_loader_len = [2000, 3000, 5000]
    for loader_len in test_loader_len:
        batch_size = 4
        num_epochs = 8
        data_loader = torch.utils.data.DataLoader(
            range(loader_len * batch_size), batch_size=batch_size
        )
        updater.on_loop_begin(
            optimizer=opt, data_loader=data_loader, num_epochs=num_epochs
        )

        for global_step_id in test_global_step_id:
            updater.on_step_begin(
                optimizer=opt,
                epoch_id=global_step_id // loader_len,
                step_id=global_step_id % loader_len,
                global_step_id=global_step_id,
            )
            step_num = global_step_id + 1
            target_lr = (
                init_lr
                * (256 ** 0.5)
                * min(step_num ** -0.5, step_num * 4000 ** -1.5)
            )
            _check_lr(opt, target_lr=target_lr)


def test_noam_lr_updater_by_epoch():
    # 准备 opt 和 model
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    # 准备 updater
    updater_config = dict(
        type="NoamLrUpdater",
        d_model=256,
        warmup_step=10,
        update_by="epoch",
    )
    updater = build_from_registry(updater_config)
    # 测试
    test_global_step_id = [
        0,
        1,
        98,
        99,
        100,
        101,
        500,
        998,
        999,
        1000,
        1001,
        1100,
        1400,
        2000,
        5000,
        7000,
        8000,
    ]
    # on_loop_begin
    loader_len = 100
    batch_size = 4
    num_epochs = 100
    data_loader = torch.utils.data.DataLoader(
        range(loader_len * batch_size), batch_size=batch_size
    )
    updater.on_loop_begin(
        optimizer=opt, data_loader=data_loader, num_epochs=num_epochs
    )
    # 测试
    for global_step_id in test_global_step_id:
        updater.on_epoch_begin(
            optimizer=opt,
            epoch_id=global_step_id // loader_len,
            global_step_id=global_step_id,
        )
        step_num = (global_step_id // loader_len) + 1
        target_lr = (
            init_lr
            * (256 ** 0.5)
            * min(step_num ** -0.5, step_num * 10 ** -1.5)
        )
        _check_lr(opt, target_lr=target_lr)


def test_noam_lr_updater_in_train():
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
                length=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
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
            loss_collector=collect_loss_by_regex("^.*loss.*"),
        ),
        device=None,
        stop_by="step",
        num_epochs=15,
        num_steps=30,
        callbacks=[
            dict(
                type="NoamLrUpdater",
                d_model=256,
                warmup_step=20,
                update_by="step",
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


def test_noam_lr_updater_in_train_gen_dataset():
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
                type="ToyGenDataset",
                batch_size=1,
                loader_len=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
            batch_size=1,
            shuffle=False,
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
        stop_by="step",
        num_epochs=15,
        num_steps=30,
        callbacks=[
            dict(
                type="NoamLrUpdater",
                d_model=256,
                warmup_step=20,
                update_by="step",
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


def test_onecycle_lr_updater_by_step():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="OneCycleUpdater",
        lr_max=init_lr,
        div_factor=10.0,
        stop_lr=init_lr / (10.0 * 1000),
        pct_start=0.4,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader, num_epochs=3)
    loader_len = len(data_loader)

    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_lr(opt, target_lr=updater.warmup_begin_lr)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_lr(
        opt,
        target_lr=updater.get_warmup_lr(warmup_end_lr=init_lr, num_update=1),
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_lr(
        opt,
        target_lr=updater.get_warmup_lr(warmup_end_lr=init_lr, num_update=2),
    )

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(opt, target_lr=updater.get_lr(begin_lr=init_lr, num_update=3))

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(opt, target_lr=updater.get_lr(begin_lr=init_lr, num_update=4))

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_lr(opt, target_lr=updater.get_lr(begin_lr=init_lr, num_update=5))


def test_onecycle_lr_updater_in_trainer():
    num_epochs = 4
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
                length=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
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
            loss_collector=collect_loss_by_regex("^.*loss.*"),
        ),
        device=None,
        stop_by="epoch",
        num_epochs=num_epochs,
        callbacks=[
            dict(
                type="OneCycleUpdater",
                lr_max=0.001,
                div_factor=10.0,
                stop_lr=0.001 / (10.0 * 1000),
                pct_start=0.4,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


def test_cycle_lr_updater_by_step():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="CyclicLrUpdater",
        target_ratio=(10, 1e-4),
        cyclic_times=1,
        step_ratio_up=0.4,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader, num_epochs=3)
    loader_len = len(data_loader)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_lr(opt, target_lr=updater.get_lr(begin_lr=init_lr, num_update=3))

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_lr(opt, target_lr=updater.get_lr(begin_lr=init_lr, num_update=4))

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_lr(opt, target_lr=updater.get_lr(begin_lr=init_lr, num_update=5))


def test_cycle_lr_updater_in_trainer():
    num_epochs = 4
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
                length=2,
                example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
                clone=True,
            ),
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
            loss_collector=collect_loss_by_regex("^.*loss.*"),
        ),
        device=None,
        stop_by="epoch",
        num_epochs=num_epochs,
        callbacks=[
            dict(
                type="CyclicLrUpdater",
                target_ratio=(10, 1e-4),
                cyclic_times=1,
                step_ratio_up=0.4,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


def test_cosineAnnealing_lr_updater_by_step():
    toy_opt = copy.deepcopy(toy_optimizer)
    init_lr = toy_opt["lr"]
    updater = dict(
        type="CosineAnnealingLrUpdater",
        warmup_len=1,
        warmup_by="step",
        warmup_lr_ratio=1.0 / 3,
        stop_lr=1e-6,
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader, num_epochs=6)
    loader_len = len(data_loader)

    updater.on_epoch_begin(
        optimizer=opt, epoch_id=2, global_step_id=2 * loader_len
    )
    _check_lr(opt, target_lr=updater.get_lr(begin_lr=init_lr, num_update=2))

    updater.on_epoch_begin(
        optimizer=opt, epoch_id=4, global_step_id=4 * loader_len
    )

    _check_lr(opt, target_lr=updater.get_lr(begin_lr=init_lr, num_update=4))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
