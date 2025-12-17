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
toy_optimizer_sgd = dict(
    type=torch.optim.SGD,
    params={"weight": dict(weight_decay=4e-5)},
    # weight_decay=4e-5,
    lr=0.001,
    momentum=0.9,
)

toy_optimizer_adamw = dict(
    type=torch.optim.AdamW,
    params={"weight": dict(weight_decay=4e-5)},
    betas=(0.95, 0.99),
    # weight_decay=4e-5,
    lr=0.001,
)


params = [
    pytest.param(toy_optimizer_sgd, "weight_decay", 4e-5),
    pytest.param(toy_optimizer_adamw, "betas", 0.95),
]


def _check_param(param_name, opt, target_factor=None, target_func=None):
    for group in opt.param_groups:
        init_param = group[f"initial_{param_name}"]
        if target_factor is not None:
            target_value = group[f"initial_{param_name}"] * target_factor
        elif target_func is not None:
            target_value = target_func(group[f"initial_{param_name}"])
        else:
            raise AssertionError("target func or factor should be provided")
        optim_param_value = group[param_name]
        if isinstance(optim_param_value, (list, tuple)):
            optim_param_value = optim_param_value[0]

        assert (
            abs(optim_param_value - target_value) < 1e-9
        ), f"init {init_param}, {optim_param_value} vs. {target_value}"


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_stepdecay_param_updater_warmup_by_epoch(
    toy_optimizer,
    param_name,
    init_param,
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="StepDecayOptimParamUpdater",
        param_name=param_name,
        update_by="epoch",
        warmup_by="epoch",
        warmup_len=1,
        param_decay_id=[2, 4],
        param_decay_factor=0.1,
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
    _check_param(param_name, opt, target_factor=0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_param(param_name, opt, target_factor=1 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=1, global_step_id=1 * loader_len
    )
    _check_param(param_name, opt, target_factor=1.0)

    # first decay
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=2, global_step_id=2 * loader_len
    )
    _check_param(param_name, opt, target_factor=0.1)

    updater.on_epoch_begin(
        optimizer=opt, epoch_id=3, global_step_id=3 * loader_len
    )
    _check_param(param_name, opt, target_factor=0.1)

    # second decay
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=4, global_step_id=4 * loader_len
    )
    _check_param(param_name, opt, target_factor=0.1 * 0.1)


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_stepdecay_param_updater_warmup_by_epoch_gen_dataset(
    toy_optimizer,
    param_name,
    init_param,
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="StepDecayOptimParamUpdater",
        param_name=param_name,
        update_by="epoch",
        warmup_by="epoch",
        warmup_len=1,
        param_decay_id=[2, 4],
        param_decay_factor=0.1,
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


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_stepdecay_param_updater_by_epoch_warmup_by_step(
    toy_optimizer,
    param_name,
    init_param,
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="StepDecayOptimParamUpdater",
        param_name=param_name,
        update_by="epoch",
        warmup_by="step",
        warmup_len=3,
        param_decay_id=[2, 4],
    )
    updater = build_from_registry(updater)
    model = build_from_registry(toy_bone)
    toy_opt["model"] = model
    opt = build_from_registry(toy_opt)
    data_loader = torch.utils.data.DataLoader(range(2), batch_size=1)

    updater.on_loop_begin(optimizer=opt, data_loader=data_loader)
    loader_len = len(data_loader)
    print("updater init param", updater._per_group_init_param)
    # 1. warmup training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=0 // loader_len,
        step_id=0 % loader_len,
        global_step_id=0,
    )
    _check_param(param_name, opt, target_factor=0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_param(param_name, opt, target_factor=1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_param(param_name, opt, target_factor=2 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_param(param_name, opt, target_factor=2 / updater.warmup_steps)

    # 2. formal training stage
    # first decay
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=2, global_step_id=2 * loader_len
    )
    _check_param(param_name, opt, target_factor=0.1)

    updater.on_epoch_begin(
        optimizer=opt, epoch_id=3, global_step_id=3 * loader_len
    )
    _check_param(param_name, opt, target_factor=0.1)

    # second decay
    updater.on_epoch_begin(
        optimizer=opt, epoch_id=4, global_step_id=4 * loader_len
    )
    _check_param(param_name, opt, target_factor=0.1 * 0.1)


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_stepdecay_param_updater_by_epoch_warmup_by_step_gen_dataset(
    toy_optimizer,
    param_name,
    init_param,
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="StepDecayOptimParamUpdater",
        param_name=param_name,
        update_by="epoch",
        warmup_by="step",
        warmup_len=3,
        param_decay_id=[2, 4],
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


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_stepdecay_param_updater_by_step(
    toy_optimizer,
    param_name,
    init_param,
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="StepDecayOptimParamUpdater",
        param_name=param_name,
        update_by="step",
        warmup_by="step",
        warmup_len=3,
        param_decay_id=[4, 6],
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
    _check_param(param_name, opt, target_factor=0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_param(param_name, opt, target_factor=1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_param(param_name, opt, target_factor=2 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_param(param_name, opt, target_factor=1.0)

    # first decay
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_param(param_name, opt, target_factor=0.1)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_param(param_name, opt, target_factor=0.1)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=6 // loader_len,
        step_id=6 % loader_len,
        global_step_id=6,
    )
    _check_param(param_name, opt, target_factor=0.1 * 0.1)


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_stepdecay_param_updater_by_step_gen_dataset(
    toy_optimizer,
    param_name,
    init_param,
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="StepDecayOptimParamUpdater",
        param_name=param_name,
        update_by="step",
        warmup_by="step",
        warmup_len=3,
        param_decay_id=[4, 6],
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
    _check_param(param_name, opt, target_factor=0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_param(param_name, opt, target_factor=1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_param(param_name, opt, target_factor=2 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    _check_param(param_name, opt, target_factor=1)

    # first decay
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_param(param_name, opt, target_factor=0.1)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_param(param_name, opt, target_factor=0.1)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=6 // loader_len,
        step_id=6 % loader_len,
        global_step_id=6,
    )
    _check_param(param_name, opt, target_factor=0.1 * 0.1)


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_stepdecay_param_updater_in_trainer(
    toy_optimizer, param_name, init_param
):
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
        optimizer=toy_optimizer,
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
                type="StepDecayOptimParamUpdater",
                param_name=param_name,
                warmup_by="epoch",
                warmup_len=1,
                param_decay_id=[1, 2],
                param_decay_factor=0.1,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_stepdecay_param_updater_in_trainer_gen_dataset(
    toy_optimizer, param_name, init_param
):
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
        optimizer=toy_optimizer,
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
                type="StepDecayOptimParamUpdater",
                param_name=param_name,
                warmup_by="epoch",
                warmup_len=1,
                param_decay_id=[1, 2],
                param_decay_factor=0.1,
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


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_cos_param_updater_warmup_by_step(
    toy_optimizer, param_name, init_param
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="CosOptimParamUpdater",
        param_name=param_name,
        warmup_by="step",
        warmup_len=3,
        stop_param=0.001,
    )
    stop_param = updater["stop_param"]
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
    _check_param(param_name, opt, target_factor=0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_param(param_name, opt, target_factor=1 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    _check_param(param_name, opt, target_factor=2 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    factor = (1 + cos(pi * (3 - updater.warmup_steps) / 5)) / 2
    _check_param(
        param_name,
        opt,
        target_func=lambda x0: (x0 - stop_param) * factor + stop_param,
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    factor = (1 + cos(pi * (4 - updater.warmup_steps) / 5)) / 2
    _check_param(
        param_name,
        opt,
        target_func=lambda x0: (x0 - stop_param) * factor + stop_param,
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    factor = (1 + cos(pi * (5 - updater.warmup_steps) / 5)) / 2
    _check_param(
        param_name,
        opt,
        target_func=lambda x0: (x0 - stop_param) * factor + stop_param,
    )


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_cos_param_updater_warmup_by_step_gen_dataset(
    toy_optimizer, param_name, init_param
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="CosOptimParamUpdater",
        param_name=param_name,
        warmup_by="step",
        warmup_len=3,
        stop_param=0.001,
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
            == "'CosOptimParamUpdater' object has no attribute step_per_epoch"
        )


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_cos_param_updater_warmup_by_epoch(
    toy_optimizer, param_name, init_param
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="CosOptimParamUpdater",
        param_name=param_name,
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
    _check_param(param_name, opt, target_factor=0 / updater.warmup_steps)

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=1 // loader_len,
        step_id=1 % loader_len,
        global_step_id=1,
    )
    _check_param(param_name, opt, target_factor=1 / updater.warmup_steps)

    # 2. formal training stage
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=2 // loader_len,
        step_id=2 % loader_len,
        global_step_id=2,
    )
    factor = (1 + cos(pi * (2 - updater.warmup_steps) / updater.max_steps)) / 2
    _check_param(
        param_name,
        opt,
        target_factor=factor,
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=3 // loader_len,
        step_id=3 % loader_len,
        global_step_id=3,
    )
    factor = (1 + cos(pi * (3 - updater.warmup_steps) / updater.max_steps)) / 2
    _check_param(
        param_name,
        opt,
        target_factor=factor,
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    factor = (1 + cos(pi * (4 - updater.warmup_steps) / updater.max_steps)) / 2
    _check_param(
        param_name,
        opt,
        target_factor=factor,
    )


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_cos_param_updater_warmup_by_epoch_gen_dataset(
    toy_optimizer, param_name, init_param
):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="CosOptimParamUpdater",
        param_name=param_name,
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


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_cos_param_updater_warmup_in_train(
    toy_optimizer, param_name, init_param
):
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
        optimizer=toy_optimizer,
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
                type="CosOptimParamUpdater",
                param_name=param_name,
                warmup_by="epoch",
                warmup_len=1,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_cos_param_updater_warmup_in_train_gen_dataset(
    toy_optimizer, param_name, init_param
):
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
        optimizer=toy_optimizer,
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
                type="CosOptimParamUpdater",
                param_name=param_name,
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


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_cycle_param_updater_by_step(toy_optimizer, param_name, init_param):
    toy_opt = copy.deepcopy(toy_optimizer)
    updater = dict(
        type="CyclicOptimParamUpdater",
        param_name=param_name,
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

    _check_param(
        param_name,
        opt,
        target_func=lambda x: updater.annealing_cos(
            x * 10, x * 1e-4, (3 - 2) / (6 - 2)
        ),
    )
    updater.on_step_begin(
        optimizer=opt,
        epoch_id=4 // loader_len,
        step_id=4 % loader_len,
        global_step_id=4,
    )
    _check_param(
        param_name,
        opt,
        target_func=lambda x: updater.annealing_cos(
            x * 10, x * 1e-4, (4 - 2) / (6 - 2)
        ),
    )

    updater.on_step_begin(
        optimizer=opt,
        epoch_id=5 // loader_len,
        step_id=5 % loader_len,
        global_step_id=5,
    )
    _check_param(
        param_name,
        opt,
        target_func=lambda x: updater.annealing_cos(
            x * 10, x * 1e-4, (5 - 2) / (6 - 2)
        ),
    )


@pytest.mark.parametrize(
    ["toy_optimizer", "param_name", "init_param"],
    params,
)
def test_cycle_param_updater_in_trainer(toy_optimizer, param_name, init_param):
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
        optimizer=toy_optimizer,
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
                type="CyclicOptimParamUpdater",
                param_name=param_name,
                target_ratio=(10, 1e-4),
                cyclic_times=1,
                step_ratio_up=0.4,
            )
        ],
    )

    trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]
    trainer = build_from_registry(trainer_cfg)
    trainer.fit()
