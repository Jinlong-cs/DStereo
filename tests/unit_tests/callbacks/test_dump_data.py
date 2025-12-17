import copy
import os
import pickle
import random
import re

import pytest
import torch

from hat.engine.launcher import build_launcher
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.registry import build_from_registry
from hat.utils.seed import seed_everything
from tests.data.toy_modules import *  # noqa: F403,F401
from tests.utils import init_test_logger

dump_dir = "/tmp/tmp_dump_dir"

# --------------- common ---------------
model = dict(
    type="ToyModel",
    backbone=dict(
        type="ToyBackbone",
        strides=(1, 2),
        channels=(3, 8),
    ),
    head=dict(
        type="ToyHead",
        in_channels=8,
        fc_filter=16,
        num_classes=10,
        with_dequant=True,
    ),
    loss=dict(type="ToyLoss"),
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="RandDataset",
        length=10,
        example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
        clone=True,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
)

# --------------- ddp trainer ---------------
trainer_cfg = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=5,
    device=None,
    callbacks=[
        dict(
            type="StatsMonitor",
            log_freq=1,
        ),
        dict(
            type="DumpData",
            output_dir=dump_dir,
            name_prefix="test_dump_data",
            save_interval=1,
            interval_by="step",
            dump_batch_data=True,
            dump_model_outs=True,
            dump_model_grad=True,
        ),
        dict(
            type="StepDecayLrUpdater",
            warmup_by="step",
            warmup_len=0,
            lr_decay_id=[1, 2],
        ),
        dict(
            type="Checkpoint",
            interval_by="step",
            save_dir=dump_dir,
            save_hash=False,
        ),
    ],
    sync_bn=True,
    sync_bn_by_host=True,
)
trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]

# --------------- predictor ---------------
predictor_cfg = dict(
    type="Predictor",
    model=model,
    data_loader=data_loader,
    batch_processor=batch_processor,
    device=0,
    callbacks=[
        dict(
            type="StatsMonitor",
            log_freq=1,
        ),
        dict(
            type="DumpData",
            output_dir=dump_dir,
            name_prefix="test_dump_data",
            save_interval=1,
            interval_by="step",
            dump_batch_data=True,
            dump_model_outs=True,
            dump_model_grad=True,
        ),
    ],
)


def _check_dumped_file(dumped_file, dumped_keys, step_nums):
    assert os.path.exists(dumped_file)
    with open(dumped_file, "rb") as f:
        datas = pickle.load(f)

    for data in datas:
        assert len(data) == step_nums
        for _, step_values in data.items():
            for key in dumped_keys:
                pkl_keys = list(
                    filter(
                        lambda x: re.match(f".*{key}", x) is not None,
                        step_values.keys(),
                    )  # noqa E501
                )
                assert len(pkl_keys) == 1


def _dump_data_ddp_trainer_main(gpu_id, tmpdir, *args):
    init_test_logger(rank=gpu_id)
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id
    cfg["callbacks"][1]["output_dir"] = tmpdir
    trainer = build_from_registry(cfg)
    trainer.fit()

    _check_dumped_file(
        dumped_file=os.path.join(tmpdir, "test_dump_data.pkl"),
        dumped_keys=[
            "batch_data",
            "grad_before_optimizer",
            "model_outs",
        ],
        step_nums=cfg["num_steps"],
    )


@pytest.mark.serial_task
def test_dump_data_in_ddp_trainer(tmpdir):
    seed_everything(os.getpid())
    launcher = build_launcher(trainer_cfg)
    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(_dump_data_ddp_trainer_main, device_ids, dist_url, args=(tmpdir,))


def test_dump_data_in_trainer(tmpdir):

    cfg = copy.deepcopy(trainer_cfg)
    cfg["type"] = "Trainer"
    cfg["data_loader"].pop("sampler")
    cfg["batch_processor"]["type"] = "BasicBatchProcessor"
    # cfg["callbacks"][-1]["output_dir"] = tmpdir
    cfg["callbacks"][1]["output_dir"] = tmpdir
    cfg.pop("sync_bn")
    cfg.pop("sync_bn_by_host")
    trainer = build_from_registry(cfg)
    trainer.fit()

    _check_dumped_file(
        dumped_file=os.path.join(tmpdir, "test_dump_data.pkl"),
        dumped_keys=[
            "batch_data",
            "grad_before_optimizer",
            "model_outs",
        ],
        step_nums=cfg["num_steps"],
    )


def test_dump_data_in_predictor(tmpdir):
    cfg = copy.deepcopy(predictor_cfg)
    cfg["batch_processor"]["need_grad_update"] = False
    cfg["callbacks"][-1]["output_dir"] = tmpdir
    predictor = build_from_registry(cfg)
    predictor.fit()

    _check_dumped_file(
        dumped_file=os.path.join(tmpdir, "test_dump_data.pkl"),
        dumped_keys=[
            "batch_data",
            "model_outs",
        ],
        step_nums=10,
    )
