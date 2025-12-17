import copy
import os
import random

import pytest
import torch

from hat.engine.launcher import build_launcher
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.profiler.pytorch_profiler import PyTorchProfiler
from hat.registry import build_from_registry
from tests.data.toy_modules import *  # noqa: F403,F401
from tests.unit_tests.base import BoringModel, FakeAuto2dDataset

trainer_cfg = dict(
    type="distributed_data_parallel_trainer",
    model=dict(
        type="ToyModel",
        backbone=dict(
            type="ToyBackbone",
            strides=(1, 2),
            channels=(3, 4),
        ),
        head=dict(
            type="ToyHead",
            in_channels=4,
            fc_filter=4,
            num_classes=3,
            with_dequant=True,
        ),
        loss=dict(type="ToyLoss"),
    ),
    data_loader=dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="RandDataset",
            length=50,
            example=(torch.randn((3, 14, 14)), random.randint(0, 3 - 1)),
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
    num_steps=10,
    stop_by="step",
    device=None,
    callbacks=[
        dict(
            type="StatsMonitor",
            log_freq=1,
        ),
        dict(
            type="StepDecayLrUpdater",
            warmup_by="step",
            warmup_len=0,
            lr_decay_id=[1, 2],
        ),
    ],
)
trainer_cfg["optimizer"]["model"] = trainer_cfg["model"]


@pytest.fixture
def pytorch_profiler(tmpdir):
    return PyTorchProfiler(dirpath=tmpdir, schedule=None)


def test_pytorch_profiler_describe(pytorch_profiler):
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    for _, data in enumerate(data_loader):
        with pytorch_profiler.profile("model_forward"):
            output = model(data)  # noqa: F841
    pytorch_profiler.describe()


def _pytorch_profiler_trainer_ddp(
    gpu_id,
    tmp_dir,
    record_functions,
    step_function,
    *args,
):

    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    profiler = dict(
        type="PyTorchProfiler",
        dirpath=tmp_dir,
        export_to_chrome=True,
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=1),
        record_functions=record_functions,
        step_function=step_function,
    )
    cfg["profiler"] = profiler

    trainer = build_from_registry(cfg)
    trainer.fit()

    pytorch_profiler = trainer.profiler

    files = set(os.listdir(pytorch_profiler.dirpath))
    json_file = [f for f in files if f.endswith(".pt.trace.json")]  # noqa E501
    assert len(json_file) == torch.cuda.device_count()


@pytest.mark.parametrize(
    ["record_functions", "step_function"],
    [
        pytest.param(None, None),
        pytest.param(
            {
                "gpu_transforms",
                "model_forward",
                "model_backward",
                "optimizer_step",
                "optimizer_zero_grad",
            },
            "optimizer_zero_grad",
        ),
    ],
)
@pytest.mark.serial_task
def test_pytorch_profiler_in_ddp_trainer(
    tmpdir,
    record_functions,
    step_function,
):
    launcher = build_launcher(trainer_cfg)
    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(
        _pytorch_profiler_trainer_ddp,
        device_ids,
        dist_url,
        args=(
            tmpdir,
            record_functions,
            step_function,
        ),
    )


def test_pytorch_profiler(tmpdir):
    torch_profiler = PyTorchProfiler(
        dirpath=tmpdir,
        schedule=None,
        use_cuda=False,
    )

    with torch_profiler.profile("a"):
        a = torch.ones(1)
        with torch_profiler.profile("b"):
            b = torch.zeros(2)
        with torch_profiler.profile("c"):
            _ = a + b

    torch_profiler.describe()

    events_name = {e.name for e in torch_profiler.function_events}
    ops = {"add", "empty", "fill_", "ones", "zero_", "zeros"}
    ops = {f"aten::{op}" for op in ops}
    assert ops == events_name
