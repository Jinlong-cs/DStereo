import copy
import os
import random

import pytest
import torch

from hat.engine.launcher import build_launcher
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.profiler.dynamo_profiler import DynamoProfiler
from hat.registry import build_from_registry
from hat.utils.dynamo import CompileBackendWrapper
from hat.utils.global_var import set_value
from hat.utils.package_helper import check_packages_available
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
    compiler=dict(
        type="Torch2Compile",
        backend="inductor",
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
    num_steps=2,
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


def _dynamo_profiler_trainer_ddp(
    gpu_id,
    tmp_dir,
    *args,
):
    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    profiler = dict(
        type="DynamoProfiler",
        dirpath=tmp_dir,
        filename="dynamo_result",
    )
    cfg["profiler"] = profiler

    trainer = build_from_registry(cfg)
    trainer.fit()

    dynamo_profiler = trainer.profiler

    files = set(os.listdir(dynamo_profiler.dirpath))
    assert len(files) > 0


@pytest.mark.serial_task
@pytest.mark.skipif(
    not check_packages_available("torch>=2.0", raise_exception=False),
    reason="require torch>=2.0",
)
def test_pytorch_profiler_in_ddp_trainer(
    tmpdir,
):
    launcher = build_launcher(trainer_cfg)
    device_ids = list(range(torch.cuda.device_count()))
    dist_url = "auto"
    launcher(
        _dynamo_profiler_trainer_ddp,
        device_ids,
        dist_url,
        args=(tmpdir,),
    )


@pytest.fixture
def dynamo_profiler(tmpdir):
    return DynamoProfiler(dirpath=tmpdir)


@pytest.mark.skipif(
    not check_packages_available("torch>=2.0", raise_exception=False),
    reason="require torch>=2.0",
)
def test_dynamo_profiler_describe(dynamo_profiler):
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    eb_backend = CompileBackendWrapper(backend="inductor")
    torch.compile(model, backend=eb_backend)
    set_value(CompileBackendWrapper.compile_wrapper_name, eb_backend)

    for _, data in enumerate(data_loader):
        with dynamo_profiler.profile("model_forward"):
            output = model(data)  # noqa: F841
    dynamo_profiler.describe()
