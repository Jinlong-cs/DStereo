import copy
import os
from copy import deepcopy
from distutils.version import LooseVersion

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from horizon_plugin_pytorch.qtensor import QTensor

from hat.engine.launcher import build_launcher
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.models.model_convert.converters import Float2QAT
from hat.registry import OBJECT_REGISTRY, build_from_registry
from hat.utils.apply_func import flatten, regroup
from hat.utils.logger import init_logger
from tests.data.toy_modules import *  # noqa: F403,F401

try:
    import hatbc
except ImportError:
    hatbc = None


@OBJECT_REGISTRY.register
class TestBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv1.weight.data.fill_(1)
        self.conv1.bias.data.fill_(1)

    def forward(self, x):
        x = self.conv1(x)
        return [x]


@OBJECT_REGISTRY.register
class TestHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.conv1.weight.data.fill_(1)
        self.conv1.bias.data.fill_(1)

    def forward(self, x, label=None):
        x = self.conv1(x[0])
        return torch.add(x, label)


@OBJECT_REGISTRY.register
class QATTestHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.conv1.weight.data.fill_(1)
        self.conv1.bias.data.fill_(1)
        self.qat_add = nn.quantized.FloatFunctional()

    def forward(self, x, label=None):
        x = self.conv1(x[0])
        return self.qat_add.add(x, label)._base


class MultitaskStructure(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        output_module: torch.nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_module = output_module

    def forward(self, x):
        feats = self.backbone(x["img"])
        if "label" in x:
            res = self.output_module(feats, x["label"])
        else:
            res = self.output_module(feats)
        return res


backbone = dict(
    type="TestBackbone",
    node_name="backbone",
)

loss1 = dict(
    type="TestHead",
    node_name="loss1",
)
qat_loss1 = dict(
    type="QATTestHead",
    node_name="loss1",
)

qat_loss2 = dict(
    type="QATTestHead",
    node_name="loss2",
)

loss2 = dict(
    type="TestHead",
    node_name="loss2",
)

model_cfg = dict(
    type="MultitaskGraphModel",
    inputs=dict(
        img=torch.tensor(
            [[[[5, 3, 2], [2, 3, 4], [5, 4, 3]]]], dtype=torch.float32
        )
    ),
    task_inputs=dict(
        loss1=dict(label=torch.tensor([1], dtype=torch.float32)),
        loss2=dict(label=torch.tensor([2], dtype=torch.float32)),
    ),
    task_modules=dict(
        loss1=dict(
            type=MultitaskStructure,
            backbone=backbone,
            output_module=loss1,
        ),
        loss2=dict(
            type=MultitaskStructure,
            backbone=backbone,
            output_module=loss2,
        ),
    ),
    lazy_forward=False,
)

qat_model_cfg = dict(
    type="MultitaskGraphModel",
    inputs=dict(
        img=torch.tensor(
            [[[[5, 3, 2], [2, 3, 4], [5, 4, 3]]]], dtype=torch.float32
        )
    ),
    task_inputs=dict(
        loss1=dict(label=torch.tensor([1], dtype=torch.float32)),
        loss2=dict(label=torch.tensor([2], dtype=torch.float32)),
    ),
    task_modules=dict(
        loss1=dict(
            type=MultitaskStructure,
            backbone=backbone,
            output_module=qat_loss1,
        ),
        loss2=dict(
            type=MultitaskStructure,
            backbone=backbone,
            output_module=qat_loss2,
        ),
    ),
    lazy_forward=False,
)

processor_cfg = dict(
    type="MultiStageBatchProcessor",
    delay_sync=False,
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
)

processor_cfg2 = dict(
    type="MultiBatchProcessor",
    delay_sync=False,
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_multi_stage_batch_processor():
    def multi_stage_batch_processor():
        data = dict(**model_cfg["inputs"], **model_cfg["task_inputs"])
        model = build_from_registry(model_cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        processor = build_from_registry(processor_cfg)

        gpu = 0
        model = model.cuda(gpu)
        flats, fmt = flatten(data)
        flats = list(flats)
        model_state_before = deepcopy(model.state_dict())
        for i in range(len(flats)):
            if isinstance(flats[i], torch.Tensor):
                flats[i] = flats[i].clone()
                if gpu is not None:
                    flats[i] = flats[i].cuda(gpu)
        new_bat, flats_idx = regroup(flats, fmt)
        assert len(flats) == flats_idx, flats_idx
        processor(
            0, [new_bat, ["loss1", "loss2"]], model, None, optimizer=optimizer
        )
        model_state_updated = deepcopy(model.state_dict())
        return model_state_before, model_state_updated

    def multi_batch_processor():
        data = dict(**model_cfg["inputs"], **model_cfg["task_inputs"])
        model = build_from_registry(model_cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        processor = build_from_registry(processor_cfg2)

        gpu = 0
        model = model.cuda(gpu)
        flats, fmt = flatten(data)
        flats = list(flats)
        model_state_before = deepcopy(model.state_dict())
        for i in range(len(flats)):
            if isinstance(flats[i], torch.Tensor):
                flats[i] = flats[i].clone()
                if gpu is not None:
                    flats[i] = flats[i].cuda(gpu)
        new_bat, flats_idx = regroup(flats, fmt)
        assert len(flats) == flats_idx, flats_idx
        processor(
            0, [new_bat, ["loss1", "loss2"]], model, None, optimizer=optimizer
        )
        model_state_updated = deepcopy(model.state_dict())
        return model_state_before, model_state_updated

    state_before_step, state_after_step = multi_stage_batch_processor()
    gt_before, gt_after = multi_batch_processor()
    for key in state_before_step.keys():
        assert torch.equal(state_before_step[key], gt_before[key])
        assert torch.equal(state_after_step[key], gt_after[key])


@OBJECT_REGISTRY.register
class LocalDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        count = 0
        flag = 1
        while True:
            if flag:
                key = "loss1"
                flag = 0
            else:
                key = "loss2"
                flag = 1
            if self.rank == 0:
                yield (self.data1, key)
            if self.rank == 1:
                yield (self.data2, key)
            count += 1
            if count >= 6:
                break

    def __len__(self):
        return 12


input1 = dict(
    img=torch.tensor([[[3, 3, 2], [2, 3, 4], [5, 6, 3]]], dtype=torch.float32),
    loss1=dict(label=torch.tensor(1, dtype=torch.float32)),
    loss2=dict(label=torch.tensor(2, dtype=torch.float32)),
)
input2 = dict(
    img=torch.tensor(
        [[[54, 33, 22], [22, 43, 24], [45, 64, 33]]], dtype=torch.float32
    ),
    loss1=dict(label=torch.tensor(5, dtype=torch.float32)),
    loss2=dict(label=torch.tensor(9, dtype=torch.float32)),
)
dataset = dict(
    type="LocalDataset",
    data1=input1,
    data2=input2,
)


trainer_cfg = dict(
    type="distributed_data_parallel_trainer",
    model=model_cfg,
    data_loader=dict(
        type=torch.utils.data.DataLoader,
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    ),
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.01,
    ),
    batch_processor=processor_cfg,
    num_epochs=1,
    device=None,
    sync_bn=True,
    sync_bn_by_host=True,
)


def _ddp_baseline(gpu_id, *args):
    init_logger(os.path.join(args[0], str(gpu_id)), rank=gpu_id)

    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id
    cfg["batch_processor"] = processor_cfg2
    trainer = build_from_registry(cfg)
    trainer.fit()
    torch.save(
        trainer.model.state_dict(), os.path.join(args[0], "result_baseline.pt")
    )


def _ddp_multistage_no_delay_sync(gpu_id, *args):
    init_logger(os.path.join(args[0], str(gpu_id)), rank=gpu_id)

    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    trainer = build_from_registry(cfg)
    trainer.fit()
    torch.save(
        trainer.model.state_dict(),
        os.path.join(args[0], "result_multistage_no_delay.pt"),
    )


def _ddp_multistage_delay_sync(gpu_id, *args):
    init_logger(os.path.join(args[0], str(gpu_id)), rank=gpu_id)

    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    if int(os.environ.get("USE_DCU", 0)) == 1:
        cfg["batch_processor"]["delay_sync"] = False
    else:
        cfg["batch_processor"]["delay_sync"] = True

    trainer = build_from_registry(cfg)
    trainer.fit()
    torch.save(
        trainer.model.state_dict(),
        os.path.join(args[0], "result_multistage_delay_sync.pt"),
    )


def _ddp_multistage_grad_accumulation(gpu_id, *args):
    init_logger(os.path.join(args[0], str(gpu_id)), rank=gpu_id)

    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id

    if int(os.environ.get("USE_DCU", 0)) == 1:
        cfg["batch_processor"]["grad_accumulation_step"] = 1
    else:
        cfg["batch_processor"]["grad_accumulation_step"] = 2

    trainer = build_from_registry(cfg)
    trainer.fit()
    torch.save(
        trainer.model.state_dict(),
        os.path.join(args[0], "result_multistage_grad_accumulation.pt"),
    )


def _ddp_multistage_channels_last(gpu_id, *args):
    init_logger(os.path.join(args[0], str(gpu_id)), rank=gpu_id)

    cfg = copy.deepcopy(trainer_cfg)
    cfg["device"] = gpu_id
    cfg["batch_processor"]["enable_channels_last"] = True
    cfg["batch_processor"]["channels_last_keys"] = ("img",)

    trainer = build_from_registry(cfg)
    trainer.fit()
    torch.save(
        trainer.model.state_dict(),
        os.path.join(args[0], "result_multistage_channels_last.pt"),
    )


@pytest.mark.timeout(500)
@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.serial_task
def test_multi_stage_batch_processor_ddp(tmp_path):
    launcher = build_launcher(trainer_cfg)
    device_ids = [0, 1]
    dist_url = "auto"
    args = (tmp_path,)
    launcher(_ddp_baseline, device_ids, dist_url, args=args)
    launcher = build_launcher(trainer_cfg)
    launcher(_ddp_multistage_no_delay_sync, device_ids, dist_url, args=args)
    launcher = build_launcher(trainer_cfg)
    launcher(_ddp_multistage_delay_sync, device_ids, dist_url, args=args)
    launcher = build_launcher(trainer_cfg)
    launcher(
        _ddp_multistage_grad_accumulation, device_ids, dist_url, args=args
    )
    torch_version = torch.__version__
    if LooseVersion(torch_version) > LooseVersion("1.13"):
        launcher = build_launcher(trainer_cfg)
        launcher(
            _ddp_multistage_channels_last, device_ids, dist_url, args=args
        )

    with open(os.path.join(tmp_path, "result_baseline.pt"), "rb") as f:
        state_baseline = torch.load(f, map_location=torch.device("cpu"))
    with open(
        os.path.join(tmp_path, "result_multistage_no_delay.pt"), "rb"
    ) as f:
        state_dict_no_delay = torch.load(f, map_location=torch.device("cpu"))
    with open(
        os.path.join(tmp_path, "result_multistage_delay_sync.pt"), "rb"
    ) as f:
        state_dict_delay_sync = torch.load(f, map_location=torch.device("cpu"))
    with open(
        os.path.join(tmp_path, "result_multistage_grad_accumulation.pt"), "rb"
    ) as f:
        state_dict_grad_accumulation = torch.load(
            f, map_location=torch.device("cpu")
        )

    if LooseVersion(torch_version) > LooseVersion("1.13"):
        with open(
            os.path.join(tmp_path, "result_multistage_channels_last.pt"), "rb"
        ) as f:
            state_dict_channels_last = torch.load(
                f, map_location=torch.device("cpu")
            )

    for key in state_baseline.keys():
        if key.startswith("_"):
            continue
        assert torch.equal(state_baseline[key], state_dict_no_delay[key])
        assert torch.equal(
            state_dict_no_delay[key], state_dict_delay_sync[key]
        )

        # TODO(zhigang.yang): May have some problems
        if int(os.environ.get("USE_DCU", 0)) == 1:
            return
        assert not torch.equal(
            state_dict_no_delay[key], state_dict_grad_accumulation[key]
        )

        if LooseVersion(torch_version) > LooseVersion("1.13"):
            # different memory format can still use torch.equal
            assert torch.equal(
                state_dict_no_delay[key], state_dict_channels_last[key]
            )


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.serial_task
def test_multi_stage_batch_processor_qtensor():
    data = dict(**model_cfg["inputs"], **model_cfg["task_inputs"])
    model = build_from_registry(qat_model_cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    processor = build_from_registry(processor_cfg)

    converter = Float2QAT()
    q_model = converter(model)

    gpu = 0
    q_model = q_model.cuda(gpu)
    flats, fmt = flatten(data)
    flats = list(flats)
    model_state_before = deepcopy(q_model.state_dict())
    for i in range(len(flats)):
        if isinstance(flats[i], torch.Tensor):
            flats[i] = flats[i].clone()
            if gpu is not None:
                flats[i] = flats[i].cuda(gpu)
            flats[i] = QTensor(
                flats[i], torch.tensor([0.2], dtype=torch.float32), "qint8"
            )
    new_bat, flats_idx = regroup(flats, fmt)
    assert len(flats) == flats_idx, flats_idx
    processor(
        0, [new_bat, ["loss1", "loss2"]], q_model, None, optimizer=optimizer
    )
    model_state_updated = deepcopy(q_model.state_dict())
    if int(os.environ.get("USE_DCU", 0)) == 1:
        return
    for key in model_state_updated.keys():
        if ("weight" in key or "bias" in key) and "q" not in key:
            assert not torch.equal(
                model_state_before[key], model_state_updated[key]
            )
