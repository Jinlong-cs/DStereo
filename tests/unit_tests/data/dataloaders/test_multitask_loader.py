from collections import OrderedDict as odict

import pytest
import torch

from hat.core.task_sampler import TaskSampler
from hat.data.dataloaders.multitask_loader import (
    MultitaskInfLoader,
    MultitaskLoader,
)
from tests.unit_tests.base import ToyIterableDataset


def test_combined_loader():
    loaders = {
        "person": torch.utils.data.DataLoader(range(256), batch_size=4),
        "vehicle": torch.utils.data.DataLoader(range(256), batch_size=8),
        "semseg": torch.utils.data.DataLoader(range(256), batch_size=8),
        "real3d": torch.utils.data.DataLoader(range(256), batch_size=4),
    }
    multitask_loader = MultitaskLoader(loaders)
    assert multitask_loader.batch_size == 24
    for _, data in enumerate(multitask_loader):
        assert isinstance(data, dict)
        assert "person" in data
        assert "vehicle" in data
        assert "semseg" in data
        assert "real3d" in data


def test_loader_with_task_sampler():
    cfg = dict(person=1, vehicle=2, lane=3, real3d=2)
    task_sampler = TaskSampler(cfg)
    loaders = {
        "person": torch.utils.data.DataLoader(range(256), batch_size=4),
        "vehicle": torch.utils.data.DataLoader(range(256), batch_size=8),
        "lane": torch.utils.data.DataLoader(range(256), batch_size=8),
        "real3d": torch.utils.data.DataLoader(range(256), batch_size=4),
    }
    multitask_loader = MultitaskLoader(loaders, task_sampler)
    assert multitask_loader.batch_size == (4 * 1 + 8 * 2 + 8 * 3 + 4 * 2)
    for _, data in enumerate(multitask_loader):
        assert isinstance(data, dict)
        assert len(data) == 1

    # test loader with task_sampler in sample_all mode
    task_sampler = TaskSampler(cfg, method="sample_all")
    multitask_loader = MultitaskLoader(loaders, task_sampler)
    for _, data in enumerate(multitask_loader):
        assert isinstance(data, dict)
        assert len(data) == 4
        assert "person" in data
        assert "vehicle" in data
        assert "lane" in data
        assert "real3d" in data

    multitask_loader = MultitaskLoader(loaders, task_sampler, return_task=True)
    for _, data in enumerate(multitask_loader):
        assert isinstance(data, tuple)
        assert isinstance(data[0], tuple)
        assert len(data) == 4
        assert data[0][1] == "person"
        assert data[1][1] == "vehicle"
        assert data[2][1] == "lane"
        assert data[3][1] == "real3d"


class ToyTrainer(object):
    def __init__(self, loader, epoch=10):
        self.model = torch.nn.Conv2d(3, 16, 3)
        self.epoch = epoch
        self.dataloader = loader

    def fit(self):
        for epoch_id in range(self.epoch):
            for _, data in enumerate(self.dataloader):
                print(
                    f"epoch: {epoch_id}, data: {list(data.keys())}, "
                    f"len: {len(self.dataloader)}"
                )
                # do forward_backward_update


@pytest.mark.parametrize(
    ["cfg", "method"],
    [
        pytest.param(
            odict(person=1, vehicle=1, lane=1, real3d=1), "sample_one"
        ),
        # multitask v1.0
        pytest.param(
            odict(person=1, vehicle=1, lane=1, real3d=1), "sample_all"
        ),
        # multitask with end_epoch
        pytest.param(
            odict(
                person=dict(sampling_factor=1, end_epoch=5),
                vehicle=dict(sampling_factor=1, end_epoch=5),
                lane=dict(sampling_factor=1, end_epoch=3),
                real3d=dict(sampling_factor=1, end_epoch=4),
            ),
            "sample_one",
        ),
    ],
)
def test_loader_in_trainer(cfg, method):
    task_sampler = TaskSampler(cfg, method=method)
    loaders = {
        "person": torch.utils.data.DataLoader(range(20), batch_size=4),  # 5
        "vehicle": torch.utils.data.DataLoader(range(16), batch_size=8),  # 2
        "lane": torch.utils.data.DataLoader(range(16), batch_size=8),  # 2
        "real3d": torch.utils.data.DataLoader(range(12), batch_size=4),  # 3
    }
    multitask_loader = MultitaskLoader(loaders, task_sampler)
    trainer = ToyTrainer(multitask_loader, epoch=5)
    trainer.fit()


def test_loader_with_iterable_dataset():
    loaders = {
        "person": torch.utils.data.DataLoader(range(8), batch_size=4),
        "task_with_iter_dataset": torch.utils.data.DataLoader(
            ToyIterableDataset(), batch_size=8
        ),
    }
    multitask_loader = MultitaskLoader(loaders)
    assert multitask_loader.__len__() == float("inf")

    for _, data in enumerate(multitask_loader):
        assert isinstance(data, dict)
        assert "person" in data
        assert "task_with_iter_dataset" in data


def test_loader_in_validation():
    loaders = {
        "person": torch.utils.data.DataLoader(range(256), batch_size=4),  # 64
        "vehicle": torch.utils.data.DataLoader(range(256), batch_size=8),  # 32
        "semseg": torch.utils.data.DataLoader(range(256), batch_size=8),  # 32
        "real3d": torch.utils.data.DataLoader(range(256), batch_size=4),  # 64
    }
    multitask_loader = MultitaskLoader(loaders, mode="validation")
    for i, data in enumerate(multitask_loader):
        assert isinstance(data, dict)
        assert "person" in data
        assert "real3d" in data
        if i <= 31:
            assert "vehicle" in data
            assert "semseg" in data
        else:
            assert "vehicle" not in data
            assert "semseg" not in data


def test_loader_with_wrap_batch():
    loaders = {
        "person": torch.utils.data.DataLoader(range(256), batch_size=4),
        "vehicle": torch.utils.data.DataLoader(range(256), batch_size=8),
        "semseg": torch.utils.data.DataLoader(range(256), batch_size=8),
        "real3d": torch.utils.data.DataLoader(range(256), batch_size=4),
    }

    # wrap_batch = False
    multitask_loader = MultitaskLoader(
        loaders, return_task=True, wrap_batch=False
    )
    assert multitask_loader.batch_size == 24
    for _, data in enumerate(multitask_loader):
        assert isinstance(data, tuple)
        assert "person" in data[0]
        assert "vehicle" in data[1]
        assert "semseg" in data[2]
        assert "real3d" in data[3]

    # wrap_batch = True
    multitask_loader = MultitaskLoader(
        loaders, return_task=True, wrap_batch=True
    )
    assert multitask_loader.batch_size == 24
    for _, data in enumerate(multitask_loader):
        assert isinstance(data, tuple)
        assert isinstance(data[0][0], dict)
        assert "person" in data[0][0]
        assert isinstance(data[1][0], dict)
        assert "vehicle" in data[1][0]
        assert isinstance(data[2][0], dict)
        assert "semseg" in data[2][0]
        assert isinstance(data[3][0], dict)
        assert "real3d" in data[3][0]


def test_multitask_inf_loader():
    loaders = {
        "person": torch.utils.data.DataLoader(
            [dict(img=[[1, 2, 3]])], batch_size=1
        ),
        "vehicle": torch.utils.data.DataLoader(
            [dict(img=[[1, 2, 3]])], batch_size=1
        ),
    }
    task_sampler = TaskSampler(
        odict(
            person=dict(sampling_factor=1),
            vehicle=dict(sampling_factor=1),
        ),
        method="sample_all",
        shuffle=True,
    )
    data_loader = MultitaskInfLoader(
        loaders=loaders,
        task_sampler=task_sampler,
        return_task=True,
    )
    data_iter = iter(data_loader)
    for _ in range(10):
        data = next(data_iter)
        assert isinstance(data, tuple)
        assert isinstance(data[0][0], dict)
        assert "person" in data[0][0]
        assert isinstance(data[1][0], dict)
        assert "vehicle" in data[1][0]


def test_multitask_inf_loader_task_mapper():

    loaders = {
        "full": torch.utils.data.DataLoader(
            [dict(img=[[1, 2, 3]])], batch_size=1
        ),
    }

    task_mapper = {
        "person": "full",
        "vehicle": "full",
    }

    task_sampler = TaskSampler(
        odict(
            person=dict(sampling_factor=1),
            vehicle=dict(sampling_factor=1),
        ),
        method="sample_all",
        shuffle=True,
    )
    data_loader = MultitaskInfLoader(
        loaders=loaders,
        task_sampler=task_sampler,
        task_mapper=task_mapper,
        return_task=True,
    )
    data_iter = iter(data_loader)
    for _ in range(10):
        data = next(data_iter)
        assert isinstance(data, tuple)
        assert isinstance(data[0][0], dict)
        assert "person" in data[0][1]
        assert "vehicle" in data[0][1]
