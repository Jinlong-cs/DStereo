import os
from collections import OrderedDict as odict

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from hat.core.task_sampler import TaskSampler
from hat.utils.distributed import (
    find_free_port,
    get_dist_info,
    get_local_process_group,
)


def normal_check(sampler):
    for _ in range(5):
        task = sampler.sample_task()
        assert task == ["person"]
        for _ in range(2):
            assert sampler.sample_task() == ["vehicle"]
        for _ in range(3):
            assert sampler.sample_task() == ["lane"]
        for _ in range(2):
            assert sampler.sample_task() == ["real3d"]


def test_simple():
    cfg = odict(
        person=1,
        vehicle=2,
        lane=3,
        real3d=2,
    )
    sampler = TaskSampler(cfg)
    assert len(sampler) == 8
    normal_check(sampler)
    assert sampler.tasks == ["person", "vehicle", "lane", "real3d"]


def test_basic():
    cfg = odict(
        person=dict(sampling_factor=1),
        vehicle=dict(sampling_factor=2),
        lane=dict(sampling_factor=3),
        real3d=dict(sampling_factor=2),
    )
    sampler = TaskSampler(cfg)
    normal_check(sampler)


def test_sampler_iterable():
    cfg = odict(
        person=dict(sampling_factor=1),
        vehicle=dict(sampling_factor=2),
        lane=dict(sampling_factor=3),
        real3d=dict(sampling_factor=2),
    )
    sampler = TaskSampler(cfg)
    for i, task in enumerate(sampler):
        if i == 0:
            assert task == ["person"]
        elif i in (1, 2):
            assert task == ["vehicle"]
        elif i in (3, 4, 5):
            assert task == ["lane"]
        elif i in (6, 7):
            assert task == ["real3d"]
        else:
            raise AssertionError()


def test_task_shuffle():
    cfg = odict(
        person=dict(sampling_factor=1),
        vehicle=dict(sampling_factor=2),
        lane=dict(sampling_factor=3),
        real3d=dict(sampling_factor=2),
    )
    sampler = TaskSampler(cfg, shuffle=True)
    with pytest.raises(AssertionError):
        normal_check(sampler)
    tasks_1 = list()
    for _ in range(8):
        tasks_1.append(sampler.sample_task())
    tasks_2 = list()
    for _ in range(8):
        tasks_2.append(sampler.sample_task())
    assert tasks_1 != tasks_2


def test_task_epoch():
    cfg = odict(
        person=dict(sampling_factor=1, start_epoch=0, end_epoch=5),
        vehicle=dict(sampling_factor=2, start_epoch=0, end_epoch=5),
        lane=dict(sampling_factor=3, start_epoch=3, end_epoch=5),
        real3d=dict(sampling_factor=2, start_epoch=4, end_epoch=5),
    )
    sampler = TaskSampler(cfg)

    for epoch in range(3):
        for _ in range(2):
            task = sampler.sample_task(epoch=epoch)
            assert task == ["person"]
            for _ in range(2):
                assert sampler.sample_task(epoch=epoch) == ["vehicle"]

    epoch = 3
    task = sampler.sample_task(epoch=epoch)
    assert task == ["person"]
    for _ in range(2):
        assert sampler.sample_task(epoch=epoch) == ["vehicle"]
    for _ in range(3):
        assert sampler.sample_task(epoch=epoch) == ["lane"]

    epoch = 4
    task = sampler.sample_task(epoch=epoch)
    assert task == ["person"]
    for _ in range(2):
        assert sampler.sample_task(epoch=epoch) == ["vehicle"]
    for _ in range(3):
        assert sampler.sample_task(epoch=epoch) == ["lane"]
    for _ in range(2):
        assert sampler.sample_task(epoch=epoch) == ["real3d"]


@pytest.mark.skipif(True, reason="Not implemented yet")
def test_task_step():
    cfg = odict(
        person=dict(sampling_factor=1, start_step=0, end_step=100),
        vehicle=dict(sampling_factor=2, start_step=0, end_step=30),
        lane=dict(sampling_factor=3, start_step=30, end_step=94),
        real3d=dict(sampling_factor=2, start_step=70, end_step=94),
    )
    sampler = TaskSampler(cfg)

    # 0-30 step
    for idx in range(10):
        assert sampler.sample_task(step=idx * 3 + 0) == "person"
        assert sampler.sample_task(step=idx * 3 + 1) == "vehicle"
        assert sampler.sample_task(step=idx * 3 + 2) == "vehicle"

    # 30-70 step
    for idx in range(10):
        assert sampler.sample_task(step=30 + idx * 4 + 0) == "person"
        assert sampler.sample_task(step=30 + idx * 4 + 1) == "lane"
        assert sampler.sample_task(step=30 + idx * 4 + 2) == "lane"
        assert sampler.sample_task(step=30 + idx * 4 + 3) == "lane"

    # 70-94 step
    for idx in range(4):
        assert sampler.sample_task(step=70 + idx * 6 + 0) == "person"
        assert sampler.sample_task(step=70 + idx * 6 + 1) == "lane"
        assert sampler.sample_task(step=70 + idx * 6 + 2) == "lane"
        assert sampler.sample_task(step=70 + idx * 6 + 3) == "lane"
        assert sampler.sample_task(step=70 + idx * 6 + 4) == "real3d"
        assert sampler.sample_task(step=70 + idx * 6 + 5) == "real3d"

    # 94-100 step
    for idx in range(6):
        assert sampler.sample_task(step=94 + idx) == "person"


def test_task_gpus():
    cfg = odict(
        person=dict(sampling_factor=1, gpu_group="0"),
        vehicle=dict(sampling_factor=2, gpu_group="1"),
        lane=dict(sampling_factor=3, gpu_group="2"),
        real3d=dict(sampling_factor=3, gpu_group="2"),
    )
    gpu_weights = {"0": 2, "1": 2, "2": 1}
    sampler = TaskSampler(cfg, gpu_weights=gpu_weights)  # noqa: F841
    assert not sampler.is_parallel()

    # need pytorch ddp plugin for pytest
    # for _ in range(100):
    #     pass


def test_task_sample_all():
    cfg = dict(
        person=dict(sampling_factor=1),
        vehicle=dict(sampling_factor=2),
        lane=dict(sampling_factor=3),
        real3d=dict(sampling_factor=2),
    )
    sampler = TaskSampler(cfg, method="sample_all")
    task = sampler.sample_task()
    assert isinstance(task, list)
    assert sorted(task) == [
        "lane",
        "lane",
        "lane",
        "person",
        "real3d",
        "real3d",
        "vehicle",
        "vehicle",
    ]


def test_task_sample_repeat1():
    cfg1 = dict(
        person=dict(sampling_factor=1),
        vehicle=dict(sampling_factor=1),
        lane=dict(sampling_factor=1),
        real3d=dict(sampling_factor=1),
        end_steps=[3, 5, 7, 9],
        chosen_tasks=[
            ["person", "vehicle", "lane"],
            ["person", "vehicle", "lane", "real3d"],
            ["person", "lane"],
            ["person"],
        ],
    )

    sampler = TaskSampler(
        cfg1,
        method="sample_repeat",
    )
    correct_task_list = [
        [
            "person",
            "vehicle",
            "lane",
        ],
        [
            "person",
            "vehicle",
            "lane",
        ],
        [
            "person",
            "vehicle",
            "lane",
        ],
        [
            "person",
            "vehicle",
            "lane",
            "real3d",
        ],
        [
            "person",
            "vehicle",
            "lane",
            "real3d",
        ],
        [
            "person",
            "lane",
        ],
        [
            "person",
            "lane",
        ],
        ["person"],
        ["person"],
    ]
    task_list = []
    for _ in range(9):
        task = sampler.sample_task()
        assert isinstance(task, list)
        task_list.append(task)
    assert task_list == correct_task_list


def test_task_sample_repeat2():
    cfg1 = dict(
        person=dict(sampling_factor=1),
        vehicle=dict(sampling_factor=1),
        lane=dict(sampling_factor=1),
        real3d=dict(sampling_factor=1),
        chosen_tasks=[
            ["person", "vehicle", "lane"],
            ["person", "vehicle", "lane", "real3d"],
            ["person", "lane"],
            ["person"],
        ],
    )

    sampler = TaskSampler(
        cfg1,
        method="sample_repeat",
    )
    correct_task_list = [
        [
            "person",
            "vehicle",
            "lane",
        ],
        [
            "person",
            "vehicle",
            "lane",
            "real3d",
        ],
        [
            "person",
            "lane",
        ],
        ["person"],
        [
            "person",
            "vehicle",
            "lane",
        ],
        [
            "person",
            "vehicle",
            "lane",
            "real3d",
        ],
        [
            "person",
            "lane",
        ],
        ["person"],
        [
            "person",
            "vehicle",
            "lane",
        ],
    ]
    task_list = []
    for _ in range(9):
        task = sampler.sample_task()
        assert isinstance(task, list)
        task_list.append(task)
    assert task_list == correct_task_list


def test_task_sample_part():
    cfg = odict(
        person=dict(sampling_factor=1),
        vehicle=dict(sampling_factor=2),
        lane=dict(sampling_factor=3),
        real3d=dict(sampling_factor=2),
    )
    with pytest.raises(NotImplementedError):
        sampler = TaskSampler(
            cfg,
            method="sample_part",
            unions=[["person", "vehicle"], ["lane", "real3d"], ["person"]],
        )
        task = sampler.sample_task()

    cfg = odict(
        person=dict(sampling_factor=1),
        vehicle=dict(sampling_factor=1),
        lane=dict(sampling_factor=2),
        real3d=dict(sampling_factor=2),
    )

    sampler = TaskSampler(
        cfg,
        method="sample_part",
        unions=[["person", "vehicle"], ["lane", "real3d"]],
    )
    for _ in range(5):
        task = sampler.sample_task()
        assert task == ["person", "vehicle"]
        task = sampler.sample_task()
        assert task == ["lane", "real3d"]


def _main_worker(rank, world_size, cfg, gpu_weights):
    # init process group
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)

    sampler = TaskSampler(cfg, gpu_weights=gpu_weights)  # noqa: F841

    # check is parallel
    assert sampler.is_parallel()

    # check sub process group
    pg = get_local_process_group()
    r, w = get_dist_info(pg)
    rws = {0: [0, 2], 1: [1, 2], 2: [0, 2], 3: [1, 2], 4: [0, 1]}
    assert r == rws[rank][0]
    assert w == rws[rank][1]

    # check task selection
    tasks = {
        0: ["person"],
        1: ["person"],
        2: ["vehicle"],
        3: ["vehicle"],
        4: ["lane", "real3d"],
    }
    needed_tasks = tasks[rank]
    assert len(needed_tasks) == len(sampler.tasks)
    for t in sampler.tasks:
        assert t in needed_tasks
    dist.destroy_process_group()


@pytest.mark.serial_task
def test_task_parallel():
    cfg = odict(
        person=dict(sampling_factor=1, gpu_group="0"),
        vehicle=dict(sampling_factor=2, gpu_group="1"),
        lane=dict(sampling_factor=3, gpu_group="2"),
        real3d=dict(sampling_factor=3, gpu_group="2"),
    )
    gpu_weights = {"0": 2, "1": 2, "2": 1}
    sampler = TaskSampler(cfg, gpu_weights=gpu_weights)  # noqa: F841
    assert not sampler.is_parallel()

    num_workers = 5  # must same as sum of gpu_weights
    host_name = "localhost"
    port = find_free_port()
    os.environ["MASTER_ADDR"] = host_name
    os.environ["MASTER_PORT"] = str(port)

    mp.spawn(
        _main_worker, nprocs=num_workers, args=(num_workers, cfg, gpu_weights)
    )
