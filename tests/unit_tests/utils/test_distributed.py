import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hat.utils.distributed import (
    all_gather_object,
    create_process_group,
    dist_initialized,
    find_free_port,
    get_device_count,
    get_dist_info,
    get_global_out,
    get_local_host,
    rank_zero_only,
    split_process_group_by_host,
)


@pytest.mark.repeat(3)
def test_find_free_port():
    free_port = find_free_port()
    assert free_port >= 10001 and free_port <= 19999


def test_rank_zero_only():
    @rank_zero_only
    def say_hello():
        print("hello, hat")

    say_hello()


def test_get_dist_info():
    rank, world_size = get_dist_info()
    assert rank == 0
    assert world_size == 1


def test_get_device_count():
    device_num = get_device_count()
    assert device_num == torch.cuda.device_count()


def test_get_local_host():
    hostid = get_local_host()
    print("hostid: ", hostid)


def test_create_process_group_with_no_dist():
    group = create_process_group([1, 2, 3])
    assert group is None


def test_all_gather_object_with_no_dist():
    objs = [None]
    obj = "abc"
    all_gather_object(objs, obj)
    assert objs[0] == obj


def test_get_global_out():
    output = "abc"
    global_rank, global_output = get_global_out(output)
    assert global_rank == 0
    assert global_output[0] == output


@rank_zero_only
def say_hello():
    print("Hello, hat")


def _main_worker(rank, world_size):
    # init process group
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)

    # test for rank_zero_only
    say_hello()

    # test for get dist info
    assert dist_initialized()
    r, w = get_dist_info()
    print(f"rank: {rank}, world: {world_size}, r: {r}, w: {w}")
    assert r == rank
    assert w == world_size

    # test for create sub process group
    ranks = [1, 2, 3]
    sub_group = create_process_group(ranks)
    assert sub_group is not None
    if rank == 1:
        r, w = get_dist_info(sub_group)
        device_num = get_device_count(sub_group)
        assert r == 0
        assert w == 3
        assert device_num == w
    if rank == 2:
        r, w = get_dist_info(sub_group)
        device_num = get_device_count(sub_group)
        assert r == 1
        assert w == 3
        assert device_num == w

    # test for all_gather_object
    obj_list = [None for _ in range(world_size)]
    data = torch.ones((1,))
    all_gather_object(obj_list, data)
    res = torch.stack(obj_list)
    res = torch.sum(res)
    assert res == world_size

    # test for split_process_group_by_host
    new_pg, ok = split_process_group_by_host()
    assert new_pg is None and ok

    ranks = [0]
    sub_group1 = create_process_group(ranks)
    ranks = [1, 2, 3]
    sub_group2 = create_process_group(ranks)
    if rank == 0:
        sub_group = sub_group1
    else:
        sub_group = sub_group2
    new_pg, created = split_process_group_by_host(sub_group)
    assert new_pg == sub_group and ok
    dist.destroy_process_group()


def test_with_dist_mode():
    num_workers = 4
    host_name = "localhost"
    port = find_free_port()
    os.environ["MASTER_ADDR"] = host_name
    os.environ["MASTER_PORT"] = str(port)

    mp.spawn(_main_worker, nprocs=num_workers, args=(num_workers,))
