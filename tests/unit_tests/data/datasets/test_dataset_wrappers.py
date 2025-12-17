import math

import numpy as np
import pytest
import torch
import torch.distributed as dist

from hat.data.datasets.dataset_wrappers import ChunkShuffleDataset
from hat.data.datasets.rand_dataset import SimpleDataset
from hat.engine.ddp_trainer import launch
from hat.registry import build_from_registry


def test_resample_dataset():
    ori_dataset_length = 100
    resample_interval = 3
    ori_dataset = dict(
        type="RandDataset",
        length=ori_dataset_length,
        example=1,
    )
    ori_dataset = build_from_registry(ori_dataset)
    ori_dataset.pack_flag = np.zeros(ori_dataset_length)
    dataset = dict(
        type="ResampleDataset",
        dataset=ori_dataset,
        with_flag=True,
        with_pack_flag=True,
        resample_interval=resample_interval,
    )

    dataset = build_from_registry(dataset)
    assert len(dataset) == math.ceil(ori_dataset_length / resample_interval)
    assert len(dataset) == len(dataset.flag)
    assert len(dataset) == len(dataset.pack_flag)


@pytest.mark.parametrize(
    "with_flag, with_pack_flag, record_index",
    [(False, False, False), (True, True, True)],
)
def test_concat_dataset(with_flag, with_pack_flag, record_index):
    dataset_length = 1
    dataset1_example = {"label": 1}
    dataset2_example = {"label": 2}
    dataset1 = dict(
        type="RandDataset",
        length=dataset_length,
        example=dataset1_example,
    )
    dataset1 = build_from_registry(dataset1)
    dataset1.flag = np.zeros(len(dataset1))
    dataset1.pack_flag = np.zeros(len(dataset1))
    for i in range(len(dataset1)):
        data = dataset1.__getitem__(i)
        dataset1.flag[i] = data["label"]

    dataset2 = dict(
        type="RandDataset",
        length=dataset_length,
        example=dataset2_example,
    )
    dataset2 = build_from_registry(dataset2)
    dataset2.flag = np.zeros(len(dataset2))
    dataset2.pack_flag = np.zeros(len(dataset2))
    for i in range(len(dataset2)):
        data = dataset2.__getitem__(i)
        dataset2.flag[i] = data["label"]

    dataset = dict(
        type="ConcatDataset",
        datasets=[dataset1, dataset2],
        with_flag=with_flag,
        with_pack_flag=with_pack_flag,
        record_index=record_index,
    )
    dataset = build_from_registry(dataset)

    assert len(dataset) == len(dataset1) + len(dataset2)
    all_result = []
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        all_result.append(data["label"])
        if with_flag:
            flag = dataset.flag[i]
            assert data["label"] == flag
        if record_index:
            assert data["index"] == i
    if with_pack_flag:
        pack_flag = dataset.pack_flag
        assert len(pack_flag) == len(dataset)
        assert np.unique(pack_flag).shape[0] == 2
    assert dataset1_example["label"] in all_result
    assert dataset2_example["label"] in all_result


def test_compose_dataset():
    dataset_length = 20
    dataset1_example, dataset2_example = 1, 2
    dataset1 = dict(
        type="RandDataset",
        length=dataset_length,
        example=dataset1_example,
    )
    dataset2 = dict(
        type="RandDataset",
        length=dataset_length,
        example=dataset2_example,
    )
    dataset = dict(
        type="ComposeDataset",
        batchsize_list=[2, 2],
        datasets=[dataset1, dataset2],
    )

    dataset = build_from_registry(dataset)

    assert len(dataset) == dataset_length * 2
    all_result = []
    for i in dataset:
        all_result.append(i)
    assert dataset1_example in all_result[0::4]
    assert dataset1_example in all_result[1::4]
    assert dataset2_example in all_result[2::4]
    assert dataset2_example in all_result[3::4]


def test_distributed_compose_dataset():
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:14567",
        world_size=1,
        rank=0,
    )
    dataset_length = 20
    dataset1_example, dataset2_example = 1, 2
    dataset1 = dict(
        type="RandDataset",
        length=dataset_length,
        example=dataset1_example,
    )
    dataset2 = dict(
        type="RandDataset",
        length=dataset_length,
        example=dataset2_example,
    )
    dataset = dict(
        type="DistributedComposeRandomDataset",
        sample_weights=[2, 2],
        datasets=[dataset1, dataset2],
    )

    dataset = build_from_registry(dataset)

    assert len(dataset) == dataset_length * 2
    all_result = []
    for i in dataset:
        all_result.append(i)
    assert dataset1_example in all_result[0::4]
    assert dataset1_example in all_result[1::4]
    assert dataset2_example in all_result[2::4]
    assert dataset2_example in all_result[3::4]
    dist.destroy_process_group()


def test_compose_iterable_dataset():
    dataset_length = 20
    dataset1_example, dataset2_example = 1, 2
    dataset1 = dict(
        type="RandDataset",
        length=dataset_length,
        example=dataset1_example,
    )
    dataset2 = dict(
        type="RandDataset",
        length=dataset_length,
        example=dataset2_example,
    )
    dataset = dict(
        type="ComposeIterableDataset",
        batchsize_list=[1, 2],
        datasets=[dataset1, dataset2],
        multi_sample_output=False,
    )

    dataset = build_from_registry(dataset)

    all_result = []
    num = 0
    for i in dataset:
        num += 1
        if num > dataset_length * 2:
            break
        all_result.append(i)

    assert dataset1_example in all_result[0::3]
    assert dataset2_example in all_result[1::3]
    assert dataset2_example in all_result[2::3]


def data_func(local_rank):
    dataset = ChunkShuffleDataset(
        dataset=SimpleDataset(length=64, start=0),
        chunk_size_in_worker=16,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    data_list = []
    for _, data in enumerate(data_loader):
        data_list.append(data)
    print(data_list)


@pytest.mark.serial_task
def test_chunk_shuffle_dataset():
    launch(
        data_func,
        device_ids=[0, 1],
        args=(),
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
