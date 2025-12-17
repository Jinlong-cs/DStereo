import os

import pytest
import torch

from hat.data.datasets.e2e_dynamic_dataset import DatumParserE2E
from hat.engine.ddp_trainer import launch
from hat.registry import RegistryContext, build_from_registry
from hat.utils.pack_type.lmdb import LmdbReadList
from hat.utils.seed import seed_everything
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_datum_parse_e2e():
    bucket_path = HAT_BUCKET_PATH
    unit_test_data = "unit_test_data/J5FSD/users/xiangyu.li/test_e2e_dynamic_auto3dv"  # noqa

    e2e_dynamic_lmdb_path = os.path.join(bucket_path, unit_test_data, "train")
    if e2e_dynamic_lmdb_path is not None:
        e2e_dynamic_lmdb = LmdbReadList(
            e2e_dynamic_lmdb_path,
            readonly=True,
            map_size=1024 ** 2 * 10,
        )
    data_key = "UT263_20220924_D_20220924-094656_247+41"
    data_bin = e2e_dynamic_lmdb.read(data_key)
    data = DatumParserE2E.parse_from_string(data_bin)
    assert len(data) == 5


dataset = {
    "type": "ANCRankSplitE2EDataset",
    "datasets": [
        {
            "type": "SimpleDataset",
            "length": 16,
            "flag": 1,
            "start": 0,
            "__lazy_build__": True,
        },
        {
            "type": "SimpleDataset",
            "length": 32,
            "flag": 2,
            "start": 16,
            "__lazy_build__": True,
        },
    ],
    "with_flag": True,
    "sub_clip_num": 4,
    "batch_size": 3,
}

data_loader = {
    "type": "ANCRankSplitE2EDataLoader",
    "dataset": dataset,
    "sampler": {
        "type": torch.utils.data.DistributedSampler,
        "dataset": dataset,
        "drop_last": True,
    },
    "batch_size": 3,
    "shuffle": False,
    "num_workers": 2,
    "pin_memory": False,
    "drop_last": True,
}


def data_func(local_rank):
    with RegistryContext():
        dataloader = build_from_registry(data_loader)

    data_history = None
    for i, data in enumerate(dataloader):  # noqa
        if i % 4 != 0:
            assert ((data - data_history) == 1).all()
        assert (data < 16).all() or (data >= 16).all()
        data_history = data
        pass


@pytest.mark.serial_task
def test_worker_split_dataset():
    seed_everything(os.getpid())
    launch(
        data_func,
        device_ids=[0, 1],
        args=(),
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
