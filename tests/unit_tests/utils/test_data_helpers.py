import torch

from hat.utils.data_helpers import get_dataloader_length, has_len_func
from tests.unit_tests.base import ToyIterableDataset


def test_get_len():
    ds = torch.utils.data.DataLoader(range(256), batch_size=4)
    assert get_dataloader_length(ds) == 64
    ds = torch.utils.data.DataLoader(ToyIterableDataset(), batch_size=8)
    assert get_dataloader_length(ds) == float("inf")


def test_has_len():
    ds = torch.utils.data.DataLoader(range(256), batch_size=4)
    assert has_len_func(ds)
    ds = torch.utils.data.DataLoader(ToyIterableDataset(), batch_size=8)
    assert not has_len_func(ds)
