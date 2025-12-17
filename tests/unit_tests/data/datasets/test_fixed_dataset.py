import pytest

from hat.registry import build_from_registry


@pytest.mark.parametrize("fixed_idx", [-1, 0, 1, 10, 99])
def test_perf_dataset(fixed_idx):
    dataset_length = 100
    perf_dataset = dict(
        type="FixedDataset",
        dataset=dict(
            type="SimpleDataset",
            start=0,
            length=dataset_length,
        ),
        fixed_idx=fixed_idx,
    )
    perf_dataset = build_from_registry(perf_dataset)
    for idx in range(10):
        if fixed_idx != -1:
            assert perf_dataset[idx] == perf_dataset[fixed_idx]
        else:
            assert perf_dataset[idx] == perf_dataset[idx + 1]
