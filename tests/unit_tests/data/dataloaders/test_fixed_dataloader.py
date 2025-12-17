import torch

from hat.registry import build_from_registry


def test_fixed_dataloaer():
    dataset_length = 100
    dataset = dict(
        type="SimpleDataset",
        start=0,
        length=dataset_length,
    )
    data_loader = dict(
        type="FixedDataLoader",
        dataloader=dict(
            type=torch.utils.data.DataLoader,
            dataset=dataset,
            batch_size=1,
        ),
    )
    dataloader = build_from_registry(data_loader)
    it = iter(dataloader)
    data0 = next(it)
    for _i in range(dataset_length - 1):
        data1 = next(it)
        assert torch.equal(data0, data1)
        data0 = data1
