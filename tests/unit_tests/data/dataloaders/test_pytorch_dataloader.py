import pytest
import torch

from hat.registry import build_from_registry


def test_pytorch_dataloader():

    dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="RandDataset",
            length=16,
            example=torch.randn((3, 32, 32)),
            clone=True,
        ),
        sampler=dict(
            type=torch.utils.data.DistributedSampler, num_replicas=1, rank=0
        ),
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    dataloader = build_from_registry(dataloader)
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    dataiter = iter(dataloader)
    data = next(dataiter)
    assert data.shape == (2, 3, 32, 32)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
