from hat.registry import OBJECT_REGISTRY

__all__ = [
    "FixedDataset",
]


@OBJECT_REGISTRY.register
class FixedDataset(object):
    """Save and reuse the dataset results.

    This dataset can be used as a wrapper of all dataset,
    the real dataset only execute once.

    Args:
        dataset: The actual dataset used to obtain data once

    """

    def __init__(
        self,
        dataset,
        fixed_idx=-1,
    ):
        self.dataset = dataset
        self.fixed_output = None
        self.fixed_idx = fixed_idx
        assert self.fixed_idx < len(self.dataset), (
            f"Idx should not exceed "
            f"{len(self.dataset) - 1}, but given {self.fixed_idx}"
        )

    def __getitem__(self, idx):
        if self.fixed_idx != -1:
            return self.dataset[self.fixed_idx]
        if self.fixed_output is None:
            self.fixed_output = self.dataset[idx]
        return self.fixed_output

    def __len__(self):
        return len(self.dataset)
