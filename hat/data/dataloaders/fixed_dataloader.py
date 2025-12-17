import math
from collections.abc import Iterable

from hat.registry import OBJECT_REGISTRY

__all__ = ["FixedDataLoader"]


@OBJECT_REGISTRY.register
class FixedDataLoader:
    """
    This DataLoader only get data once and reuse the output.

    Arguments:
        dataloader: actual used dataloader
    """

    def __init__(self, dataloader: Iterable):
        if hasattr(dataloader, "__len__"):
            self.length = len(dataloader)
        else:
            self.length = math.inf
        self.it = iter(dataloader)
        self.get_data_count = 0
        self.batch_size = dataloader.batch_size

    def __iter__(self):
        data = next(self.it)
        while self.get_data_count < self.length:
            self.get_data_count += 1
            yield data
        raise StopIteration("The data obtained exceeds the actual quantity")

    def __len__(self):
        return self.length
