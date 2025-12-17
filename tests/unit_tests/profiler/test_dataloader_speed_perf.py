import torch

from hat.data.dataloaders.multitask_loader import MultitaskLoader
from hat.profiler.dataloader_speed_perf import DataloaderSpeedPerf


def test_data_loader_speed_perf():
    loaders = {
        "person": torch.utils.data.DataLoader(range(256), batch_size=4),
        "vehicle": torch.utils.data.DataLoader(range(256), batch_size=8),
        "semseg": torch.utils.data.DataLoader(range(256), batch_size=8),
        "real3d": torch.utils.data.DataLoader(range(256), batch_size=4),
    }
    loader = MultitaskLoader(loaders)
    speed_perf = DataloaderSpeedPerf(loader, iter_nums=2000, frequent=1)
    speed_perf.run()
