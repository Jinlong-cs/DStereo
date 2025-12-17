import os

import torch

from hat.registry import build_from_registry
from hat.utils.config import Config
from .utils import config_names


def test_build_dataloader():
    os.system("ln -s tmp_data data")
    for config in config_names:
        print("test build dataloader: " + config)
        config_path = "examples/" + config
        assert os.path.exists(config_path)
        cfg = Config.fromfile(config_path)

        if "train_dataloader" in cfg:
            if "sampler" in cfg.train_dataloader and cfg.train_dataloader[
                "sampler"
            ]["type"] == (torch.utils.data.DistributedSampler):
                cfg.train_dataloader["sampler"] = None
            if "shuffle" in cfg.train_dataloader:
                cfg.train_dataloader["shuffle"] = False
            train_dataloader = build_from_registry(cfg.train_dataloader)
            assert train_dataloader is not None

            train_dataloader = iter(train_dataloader)
            data = next(train_dataloader)
            assert data is not None

        if "val_dataloader" in cfg:
            val_dataloader = build_from_registry(cfg.val_dataloader)
            assert val_dataloader is not None

            val_dataloader = iter(val_dataloader)
            data = next(val_dataloader)
            assert data is not None
    os.system("rm data")
