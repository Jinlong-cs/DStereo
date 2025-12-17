import os

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from hat.callbacks.freeze_bn import FreezeBNStatistics
from hat.registry import build_from_registry
from hat.utils.config import Config
from hat.utils.model_helpers import get_binding_module
from tests import root
from tests.data.toy_modules import *  # noqa: F403,F401


def _load_cfg():
    cfg_file = os.path.join(root, "tests/data/resnet18_with_deploy_model.py")
    cfg = Config.fromfile(cfg_file)
    return cfg


def test_freeze_bn():

    # build model
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    model = torch.nn.DataParallel(model, device_ids=[0])

    # build FreezebnModule
    callback = FreezeBNStatistics()

    model.train()
    callback.on_epoch_begin(model=model)
    model = get_binding_module(model)
    for _, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            assert not m.training
