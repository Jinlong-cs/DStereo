import os

import horizon_plugin_pytorch as horizon
import pytest
import torch
from horizon_plugin_pytorch.qat_mode import QATMode, set_qat_mode

from hat.callbacks.online_model_trick import FreezeModule, FuseBN
from hat.registry import build_from_registry
from hat.utils.config import Config
from hat.utils.model_helpers import has_normalization
from tests import root
from tests.data.toy_modules import *  # noqa: F403,F401

try:
    import hatbc
except ImportError:
    hatbc = None


def _load_cfg():
    cfg_file = os.path.join(root, "tests/data/toy_graph_model_cfg.py")
    cfg = Config.fromfile(cfg_file)
    return cfg


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.parametrize(
    ["update_by", "freeze_step_id", "freeze_nodes", "elastic_resume"],
    [
        pytest.param("step", [1, 2], [["backbone"], ["task1"]], False),
        pytest.param("epoch", [2], [["backbone", "task1", "task2"]], False),
        pytest.param("step", [1, 2], [["backbone"], ["task1"]], True),
        pytest.param("epoch", [2], [["backbone", "task1", "task2"]], True),
    ],
)
def test_freeze_module(
    update_by, freeze_step_id, freeze_nodes, elastic_resume
):
    # build model
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    model = torch.nn.DataParallel(model, device_ids=[0])

    if elastic_resume:
        # set env for elastic resume
        os.environ["USE_ELASTIC"] = "1"
        os.environ["ELASTIC_NEED_RESUME"] = "1"
        start_idx = freeze_step_id[-1]
    else:
        start_idx = 0

    # build FreezeModule
    callback = FreezeModule(freeze_nodes, freeze_step_id, update_by)

    model.train()
    for i in range(start_idx, max(freeze_step_id) + 1):
        callback.on_loop_begin(model=model)
        callback.on_epoch_begin(epoch_id=i, model=model)
        callback.on_step_begin(step_id=i, global_step_id=i, model=model)
        for ii, nodes in zip(freeze_step_id, freeze_nodes):
            if i < ii:
                for node in nodes:
                    assert getattr(model.module, node).training
                    for param in getattr(model.module, node).parameters():
                        assert param.requires_grad
            else:
                for node in nodes:
                    assert not getattr(model.module, node).training
                    for param in getattr(model.module, node).parameters():
                        assert not param.requires_grad

    # build FreezeModule with only_batchnorm=True
    callback = FreezeModule(
        freeze_nodes, freeze_step_id, update_by, only_batchnorm=True
    )

    model.train()
    for i in range(start_idx, max(freeze_step_id) + 1):
        callback.on_loop_begin(model=model)
        callback.on_epoch_begin(epoch_id=i, model=model)
        callback.on_step_begin(step_id=i, global_step_id=i, model=model)
        for ii, nodes in zip(freeze_step_id, freeze_nodes):
            if i < ii:
                for node in nodes:
                    assert getattr(model.module, node).training
            else:
                for node in nodes:
                    assert not getattr(model.module, node).training

    if elastic_resume:
        # delete env
        del os.environ["USE_ELASTIC"]
        del os.environ["ELASTIC_NEED_RESUME"]


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.parametrize("update_by", ["step", "epoch"])
@pytest.mark.parametrize(
    "nodes, step_id, elastic_resume",
    [
        ([["backbone"], ["task1"]], [1, 2], False),
        ([["backbone", "task1", "task2"]], [2], False),
        ([["backbone"], ["task1"]], [1, 2], True),
        ([["backbone", "task1", "task2"]], [2], True),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "qat_mode", [QATMode.FuseBN, QATMode.WithBN, QATMode.WithBNReverseFold]
)
def test_fuse_bn(update_by, step_id, nodes, elastic_resume, inplace, qat_mode):
    # build model
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    model = torch.nn.DataParallel(model, device_ids=[0])

    if elastic_resume:
        # set env for elastic resume
        os.environ["USE_ELASTIC"] = "1"
        os.environ["ELASTIC_NEED_RESUME"] = "1"
        start_idx = step_id[-1]
    else:
        start_idx = 0

    # build FuseBN
    callback = FuseBN(nodes, step_id, update_by, inplace=inplace)
    if qat_mode != QATMode.FuseBN:
        model.qconfig = horizon.quantization.get_default_qat_qconfig()
        set_qat_mode(qat_mode)
        model.module.fuse_model()
        horizon.quantization.prepare_qat(model, inplace=True)
    model.train()

    for i in range(start_idx, max(step_id) + 1):
        callback.on_loop_begin(model=model)
        callback.on_epoch_begin(epoch_id=i, model=model)
        callback.on_step_begin(step_id=i, global_step_id=i, model=model)

        for ii, _nodes in zip(step_id, nodes):
            if i < ii:
                for node in _nodes:
                    assert has_normalization(
                        getattr(model.module, node), check_list=["bn"]
                    )
            else:
                for node in _nodes:
                    assert not has_normalization(
                        getattr(model.module, node), check_list=["bn"]
                    )

    set_qat_mode("fuse_bn")

    if elastic_resume:
        # delete env
        del os.environ["USE_ELASTIC"]
        del os.environ["ELASTIC_NEED_RESUME"]
