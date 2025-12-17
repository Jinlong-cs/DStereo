import os
import re
import time

import pytest
import torch

from hat.engine.launcher import get_launcher
from hat.registry import build_from_registry
from hat.utils.apply_func import (
    _as_list,
    flatten,
    regroup,
    to_flat_ordered_dict,
)
from hat.utils.config import Config
from hat.utils.seed import seed_everything
from tests import root
from tests.data.toy_modules import *  # noqa: F403,F401

try:
    import hatbc
except ImportError:
    hatbc = None

TMP_DIR = "/tmp"


def _load_cfg():
    cfg_file = os.path.join(
        root, "tests/data/toy_multitask_graph_model_cfg.py"
    )
    cfg = Config.fromfile(cfg_file)
    return cfg


def _clone_batch(batch, keys=None, task_inputs=None, task_keys=None, gpu=None):
    if keys is None:
        keys = batch.keys()
    else:
        keys = _as_list(keys)

    new_bat = dict()
    for k in keys:
        new_bat[k] = batch[k]
        if task_inputs is not None:
            if task_keys is not None:
                task_keys = _as_list(task_keys)
                for task_key in task_keys:
                    new_bat[task_key] = task_inputs[task_key]
            else:
                new_bat.update(task_inputs)

    flats, fmt = flatten(new_bat)
    flats = list(flats)
    for i in range(len(flats)):
        if isinstance(flats[i], torch.Tensor):
            flats[i] = flats[i].clone()
            if gpu is not None:
                flats[i] = flats[i].cuda(gpu)

    new_bat, flats_idx = regroup(flats, fmt)
    assert len(flats) == flats_idx, flats_idx
    return new_bat


def _run_step(batches, graph_model, is_train, optimizer=None):
    if is_train:
        optimizer.zero_grad()

    loss_re = re.compile("^.*[lL]oss.*")

    for batch in _as_list(batches):
        results = graph_model(*batch)
        if len(batch) > 1:
            out_names = _as_list(batch[1])
            assert len(results) == len(out_names), "%d vs. %d" % (
                len(results),
                len(out_names),
            )

        flat_res = to_flat_ordered_dict(results)

        if is_train:
            losses = []
            for k, v in flat_res.items():
                if loss_re.match(k):
                    losses.append(v.sum())

            total_loss = sum(losses)
            total_loss.backward()
        else:
            for r in flat_res.values():
                assert isinstance(r, torch.Tensor), (
                    "only `torch.Tensor` can be trace by `torch.jit.trace()`"
                    ", but get %s" % type(r)
                )

    if is_train:
        optimizer.step()


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_init_from_cfg():
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    test_model = build_from_registry(cfg["test_model"])  # noqa: F841
    # check params
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # noqa: F841
    assert "ToyBackbone2" not in model.graph.get_children_name(recursive=True)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_all_task_forward_backward_step():
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = cfg["model"]["inputs"]
    task_inputs = cfg["model"]["task_inputs"]
    gpu = 0
    model = model.cuda(gpu)
    model.train()
    out_names = model.output_names
    assert len(out_names) > 0

    # zero_grad -> fw_all -> bw_all -> step
    batches = [
        [
            _clone_batch(
                inputs, keys=["img"], task_inputs=task_inputs, gpu=gpu
            )
        ]  # inputs, refer GraphModel.forward()
    ]
    _run_step(batches, model, is_train=True, optimizer=optimizer)

    # zero_grad -> fw_all -> bw_all -> fw_all -> bw_all -> step
    batches = [
        [_clone_batch(inputs, keys=["img"], task_inputs=task_inputs, gpu=gpu)],
        [
            _clone_batch(
                inputs, keys=["img"], task_inputs=task_inputs, gpu=gpu
            ),
            out_names,
        ],
    ]

    _run_step(batches, model, is_train=True, optimizer=optimizer)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_split_module():
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    model.train()
    common_model, split_model, _a, _b = model.split_module(model.output_names)
    assert isinstance(common_model, torch.nn.Module)
    assert isinstance(split_model, torch.nn.Module)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_one_task_forward_backward_step():
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = cfg["model"]["inputs"]
    task_inputs = cfg["model"]["task_inputs"]
    gpu = 0
    model = model.cuda(gpu)
    model.train()
    out_names = model.output_names
    assert len(out_names) > 0

    # task index: 0, 1, 2 ...
    # zero_grad -> fw_0 -> bw_0 -> fw_1 -> bw_1 ... -> step
    batches = []
    for n in out_names:
        batches.append([_clone_batch(inputs, ["img"], task_inputs, n, gpu), n])

    _run_step(batches, model, is_train=True, optimizer=optimizer)


def _run_partial_task(cfg, model, gpu):
    if isinstance(
        model,
        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
    ):
        graph_model = model.module
    else:
        graph_model = model

    out_names = graph_model.output_names
    assert len(out_names) > 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch = cfg["model"]["inputs"]
    task_inputs = cfg["model"]["task_inputs"]

    # task index: 0, 1, 2 ...
    # zero_grad -> fw_0&1 -> bw_0&1 -> fw_1&2 -> bw_1&2 ... -> step
    batches = []
    for i in range(len(out_names) - 1):
        tasks = out_names[i : i + 2]
        batches.append(
            [
                _clone_batch(
                    batch,
                    ["img"],
                    task_inputs=task_inputs,
                    task_keys=tasks,
                    gpu=gpu,
                ),  # inputs, refer GraphModel.forward()  # noqa
                tasks,  # out_names
            ]
        )

    _run_step(batches, model, is_train=True, optimizer=optimizer)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_part_task_forward_backward_step():
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    gpu = 0
    model = model.cuda(gpu)
    _run_partial_task(cfg, model.train(), gpu)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_model_validation():
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    inputs = cfg["model"]["inputs"]
    task_inputs = cfg["model"]["task_inputs"]
    gpu = 0
    model = model.cuda(gpu)

    # forward_all
    batches = [
        [_clone_batch(inputs, keys=["img"], task_inputs=task_inputs, gpu=gpu)]
    ]
    _run_step(batches, model.eval(), is_train=False, optimizer=None)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_test_model_predict():
    cfg = _load_cfg()
    test_model = build_from_registry(cfg["test_model"])
    inputs = cfg["test_model"]["inputs"]
    gpu = 0
    test_model = test_model.cuda(gpu)

    # forward_all
    batches = [[_clone_batch(inputs, ["img"], gpu=gpu)]]
    _run_step(batches, test_model.eval(), is_train=False, optimizer=None)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_resume_trace_save():
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    test_model = build_from_registry(cfg["test_model"])
    fn_prefix = "%s/%s" % (TMP_DIR, time.time())

    # 1. model save
    state = {
        "state_dict": model.state_dict(),
    }
    model_ckpt_fn = fn_prefix + ".pth.tar"
    torch.save(state, model_ckpt_fn)

    # 2. resume model to init export_mode
    test_inputs = cfg["test_model"]["inputs"]
    model_ckpt = torch.load(model_ckpt_fn, map_location="cpu")
    test_model.load_state_dict(model_ckpt["state_dict"])

    # 3. export_mode trace
    test_model_ckpt_fn = fn_prefix + "-export.pt"
    script_module = torch.jit.trace(
        func=test_model.eval(),
        example_inputs=test_inputs,
    )
    torch.jit.save(script_module, test_model_ckpt_fn)

    # 4. load
    script_module = torch.jit.load(test_model_ckpt_fn, map_location="cpu")
    outs = script_module(test_inputs)  # noqa: F841

    # clean
    os.remove(model_ckpt_fn)
    os.remove(test_model_ckpt_fn)


def _ddp_main(gpu, *args):
    cfg = _load_cfg()
    model = build_from_registry(cfg["model"])
    model = model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True, device_ids=[gpu]
    )
    torch.autograd.set_detect_anomaly(True)
    _run_partial_task(cfg, model.train(), gpu)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.serial_task
def test_ddp_model():
    device_ids = list(range(min(2, torch.cuda.device_count())))
    dist_url = "auto"
    seed_everything(os.getpid())
    launcher = get_launcher("distributed_data_parallel_trainer")
    launcher(_ddp_main, device_ids, dist_url)


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
def test_get_sub_modules():
    cfg = _load_cfg()
    model = build_from_registry(cfg["test_model"])
    # here the heads are: task1, task2, task3
    task1_param = {}
    for n, p in model.named_parameters_by_outname(["task1"]):
        task1_param[n] = p

    task2_param = {}
    for n, p in model.named_parameters_by_outname(["task2"]):
        task2_param[n] = p
    task3_param = {}
    for n, p in model.named_parameters_by_outname(["task3"]):
        task3_param[n] = p

    total_param = {}
    for n, p in model.named_parameters():
        total_param[n] = p

    # check all test param in total param
    for k in task1_param:
        assert k in total_param
    for k in task2_param:
        assert k in total_param
    for k in task3_param:
        assert k in total_param

    # check all total param in task1 or task2 or task3
    for k in total_param:
        assert k in task1_param or k in task2_param or k in task3_param

    # test for buffers
    task1_buffer = {}
    for n, p in model.named_buffers_by_outname(["task1"]):
        task1_buffer[n] = p
    task2_buffer = {}
    for n, p in model.named_buffers_by_outname(["task2"]):
        task2_buffer[n] = p
    task3_buffer = {}
    for n, p in model.named_buffers_by_outname(["task3"]):
        task3_buffer[n] = p

    total_buffer = {}
    for n, p in model.named_buffers():
        total_buffer[n] = p

    # check all test buffer in total buffer
    for k in task1_buffer:
        assert k in total_buffer
    for k in task2_buffer:
        assert k in total_buffer
    for k in task3_buffer:
        assert k in total_buffer

    # check all total buffer in task1 or task2 or task3
    for k in total_buffer:
        assert k in task1_buffer or k in task2_buffer or k in task3_buffer
