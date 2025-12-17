import os
import random
import shutil

import pytest
import torch

from hat.callbacks.checkpoint import Checkpoint
from hat.utils.hash import get_hash_file_if_hashed_and_local
from tests.unit_tests.base import BoringMetric, BoringTraceModel


@pytest.fixture
def boring_model():
    model = BoringTraceModel()
    return model


@pytest.fixture
def boring_metric():
    metric = BoringMetric()
    return metric


@pytest.fixture(
    params=[
        ["epoch", None],
        ["epoch", "min"],
        ["epoch", "max"],
        ["step", None],
        ["step", "min"],
        ["step", "max"],
    ]
)
def checkpoint_wo_metric(request, tmpdir, boring_model):
    interval_by = request.param[0]
    mode = request.param[1]
    checkpoint = Checkpoint(
        save_dir=os.path.join(tmpdir, "wo_%s_%s" % (interval_by, mode)),
        interval_by=interval_by,
        mode=mode,
        monitor_metric_key="accuracy",
    )
    return checkpoint


@pytest.fixture(
    params=[
        ["epoch", None],
        ["epoch", "min"],
        ["epoch", "max"],
        ["step", None],
        ["step", "min"],
        ["step", "max"],
    ]
)
def checkpoint_with_metric(request, tmpdir, boring_model, boring_metric):
    interval_by = request.param[0]
    mode = request.param[1]
    checkpoint = Checkpoint(
        save_dir=os.path.join(tmpdir, "with_%s_%s" % (interval_by, mode)),
        interval_by=interval_by,
        mode=mode,
        best_refer_metric=boring_metric,
    )
    return checkpoint


@pytest.fixture(
    params=[
        ["epoch", None],
        ["epoch", "min"],
        ["epoch", "max"],
        ["step", None],
        ["step", "min"],
        ["step", "max"],
    ]
)
def checkpoint_using_ema_model(request, tmpdir, boring_model, boring_metric):
    interval_by = request.param[0]
    mode = request.param[1]
    checkpoint = Checkpoint(
        save_dir=os.path.join(tmpdir, "with_%s_%s" % (interval_by, mode)),
        interval_by=interval_by,
        mode=mode,
        best_refer_metric=boring_metric,
    )
    return checkpoint


class TestCheckpointWOMetric(object):
    def test_on_step_end(
        self, checkpoint_wo_metric, boring_model, boring_metric
    ):
        save_dir = checkpoint_wo_metric.save_dir
        interval_by = checkpoint_wo_metric.interval_by
        mode = checkpoint_wo_metric.mode
        if interval_by == "step":
            checkpoint_wo_metric.on_step_end(
                epoch_id=0,
                step_id=0,
                global_step_id=0,
                val_metrics=[boring_metric],
                model=boring_model,
                ema_model=None,
                optimizer=None,
                num_steps=None,
            )
            assert os.path.exists(
                get_hash_file_if_hashed_and_local(
                    os.path.join(save_dir, "checkpoint-step-0.pth.tar")
                )
            )
            if mode is not None:
                assert os.path.exists(
                    os.path.join(save_dir, "checkpoint-best.pth.tar")
                )
            shutil.rmtree(save_dir)

    def test_on_epoch_end(
        self, checkpoint_wo_metric, boring_model, boring_metric
    ):
        save_dir = checkpoint_wo_metric.save_dir
        interval_by = checkpoint_wo_metric.interval_by
        mode = checkpoint_wo_metric.mode
        if interval_by == "epoch":
            checkpoint_wo_metric.on_epoch_end(
                epoch_id=0,
                global_step_id=0,
                model=boring_model,
                ema_model=None,
                optimizer=None,
                num_epochs=None,
                val_metrics=[boring_metric],
            )
            assert os.path.exists(
                get_hash_file_if_hashed_and_local(
                    os.path.join(save_dir, "checkpoint-epoch-0000.pth.tar")
                )
            )
            if mode is not None:
                assert os.path.exists(
                    os.path.join(save_dir, "checkpoint-best.pth.tar")
                )
            shutil.rmtree(save_dir)

    def test_on_loop_end(self, checkpoint_wo_metric, boring_model):
        epoch_id = random.randint(0, 10)
        step_id = random.randint(0, 10000)
        checkpoint_wo_metric.on_loop_end(
            model=boring_model,
            ema_model=None,
            optimizer=None,
            storage=None,
            epoch_id=epoch_id,
            global_step_id=step_id,
        )
        save_dir = checkpoint_wo_metric.save_dir
        save_path = os.path.join(save_dir, "checkpoint-last.pth.tar")
        assert os.path.exists(save_path)
        ckpt = torch.load(save_path)
        assert ckpt["epoch"] == epoch_id
        assert ckpt["step"] == step_id
        shutil.rmtree(save_dir)


class TestCheckpointWithMetric(object):
    def test_on_step_end(self, checkpoint_with_metric, boring_model):
        save_dir = checkpoint_with_metric.save_dir
        interval_by = checkpoint_with_metric.interval_by
        mode = checkpoint_with_metric.mode
        if interval_by == "step":
            checkpoint_with_metric.on_step_end(
                epoch_id=0,
                step_id=0,
                global_step_id=0,
                val_metrics=None,
                model=boring_model,
                ema_model=None,
                optimizer=None,
                num_steps=None,
            )
            assert os.path.exists(
                get_hash_file_if_hashed_and_local(
                    os.path.join(save_dir, "checkpoint-step-0.pth.tar")
                )
            )
            if mode is not None:
                assert os.path.exists(
                    os.path.join(save_dir, "checkpoint-best.pth.tar")
                )
            shutil.rmtree(save_dir)

    def test_on_epoch_end(self, checkpoint_with_metric, boring_model):
        save_dir = checkpoint_with_metric.save_dir
        interval_by = checkpoint_with_metric.interval_by
        mode = checkpoint_with_metric.mode
        if interval_by == "epoch":
            checkpoint_with_metric.on_epoch_end(
                epoch_id=0,
                global_step_id=0,
                model=boring_model,
                ema_model=None,
                optimizer=None,
                num_epochs=None,
                val_metrics=None,
            )
            assert os.path.exists(
                get_hash_file_if_hashed_and_local(
                    os.path.join(save_dir, "checkpoint-epoch-0000.pth.tar")
                )
            )
            if mode is not None:
                assert os.path.exists(
                    os.path.join(save_dir, "checkpoint-best.pth.tar")
                )
            shutil.rmtree(save_dir)

    def test_on_loop_end(self, checkpoint_with_metric, boring_model):
        epoch_id = random.randint(0, 10)
        step_id = random.randint(0, 10000)
        checkpoint_with_metric.on_loop_end(
            model=boring_model,
            ema_model=None,
            optimizer=None,
            storage=None,
            epoch_id=epoch_id,
            global_step_id=step_id,
        )
        save_dir = checkpoint_with_metric.save_dir
        save_path = os.path.join(save_dir, "checkpoint-last.pth.tar")
        assert os.path.exists(save_path)
        ckpt = torch.load(save_path)
        assert ckpt["epoch"] == epoch_id
        assert ckpt["step"] == step_id
        shutil.rmtree(save_dir)


class TestCheckpointUsingEMAModel(object):
    def test_on_step_end(self, checkpoint_using_ema_model, boring_model):
        save_dir = checkpoint_using_ema_model.save_dir
        interval_by = checkpoint_using_ema_model.interval_by
        mode = checkpoint_using_ema_model.mode
        if interval_by == "step":
            checkpoint_using_ema_model.on_step_end(
                epoch_id=0,
                step_id=0,
                global_step_id=0,
                val_metrics=None,
                model=boring_model,
                ema_model=boring_model,
                optimizer=None,
                num_steps=None,
            )
            assert os.path.exists(
                get_hash_file_if_hashed_and_local(
                    os.path.join(save_dir, "checkpoint-step-0.pth.tar")
                )
            )
            if mode is not None:
                assert os.path.exists(
                    os.path.join(save_dir, "checkpoint-best.pth.tar")
                )
            shutil.rmtree(save_dir)

    def test_on_epoch_end(self, checkpoint_using_ema_model, boring_model):
        save_dir = checkpoint_using_ema_model.save_dir
        interval_by = checkpoint_using_ema_model.interval_by
        mode = checkpoint_using_ema_model.mode
        if interval_by == "epoch":
            checkpoint_using_ema_model.on_epoch_end(
                epoch_id=0,
                global_step_id=0,
                model=boring_model,
                ema_model=boring_model,
                optimizer=None,
                num_epochs=None,
                val_metrics=None,
            )
            assert os.path.exists(
                get_hash_file_if_hashed_and_local(
                    os.path.join(save_dir, "checkpoint-epoch-0000.pth.tar")
                )
            )
            if mode is not None:
                assert os.path.exists(
                    os.path.join(save_dir, "checkpoint-best.pth.tar")
                )
            shutil.rmtree(save_dir)

    def test_on_loop_end(self, checkpoint_using_ema_model, boring_model):
        epoch_id = random.randint(0, 10)
        step_id = random.randint(0, 10000)
        checkpoint_using_ema_model.on_loop_end(
            model=None,
            ema_model=boring_model,
            optimizer=None,
            storage=None,
            epoch_id=epoch_id,
            global_step_id=step_id,
        )
        save_dir = checkpoint_using_ema_model.save_dir
        save_path = os.path.join(save_dir, "checkpoint-last.pth.tar")
        assert os.path.exists(save_path)
        ckpt = torch.load(save_path)
        assert ckpt["epoch"] == epoch_id
        assert ckpt["step"] == step_id
        shutil.rmtree(save_dir)
