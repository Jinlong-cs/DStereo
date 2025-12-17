import os
import time
import uuid

import pytest

from hat.callbacks.aidi_expmodel import (
    ARTIFACT_NAME_FORMAT,
    AIDIExperimentManager,
)
from hat.callbacks.checkpoint import Checkpoint
from hat.callbacks.metric_updater import (
    MetricUpdater,
    update_metric_using_regex,
)
from hat.metrics.metric import EvalMetric
from hat.utils.aidi import get_aidi_client
from tests.unit_tests.base.boring_modules import BoringTrainer

MODEL_NAME = str(int(time.time()))


@pytest.fixture(
    params=[
        [False, False, None],
        [True, True, "v0.0.1"],
    ]
)
def aidi_exp_model_callback(request):
    aidi_exp_model = AIDIExperimentManager(
        model_name=MODEL_NAME,
        save_model="last",
        model_version=request.param[2],
        upload_progressive_checkpoint=request.param[0],
        overwrite_file=request.param[1],
    )
    return aidi_exp_model


class TestLossMetric(EvalMetric):
    def __init__(self, name, **kwargs):
        super(TestLossMetric, self).__init__(name=name, **kwargs)
        self.cnt = 0

    def update(self, preds):
        self.cnt += 1

    def get(self):
        return self.name, self.cnt


class TestMetric(EvalMetric):
    def __init__(self, name, **kwargs):
        super(TestMetric, self).__init__(name=name, **kwargs)
        self.cnt = 0

    def update(self, labels, preds):
        self.cnt += 1

    def get(self):
        return self.name, self.cnt


@pytest.fixture
def checkpoint_callback(tmpdir):
    checkpoint = Checkpoint(
        save_dir=tmpdir,
        name_prefix="float-",
    )

    def pass_end(*args, **kwargs):
        pass

    def save_checkpoint_file(*args, **kwargs):
        with open(
            os.path.join(checkpoint.save_dir, "float-checkpoint-last.pth.tar"),
            "w",
        ) as f:
            f.write("HAT_UT_CHECKPOINT")
            pass

    checkpoint.on_step_end = save_checkpoint_file
    checkpoint.on_epoch_end = pass_end
    checkpoint.on_loop_end = save_checkpoint_file
    checkpoint.on_loop_begin = pass_end

    return checkpoint


@pytest.fixture
def metric_updater():
    metric_updater = MetricUpdater(
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=[  # corresponding to metrics
                dict(label_pattern=None, pred_pattern="^.*loss.*"),
                dict(label_pattern="^.*label.*", pred_pattern="^.*predict.*"),
            ]
        ),
        step_log_freq=1,
        epoch_log_freq=1,
        log_prefix="ToyTask",
    )
    return metric_updater


@pytest.mark.skip(
    reason="test AIDIExperimentManager callback in intergration tests, skip this unit test.",  # noqa E501
)
def test_aidi_experiment(
    checkpoint_callback, aidi_exp_model_callback, metric_updater
):
    os.environ["HAT_TRAINING_STEP"] = "float"
    os.environ["HAT_ENABLE_MODEL_TRACKING"] = "1"

    experiment_name = "HAT_UNIT_TESTS-test_aidi_expmodel_callback"

    client = get_aidi_client()
    if client.experiment.get_experiment(experiment_name) is None:
        client.experiment.create_experiment(
            name=experiment_name,
            project_id="RDS20220011",
            experiment_path="dmpv2://HDLTAlgorithm/aidi_exp/HAT_CICD/",
        )
    assert client.experiment.get_experiment(experiment_name) is not None

    with client.experiment.init(
        experiment_name=experiment_name,
        run_name=f"RUN-hat_unittest_{str(uuid.uuid1())}",
        enabled=True,
    ) as run:
        run.log_runtime("local", config_file=None)

    client.experiment.init_group("group-hat_unittest")

    train_metrics = [TestLossMetric("loss_metric"), TestMetric("pred_metric")]

    callbacks = [
        metric_updater,
        checkpoint_callback,
        aidi_exp_model_callback,
    ]

    trainer = BoringTrainer(callbacks=callbacks, train_metrics=train_metrics)
    trainer.fit()

    training_step = os.getenv("HAT_TRAINING_STEP")
    artifact_name = ARTIFACT_NAME_FORMAT % (
        MODEL_NAME,
        training_step,
        "last",
    )

    # test
    artifact = client.experiment.use_artifact(artifact_name, trace=False)
    assert artifact
    assert artifact.get_file("float-checkpoint-last.pth.tar")

    if aidi_exp_model_callback.model_version is not None:
        artifact_name_version = ARTIFACT_NAME_FORMAT % (
            MODEL_NAME,
            training_step,
            aidi_exp_model_callback.model_version,
        )
        artifact = client.experiment.use_artifact(
            artifact_name_version, trace=True
        )
        assert artifact
        assert artifact.get_file("float-checkpoint-last.pth.tar")

    # delete
    # if client.experiment.get_experiment(experiment_name) is not None:
    #     client.experiment.delete_experiment(experiment_name)
