import os

import pytest

from hat.utils.setup_env import setup_args_env, setup_hat_env


@pytest.mark.parametrize(
    ["step", "pipeline_test"],
    [
        pytest.param(
            "float",
            True,
        ),
        pytest.param("qat", False),
    ],
)
def test_setup_hat_env(step, pipeline_test):
    setup_hat_env(step, pipeline_test)
    assert os.environ.get("HAT_TRAINING_STEP") == step
    assert os.environ.get("HAT_PIPELINE_TEST") == str(int(pipeline_test))
    assert os.environ.get("NCCL_ASYNC_ERROR_HANDLING") == "1"


def test_setup_args_env():
    args_env = ["sample-env1=xxx1", "sample-env2", "xxx2", "sample_env3=xxx3"]
    setup_args_env(args_env)
    assert os.environ.get("SAMPLE_ENV1") == "xxx1"
    assert os.environ.get("SAMPLE_ENV2") == "xxx2"
    assert os.environ.get("SAMPLE_ENV3") == "xxx3"
