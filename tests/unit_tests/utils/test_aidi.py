import os
import uuid

import pytest

from hat.utils import aidi as aidi_utils

try:
    import aidisdk
except ImportError:
    aidisdk = None


@pytest.mark.skipif(aidisdk is None, reason="`need aidisdk")
def test_get_aidi_client():
    assert aidi_utils.get_aidi_client()


@pytest.mark.skipif(aidisdk is None, reason="`need aidisdk")
def test_get_aidi_token():
    assert aidi_utils.get_aidi_token()


@pytest.mark.skipif(aidisdk is None, reason="`need aidisdk")
def test_aidi_experiment_logger(tmpdir):

    tmp_file = os.path.join(tmpdir, "artifact_file.txt")
    with open(tmp_file, "w") as f:
        f.write("hat_ut")

    tracking = True
    os.environ["HAT_ENABLE_MODEL_TRACKING"] = "1"

    aidi_logger = aidi_utils.AIDIExperimentLogger()

    experiment_name = "HAT_UNIT_TESTS-test_aidi_utils"

    aidi_logger.init(
        experiment_name=experiment_name,
        project_id="RDS20220011",
        experiment_path="dmpv2://HDLTAlgorithm/aidi_exp/HAT_CICD/",
        enable_tracking=tracking,
        run_name=f"RUN-hat_unittest_{str(uuid.uuid1())}",
        runtime="local",
        config_file=None,
    )

    aidi_logger.init_group("hat_utils_ut-group")
    aidi_logger.log_config({"enable_tracking": tracking})
    aidi_logger.log_artifact(
        artifact_name="hat_utils_ut_artifact-float",
        artifact_type="model",
        artifact_aliases=["float-latest", "last"],
        artifact_tags=["latest"],
        files=[tmp_file],
    )

    assert aidi_logger.enabled_tracking() == tracking

    if tracking:
        artifact_url = "aidi_artifact://hat_utils_ut_artifact/float/last/artifact_file.txt"  # noqa E501
        artifact_file = aidi_logger.download_checkpoint_from_artifact(
            artifact_path=artifact_url,
        )
        assert artifact_file is not None

    # delete
    # if client.experiment.get_experiment(experiment_name) is not None:
    # client.experiment.delete_experiment(experiment_name)
