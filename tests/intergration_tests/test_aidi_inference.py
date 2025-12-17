import pytest

from hat.utils.aidi import get_aidi_client
from tests import root
from tests.utils import execute_cmd


def test_aidi_inference():
    version = "v0.0.1"
    script = f"tests/intergration_tests/run_aidi_inference.sh {version}"
    cmd = f"""
    cd {root}
    bash {script}
    """
    execute_cmd(cmd)

    client = get_aidi_client()
    # TODO aidisdk do not support delete model card
    if client.model_registry.exist(
        name=f"test_aidi_model_inference:{version}"
    ):
        client.model_registry.delete(
            name=f"test_aidi_model_inference:{version}"
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
