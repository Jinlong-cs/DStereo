import pytest

from tests import root
from tests.utils import execute_cmd


def test_resnet18_fx_float_qat_int_pipeline():
    script = (
        "tests/intergration_tests/classification/run_resnet18_fx.sh"  # noqa
    )
    cmd = f"""
    cd {root}
    bash {script}
    """
    execute_cmd(cmd)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
