import pytest

from tests import root
from tests.utils import execute_cmd


def test_toy_mul_float_qat_int_pipeline():
    script = "tests/intergration_tests/run_toy_multitask.sh"
    cmd = f"""
    cd {root}
    bash {script}
    """
    execute_cmd(cmd)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
