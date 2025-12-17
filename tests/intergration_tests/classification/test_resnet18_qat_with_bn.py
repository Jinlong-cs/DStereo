import pytest

from tests import root
from tests.utils import execute_cmd


def test_resnet18_qat_with_bn():
    script = "tests/intergration_tests/classification/run_resnet18_qat_with_bn.sh"  # noqa
    cmd = f"""
    cd {root}
    bash {script}
    """
    execute_cmd(cmd)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
