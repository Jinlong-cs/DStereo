import pytest

from hat.utils.package_helper import check_packages_available
from tests import root
from tests.utils import execute_cmd

# test when torch>=1.13.0
skip = not check_packages_available("torch>=1.13.0", raise_exception=False)


@pytest.mark.skipif(skip, reason="Need torch>=1.13.0")
def test_hbdk4_pipeline():
    script = "tests/intergration_tests/run_hbdk4_pipeline.sh"
    cmd = f"""
    cd {root}
    bash {script}
    """
    execute_cmd(cmd)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
