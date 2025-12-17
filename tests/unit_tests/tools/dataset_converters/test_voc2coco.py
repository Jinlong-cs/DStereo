from tests import root
from tests.utils import execute_cmd


def test_voc2coco():
    script = "tests/unit_tests/tools/dataset_converters/run_voc2coco.sh"
    cmd = f"""
    cd {root}
    bash {script}
    """
    execute_cmd(cmd)
