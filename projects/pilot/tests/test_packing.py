import glob
import os
import subprocess

import pytest
from filelock import FileLock

from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


def bucket_exist(path):
    """If user does not have permission of some horzion buckets,
    like HDLTAlgorithm, it will raise PermissionError. This function
    will catch this error.
    """
    try:
        common_exists = os.path.exists(path) and len(os.listdir(path)) > 0
    except PermissionError:
        common_exists = False
    return common_exists


ROOT_PATH = os.path.split(os.path.realpath(__file__))[0]
HAT_PATH = f"{ROOT_PATH}/../../../"


# step 1 prepare data
@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS,
    reason="requiring HAT_BUCKET bucket",
)
@pytest.fixture(scope="module")
def prepare_data():

    root = f"{HAT_BUCKET_PATH}/users/zihan.qiu/jenkins_test_pack_raw_data"

    cmd = f"""
    echo prepare data
    cd {HAT_PATH}/projects/pilot/pack_tools/
    if [ -d data ]; then
        rm -r data
    fi
    mkdir -p data/train
    ln -s {root}/* data/train/

    """
    with FileLock("prepare_data.lock"):
        subprocess.check_call(cmd, shell=True)
    return True


packing_configs = glob.glob(f"{HAT_PATH}/projects/pilot/pack_tools/configs/*")
tasks = [cfg_i.split("/")[-1] for cfg_i in packing_configs]

tasks = sorted(tasks)
project_path = f"{HAT_PATH}/projects/pilot"


@pytest.mark.usefixtures("prepare_data")
@pytest.mark.parametrize("task", tasks)
def test_packing(task):
    cmd = f"""
    cd {project_path}/pack_tools/
    pwd
    echo starting {task} packing data
    python3 pack.py --config configs/{task}/train.py --visualize
    """
    subprocess.check_call(cmd, shell=True)
