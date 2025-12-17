import os
import sys

import horizon_plugin_pytorch as horizon

# HAT root path
root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, root)
import hat  # noqa


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


BasicAlgorithm_BUCKET_PATH = "/horizon-bucket/BasicAlgorithm"
BasicAlgorithm_BUCKET_EXISTS = bucket_exist(BasicAlgorithm_BUCKET_PATH)

AIDI_PUBLIC_DATA_PATH = os.path.join(root, "tmp_bucket/aidi_public_data")
AIDI_PUBLIC_DATA_EXISTS = bucket_exist(AIDI_PUBLIC_DATA_PATH)

J5FSD_BUCKET_PATH = "/horizon-bucket/J5FSD"
J5FSD_BUCKET_EXISTS = bucket_exist(J5FSD_BUCKET_PATH)

J5FSD_2_BUCKET_PATH = "/horizon-bucket/J5FSD_2"
J5FSD_2_BUCKET_EXISTS = bucket_exist(J5FSD_2_BUCKET_PATH)

SD_ALGO_BUCKET_PATH = "/horizon-bucket/SD_Algorithm"
SD_ALGO_BUCKET_EXISTS = bucket_exist(SD_ALGO_BUCKET_PATH)

MATRIX_BUCKET_PATH = "/horizon-bucket/matrix"
MATRIX_BUCKET_EXISTS = bucket_exist(MATRIX_BUCKET_PATH)

AUTO_JENKINS_TEST_BUCKET_PATH = "/horizon-bucket/auto_jenkins_test"
AUTO_JENKINS_TEST_BUCKET_EXISTS = bucket_exist(AUTO_JENKINS_TEST_BUCKET_PATH)

AIDI_PUBLIC_DATA_BUCKET_PATH = "/horizon-bucket/aidi_public_data"
AIDI_PUBLIC_DATA_BUCKET_EXISTS = bucket_exist(AIDI_PUBLIC_DATA_BUCKET_PATH)

HAT_BUCKET_PATH = (
    "/home/cidata"
    if os.path.exists("/home/cidata")
    else "/horizon-bucket/HDLTAlgorithm"
)
HAT_BUCKET_URL_PATH = (
    "/home/cidata"
    if os.path.exists("/home/cidata")
    else "dmpv2://HDLTAlgorithm"
)

HAT_BUCKET_EXISTS = bucket_exist(HAT_BUCKET_PATH)
HAT_BUCKET_URL_EXISTS = bucket_exist(HAT_BUCKET_URL_PATH)

SD_AlGORITHM_BUCKET_PATH = "/horizon-bucket/SD_Algorithm"
SD_AlGORITHM_BUCKET_EXISTS = bucket_exist(SD_AlGORITHM_BUCKET_PATH)

# default march of plugin is BAYES.
horizon.march.set_march(horizon.march.March.BAYES)
horizon.quantization.set_qat_mode("fuse_bn")

MONO_BUCKET_PATH = "/horizon-bucket/mono_jfs"


def setup_tests_env():

    env_keys = [
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENCV_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ]
    for k in env_keys:
        os.environ[k] = "12"


# speedup on AIDI_CI
setup_tests_env()
