"""Job Settings."""
import hat

job_name = "gaze_two_head_baseline"
job_password = "hat"

num_machines = 1
num_gpus_per_machine = 8

framework = "pytorch"
task_label = "HAT"
project_id = "PDT2021005"  # default ProjectID, replace with your ProjectID
input_bucket = "HDLTAlgorithm"  # default use HDLTAlgorithm Bucket

priority = 5
version = hat.__version__.split(".dev")[0]
docker_image = (
    "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-%s" % version
)
max_jobtime = 10000  # default 7200 = 5days

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../projects",
    "../../tmp_models",
]
job_list = [
    "python3 tools/prepare_bucket.py --bucket 'HDLTAlgorithm'  --mount --create-link",  # noqa E501
    "pip3 install --pre mxnet-horizon-cu111  -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",  # noqa E501
    "pip3 install albumentations -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",  # noqa E501
    f"hdfs dfs -mkdir hdfs://hobot-bigdata/user/fan.lv/output/hat/gaze_eyeldmk/{job_name}",  # noqa E501
    "python3 tools/train.py --config projects/halo/configs/gaze/vargnet_gaze_cluter.py --step float",  # noqa E501
    "python3 tools/train.py --config projects/halo/configs/gaze/vargnet_gaze_cluter.py --step qat",  # noqa E501
    "python3 tools/train.py --config projects/halo/configs/gaze/vargnet_gaze_cluter.py --step int_infer",  # noqa E501
    f"hdfs dfs -put -f tmp_models/* hdfs://hobot-bigdata/user/fan.lv/output/hat/gaze_eyeldmk/{job_name}",  # noqa E501
]
