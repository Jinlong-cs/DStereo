"""Job Settings."""

import hat

job_name = "torch-hat-k8s-example"
job_password = "newk8s666"

num_machines = 1
num_gpus_per_machine = 8

framework = "pytorch"
task_label = "HAT"
project_id = "RDS20220011"  # default ProjectID, replace with your ProjectID
input_bucket = "HDLTAlgorithm"  # default use HDLTAlgorithm Bucket

priority = 5
version = hat.__version__.split(".dev")[0]
docker_image = (
    "docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-%s"
    % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-%s" % version
)
max_jobtime = 10000  # default 7200 = 5days

# launcher only for multi-machines
launcher = "torch"

# elastic setting
job_max_restarts = 3
job_restart_mode = "Always"
elastic_pattern_files = ["default_elastic_patterns.yaml"]
# launcher = "torchrun"  # elastic launch method

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../examples",
    "url2IP.py",
    "ssh_launcher.py",
    "watch_dog.py",
    "elastic_launcher.py",
    "default_elastic_patterns.yaml",
]

job_list = [
    "python3 tools/train.py --stage float --config examples/classification/mobilenetv1.py",  # noqa E501
    "python3 tools/predict.py --stage float --config examples/classification/mobilenetv1.py",  # noqa E501
    "python3 tools/deploy/model_checker.py --config examples/classification/mobilenetv1.py",  # noqa E501
    "python3 tools/train.py --stage calibration --config examples/classification/mobilenetv1.py",  # noqa E501
    "python3 tools/train.py --stage qat --config examples/classification/mobilenetv1.py",  # noqa E501
    "python3 tools/predict.py --stage qat --config examples/classification/mobilenetv1.py",  # noqa E501
    "python3 tools/predict.py --stage int_infer --config examples/classification/mobilenetv1.py",  # noqa E501
]

custom_cmds_before_job_list = [
    "ln -s /horizon-bucket/%s/data/pack_data ${WORKING_PATH}/tmp_data" % input_bucket,
    "ln -s /horizon-bucket/%s/models/bayes_release_models ${WORKING_PATH}/tmp_pretrained_models"  # noqa
    % input_bucket,
    "mkdir -p /job_data/models",
    "ln -s /job_data/models ${WORKING_PATH}/tmp_models",
]
