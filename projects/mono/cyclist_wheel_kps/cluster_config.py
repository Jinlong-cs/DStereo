"""Job Settings."""
# job_name = "cyckps_debug_0824_loadpretrain_poly"  # same as cfg_file
job_name = "cyckps_v002"
cfg_file = "projects/mono/cyclis_wheel_kps/example.py"
job_password = "hat"
DEBUG = False
num_machines = 1
num_gpus_per_machine = 2 if DEBUG else 8
# num_gpus_per_machine = 2
input_bucket = "SD_Algorithm"
framework = "pytorch"
task_label = "HAT"
project_id = "PDT20220004"  # default ProjectID, replace with your ProjectID
output_bucket = "mono"
# queue = "share-debug-queue-tcloud"
queue = "project-3090-mono-tcloud"
priority = 5
docker_image = (
    "docker.hobot.cc/imagesys/hat:fsd_multitask-cu11-20230823-white-box-v3.9"
)
max_jobtime = 60 if DEBUG else 20000  # default 7200 = 5days
# max_jobtime = 120

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "./hat",
    "./tools",
    # "./configs",
    "./projects",
]  # noqa E501

if num_machines > 1:
    folder_list.append("url2IP.py")


# model_dir = "/horizon-bucket/interaction/models/yisu.zhou/"
job_list = [
    f"python3 -W ignore tools/train.py --config {cfg_file} --stage float",  # noqa E501
    f"export HAT_MARCH=bayes && python3 -W ignore tools/train.py --config {cfg_file} --stage qat",  # noqa E501
    f"export HAT_MARCH=bayes && python3 -W ignore tools/train.py --config {cfg_file} --stage int_infer",  # noqa E501
    f"export HAT_MARCH=bernoulli2 && python3 -W ignore tools/train.py --config {cfg_file} --stage qat",  # noqa E501
    f"export HAT_MARCH=bernoulli2 && python3 -W ignore tools/train.py --config {cfg_file} --stage int_infer",  # noqa E501
    f"export HAT_MARCH=bernoulli && python3 -W ignore tools/train.py --config {cfg_file} --stage qat",  # noqa E501
    f"export HAT_MARCH=bernoulli && python3 -W ignore tools/train.py --config {cfg_file} --stage int_infer",  # noqa E501
]

custom_cmds_before_job_list = [
    "ln -s /job_data/ ${WORKING_PATH}/tmp_models",
    "cd ${WORKING_PATH}",
    "export PYTHONUNBUFFERED=0",
    f"cp  {cfg_file} /job_data/",
]
