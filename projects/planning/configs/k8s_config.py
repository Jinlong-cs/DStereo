"""Job Settings."""
import hat

job_name = "planning_vin_hat"
job_password = "newk8s666"

num_machines = 2
num_gpus_per_machine = 8

framework = "pytorch"
task_label = "Imitation"
project_id = "PDT2021004-PNC"  # default ProjectID, replace with your ProjectID
input_bucket = "SD_Algorithm"  # default Bucket

priority = 5
version = hat.__version__.split(".dev")[0]

docker_image = "docker.hobot.cc/imagesys/hat:imitation_runtime-py3.8-torch1.13.0-cu116-1.3.3"  # noqa
max_jobtime = 20160  # default 20160 = 14days, 60 for debug
job_type = "debug"

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../projects/planning",
    "url2IP.py",
]
job_list = []
