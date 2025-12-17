"""Job Settings."""
import hat

job_name = "torch-hat-k8s-example"
job_password = "newk8s666"

num_machines = 1
num_gpus_per_machine = 4

framework = "pytorch"
task_label = "HAT"
project_id = "TD20230010"  # default ProjectID, replace with your ProjectID
input_bucket = "NeuralISP"  # default Bucket

priority = 5
version = hat.__version__.split(".dev")[0]
docker_image = (
    "docker.hobot.cc/imagesys/hat-isp:runtime-py3.8-torch1.10.2-cu111-%s"
    % version
)
# docker_image = (
#     "docker.hobot.cc/imagesys/isp:runtime-py3.8-torch1.10.2-cu111-1.3.3-crop-roi"
# )

max_jobtime = 7200  # default 7200 = 5days

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../projects",
]
job_list = []
