"""Job Settings."""
import hat

job_name = ""
job_password = "hat"

num_machines = 1
num_gpus_per_machine = 8
framework = "pytorch"
task_label = "HAT"
project_id = "PDT2021005"  # default ProjectID, replace with your ProjectID
input_bucket = "HDLTAlgorithm,interaction"
output_bucket = "MultiMode_2"  # default use HDLTAlgorithm Bucket

priority = 5
version = hat.__version__.split(".dev")[0]

docker_image = (
    "docker.hobot.cc/imagesys/face3d:centos7.6-gcc7.3-py3.6-cuda11.1-face3d"
)
# docker_image = "docker.hobot.cc/imagesys/imagesys:face_mtl"
max_jobtime = 10000  # default 7200 = 5days
# max_jobtime = 60

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../configs",
    "../../projects",
    "url2IP.py",
    "ssh_launcher.py",
]
job_list = [
    # epoch
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/pupil_segmentation/pupil_segmentation_train.py --stage float",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/pupil_segmentation/pupil_segmentation_train.py --stage qat",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/pupil_segmentation/pupil_segmentation_train.py --stage int_infer",  # noqa E501
]
