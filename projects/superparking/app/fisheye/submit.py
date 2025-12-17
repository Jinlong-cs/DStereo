from common import job_name, num_gpus_per_machine, num_machines, project_id

# you must choose cu111 if want to train full fisheyemultitask!
# mxnet cannot be installed for cu116, so you can only train stage-one tasks !
use_docker = "cu111"  # for gpu >= 3090
dockers = dict(
    cu111="docker.hobot.cc/imagesys/hat:runtime-py3.8-torch1.10.2-cu111-1.2.1-SP-FE",
    cu116="docker.hobot.cc/imagesys/hat:runtime-py3.8-torch1.13.0-cu116_SP_FE_MT",
)
docker = dockers[use_docker]

cfg_file = "projects/superparking/app/fisheye/multitask.py"
job_list = [
    f"python3 -W ignore tools/train.py --config {cfg_file} --stage float",
    f"python3 -W ignore tools/train.py --config {cfg_file} --stage qat",
]

k8s_config = dict(
    job_name=job_name,
    dag_name=job_name,
    job_password="newk8s666",
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework="pytorch",
    task_label="fisheye_multitask",
    project_id=project_id,
    input_bucket="SuperParking,mono",
    # priority description:
    # 3: common experiment job, default
    # 4: model publish job
    # 5: data-related job, used by model publish
    priority=5,
    docker_image=docker,
    # default 7200 = 5days
    max_jobtime=12000,
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        "../../hat",
        "../../tools",
        "../../projects",
        "url2IP.py",
        # "../../Makefile",
    ],
    job_list=job_list,
)
