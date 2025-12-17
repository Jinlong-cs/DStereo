"""Job Settings."""
import importlib.util

# 给定的文件路径
cfg_file = "projects/halo/cv/configs/face3d/gridsample_example.py"
# 获取文件名（不含扩展名）
module_name = cfg_file.split("/")[-1].split(".")[0]
# 使用importlib.util模块创建一个新的spec
spec = importlib.util.spec_from_file_location(module_name, cfg_file)
# 使用新的spec来导入模块
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
# 现在你可以使用导入的模块进行后续操作
job_name = cfg.task_name
job_password = "hat"
DEBUG = cfg.DEBUG
num_machines = 1
num_gpus_per_machine = 2 if DEBUG else 8

framework = "pytorch"
task_label = "HAT"
project_id = "TD20220003"  # default ProjectID, replace with your ProjectID
output_bucket = "interaction,HDLTAlgorithm"

priority = 5
docker_image = "docker.hobot.cc/imagesys/hat:halo-cv"
max_jobtime = 60 if DEBUG else 10000  # default 7200 = 5days

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "hat",
    "tools",
    "configs",
    "projects",
    # "pretrain_models",
]
model_dir = "/horizon-bucket/interaction/models/yuhao.dou/"
job_list = [
    "cd ${WORKING_PATH}",
    "rm -rf ~//.cache/torch_extensions/py38_cu111/nvdiffrast_plugin/lock",
    # f"python3 -W ignore tools/train.py --config {cfg_file} --stage pretrain",
    # f"cp -r tmp_models/* {model_dir}",
    f"python3 -W ignore tools/train.py --config {cfg_file} --stage float",
    f"cp -r tmp_models/* {model_dir}",
    f"python3 -W ignore tools/predict.py --config {cfg_file} --stage float",
]

custom_cmds_before_job_list = [
    # "ln -s /horizon-bucket/MultiMode_3/yuhao.dou/models/ ${WORKING_PATH}/tmp_models",
    # "ln -s /horizon-bucket/interaction/models/yuhao.dou/ ${WORKING_PATH}/tmp_models",
    "ln -s /horizon-bucket/interaction/active/face3d/face3d_data ${WORKING_PATH}/face3d_data",  # noqa E501
    "cd ${WORKING_PATH}",
    f"cp  {cfg_file} /job_data/",
]
