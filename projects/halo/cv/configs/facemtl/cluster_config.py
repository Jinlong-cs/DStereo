"""Job Settings."""
import hat

job_name = "facemtl_mixvargenet0.75"
job_password = "hat"

num_machines = 1
num_gpus_per_machine = 8
framework = "pytorch"
task_label = "HAT"
project_id = "PDT2021005"  # default ProjectID, replace with your ProjectID
# project_id = "GA2020005"
input_bucket = "HDLTAlgorithm,interaction"
output_bucket = "MultiMode_2"  # default use HDLTAlgorithm Bucket

priority = 5
version = hat.__version__.split(".dev")[0]
# docker_image = (
#     # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu102-%s"
#     # % version
#     "docker.hobot.cc/imagesys/hat:pilot-runtime-cu111-hdflow-20220721-torch191"
#     # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-%s" % version
#     # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.10.2-cu102-%s" % version
#     # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.10.2-cu111-%s" % version
#     # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.9.1-cu102-%s" % version
#     # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.9.1-cu111-%s" % version
#     # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.9.1-cu102-%s" % version
#     # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.9.1-cu111-%s" % version
# )
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
    "../../face3d_data",
    "url2IP.py",
    "ssh_launcher.py",
]
job_list = [
    # "hdfs dfs -get hdfs://hobot-bigdata-aliyun/user/yuhao.dou/data/3D_Face/face3d_data /running_package/k8s_job",  # noqa E501
    # "hdfs dfs -get hdfs://hobot-bigdata-ucloud/user/jiaqi.quan/model/facemtl/face_ldmk_pretrain/float-checkpoint-best.pth.tar",  # noqa E501
    # "hdfs dfs -get hdfs://hobot-bigdata-aliyun/user/xiang.yan/Data/lpips-0.1.4-py3-none-any.whl /running_package/k8s_job",  # noqa E501
    # "pip3 install lpips-0.1.4-py3-none-any.whl",
    # "pip3 install aidisdk shapely hatbc horizon_plugin_pytorch_cu111 -U -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",  # noqa E501
    # "pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f http://art.k8s-idc.hobot.cc/artifactory/list/community-local/torch --trusted-host art.k8s-idc.hobot.cc --use-deprecated=legacy-resolver",  # noqa E501
    # "pip3 install torchaudio==0.9.1 -U -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",  # noqa E501
    # "pip3 install --pre mxnet-horizon-cu111 -U -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",  # noqa E501
    # "sleep 10000",
    # step
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_step.py --stage float",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_step.py --stage freeze_bn",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_step.py --stage qat",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_step.py --stage int_infer",  # noqa E501
    # epoch
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train.py --stage float",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train.py --stage freeze_bn",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train.py --stage qat",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train.py --stage int_infer",  # noqa E501
    # stage
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_stage.py --stage freeze_backbone",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_stage.py --stage float",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_stage.py --stage freeze_bn",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_stage.py --stage qat",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_stage.py --stage int_infer",  # noqa E501
    # freeze backbone -> allin training -> freeze_bn
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_stage.py --stage freeze_backbone",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_stage.py --stage float",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_train_stage.py --stage freeze_bn",  # noqa E501
    # qat finetune head
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_finetune_head_qat.py --stage qat",  # noqa E501
    # "python3 -W ignore tools/train.py --config projects/halo/cv/configs/facemtl/facemtl_finetune_head_qat.py --stage int_infer",  # noqa E501
]
