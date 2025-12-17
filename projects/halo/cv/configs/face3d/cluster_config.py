"""Job Settings."""
job_name = "vargenet_face3d_id01"
job_password = "hat"

num_machines = 1
num_gpus_per_machine = 8

framework = "pytorch"
task_label = "HAT"
project_id = "PDT2021005"  # default ProjectID, replace with your ProjectID
input_bucket = "HDLTAlgorithm"  # default use HDLTAlgorithm Bucket

priority = 5
docker_image = "docker.hobot.cc/imagesys/gcc:test2"
max_jobtime = 10000  # default 7200 = 5days

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = ["../../hat", "../../tools", "../../configs", "../../tmp_models"]
job_list = [
    "hdfs dfs -get hdfs://hobot-bigdata-aliyun/user/yuhao.dou/data/3D_Face/face3d_data /running_package/k8s_job",  # noqa E501
    "hdfs dfs -get hdfs://hobot-bigdata/user/xiaolong.sun/lmdb/3D_Face/FacePublic_lmdb.tar.gz /running_package/k8s_job",  # noqa E501
    "hdfs dfs -get hdfs://hobot-bigdata/user/xiaolong.sun/lmdb/3D_Face/FaceSynthetics_lmdb.tar.gz /running_package/k8s_job",  # noqa E501
    "tar -xf FacePublic_lmdb.tar.gz",
    "tar -xf FaceSynthetics_lmdb.tar.gz",
    "hdfs dfs -get hdfs://hobot-bigdata-aliyun/user/xiang.yan/Data/lpips-0.1.4-py3-none-any.whl /running_package/k8s_job",  # noqa E501
    "pip install lpips-0.1.4-py3-none-any.whl",
    "pip install aidisdk shapely hatbc horizon_plugin_pytorch_cu111  -U -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",  # noqa E501
    "pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple --use-deprecated=legacy-resolver",  # noqa E501
    f"hdfs dfs -mkdir hdfs://hobot-bigdata-aliyun/user/yuhao.dou/output/hat/face_3d/{job_name}",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/configs/face3d/vargnet_face3d_pretrain.py --stage float",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/configs/face3d/vargnet_face3d_finetune.py --stage float",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/configs/face3d/vargnet_face3d_finetune.py --stage qat",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/configs/face3d/vargnet_face3d_finetune.py --stage int_infer",  # noqa E501
    "python3 -W ignore tools/deploy/compile_perf.py -c projects/halo/configs/face3d/vargnet_face3d_finetune.py",  # noqa E501
    f"hdfs dfs -put -f tmp_models/* hdfs://hobot-bigdata-aliyun/user/yuhao.dou/output/hat/face_3d/{job_name}",  # noqa E501
]

custom_cmds_before_job_list = [
    # "ln -s /bucket/input/%s/data/pack_data ${WORKING_PATH}/tmp_data"
    # % input_bucket,
    # "ln -s /bucket/input/%s/models/bayes_release_models ${WORKING_PATH}/tmp_pretrained_models"  # noqa
    # % input_bucket,
    # "ln -s /job_data/models ${WORKING_PATH}/tmp_models",
]
