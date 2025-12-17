"""Job Settings."""
import hat

# job_name = "face_quality_mtl_mouth_vargnet_gluon"
job_name = "face_quality_mtl_glass_mask_vargnet_gluon_master_1"
job_password = "hat"

num_machines = 1
num_gpus_per_machine = 4

framework = "pytorch"
task_label = "HAT"
project_id = "PDT2021005"  # default ProjectID, replace with your ProjectID
input_bucket = "HDLTAlgorithm"
output_bucket = "MultiMode_2"  # default use HDLTAlgorithm Bucket

priority = 5
version = hat.__version__.split(".dev")[0]
docker_image = (
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu102-%s"
    # % version
    "docker.hobot.cc/imagesys/hat:pilot-runtime-cu111-hdflow-20220721-torch191"
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.10.2-cu102-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.10.2-cu111-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.9.1-cu102-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.9.1-cu111-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.9.1-cu102-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.9.1-cu111-%s" % version
)
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
]
job_list = [
    f"hdfs dfs -mkdir hdfs://hobot-bigdata-ucloud/user/jiaqi.quan/model/facequality_mtl_model{job_name}",
    "python3 -W ignore tools/train.py --config projects/halo/configs/facequality/vargnet_facequality_mtl.py --stage float",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/configs/facequality/vargnet_facequality_mtl.py --stage qat",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/configs/facequality/vargnet_facequality_mtl.py --stage int_infer",  # noqa E501
    # "python3 tools/predict.py --stage float --config projects/halo/configs/facequality/vargnet_facequality_mtl.py",  # noqa
    # "python3 -W ignore tools/deploy/compile_perf.py -c projects/halo/configs/facequality/vargnet_facequality_mtl.py",  # noqa E501
    # "python3 tools/predict.py --stage qat --config projects/halo/configs/facequality/vargnet_facequality_mtl.py",  # noqa
    # "python3 tools/predict.py --stage int_infer --config projects/halo/configs/facequality/vargnet_facequality_mtl.py",  # noqa
    # f"hdfs dfs -put -f tmp_models/* hdfs://hobot-bigdata-ucloud/user/jiaqi.quan/model/facequality_mtl_model{job_name}",
]
