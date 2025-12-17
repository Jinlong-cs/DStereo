"""Job Settings."""
import hat

job_name = (
    "torch-hat-eye-status-mix-large-fc-multi-task-qat-no-freeze-37-fuse-bn"
)
job_password = "newk8s666"

num_machines = 1
num_gpus_per_machine = 2

framework = "pytorch"
task_label = "HAT"
project_id = "TD20220003"  # default ProjectID, replace with your ProjectID
input_bucket = "HDLTAlgorithm"  # default use HDLTAlgorithm Bucket

priority = 5
# version = "e340de2"
version = hat.__version__.split("dev")[0]
docker_image = (
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu102-%s"
    # % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.10.2-cu102-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.10.2-cu111-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.9.1-cu102-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.9.1-cu111-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.9.1-cu102-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.6-torch1.9.1-cu111-%s" % version
    # "docker.hobot.cc/imagesys/hat:pilot-runtime-cu111-hdflow-20220721-torch191"
    "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-1.3.1"
)
max_jobtime = 120  # default 7200 = 5days
# max_jobtime = 60  # default 7200 = 5days
output_bucket = "MultiMode"

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../configs",
    "url2IP.py",
]
job_list = [
    # "python3 tools/train.py --stage qat --config configs/classification/debug_eye_status_filelist_classifier_mixvargenet_single_cls.py --device-ids 0",  # noqa E501
    # "python3 tools/train.py --stage qat --config configs/classification/debug_eye_status_filelist_classifier_mixvargenet_single_cls.py --device-ids 0",  # noqa E501
    # "python3 tools/train.py --stage float --config configs/classification/eye_status_filelist_classifier_mixvargenet_double_cls_pretrain_large_model.py",  # noqa E501
    "python3 tools/train.py --stage float --config projects/halo/cv/configs/eye_status/configs/eye_status_filelist_classifier_mixvargenet_double_cls_finetune.py --device-ids 0,1",  # noqa E501
    "python3 tools/train.py --stage qat --config projects/halo/cv/configs/eye_status/configs/eye_status_filelist_classifier_mixvargenet_double_cls_finetune.py --device-ids 0,1",  # noqa E501
    "python3 tools/predict.py --stage float --config projects/halo/cv/configs/eye_status/configs/eye_status_filelist_classifier_mixvargenet_double_cls_finetune.py --device-ids 0,1",  # noqa E501
    "python3 tools/predict.py --stage qat --config projects/halo/cv/configs/eye_status/configs/eye_status_filelist_classifier_mixvargenet_double_cls_finetune.py --device-ids 0,1",  # noqa E501
    # "python3 tools/train.py --stage float --config configs/classification/eye_status_filelist_classifier_mixvargenet_multitask_large_fc.py",  # noqa E501
    # "python3 tools/train.py --stage qat --config configs/classification/eye_status_filelist_classifier_mixvargenet_multitask_large_fc.py",  # noqa E501
    # "python3 tools/predict.py --stage float --config configs/classification/eye_status_filelist_classifier_mixvargenet_multitask_large_fc.py",  # noqa E501
    # "python3 tools/predict.py --stage qat --config configs/classification/eye_status_filelist_classifier_mixvargenet_multitask_large_fc.py",  # noqa E501
    # "python3 tools/predict.py --stage float --config configs/classification/eye_status_filelist_classifier_mixvargenet_double_cls_pretrain_large_model.py",  # noqa E501
    # "python3 tools/predict.py --stage float --config configs/classification/debug_eye_status_filelist_classifier_mixvargenet_double_cls.py --device-ids 0",  # noqa E501
    # "python3 tools/train.py --stage float --config configs/classification/debug_eye_status_filelist_classifier_mixvargenet.py --device-ids 0,1",  # noqa E501
    # "python3 tools/predict.py --stage float --config configs/classification/debug_eye_status_filelist_classifier_mixvargenet.py --device-ids 0",  # noqa E501
    # "python3 tools/train.py --stage float --config configs/classification/debug_eye_status_filelist_classifier.py",  # noqa E501
    # "python3 tools/predict.py --stage float --config configs/classification/debug_eye_status_filelist_classifier.py --device-ids 0",  # noqa E501
    #  "python3 tools/trainv2.py --stage float --config configs/classification/2080_eye_status_filelist_classifier.py",  # noqa E501
    # "python3 tools/train.py --config configs/classification/resnet18.py --step qat",  # noqa E501
    # "python3 tools/train.py --config configs/classification/resnet18.py --step int_infer",  # noqa E501
]
