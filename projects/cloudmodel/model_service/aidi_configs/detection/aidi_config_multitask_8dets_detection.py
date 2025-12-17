import os

from aidisdk.model import InferenceInstanceConfig

import hat
from projects.cloudmodel.model_service.models.torch_model import HatModel

model_cls = HatModel
model_tags = dict(
    algo_type="detection",
    framework="PyTorch",
    owner="CloudModel.Detection",
)
model_name = "cloudmodel_multitask_8dets_detection_swinb_w12_bifpn_fcos"
model_desc = ""

model_config = "projects/cloudmodel/model_service/model_configs/detection/multitask_8dets_swinb_w12_bifpn_fcos.py"
param_file = "/jfs-public/adas/fei01.he/model_zoo/multitask_8dets_swinb_w12_bifpn_fcos/big_model_checkpoint-last.pth.tar"  # noqa
# envs
docker_image = "docker.hobot.cc/imagesys/cloudmodel:runtime-py3.8-torch1.13.0-cu116-1.3.3-mmdet3.0-service-TRT"  # noqa
instance_config = InferenceInstanceConfig(gpu=1, threads_per_process=1)
py_deps = [hat]

# when enabled, preprocess and posprocess will be splited and paralled in aidi serving.
split_dataprocess_on_aidi = False
# we can not use tensorRT with chinese charater imported
if not os.environ.get("INFERENCE_TENSORRT_OPT", False):
    from _md_template import _gen_template

    readme_file = _gen_template(
        "/tmp",
        model_name,
        model_arch_desc=model_desc,
        training_datasets="Mono/Pilot",
        model_accuracy="NA",
        model_speed="NA",
        usage="数据挖掘、标注预刷",
    )
