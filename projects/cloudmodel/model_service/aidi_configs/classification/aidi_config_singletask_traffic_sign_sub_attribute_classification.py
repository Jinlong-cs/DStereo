import os

from aidisdk.model import InferenceInstanceConfig

import hat
from projects.cloudmodel.model_service.models.torch_model import HatModel

# models
model_cls = HatModel
model_tags = dict(
    algo_type="classification",
    framework="PyTorch",
    owner="CloudModel.Classification",
)

model_name = "cloudmodel_traffic_sign_sub_attribute"
model_desc = ""
model_config = "projects/cloudmodel/model_service/model_configs/classification/singletask_traffic_sign_sub_attribute_config.py"
param_file = "/horizon-bucket/adas/big_model/model_zoo/attribute_models/traffic_sign_sub/big_model_checkpoint-step-344999.pth.tar"
# envs
docker_image = "docker.hobot.cc/imagesys/cloudmodel:runtime-py3.8-torch1.13.0-cu116-1.3.3-mmdet3.0-service-TRT"
instance_config = InferenceInstanceConfig(gpu=1, threads_per_process=1)
py_deps = [hat]
# when enabled, preprocess and posprocess will be splited and paralled in aidi serving.
split_dataprocess_on_aidi = True
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
