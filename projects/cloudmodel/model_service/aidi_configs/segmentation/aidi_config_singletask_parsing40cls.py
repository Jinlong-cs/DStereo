import os

from aidisdk.model import InferenceInstanceConfig

import hat
from projects.cloudmodel.model_service.models.torch_model import HatModel

model_cls = HatModel
model_tags = dict(
    algo_type="segmentation",
    framework="PyTorch",
    owner="CloudModel.Parsing",
)

model_name = "cloudmodel_parsing40cls_swinb_bifpn_semanticfpn"
model_desc = ""
model_config = "projects/cloudmodel/model_service/model_configs/segmentation/singletask_parsing40cls_swinb_bifpn_semanticfpn_config.py"
param_file = "/horizon-bucket/adas/big_model/model_zoo/cloudmodel_parsing40cls_swinb_bifpn_semanticfpn_3897336.pth.tar"  # noqa
# envs
docker_image = "docker.hobot.cc/imagesys/cloudmodel:runtime-py3.8-torch1.13.0-cu116-1.3.3-mmdet3.0-service-TRT"
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
        "semantic_seg",
        model_arch_desc=model_desc,
        training_datasets="Mono/Pilot",
        model_accuracy="NA",
        model_speed="NA",
        usage="数据挖掘、标注预刷",
    )
