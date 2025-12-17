from plugins.aidi_inference.models.example_torch_model import HatModel

from _md_template import _gen_template
from aidisdk.model import InferenceInstanceConfig

import hat

model_cls = HatModel
model_tags = dict(
    algo_type="detection",
    framework="PyTorch",
    owner="BigModel.Detection",
)
model_name = "mvt4dv2_galaxy"
model_desc = "多视角检测大模型"

model_config = "projects/bigmodel/configs/mvt4d/aidi_inference.py"
param_file = ""  # noqa
# envs
docker_image = ""  # noqa
instance_config = InferenceInstanceConfig(gpu=1, threads_per_process=1)
py_deps = [hat]

# when enabled, preprocess and posprocess will be splited and paralled in aidi serving.
split_dataprocess_on_aidi = False
# we can not use tensorRT with chinese charater imported


readme_file = _gen_template(
    "/tmp",
    model_name,
    model_arch_desc=model_desc,
    training_datasets="Mono/Pilot",
    model_accuracy="NA",
    model_speed="NA",
    usage="数据挖掘、标注预刷",
)
