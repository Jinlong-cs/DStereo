import os

from plugins.aidi_inference.models.example_torch_model import HatModel

from _md_template import _gen_template
from aidisdk.model import InferenceInstanceConfig

import hat

model_cls = HatModel
model_tags = dict(
    algo_type="segmentation",
    framework="PyTorch",
    owner="Pilot.Parsing",
)

os.environ["HAT_PILOT_MODEL_SETTING"] = "x3c"
os.environ["version"] = "14.0"

model_name = "pilot5_image_fail_segmentation"
model_desc = ""
model_config = (
    "projects/pilot/configs/single_task/image_fail_segmentation_inference.py"
)
param_file = ""
# envs
docker_image = ""
instance_config = InferenceInstanceConfig(gpu=1, threads_per_process=1)
py_deps = [hat]

# when enabled, preprocess and posprocess will be splited and paralled in aidi serving.
split_dataprocess_on_aidi = False

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
