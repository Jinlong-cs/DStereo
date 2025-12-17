from aidisdk.model import InferenceInstanceConfig

import hat
from projects.cloudmodel.model_service.models.torch_model import HatModel

model_cls = HatModel
model_tags = dict(
    algo_type="classification",
    framework="PyTorch",
    owner="unittest",
)
model_name = "cloudmodel_test"
model_desc = "aidi_config used for AIDIPredictor test"

model_config = "examples/aidipredictor.py"
param_file = None  # noqa
# envs
docker_image = "docker.hobot.cc/imagesys/cloudmodel:runtime-py3.8-torch1.13.0-cu116-1.3.3-mmdet3.0-service-TRT"  # noqa
instance_config = InferenceInstanceConfig(gpu=1, threads_per_process=1)
py_deps = [hat]

# when enabled, preprocess and posprocess will be splited and paralled in aidi serving. # noqa
split_dataprocess_on_aidi = False
# we can not use tensorRT with chinese charater imported
