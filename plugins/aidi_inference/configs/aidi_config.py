import torch
from aidisdk.model import InferenceInstanceConfig

import hat

model_name = "example_resnet18"

model_tags = dict(
    algo_type="classification",
    framework="PyTorch",
    owner="unittest",
)
model_desc = "hat torch model for inference example"

model_config = "examples/inference.py"
param_file = "./tmp_models/resnet18_cls/float-checkpoint-best.pth.tar"
instance_config = InferenceInstanceConfig(gpu=1)

example_input = {"img": torch.randn(1, 3, 224, 224)}

version = hat.__version__.split(".dev")[0]
docker_image = (
    "docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-%s"
    % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-%s" % version
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-%s" % version
)
py_deps = [hat]
