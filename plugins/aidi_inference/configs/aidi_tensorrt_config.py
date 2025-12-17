import torch
from aidisdk.model import InferenceInstanceConfig
from models.example_tensorrt_model import HatTensorRTModel

import hat

model_name = "example_tensorrt_resnet18"

model_tags = dict(
    algo_type="classification",
    framework="PyTorch",
    owner="unittest",
)
model_desc = "hat tensorrt model for inference example"
instance_config = InferenceInstanceConfig(gpu=1)

model_config = "examples/inference.py"
param_file = "./tmp_models/resnet18_cls/float-checkpoint-best.pth.tar"
model_cls = HatTensorRTModel
example_input = torch.randn(1, 3, 224, 224)

docker_image = "docker.hobot.cc/aitools/torchtrt-py3.8-torch1.10.2-cu111-torchtrt1.0.0-trt8016:0.0.1"  # noqa
py_deps = [hat]
