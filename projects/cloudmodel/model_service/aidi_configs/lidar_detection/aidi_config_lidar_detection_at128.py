import os

from aidisdk.model import InferenceInstanceConfig

import hat
from projects.cloudmodel.model_service.models.torch_model import HatModel

model_cls = HatModel
model_name = "cloudmodel_lidar_detection_at128"
model_tags = dict(
    algo_type="Detection3D",
    framework="PyTorch",
    owner="LidarDetectionTeam",
)

model_config = "projects/cloudmodel/model_service/model_configs/lidar_detection/detection_at128_config.py"
param_file = "/jfs-public/adas/xiangyu.wei/models/lidar_gt_pilot_at128/20240115_baseline_7class/float-checkpoint-last.pth.tar"

# envs
docker_image = "docker.hobot.cc/imagesys/hat:pilot5.1-py3.8-torch1.10.2-cu111-c1c2cc7_20231130"
instance_config = InferenceInstanceConfig(gpu=1, threads_per_process=1)
py_deps = [hat]
