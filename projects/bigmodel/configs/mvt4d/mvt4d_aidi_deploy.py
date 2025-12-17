import os
import pickle

import cv2
import yapf
from aidisdk.model import InferenceInstanceConfig
from hatbc.message import CameraFrame, Image, MessageMeta, SyncMessages

import hat

# TODO(yilin.xiong, ?): to verify consistency deployed model & local model

model_name = "test_mvt4dv2_model"
# 模型权重
param_file = "/horizon-bucket/matrix/users/zixiang.pei/workspace/pretrained_models/float-checkpoint-last-00741c7f.pth.tar"  # noqa
# 模型构建config， 包含模型初始化，模型前后处理实现
model_config = "projects/bigmodel/configs/mvt4d/aidi_inference.py"
# 基础镜像
docker_image = "docker.hobot.cc/imagesys/hat:inference-py3.8-torch1.10.2-mmcv1.4.2-cu111-1.4.2-230808"  # noqa
instance_config = InferenceInstanceConfig(gpu=1)
# 模型tags
model_tags = dict(
    algo_type="classification",
    framework="PyTorch",
    owner="yilin.xiong",
)
model_desc = "mvt4dv2 cloud model"  # 模型描述
py_deps = [hat, yapf]  # 运行所需py依赖
# 模型输入示例
to_rgb = True
sample_dir = "/horizon-bucket/HDLTAlgorithm/users/yilin.xiong/pilot/data/cloud_model/unit_test/mvt4d/example_data"  # noqa
view_orders = [
    "front_right",
    "rear_right",
    "front_left",
    "rear_left",
    "rear",
]

with open(os.path.join(sample_dir, "calib_v2.pkl"), "rb") as f:
    calib = pickle.load(f)
frame_view = []
calib_new = dict()
for view in view_orders:
    for file in os.listdir(sample_dir):
        if file.endswith(".jpg"):
            if view == file.split("__")[1]:
                _img = cv2.imread(os.path.join(sample_dir, file))
                if to_rgb:
                    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
                img = Image(
                    data=_img,
                )
                cam_params = calib[view]
                frame = CameraFrame(
                    image=img,
                    camera_param=cam_params,
                    topic="camera_frame",
                )
                frame_view.append(frame)

sync_messages = SyncMessages(
    meta=MessageMeta(
        timestamp=frame_view[0].camera_param.timestamp,
    ),
    messages=frame_view,
)

num_batch = 1
example_input = [sync_messages for _ in range(num_batch)]
