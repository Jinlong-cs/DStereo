# CloudModelInference

## 简介

这里是云端模型组维护的深度学习模型推理支持，允许用户使用 aidisdk 提供的 api，方便、快速的实现深度学习模型推理。

## Getting Started

模型推理服务基于aidi 提供的模型管理、模型推理服务能力。这里推荐python用户使用 aidisdk 完成推理服务。

### 安装依赖

```shell
pip3 install -U pip -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --default-timeout=1000 --use-deprecated=legacy-resolver
pip3 install 'aidisdk>=0.15.0' -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --default-timeout=1000 --use-deprecated=legacy-resolver --pre
```

### 使用AIDI模型推理服务

下面以以语义分割40类模型为例，介绍模型部署服务使用方法

1. 在[ModelZoo](./ModelZoo.md)中，选择需要推理的模型与版本。

例如: cloudmodel_parsing40cls_swinb_bifpn_semanticfpn v0.7.0

2.  对有每个模型版本，用户可以通过aidisdk或者AIDI前端web创建推理服务。

假设我们想创建/使用一个名为`CloudModelInferTest-v0.7.0` 的推理服务：
```python
import numpy as np
from PIL import Image as pil_image
from aidisdk import AIDIClient

client = AIDIClient()
model_name = "cloudmodel_parsing40cls_swinb_bifpn_semanticfpn"
model_version = "v0.7.0"
service_name = "CloudModelInferTest-v0.7.0"
project_id = "PD20230003"  # 项目号，使用者需要在项目中
resource_pool = "model-deploy-titanx"  # 队列名
service_path = f"{model_name}:{service_name}"
print(f"service exist:{client.model_registry.inference_service_exist(service_path)}")
# 2. 如果 不存在，可以通过下面的接口创建推理服务
if not client.model_registry.inference_service_exist(service_path):
    client.model_registry.create_inference_service(
                service_path=service_path,
                name=f"{model_name}:{model_version}",
                project_id=project_id,
                resource_pool=resource_pool,
                instance_min=0,
                instance_max=2,
            )
# 3. 获取推理服务
inference = client.model_registry.acquire_inference_api(
    service_path=service_path,
    instances=1,
)
inference.wait()
# 4. load图片，完成推理
from hatbc.message import CameraFrame
from hatbc.message import Image
img = pil_image.open(
    "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20210304-165302_633_0__93449_1614848188663_0.jpg"  # noqa
).convert("RGB")
img = np.array(img.resize((2048, 1280)), dtype=np.uint8)
frame = CameraFrame(
    image=Image(data=img, layout="hwc", color_space="rgb")
)
inputs = [frame]
output_list = inference.forward(
    input=inputs,
)
for i in output_list:
    print(i)
print("Inference Done")
inference.release()
```
关于推理服务的设计、使用细节请参考文档：
[模型仓库、部署和模型推理服务使用用户手册](https://horizonrobotics.feishu.cn/wiki/B77Gw88xaiB1FakDYFQcQ3donrb)


## HAT云端模型的四种推理方式

除了AIDI提供的云端推理支持，我们也支持不同的本地/云端，服务化/非服务化推理功能，大家可以按需灵活使用。

### 云端服务化推理

这种方式，推理实现在云端，初始化时，服务会调用 aidi试验模型中的配置，完成推理。

```python
from projects.cloudmodel.model_service.tests.test_segmentation import TestSegmentationModel
from projects.cloudmodel.tools.instance_engine import CloudModelInference

inputs, _ = TestSegmentationModel.make_test_images()
base_model_name = "cloudmodel_parsing40cls_swinb_bifpn_semanticfpn"
base_model_version = "v0.7.0"
init_cfg = dict(
    model_name=base_model_name,
    model_version=base_model_version,
    model_serving_config=dict(
        service_path=f"{base_model_name}:CloudModelInferTest-v0.7.0",
        project_id="PD20230003",
        resource_pool="model-deploy-titanx",
        instance_min=0,
        instance_max=1
    )
)
infer_engine = CloudModelInference(
    infer_mode=CloudModelInference.InferenceMode.AidiInferServiceRemote,
    init_config=init_cfg
)
for input_data in inputs:
    res = infer_engine(input_data)
    print(res)
```

### 本地服务化推理

这种方式，用户通过服务化部署的方式，调用本地部署的服务，完成推理。初始化时，服务会调用 aidi试验模型中的配置，完成推理。
（这种模式通常是debug时会使用）

1. 首先，根据模型名、版本在开发机本地部署推理服务

```shell
aidisdk --loglevel INFO model_registry launch_service --name "cloudmodel_traffic_sign_medium_attribute:v0.7.0"  --port 8070 --resource gpu:0 --use_docker   --pip_index_urls https://pypi.hobot.cc/hobot-local/simple,https://pypi.hobot.cc/hobot-local/simple,https://mirrors.aliyun.com/pypi/simple
```

```python
from projects.cloudmodel.model_service.tests.test_segmentation import TestSegmentationModel
from projects.cloudmodel.tools.instance_engine import CloudModelInference

inputs, _ = TestSegmentationModel.make_test_images()
init_cfg = dict(
    model_name="cloudmodel_parsing40cls_swinb_bifpn_semanticfpn",
    local_service_port="8070"
)

infer_engine = CloudModelInference(
    infer_mode=CloudModelInference.InferenceMode.AidiInferServiceLocal,
    init_config=init_cfg
)
for input_data in inputs:
    res = infer_engine(input_data)
    print(res)
```

### AIDI实验模型推理

这种模式下，会使用 aidi 实验模型中保存的模型推理信息初始化模型，完成推理。
这里就不涉及任何推理服务的启动/使用。

```python
from projects.cloudmodel.model_service.tests.test_segmentation import TestSegmentationModel
from projects.cloudmodel.tools.instance_engine import CloudModelInference
from aidisdk.model import DeviceMeta

inputs, _ = TestSegmentationModel.make_test_images()
init_cfg = dict(
    model_name="cloudmodel_parsing40cls_swinb_bifpn_semanticfpn",
    model_version="v0.7.0",
    run_device=DeviceMeta("gpu", 0),
)

infer_engine = CloudModelInference(
    infer_mode=CloudModelInference.InferenceMode.InferModelRemote,
    init_config=init_cfg
)
for input_data in inputs:
    res = infer_engine(input_data)
    print(res)
```

### 本地模型推理

这种模式下，直接使用本地的配置，基于 HAT代码库，即可完成模型推理。
这里就不涉及任何推理服务的启动/使用，也不会用到 AIDI 的实验模型管理工具。

```python
from projects.cloudmodel.model_service.tests.test_segmentation import TestSegmentationModel
from projects.cloudmodel.tools.instance_engine import CloudModelInference
from aidisdk.model import DeviceMeta

inputs, _ = TestSegmentationModel.make_test_images()
init_cfg = dict(
    aidi_config="projects/cloudmodel/model_service/aidi_configs/segmentation/aidi_config_singletask_parsing40cls.py",
    run_device=DeviceMeta("gpu", 0),
)

infer_engine = CloudModelInference(
    infer_mode=CloudModelInference.InferenceMode.InferModelLocal,
    init_config=init_cfg
)
res = infer_engine(inputs)
# render results
# 可视化渲染的代码目前在只在hat中提供
import os
import uuid
from projects.cloudmodel.tools.visualizer.render_camera_frame import RenderCameraFrame
from projects.cloudmodel.tools.visualizer.class_metas.cloudmodel.class_meta_cloudmodel import (
    ClassMetaCloudModel,
)
results = infer_engine.result2cameraframe(res, inputs)
visualizer = RenderCameraFrame(image_scale=1.0)
class_meta = ClassMetaCloudModel()
for res_i in results:
    render_res = visualizer(res_i, metadata=class_meta)
    save_dir = os.path.join("tmp", res_i.topic, f"{uuid.uuid4().hex}.png")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    render_res.save(save_dir)

```

## ModelZoo

### 使用
在[ModelZoo](./ModelZoo.md)中, 收录了当前云端模型小组提供的所有模型。它们都可以使用上面的方法完成模型的推理。

### 更新
如果想部署一个新模型，并创建云端推理服务，可以参考下面的步骤：

#### 创建Model card
ModelCard相关的配置都存放在[aidi_configs](./aidi_configs)文件夹中。

以分割为例，
```python
from plugins.aidi_inference.models.example_torch_model import HatModel

from _md_template import _gen_template
from aidisdk.model import InferenceInstanceConfig

import hat

model_cls = HatModel
model_tags = dict(
    algo_type="segmentation",
    framework="PyTorch",
    owner="CloudModel.Parsing",
)
# AIDI model card中模型的名字
model_name = "cloudmodel_parsing40cls_swinb_bifpn_semanticfpn"
# 模型的描述信息，这个信息会反应到aidi model card的README文件中
model_desc = ""
# 模型的配置文件，这里保存了模型结构、数据的预处理、后处理等信息
model_config = "projects/cloudmodel/model_service/model_configs/segmentation/singletask_parsing40cls_swinb_bifpn_semanticfpn_config.py"
# checkpoint 文件
param_file = "/horizon-bucket/adas/big_model/model_zoo/cloudmodel_parsing40cls_swinb_bifpn_semanticfpn_3897336.pth.tar"  # noqa
# docker
docker_image = "docker.hobot.cc/imagesys/cloudmodel:runtime-py3.8-torch1.13.0-cu116-1.3.3-mmdet3.0-service-TRT"
# 推理实例相关配置，指定一个实例占用的资源
instance_config = InferenceInstanceConfig(gpu=1, threads_per_process=1)
# 相关的环境依赖
py_deps = [hat]
# 生成markdown格式的README文件
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
# 开启这个flag后，会使用aidi提供的云端推预处理、后处理加速功能，默认不开启
split_dataprocess_on_aidi = True
```
确认好配置之后，可以通过下面的命令上传到aidi模型仓库
```shell
python3 plugins/aidi_inference/aidi_deploy.py --config projects/cloudmodel/model_service/aidi_configs/segmentation/aidi_config_singletask_parsing40cls.py --publish-version v0.0.1 --project PD20230003  --disable-inference 
# --config: aidi_configs文件路径
# --publish-version 模型对应的版本
# --project 项目号，PDXXXXXX
# --disable-inference 加上这个flag之后，在创建完成模型仓库后即返回。否则，在创建完成后，会根据modelcard创建一个镜像用于推理服务。
```
#### 开启/使用AIDI推理服务
AIDI 推理服务的使用参考文档：

[模型仓库、部署和模型推理服务使用用户手册](https://horizonrobotics.feishu.cn/wiki/B77Gw88xaiB1FakDYFQcQ3donrb)