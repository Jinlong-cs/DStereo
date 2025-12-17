import os
import uuid


def _gen_template(
    cache_dir: str,
    model_name,
    model_arch_desc: str = "",
    training_datasets: str = "",
    model_accuracy: str = "",
    model_speed: str = "",
    usage: str = "",
):
    template = """# **{MODEL_NAME}**
# 模型基本信息
## 模型描述
1. 模型结构

    {MODEL_ARCH_DESC}
2. 训练数据

    {MODEL_TRAINING_DATA}
3. 精度

    {MODEL_ACCURACY}
4. 速度

    {MODEL_SPEED}
5. 使用场景

    {MODEL_USECASE}
6. 模型输入/输出:

```python
# 输入数据结构：
# inputs : List[hatbc.message.message.SyncMessages]
# 表示len(inputs)组图片。每组图片组成一个SyncMessage。
# 输出：
# outputs : List[List[hatbc.message.frame.Frame]]
# 第一层list元素表示每个batch的结果。第二层list元素表示batch中每个SyncMessage的结果。
```
# 使用说明

模型推理服务基于aidi 提供的模型管理、模型推理服务能力。这里推荐python用户使用 aidisdk 完成推理服务。

## 安装依赖

```shell
pip3 install -U pip -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --default-timeout=1000 --use-deprecated=legacy-resolver
pip3 install horizon-hdflow --index-url https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --use-deprecated=legacy-resolver
```

## HAT推理


## HDFLOW推理

1.对于Pack格式的原始输入数据
```python
import torch

from hdflow.v2.dataset.multiview_dir_dataset import MultiViewDirDataset
from hdflow.v2.hub.model_zoo import ModelLocation, ModelZoo

model_zoo = ModelZoo(location="git")
model = model_zoo.get_model(
    name="mvt4dv2_galaxy",
    version="1.0.0",
)
predictor = model.get_predictor(
    device=torch.device("cuda:0"),
)

img_dir = "/horizon-bucket/auto_jenkins_test/hdflow_workspace/test_v2/test_dataset/test_multiview_dir_dataset/pack_imgs"  # noqa
idx2cam = dict(
    zip(
        ["0", "1", "2", "3", "4"],
        ["front_right", "rear_right", "front_left", "rear_left", "rear"],
    )
)

dataset = MultiViewDirDataset(
    img_dir,
    idx2cam=idx2cam,
    to_rgb=True,
    max_samples=-1,
)
data = dataset[0]

output = predictor.predict_single([data])
```

""".format(
        MODEL_NAME=model_name,
        MODEL_ARCH_DESC=model_arch_desc,
        MODEL_TRAINING_DATA=training_datasets,
        MODEL_ACCURACY=model_accuracy,
        MODEL_SPEED=model_speed,
        MODEL_USECASE=usage,
    )
    save_path = os.path.join(cache_dir, uuid.uuid4().hex, "README.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(template)

    return save_path
