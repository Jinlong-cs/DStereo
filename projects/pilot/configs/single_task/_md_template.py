import os
import uuid


def _gen_template(
    cache_dir: str,
    model_name: str,
    seg_type: str,
    model_arch_desc: str = "",
    training_datasets: str = "",
    model_accuracy: str = "",
    model_speed: str = "",
    usage: str = "",
):
    if seg_type == "semantic_seg":
        model_out_type_hint_str = "List[List[hatbc.message.objects.DenseMap]]"
    elif seg_type == "instance_seg":
        model_out_type_hint_str = "List[List[hatbc.message.objects.Instance]]"
    else:
        raise ValueError(f"invalid model seg type:{seg_type}")
    template = """# **{MODEL_NAME}**
# 模型基础信息
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
# inputs : List[hatbc.message.frame.CameraFrame]
# 表示len(inputs)个图片。注意，在推理时会把这 N个图片变成NCHW 送入模型。
# 输出：
# outputs : {MODEL_OUT_TYPE}
# 第一层list元素表示每个 batch的结果。第二层list元素表示batch中每张image的结果。
```
# 使用说明

通过HDFlow中的Model Zoo进行模型的调用，使用方式详见HDFlow推理。

## 安装依赖

```shell
pip3 install -U pip -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --default-timeout=1000 --use-deprecated=legacy-resolver
pip3 install horizon-hdflow --index-url https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --use-deprecated=legacy-resolver```
```

## HAT推理


## HDFLOW推理

1. 对于Pack格式的原始输入数据
```python
from tat.matrix.pack_sdk import TopicChannel

from hdflow.v2.dataset.horizon_pack_dataset import HorizonPackDataset
from hdflow.v2.format.hat_msg.camera_frame import ComposePerceptions
from hdflow.v2.hub.model_zoo import ModelLocation, ModelZoo

dataset = HorizonPackDataset(
    handle="/horizon-bucket/auto_jenkins_test/test_horizon_pack_dataset/ADAS_20230110-220210_302_1.pack",  # noqa
    topic_channel=[
        TopicChannel("image", 0),
        TopicChannel("camera_runtime", 0),
    ],
    item_with_pack_meta=True,
)
location = ModelLocation.GIT
model_cond = dict(name="pilot_image_fail_parsing")
model = ModelZoo(location=location).get_model(**model_cond)
predictor = model.get_predictor(
    device=0,
    post_processor=ComposePerceptions(
        msg_meta_key=["meta", "camera_param"],
        msg_topic="pilot_image_fail_parsing",
    ),
)
frame_transform = model.get_transform(data_type="pack")

frame = frame_transform(dataset[0])
output = predictor.predict_single([frame])
```

2. 对于Image格式的原始输入数据
```
import numpy as np
from PIL import Image

from hdflow.utils.path import url_to_local_path
from hdflow.v2.format.hat_msg.camera_frame import ComposePerceptions
from hdflow.v2.hub.model_zoo import ModelLocation, ModelZoo

img = np.array(
    Image.open(
        url_to_local_path(
            "dmpv2://auto_jenkins_test/hdflow_workspace/test_modelzoo/ADAS_20220315-142130_023_4__29219_1647325476006_0.jpg"  # noqa
        )
    )
)
location = ModelLocation.GIT
model_cond = dict(name="pilot_image_fail_parsing")
model = ModelZoo(location=location).get_model(**model_cond)
predictor = model.get_predictor(
    device=0,
    post_processor=ComposePerceptions(
        msg_meta_key=["meta", "camera_param"],
        msg_topic="pilot_image_fail_parsing",
    ),
)
frame_transform = model.get_transform(data_type="image")
frame = frame_transform(img)
output = predictor.predict_single([frame])
```

""".format(
        MODEL_NAME=model_name,
        MODEL_ARCH_DESC=model_arch_desc,
        MODEL_TRAINING_DATA=training_datasets,
        MODEL_ACCURACY=model_accuracy,
        MODEL_SPEED=model_speed,
        MODEL_USECASE=usage,
        MODEL_OUT_TYPE=model_out_type_hint_str,
    )
    save_path = os.path.join(cache_dir, uuid.uuid4().hex, "README.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(template)

    return save_path
