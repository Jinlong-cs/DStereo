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
# inputs : List[hatbc.message.frame.CameraFrame]
# 表示len(inputs)个图片。注意，在推理时会把这 N个图片变成NCHW 送入模型。
# 输出：
# outputs : List[List[hatbc.message.objects.Instance]]
# 第一层list元素表示每个 batch的结果。第二层list元素表示batch中每张image的结果。
```
# 使用说明

模型推理服务基于aidi 提供的模型管理、模型推理服务能力。这里推荐python用户使用 aidisdk 完成推理服务。

## 安装依赖

```shell
pip3 install -U pip -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --default-timeout=1000 --use-deprecated=legacy-resolver
pip3 install hatbc[all]==0.10.0b202309070700+b39a57d -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --default-timeout=1000 --use-deprecated=legacy-resolver
pip3 install -U 'aidisdk>=0.15.0' -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple --default-timeout=1000 --use-deprecated=legacy-resolver
```

## 【demo】使用AIDI模型推理服务

1. 安装环境依赖
2. 通过 aidisdk 调用模型，完成推理.

云端推理的使用方法请参考文档：

[使用AIDI模型推理服务](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/tree/master/projects/cloudmodel/model_service#%E4%BD%BF%E7%94%A8aidi%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1)

## HAT推理

除了AIDI提供的云端推理支持， 在 HAT中我们也支持不同的云端模型服务化/非服务化推理功能，大家可以按需灵活使用。
具体细节请参考[云端模型HAT推理](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/blob/master/projects/cloudmodel/model_service/README.md)

## HDFLOW推理

Comming soon...
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
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(template)

    return save_path
