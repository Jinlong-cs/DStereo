# 简介
(torchdynamo/how_to_use_torchdynamo_speedup_inference)=
## 如何使用TorchDynamo编译加速推理模型

借助于强大的`TorchDynamo`，HAT支持通过`TensorRT`(以下简称trt)一键加速模型推理。TorchDynamo提取模型前向计算图，并通过FX作为IR(中间表示)，进一步地我们可以利用trt来加速FX计算图。

我们可以在HAT推理类[Inference](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/blob/master/hat/engine/inference.py)中，指定dynamo backends，从而完成编译优化。

下面我们给出示例。

```python
# 定义模型，以ResNet18为例
deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="ResNet18",
        num_classes=1000,
        bn_kwargs={},
        flat_output=False,
    ),
    losses=None,
)
# 定义dynamo tensorrt backends
# 您可以使用官方提供的backend
inference = dict(
    type="Inference",
    device=None,
    march=March.BERNOULLI2,
    pre_processors=None,
    post_processors=None,
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="TorchCompile",
                compile_backend="fx2trt",
            )
        ],
    ),
)

# 或使用HAT提供的自定义backend
# 该backend对于model计算图较为复杂(如使用MultitaskGraphModel封装时)
# 有更好的优化表现
from hat.utils.compile_backends import tensorRT_backend

inference = dict(
    type="Inference",
    device=None,
    march=March.BERNOULLI2,
    pre_processors=None,
    post_processors=None,
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="TorchCompile",
                # fp16=True，使用fp16优化；否则fp32
                compile_backend=tensorRT_backend(fp16=True),
            )
        ],
    ),
)
```

实例化该`Inference`类，即可获得一个支持trt加速的推理模型。
**需要注意**的是，实际的编译优化发生在第一次推理时；因此初次运行Inference时，编译会花费一些时间，为正常现象。

## 原理介绍
从上面的使用示例中，我们不难发现利用torchDynamo进行tensorrt优化具有如下优势：

- 优化对象支持任意python函数。在过去，若希望优化为trt模型，往往需要将`torch.nn.Module`模型转化为IR，常用的手段有torchscript、FX，亦或者转化为onnx模型。然而目前大部分转化对Module的输入格式存在限制，要求为`Tensor`、`List[Tensor]`。尴尬的是，当前HAT中存在的模型往往使用Dict作为输入；若要满足这些转化的需求，大量的代码重写是必不可少的。此外，这些IR在记录计算图的过程中，遇到不支持的算子，往往会直接报错，当我们使用GraphModel这类多任务封装module时，几乎不可能完整记录整个计算图，这意味着我们需要进行大量的工作来处理多个计算图的拆分与串联问题。

- 支持自定义backends。`torchDynamo`支持自定义backends，用户可以通过自定义backends来完成细粒化的编译优化实现。我们将在后续章节中说明，如何增加一个自定义backends。

CPython的每次函数调用会生成一个Frame（或者叫 Stack），Frame 中带有的代码部分就是 ByteCode。CPython 运行时支持基于现有的 Frame 去设置一个自定义的 Frame，然后后面执行的就是自定义的 Frame。

`TorchDynamo`的工作原理就是在运行时设置一个自定义的Frame，该Frame中的ByteCode支持CallBack到Python层去修改。其提供的典型的修改接口是FX Graph，也就是说`TorchDynamo`会分析ByteCode，生成对应的FX Graph，然后提供FX Graph的接口供用户自定义计算图。

关于`TorchDynamo`，更多资料信息建议查阅[pytorch2.0](https://pytorch.org/get-started/pytorch-2.0)