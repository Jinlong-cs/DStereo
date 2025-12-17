# 在HAT中创建一个模型应用

## 模型应用的定义

针对一个行为符合预期（基于输入得到合理输出）的模型，构建一个可运行的程序，使得模型可被用来处理数据，这个可运行的程序即为模型应用。

构建时，需要确定好模型输入输出的数据结构，方便一个模型应用被集成到各类应用场景中。

## 配置方法

形式上，在相关config中，定义一个 `inference`变量来描述推理行为即可。这里使用了 `HAT`提供的{py:class}`Inference <hat.engine.inference.Inference>`类，来支持模型应用的构建。

```python

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

inference = dict(
    type="Inference",
    device=None,
    model=deploy_model,
    pre_processors=None,
    post_processors=None,
    model_convert_pipeline=None,
)
```

## 输入输出接口与前后处理

这里的输入输出接口，指对于一个具体的模型应用，运行{py:class}`Inference <hat.engine.inference.Inference>`类d的py:meth}`~hat.engine.inference.Inference.forward`方法时输入、输出分别所需要对应的结构。

为了保证一个模型应用的可用性和易用性，模型生产者应该在发布模型时给出明确的接口介绍。

可以看到，在{py:class}`Inference <hat.engine.inference.Inference>`类中，除了可以使用模型外，还提供了 `pre_processors`和 `post_processors`两个接口，用于在模型推理前后处理数据。这样，对于执行相同处理任务的模型，即使其本身输入输出接口不同，我们也可以通过定义 `pre_processors`和 `post_processors`来保证外部接口的统一。

无论输入、输出，我们都推荐大家使用[HATBC message](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/hatbc/-/tree/master/python/hatbc/message)中定义好的结构来表示。

### `pre_processors`与 `post_processors`的实现方式

在{py:class}`Inference <hat.engine.inference.Inference>`的实现中，为了降低配置复杂度，`pre_processors`、模型推理和 `post_processors`的执行是单进程串联进行的。因此从运行效率和利用率角度考虑，`pre_processors`和 `post_processors`中应尽量不包含大计算量的内容，只做格式转换的操作。

另外，从通信量的角度，由于{py:class}`Inference <hat.engine.inference.Inference>`内的计算操作可能在远程执行，对于一个模型应用应该降低输入输出接口对应的数据量，避免因为网络阻塞影响整体的运行效率。
