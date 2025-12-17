# 模型转换流程介绍

## 背景

1. 存在量化训练、部署的需求，使得模型在整个声明周期中可能存在多个形态（对应多个阶段），如常见的`float`、`calibration`、`qat`、`int_infer`，或在纯量化训练的语境中的`with_bn`, `fuse_bn`。

2. 在训练过程中存在的对于checkpoint不同的使用方式（pretrain、pre_stage、resume），又与各阶段模型的获取耦合在了一起，可能需要获取模型转换过程中的中间结果。

3. 不同形态的模型一般可以通过一个模型config定义，只是在构建过程中被转化成不同的形态。

综上，我们的模型构建方法应该是：在根据config里的定义初始化模型后，通过一个灵活、透明的模型转换流程（convert pipeline）得到最终用于训练、预测的模型，来支持多样化的使用方式。

## 调用方式

由于从整体框架上，我们用`Loop`描述模型的一次完整行为（如完成一个训练、完成一次评测等），因此将调用方式设定为：在基类`LoopBase`的初始化阶段进行模型转换的相关操作，使得后续流程可以正常进行。

## 实现方式

整体分为`converter`和`pipeline`两层

### converter

其基类为`BaseConverter`。

每个`BaseConverter`的子类应该定义一个简单的不可再分的模型转换行为。比如`Float2QAT`将浮点模型转换为qat模型，而`LoadCheckpoint`则将checkpoint文件中的参数以自定义方式加载到模型上。

### pipeline

描述一个完整的模型转换流程，具体行为是：

1. 设定一些与模型相关的全局化的变量，比如`qat_mode`，`qconfig_params`与模型的量化行为相关。

2. 调用各种converter，进行具体的模型转换。

一个基础的实现为`ModelConvertPipeline`，其功能是：通过顺序执行在初始化阶段接收的多个`ModelConverter`实例，完成模型的转换。

这里举一个简单的例子，展示如何完成qat模型的转换并读取相应的qat checkpoint。

```python
model_convert_pipeline = ModelConvertPipeline(
    qat_mode="fuse_bn",
    qconfig_params=dict(
        activation_qat_qkwargs=dict(
            averaging_constant=0.0,  # 0.0 ~ 1.0 之间的浮点数
        ),
        weight_qat_qkwargs=dict(
            averaging_constant=1.0,  # 0.0 ~ 1.0 之间的浮点数
        ),
    ),
    converters=[
        Float2QAT(),
        LoadCheckpoint(,
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
        ),
    ],
)
```
如果使用者的需求与上述例子不同，比如想读取一个浮点checkpoint，并在qat训练中使用，则可以将`converters`部分稍加修改
```python
converters=[
    LoadCheckpoint(,
        checkpoint_path=os.path.join(
            ckpt_dir, "float-checkpoint-best.pth.tar"
        ),
    ),
    Float2QAT(),  
]
```
就能继续使用。

## 配置方式

在`{stage}_trainer`中的`model_convert_pipeline`关键字下定义相应操作即可。

## 扩展方法

欢迎贡献代码！

### converter层面

如果`HAT`中现有的converter实现无法满足您的需求，直接基于`BaseConverter`基类实现相关功能即可。

## pipeline层面

如果您发现，目前的`ModelConvertPipeline`实现无法满足您的需求，或使用起来不够方便（比如存在convert数量较多但模式较为固定的情况，可以通过实现一个子类减轻config负担），也可以实现符合您自己需求的pipeline方法。除非需求本身较为特殊，也建议基于现有的`ModelConvertPipeline`进行开发。

`QATFuseBNConvertPipeline`即为一个扩展pipeline的例子。

before:

```python
# pre_step
ModelConvertPipeline(
    qat_mode=xxx
    qconfig_params=yyy
    converters=[
        Float2Qat(),
        QATFusePartBN(qat_fuse_patterns=pre_step_patterns),
        LoadCheckpoint(checkpoint_path),
        QATFusePartBN(qat_fuse_patterns=cur_step_patterns),
    ],
)

# resume
ModelConvertPipeline(
    qat_mode=xxx
    qconfig_params=yyy
    converters=[
        Float2Qat(),
        QATFusePartBN(qat_fuse_patterns=pre_stage_fuse_patterns),
        QATFusePartBN(qat_fuse_patterns=cur_stage_fuse_patterns),
        LoadCheckpoint(checkpoint_path),
    ],
)
```

after:

```python
QATFuseBNConvertPipeline(
    qat_mode=xxx
    qconfig_params=yyy
    pre_stage_fuse_patterns=pre_stage_fuse_patterns,
    cur_stage_fuse_patterns=cur_stage_fuse_patterns,
    checkpoint_mode={"pre_step" or "resume"},
    checkpoint_configs=dict(
        checkpoint_path=checkpoint_path,
        ...
    )
)
```

通过以上的相同pipeline声明方式的对比，可以看出，在应用方式相对确定的情况下，通过简单的扩展，即可降低配置的复杂程度，提供更易用的接口。
