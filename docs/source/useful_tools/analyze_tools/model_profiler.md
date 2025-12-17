# 如何使用各类模型检查分析工具

## 介绍

量化训练任务中经常会碰到精度损失的问题，用户可以使用这里提供的一系列工具来分析定位量化模型结构是否符合预期、模型精度损失具体发生在哪一层等。下面会介绍如何在HAT中配置使用这些工具，更多使用时的注意点和相关说明见 [模型精度DEBUG工具](http://model.aidi.hobot.cc/api/docs/horizon_plugin_pytorch/latest/html/tutorials/quant_profiler.html)

## 使用步骤

按照如下流程进行配置使用检查分析工具
### 1. 配置 config
在原有`config`文件的基础上增加`model_profiler_solver`关键字
```python
model_profiler_solver = dict(
    model=deploy_model,
    inputs=deploy_inputs,
    model_convert_pipeline=[
        dict(
            type="ModelConvertPipeline",
            qat_mode="with_bn",
            converters=[
                dict(type="Float2QAT"),
                dict(
                    type="LoadCheckpoint",
                    checkpoint_path=os.path.join(
                        ckpt_dir, "qat-checkpoint-best.pth.tar"
                    ),
                ),
            ],
        ),
        dict(
            type="ModelConvertPipeline",
            qat_mode="with_bn",
            converters=[
                dict(type="Float2QAT"),
                dict(
                    type="LoadCheckpoint",
                    checkpoint_path=os.path.join(
                        ckpt_dir, "qat-checkpoint-best.pth.tar"
                    ),
                ),
                dict(type="QAT2Quantize"),
            ],
        ),
    ],
    tool=dict(
        type="FeaturemapSimilarity",
        similarity_func="Cosine",
        threshold=None,
    )
)
```
若是在`predictor`中有定义`model_convert_pipeline`，也可直接使用`predictor`中定义的
```python
model_profiler_solver = dict(
    model=deploy_model,
    inputs=deploy_inputs,
    model_convert_pipeline=[
        qat_predictor["model_convert_pipeline"],
        int_infer_predictor["model_convert_pipeline"],
    ],
    tool=dict(
        type="FeaturemapSimilarity",
        similarity_func="Cosine",
        threshold=None,
    )
)
```
其中各个参数含义如下：  
`model`：表示要进行分析的模型结构  
`inputs`：表示模型的输入
`model_convert_pipeline`：表示模型的convert过程。若使用相似度计算工具、weight比较工具或者集成显示工具model_profiler，需要传入一个长度为2的list类型指定两个`model_convert_pipeline`（见上述示例）。而其他工具仅需指定一个，如`model_convert_pipeline=qat_predictor["model_convert_pipeline"]`
`tool`：表示当前使用的分析工具，同样以`dict`的形式指定要使用的分析工具即其参数。目前支持的如下几种分析工具都通过注册的机制注册在HAT中，用户也可以参考`hat/profiler/model_profiler.py`实现自定义的分析工具。

#### type=ModelProfiler
会调用其他debug工具，并将结果集中显示到一个html页面中
```python
tool=dict(
    type="ModelProfiler",
    mode="FvsQ",
    out_dir=None,
    kwargs_dict=dict(
        FeaturemapSimilarity=dict(
            similarity_func="Cosine"
        )
    )
)
```

- `mode`：表示进行比较的是哪两个模型，仅支持以下三种模式
    - `FvsQ`：float模型和qat/calibration模型对比
    - `QvsQ`：qat模型和quantized模型对比
    - `CvsQ`：calibration模型和qat模型对比
- `out_dir`：指定输出的结果文件`profiler.html`和所有debug工具调用结果的路径。默认为`None`，会在`ckpt_dir`指定的目录下或当前目录下生成`profiler`目录，并将所有结果存储在该目录下。
- `kwargs_dict`：调用其他debug工具时的参数，以`dict`的形式给出。**具体的参数可以参考下面每个工具的具体介绍**。支持7个key值
    - `FeaturemapSimilarity`：相似度
    - `ProfileFeaturemap`：统计量函数，输出模型中每一层结果的最大最小值，均值和方差等
    - `CheckShared`：检查模型是否有共享op
    - `CheckFused`：检查模型是否有未fuse的pattern
    - `CompareWeights`：比较两个模型中weight的相似度
    - `CheckQConfig`：检查QAT模型中的Qconfig配置


#### type=FeaturemapSimilarity
相似度计算，打印两个模型每一层输出结果的相似度
```python
tool=dict(
    type="FeaturemapSimilarity",
    similarity_func="Cosine",
    threshold=None,
    devices=torch.device("cpu"),
    out_dir=None,
)
```
- `similarity_func`：计算相似度时使用的函数，目前支持`Cosine`、`MSE`、`L1`、`KL`，用户也可以传入自定义的相似度计算函数
- `threshold`：使用`Cosine`函数时，值越接近1相似度越高，`threshold`默认为0，小于`threshold`的值会以红色高亮打印；使用其他函数时，值越小相似度越高，`threshold`默认为1，大于`threshold`的值会以红色高亮打印
- `devices`：指定计算相似度时模型在哪个`device`上进行`forward`。若为`None`，则默认在模型输入时的 device 上进行 forward；若仅有一个参数如`torch.device("cpu")`，则会把两个模型均移动到指定的 device 上 forward；若指定了两个值如`(torch.device("cpu"), torch.device("cuda"))`，则会把两个模型分别移动到对应的 device 上 forward。一般用于比较同一个模型同一个阶段的 `CPU/GPU` 的中间结果。
- `out_dir`：指定输出的结果文件和图片的路径。默认为`None`，会保存到`ckpt_dir`参数指定的目录或当前目录下的`profiler`目录。

#### type=ProfileFeaturemap
统计量工具，统计每一层输入输出及每一层参数的最大最小值均值方差等并输出
```python
tool=dict(
    type="ProfileFeaturemap",
    # get_raw_features args
    prefixes=(),
    types=(),
    device=None,
    preserve_int=False,
    use_class_name=False,
    skip_identity=False,
    # profile_featuremap args
    with_tensorboard=False,
    tensorboard_dir=None,
    print_per_channel_scale=False,
    show_per_channel=False,
    out_dir=None,
    file_name=None,
    profile_func=None,
)
```
统计量工具内部通过两个函数配合使用
- `get_raw_features`：记录模型中每一层的输出结果。其参数如下
    - `prefixes`：指定要输出统计量的 op 在模型中对应的 layer name（以 prefixes 开头的 layer）
    - `types`：指定要输出统计量的 op 的类型
    - `device`：指定模型在 CPU/GPU 上 forward
    - `preserve_int`：是否以定点数值的形式输出。默认输出为浮点值。该参数仅对 qat 和定点模型生效，且只会在该层输出有 scale 的情况下生效（比如，dequant 层输出的结果是浮点，该参数就不起效果）
    - `use_class_name`：是否打印每一层 op 的 name，默认打印的是 op 的类型
    - `skip_identity`：是否跳过 Identity op 的统计。默认所有类型的 op 都会输出统计量
- `profile_featuremap`：统计每一层结果的最大最小值、均值方差等。其参数如下
    - `with_tensorboard`：是否使用tensorboard可视化数据分布，默认为`False`
    - `tensorboard_dir`：tensorboard使用的log文件路径，仅在`with_tensorboard=True`时生效。默认`None`，会在当前目录下生成log文件
    - `print_per_channel_scale`：是否打印`per_channel`量化的数据的scale，默认为`False`
    - `show_per_channel`：在 tensorboard 中是否以 per channel 的方式显示 feature 中每个 channel 的数据直方图。默认为 False。
    - `out_dir`：指定输出的结果文件和图片的路径。若未指定，则保存到`ckpt_dir`参数指定的目录或当前目录下的`profiler`目录。
    - `file_name`：保存的文件和图片的名字。若未指定，默认为`statistic.txt`和一个可交互的`statistic.html`。
若想要其他一些统计信息，可以通过`profile_func`参数传递一个自定义的数据统计函数。
  
#### type=CheckShared
检查模型中的共享op，打印模型中每一个op的调用次数，超过1次即为共享op
```python
tool=dict(
    type="CheckShared",
    check_leaf_module=None,
    print_tabulate=True,
)
```
- `check_leaf_module`：判断leaf module的函数，默认为`None`，将torch社区和plugin中定义的op作为leaf module，记录被调用的次数。用户也可以传入自定义的leaf module判断方法
- `print_tabulate`：是否打印结果，默认`True`

#### type=CheckFused
检查模型中是否有未fuse的pattern，会打印出模型中可以fuse的pattern，目前仅支持检查float模型或者fused之后还未转qat的模型
```python
tool=dict(
    type="CheckFused",
    print_tabulate=True,
)
```
- `print_tabulate`：是否打印结果，默认`True`

#### type=CheckQConfig
检查qat模型中每一层的量化配置，包括模型中每一层的输出 activation 和 weight 的量化配置，以及模型中每一层的输入输出类型。配置信息会保存在`qconfig_info.txt`中。在检查到下列情况时会在屏幕输出提示信息。
- 输出层 activation 没有量化
- 固定 scale
- 非 int8 量化的 weight（目前仅支持 int8 量化的 weight）
- 模型输入输出类型不一样

```python
tool=dict(
    type="CheckQConfig",
    prefixes=(),
    types=(),
    custom_check_func=None,
    out_dir=None,
)
```
- `prefixes`：指定要检查量化配置的 op 在模型中对应的 layer name（以 prefixes 开头的 layer）
- `types`：指定要检查量化配置的 op 的类型
- `custom_check_func`：用户自定义函数，用于检查其他信息。这个函数在 module 的 hook 中调用，因此需要定义为格式：`func(module, input, output) -> None`
- `out_dir`：保存结果文件`qconfig_info.txt`的路径。若为 None，则保存到`ckpt_dir`参数指定的目录或当前目录下的`profiler`目录

#### type=CompareWeights
比较`float/qat/quantized`模型中的`weights`。比较模型中每一层的weight。weight 相似度和 atol 将会打印到屏幕同时保存到“weight_comparison.txt”。用户还可以设置 with_tensorboard=True，将 weight 直方图通过 tensorboard 打印。
```python
tool=dict(
    type="CompareWeights",
    similarity_func="Cosine",
    with_tensorboard=False,
    tensorboard_dir=None,
    out_dir=None,
)
```
- `similarity_func`：相似度计算函数。支持 Cosine/MSE/L1/KL/SQNR 和任意用户自定义的相似度计算函数。如果是自定义的函数，须返回标量或者仅含一个数的 tensor，否则结果显示可能不符合预期。默认为 Cosine
- `with_tensorboard`：是否使用 tensorboard，默认为 False
- `tensorboard_dir`：tensorboard 日志文件路径。默认为 None
- `out_dir`：保存 txt 结果的路径。默认为 None, 保存到`ckpt_dir`参数指定的目录或当前目录下的`profiler`目录

#### type=CheckDeployDevice
检查异构模型部署时的device，输入必须是在convert时指定`convert_mode="fx"`得到的模型。默认会在屏幕打印结果，并将结果保存到`deploy_device.txt`文件。
```python
tool=dict(
    type="CheckDeployDevice",
    print_tabulate=True,
    out_dir=None,
)
```
- `print_tabulate`：是否打印结果，默认`True`
- `out_dir`：保存 txt 结果的路径。默认为 None, 会保存到`ckpt_dir`参数指定的目录或当前目录下的`profiler`目录

### 2. 运行分析工具
配置好`config`文件后即可以使用`tools/analyze/model_profiler.py`脚本运行
```shell
python3 tools/analyze/model_profiler.py -c config.py
```
脚本仅需输入`config`参数
- `-c, --config`：必选参数，为config配置文件

结果将保存到 txt 文件，同时绘制相似度的变化曲线，保存成图片。会生成如下文件：
  - similarity.txt：按照模型 forward 的顺序打印每一层的相似度和单算子误差等结果
  - ordered_op_error_similarity.txt：按照**相同输入下单算子误差**从高到低进行排序的结果，方便用户快速定位是哪个 op 的 convert 误差较大
  - similarity.html：一个可交互的图片，显示随着模型 forward，每一层相似度的变化曲线。可以放大缩小，光标移动到对应的点可以显示具体的相似度数值。
```text
---------------------------------------------------------------
Note:
* Suffix '(I)' means this layer is Identity in one model
* Suffix '(I vs I)' means this layer is Identity in both models
* Suffix '(i)'(i >= 1) means this op is shared i times
---------------------------------------------------------------
+---------+------------------------------+--------------------------------------------------------------+--------------+-----------+----------------+------------------+------------------------+
| Index   | Module Name                  | Module Type                                                  | Similarity   | qscale    | Acc Error      | Acc Error        | Op Error with Same     |
|         |                              |                                                              |              |           | (float atol)   | (N out_qscale)   | Input (N out_qscale)   |
|---------+------------------------------+--------------------------------------------------------------+--------------+-----------+----------------+------------------+------------------------|
| 0       | backbone.quant               | <class 'horizon_plugin_pytorch.nn.qat.stubs.QuantStub'>      | 1.0000000    | 0.0078125 | 0.0000000      | 0                | 0                      |
| 1       | backbone.mod1.0.0            | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9999979    | 0.0179124 | 0.0179124      | 1                | 1                      |
| 2       | backbone.mod1.1              | <class 'horizon_plugin_pytorch.nn.qat.max_pool2d.MaxPool2d'> | 0.9999993    | 0.0179124 | 0.0179124      | 1                | 0                      |
| 3       | backbone.mod2.0.conv.0.0     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9999742    | 0.0118339 | 0.0118339      | 1                | 1                      |
| 4       | backbone.mod2.0.short_add    | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvAddReLU2d'> | 0.9999862    | 0.0214140 | 0.0214140      | 1                | 1                      |
| 5       | backbone.mod2.1.conv.0.0     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9998535    | 0.0101941 | 0.0203882      | 2                | 1                      |
| 6       | backbone.mod2.1.short_add    | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvAddReLU2d'> | 0.9999571    | 0.0219844 | 0.0439688      | 2                | 1                      |
| 7       | backbone.mod3.0.conv.0.0     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9996806    | 0.0123873 | 0.0247746      | 2                | 1                      |
| 8       | backbone.mod3.0.downsample.0 | <class 'horizon_plugin_pytorch.nn.qat.conv2d.Conv2d'>        | 0.9997617    | 0.0125587 | 0.0251174      | 2                | 1                      |
| 9       | backbone.mod3.0.short_add    | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvAddReLU2d'> | 0.9995903    | 0.0132832 | 0.0265664      | 2                | 1                      |
| 10      | backbone.mod3.1.conv.0.0     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9993098    | 0.0052046 | 0.0156139      | 3                | 1                      |
| 11      | backbone.mod3.1.short_add    | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvAddReLU2d'> | 0.9994799    | 0.0132696 | 0.0398088      | 3                | 1                      |
| 12      | backbone.mod4.0.conv.0.0     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9989703    | 0.0045097 | 0.0180387      | 4                | 1                      |
| 13      | backbone.mod4.0.downsample.0 | <class 'horizon_plugin_pytorch.nn.qat.conv2d.Conv2d'>        | 0.9990988    | 0.0064077 | 0.0256307      | 4                | 1                      |
| 14      | backbone.mod4.0.short_add    | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvAddReLU2d'> | 0.9989563    | 0.0056222 | 0.0224889      | 4                | 1                      |
| 15      | backbone.mod4.1.conv.0.0     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9989240    | 0.0020031 | 0.0120185      | 6                | 1                      |
| 16      | backbone.mod4.1.short_add    | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvAddReLU2d'> | 0.9989610    | 0.0058822 | 0.0235289      | 4                | 1                      |
| 17      | backbone.mod5.0.conv.0.0     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9992908    | 0.0021313 | 0.0106564      | 5                | 1                      |
| 18      | backbone.mod5.0.downsample.0 | <class 'horizon_plugin_pytorch.nn.qat.conv2d.Conv2d'>        | 0.9994005    | 0.0025097 | 0.0125486      | 5                | 1                      |
| 19      | backbone.mod5.0.short_add    | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvAddReLU2d'> | 0.9993593    | 0.0027239 | 0.0136197      | 5                | 1                      |
| 20      | backbone.mod5.1.conv.0.0     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvReLU2d'>    | 0.9996194    | 0.0013942 | 0.0055767      | 4                | 1                      |
| 21      | backbone.mod5.1.short_add    | <class 'horizon_plugin_pytorch.nn.qat.conv2d.ConvAddReLU2d'> | 0.9994555    | 0.0026770 | 0.0133850      | 5                | 1                      |
| 22      | head.0                       | <class 'horizon_plugin_pytorch.nn.qat.avg_pool2d.AvgPool2d'> | 0.9999515    | 0.0020154 | 0.0020154      | 1                | 0                      |
| 23      | head.1.0                     | <class 'horizon_plugin_pytorch.nn.qat.conv2d.Conv2d'>        | 0.9999067    | 0.0012452 | 0.0012452      | 1                | 1                      |
| 24      | dequant                      | <class 'horizon_plugin_pytorch.nn.qat.stubs.DeQuantStub'>    | 0.9999067    |           | 0.0012452      | 0.00124524       | 0                      |
+---------+------------------------------+--------------------------------------------------------------+--------------+-----------+----------------+------------------+------------------------+
```

## 其他
更多debug分析工具可查看下列文档：
- [DEBUG工具使用指南](https://horizonrobotics.feishu.cn/wiki/wikcnqH3VgKZg0H41E0vTjdZYzJ)