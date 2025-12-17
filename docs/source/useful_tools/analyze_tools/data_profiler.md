# 如何使用和数据相关的模型训练性能分析工具

在训练速度优化中，需要分别测试数据读取、数据增广操作、dataloader 等对训练性能的影响以便定位瓶颈指导后续优化方向。此外，不同存储系统或者是机器甚至不同时间段用户数量都可能影响数据读取性能，对训练性能影响较大。为了测试稳定可复现的训练速度 baseline，一般需要屏蔽数据读取的不稳定带来的影响。本文介绍一些实用的屏蔽数据影响以及分析数据相关操作对性能影响的工具及用法。

此外本节还将介绍一些对数据读取进行测速，对 transform 过程进行瓶颈分析的工具。

## 屏蔽数据相关操作给训练带来的影响

屏蔽数据影响的原理是针对各个数据处理的进程，仅执行一次数据相关操作（如读数据、transform、dataloader）将其输出保存，在之后的训练流程中该操作将被去除，被保存的结果作为后续操作的输入。

### 去除数据读取操作

(1) 作用

去除数据读取操作可以屏蔽读数据带来的不稳定性，同时可以通过对比训练过程有无读取数据评估读取数据对训练性能的影响。建议在测试训练速度的 baseline 时将该功能打开。

(2) 限制

目前仅支持通过 Lmdb 读取数据的方式。

(3) 用法

在 config 文件的 dataset 中配置 pack_kwargs 参数，示例如下：

```python
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        ...
    ),
    ...
)
data_loader["dataset"]["pack_kwargs"] = {"fixed_read_data": True}
```

### 去除读取数据、解码、及 transform 等操作（固定数据集）

(1) 作用

本操作将去除 dataset 获取实现层面的数据读取、解码、transforms 等操作。可以通过对比训练过程有无 dataset 系列操作来评估其对训练性能的影响，可以用于定位 dataset 的获取是否是瓶颈。

(2) 用法

在 config 文件的 dataset 中配置使用 FixedDataset。FixedDataset 将会对原来的 dataset 进行封装。示例如下：

```python
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        ...
    ),
    ...
)
data_loader["dataset"] = dict(
    type="FixedDataset",
    dataset=data_loader["dataset"]
)
```

### 每个进程始终读取单个固定的数据

(1) 作用

长时间读取单个数据，从第二次读取开始就是从缓存中读取数据。通过对比读取单个数据与否训练的性能的区别来评估是否从缓存中读取数据对训练的影响。同时可以结合数据读取测速工具对比是否使用缓存读取数据的速度快慢。

(2) 用法

在 config 文件的 dataset 中配置使用 FixedDataset，并且设置一个大于 0 的 fixed_idx 参数。

```python
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        ...
    ),
    ...
)
data_loader["dataset"] = dict(
    type="FixedDataset",
    dataset=data_loader["dataset"],
    fixed_idx=1,
)
```

### 全程使用随机生成的固定数据

(1) 作用

使用将随机生成的数据作为模型输入。由于用户需要具备在配置文件中添加生成随机数据的能力，且一些模型的输入较为复杂，因此推荐在完全无法获取数据的情况下使用该功能，否则推荐固定数据集(FixedDataset)来达到相似的目的。

(2) 用法

在配置文件中使用 RandDataset，并将随机数据的各个输入项都配置为随机数。以随机生成常见的分类网络数据集示例：

```python
import torch
import random
dataset = dict(
    type="RandDataset",
    length=10000,
    example=dict(
        img=torch.randn(3, 224, 224),
        labels=random.randint(0, 999),
    ),
)
```

在配置随机数据集的时候比较复杂的是搞明白模型的输入数据有哪些项以及形状，如 YOLO 模型的输入就会比较复杂：

```python
dataset = dict(
    type="RandDataset",
    length=10000,
    example=dict(
        img=torch.randn((3, 416, 416)),
        gt_bboxes=torch.randn((10, 4)),
        gt_classes=torch.randint(0, 20, (10,)),
        gt_difficults=torch.rand((10,)),
        gt_labels=torch.cat(
            (
                torch.randn((10, 4)),
                torch.randint(0, 20, (10,)).unsqueeze(-1),
            ),
            -1,
        ),
        layout="hwc",
        color_space="rgb",
    ),
)
```

### 去除 dataloader 过程

(1) 作用

dataloader 过程包含了从读取数据到合成 batch 的完整过程，去除 dataloader 操作意味着完全去除数据对训练的影响，可用于粗粒度地辅助定位 dataloader 是否是训练性能瓶颈。

(2) 用法

在 config 中配置使用 FixedDataloader。示例如下：

```
data_loader=dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        ...
    ),
    ...
)
data_loader = dict(
    type="FixedDataLoader",
    dataloader=data_loader,
)
```

### 为 dataloader 指定输出

(1) 作用

为 dataloader 指定一个输出

(2) 用法

在 config 中配置使用 PassThroughDataLoader.示例：

data_loader = dict(
    type="PassThroughDataLoader",
    example=torch.randn(100, 3, 224, 224),
    length=100,
)

## 对速度读取进行测速

(1) 作用

可以在训练 log 中看到读取数据的速度。目前仅支持了 Lmdb 读取。

(2) 用法

目前使用 Lmdb 读数据已经默认含有测速功能，无需额外设置，默认每十分钟将速度输出到监控日志中，用户可以通过 HAT_MONITOR_INTERVAL 环境变量来设置输出到监控日志中的间隔时长，单位为秒，默认值为 600；通过设置 HAT_MONITOR_ALL_TIME=1 将会打印每一次数据读取的速度及读取文件名信息。监控日志一般存储在统一的 log 目录下，文件名为 data-profiler, 包含所有进程读取数据的速度信息。

## 对 transform 操作进行性能瓶颈分析

transform 瓶颈分析工具可以基于 PythonProfiler 和 SimpleProfiler 进行。使用 PythonProfiler perf transform 可以分析出其中具体耗时的操作，使用 SimpleProfiler 可以看出各个 transform 耗时的多少及在所有 transform 中耗时的占比。
用法是在 config 文件中配置 transform 为 PerfTransform 即可，示例如下：

```python
from hat.utils.logger import LOG_DIR
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        ...
        transforms=[
            ....
        ],
    ),
    ...
)

data_loader["dataset"]["transforms"] = dict(
    type="PerfTransforms",
    transforms=data_loader["dataset"]["transforms"],
    profiler=dict(
        type="PythonProfiler",
        # type="SimpleProfiler",
        dirpath="./",
        filename="profiler_log",
    ),
    perf_data_len=200,
)
```

注意，对 transform 的 perf 时长取决于 perf_data_len，如果训练时间太短，而 perf_data_len 设置得较大会导致 transform 的 perf 结果没有保存下来。

## 分析 dataset 的内存占用

在 config 文件中配置使用 PerfDataset 搭配 StageCPUMemoryProfiler 对 dataset 内存进行分析，示例如下：
```python

profiler=dict(
    type="StageCPUMemoryProfiler",
    profile_action_name="perf_dataset",
    leaks=False,
    dirpath="perf_log1",
    filename="stage_cpu_profiler",
)
dataset=dict(
    type="ImageNet",
    data_path="./tmp_data/imagenet/train_lmdb/",
    transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="RandomResizedCrop",
            size=224,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        ),
    ],
)
dataset=dict(
    type="PerfDataset",
    dataset=dataset,
    profiler=profiler,
    perf_interval=128,
)
```

由于 StageCPUMemoryProfiler 只能记录某个代码段单次执行的内存使用过程，暂时还做不到统计多次执行的内存占用情况，为了防止记录文件过多，用户需要手动设置一下 perf_interval 参数，用于控制 perf 内存的间隔，每隔 perf_interval 对内存占用进行一次统计和记录文件导出。

perf 结束之后，会在 perf_log1，也就是用户在 profiler 中所设置的目录下生成后缀为 .html 的文件，将 html 文件用浏览器打开就可以看到各个操作的内存耗时情况。
## 分析 dataset 的内部操作耗时

在 config 文件中配置使用 PerfDataset 搭配 PythonProfiler 对 dataset 进行耗时分析，示例如下：

```python
python_profiler = dict(
    type="PythonProfiler",
    dirpath="work_dirs/hat_logss",
    filename="python_profiler",
)
dataset = dict(
    type="PerfDataset",
    dataset=dict(
        type="SimpleDataset",
        start=0,
        length=dataset_length,
    ),
    profiler=python_profiler,
    perf_data_len=100,
)
```

在 perf 结束之后会在 work_dirs/hat_logss 目录下生成后缀为 python_profiler.txt 的 log, 其内部格式示例如下：

```shell
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     3000   64.479    0.000 3140.630    1.047 module_patch.py:41(_wrap)
     3000   34.494    0.000 3140.362    1.047 module.py:1096(_call_impl)
     3000    5.815    0.002 3140.213    1.047 distributed.py:852(forward)
     3000    1.484    0.000 3002.369    1.001 graph_model.py:458(forward)
     3000    0.135    0.000 2991.150    0.997 basic.py:59(__call__)
     3018    0.095    0.000 2990.649    0.991 symbol.py:275(post_order_dfs_visit)
     3018    2.399    0.001 2990.548    0.991 graph_algo.py:6(post_order_dfs_visit)
      500    0.464    0.000 2982.733    0.019 basic.py:170(inner)
      500    3.040    0.000 2982.195    0.019 basic.py:87(fvisit)
      500   14.904    0.000 1562.301    0.005 container.py:139(forward)
       80   11.895    0.000 1417.618    0.002 conv_module.py:76(forward)
       20    1.346    0.000 1175.914    0.098 mixvargenet.py:254(forward)
```

主要关注第四列 cumtime，记录了某个操作(包含其子操作)的耗时。
