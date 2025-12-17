# 如何开启channels last

Channels last是torch在训练过程中，已经支持的存储格式。而且channels last配合AMP使用，还会有进一步的速度提升效果。详见[pytorch官方文档](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

HAT 中已经为channels last做好相关的封装工作，用户只需要在定义config文件中的`batch_processor`字段时将`enable_channels_last`参数设置为`True`即可。这样设置可以把数据和模型同时设置成channels last的数据格式。同时配合`channels_last_keys`的字段可以把数据中指定的key转换成channels last的格式，比如常见的图像任务中的`imgs`。


```python

# 使用 BasicBatchProcessor
batch_processor = dict(
    type='BasicBatchProcessor',
    need_grad_update=...,
    batch_transforms=...,
    enable_channels_last=True,
    channels_last_keys=("imgs",), # 也可以不设置，不设置的时候默认所有的4d-tensor都会转换
)

# 使用 MultiBatchProcessor
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=...,
    batch_transforms=...,
    loss_collector=...,
    enable_channels_last=True,
    channels_last_keys=("imgs",), # 也可以不设置，不设置的时候默认所有的4d-tensor都会转换
)
```

需要说明的是，并不是所有的算子都支持channels_last，常见的算子支持列表在[这里](https://github.com/pytorch/pytorch/wiki/Operators-with-Channels-Last-support)。但是不支持的算子不会出错。算子在channel_last转换过程中有以下几种特征：

1. 不支持算子转channels_last的过程不会出错。

2. 不支持算子不管输入的memory_format是什么，输出格式都会变成contiguous_format。

3. 已经转成channels_last的算子，不管输入的memory_format是什么，输出格式都会变成channels_last。典型的算子如Conv，这种情况下Conv默认采用channels_last的计算方式。

4. 已经转成channels_last的算子，当输入是channels_last的时候，不管算子参数的memory format是什么，输出格式都会变成channels_last。典型的算子如Conv，这个情况下Conv默认采用channels_last的计算方式。
