# 算子扩充

在之前的章节中，我们介绍了如何将一个模型通过torchdynamo提取FX graph，并进一步通过tensorRT编译优化。
然而在实践中，我们常常会遇到不支持的算子，或需要将不支持的算子转化为等价支持算子。

我们的所有扩展实现存放在[hat/utils/trt_fx_extension.py](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/tree/master/hat/utils/trt_fx_extension.py)。

接下来，我们将介绍如何为我们的推理模型扩充算子，我们根据自身经验总结为以下几种情况：

## 1. 不支持算子转化为等效支持算子

在优化过程中，我们常常会遇到算子支持不好，但通常能够将其做一定转化(或许需要增加一些限制条件)，进而做到等效替换。

下面我们给出一个实际优化案例：

### 案例1：mean替换AdaptiveAvgPool2d

我们发现对于算子`nn.AdaptiveAvgPool2d(output_size)`，其通常使用在SE block中，`output_size`常设为1。
当处理的feature map size大于一定分辨率时，trt会编译报错。
不难想到，我们可以使用`torch.mean()`来替代`output_size=1`时的情况。

我们通过如下流程来完成对算子的替换。
首先，我们需要将`nn.AdaptiveAvgPool2d(output_size)`注册到`acc_ops`([参考资料](https://github.com/pytorch/TensorRT/blob/main/docsrc/tutorials/getting_started_with_fx_path.rst))

```python
op_and_target=("call_function", nn.functional.adaptive_avg_pool2d)
@register_acc_op_mapping(
    op_and_target=op_and_target
)
@register_acc_op
def adaptive_avg_pool2d(*, input, output_size):
    if output_size == 1:
        return torch.mean(input, dim=[2, 3], keepdim=True)
    else:
        return nn.functional.adaptive_avg_pool2d(input=input, output_size=output_size)
```

通过`register_acc_op`将`nn.functional.adaptive_avg_pool2d`注册为`acc_ops`，
同时利用`register_acc_op_mapping`将算子进行映射。

然后，我们需要指定不同条件下的trt编译方式

```python
from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters.converter_utils import add_reduce_layer

@tensorrt_converter(adaptive_avg_pool2d)
def acc_ops_adaptive_avg_pool2d(network, target, args, kwargs, name):
    if kwargs["output_size"] == 1:
        # 不能支持对kwargs进行修改
        _kwargs = {}
        _kwargs["input"] = kwargs["input"]
        _kwargs["keepdim"] = True
        _kwargs["dim"] = [2, 3]
        return add_reduce_layer(
            network, target, args, _kwargs, trt.ReduceOperation.AVG, name
        )
    else:
        return acc_ops_adaptive_avg_poolnd(network, target, args, kwargs, name)
```

如此，我们便完成了当`output_size=1`时，替换为`torch.mean`的操作，并支持trt编译。
`torch_tensorrt`中将绝大多数torch常见op转化为`acc_op`，并支持trt优化。
大家可以利用这里的实现，熟悉不同op的处理方式。
[参考代码](https://github.com/pytorch/TensorRT/blob/master/py/torch_tensorrt/fx/converters/acc_ops_converters.py)


## 2. TensorRT提供plugin实现

TensorRT官方提供了大量扩充算子的实现，它们存放在`Plugin`中，我们可以利用极大地扩充支持的算子库。

下面，我们以`group_norm`为例，给出相关示例

### 案例2：Plugin扩展group_norm

和案例1类似，我们首先扩展`acc_ops`，这里不需要进行任何变化。
```python
@register_acc_op_mapping(
    op_and_target=("call_function", nn.functional.group_norm)
)
@register_acc_op
def group_norm(*, input, num_groups, weight, bias, eps):
    return nn.functional.group_norm(input=input, num_groups=num_groups, weight=weight, bias=bias, eps=eps)
```

然后，我们完成trt convert。

```python
from torch_tensorrt.fx.converters.converter_utils import get_trt_plugin, get_trt_tensor

@tensorrt_converter(group_norm)
def acc_ops_group_norm(network, target, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]
    num_group = kwargs["num_groups"]
    weight = get_trt_tensor(network, kwargs["weight"], f"{layer_name}_weight")
    bias = get_trt_tensor(network, kwargs["bias"], f"{layer_name}_bias")
    eps = kwargs["eps"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"GroupNorm2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    plugin_name = "GroupNormalizationPlugin"
    num_groups_field = trt.PluginField("num_groups", np.array(num_group, dtype=np.int32), trt.PluginFieldType.INT32)
    eps_field = trt.PluginField("eps", np.array(eps, dtype=np.float32), trt.PluginFieldType.FLOAT32)
    field_collection = trt.PluginFieldCollection(
        [num_groups_field, eps_field]
    )
    try:
        plugin = get_trt_plugin(plugin_name, field_collection, "1")
    except AssertionError:
        logger.info("Unable to find group norm plugin, fall back to TensorRT implementation.")
    layer = network.add_plugin_v2([input_val, weight, bias], plugin)
    layer.name = layer_name
    return layer.get_output(0)
```

需要注意的是，部分常量如num_group、eps，需要通过`trt.PluginField()`转化算子参数；
部分参数如weight、bias，其传进来时为`numpy.ndarray`，需要通过`get_trt_tensor`方法将其转化为tensorrt tensor。

我们可以前往[NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/plugin)寻找所需扩展。

## 3. 自定义plugin

待补充。。。


# FAQ

1. 如何确定模型中某个OP是否被支持？

首先我们需要明确的是，一个OP能够被tensorRT支持(在dynamo + fx这套框架下)，首先需要将其注册为`acc_ops`，即上面案例中第一步`register_acc_op`；然后，我们可选得提供该`acc_op`的tensorRT实现，当模型中某个OP未注册`acc_op`或未提供tensorRT的实现时，后端会进行相应提示，例如以下log:
```
Supported node types in the model:                                                                                                                  
acc_ops.pad: ((), {'input': torch.float32})
acc_ops.conv2d: ((), {'input': torch.float32, 'weight': torch.float32})
acc_ops.conv2d: ((), {'input': torch.float32, 'weight': torch.float32, 'bias': torch.float32})
acc_ops.batch_norm: ((), {'input': torch.float32, 'running_mean': torch.float32, 'running_var': torch.float32, 'weight': torch.float32, 'bias': torch.float32})
acc_ops.sigmoid: ((), {'input': torch.float32})             
acc_ops.mul: ((), {'input': torch.float32, 'other': torch.float32})
hat.utils.trt_fx_extension.adaptive_avg_pool2d: ((), {'input': torch.float32})
acc_ops.add: ((), {'input': torch.float32, 'other': torch.float32})

Unsupported node types in the model:                                                                                           acc_ops.group_norm: ((), {'input': torch.float32, 'running_mean': torch.float32, 'running_var': torch.float32, 'weight': torch.float32, 'bias': torch.float32})
```