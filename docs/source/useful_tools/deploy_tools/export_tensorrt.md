# 如何将模型导出TensorRT

## 介绍

HAT集成使用torch-tensorrt将模型导出支持TensorRT运行模式的工具。其中torch-tensorrt工具的安装和使用方法可以参考[飞书文档](https://horizonrobotics.feishu.cn/wiki/wikcnrMtwgFz7wfeQGnzNQKWZie)

这里需要注意的是，环境的安装必须严格按照上面文档的要求。

HAT中已经内置了安装torch-tensort的自动化脚本，根据版本的区别，分别为:

```bash

# torch2.0.1+cu11.8
source ./dev/install_torch201_tensorrt.sh


# torch1.13.0+cu11.6
source ./dev/install_torch1130_tensorrt.sh

# torch1.10.2
source ./dev/install_torch1102_tensorrt.sh

```

选择一种安装即可。


## 使用步骤

### 配置config

在config文件的基础上增加`tensorrt_cfg`的关键字，如下：

```python
tensorrt_cfg = dict(
    model=deploy_model["backbone"],
    example_inputs=deploy_inputs["img"],
    device="0",
    input_cfg=dict(
        shape=[1, 3, 224, 224],
        dtype=torch.float32,
    ),
    compile_cfg=dict(
        enabled_precisions={torch.float32},
        truncate_long_and_double=True,
    )
)
```

其中，各个参数的含义如下：

`model`:表示需要导出TenosrRT的模型。

`example_inputs`: 表示需要导出TensorRT模型的输入情况。

`out_dir`: 需要保存TensorRT的目录地址。

`device`: TensorRT指定运行的device。

`input_cfg`: torch_tensorrt.ts.compile接口需要的input信息，其中dtype也可以使用torch.half。

`compile_cfg`: torch_tensorrt.ts.compule接口需要的各项参数，详细信息可以参考[官方文档](https://pytorch.org/TensorRT/py_api/ts.html)


### 运行脚本

直接执行tools/deploy/export_tensorrt.py的脚本即可。

```shell
python3 tools/deploy/export_tensorrt.py --config {path_of_config}
```

### 预测方法

保存下来的`ts`文件是可以直接运行，或者嵌入到torch.nn.Module中执行的。执行的方法如下：

```python
trt_ts_module = torch.jit.load("trt_ts_module.ts")
result = trt_ts_module(input_data)
```
