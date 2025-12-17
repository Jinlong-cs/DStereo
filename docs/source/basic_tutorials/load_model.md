# 如何加载模型

## 介绍

HAT集成了加载模型参数的功能。支持加载checkpoint或者torchscript pt文件，加载过程中可以对`horizon-plugin-pytorch`的版本进行检查，对版本不一致的情况进行warning提示。

## 使用示例

使用HAT算法包进行模型训练时，保存的checkpoint和者torchscript pt文件会自动添加`horizon-plugin-pytorch`的版本号。
加载模型时，可参照如下示例检查`horizon-plugin-pytorch`版本的一致性。

### 1.load checkpoint文件

使用HAT集成的`load_checkpoint`接口可加载checkpoint文件，使用示例如下：
```python
from hat.utils.checkpoint import load_checkpoint
checkpoint = load_checkpoint(
    path_or_dict="your_model_path",
    map_location="cpu",
    state_dict_update_func=default_update_func,
    check_hash=True,
    check_plugin_version=True,
)
```
其中各个参数含义如下：

`path_or_dict`：表示要加载的模型参数地址

`map_location`：模型加载的目标设备

`state_dict_update_func`：`state_dict`更新函数

`check_hash`：是否检查文件hash值

`check_plugin_version`：是否检查文件中的`horizon-plugin-pytorch`版本号，没有版本号或者版本号不一致会给出warning提示

其中，`path_or_dict`为必须配置的选项，如果需要检查`horizon-plugin-pytorch`的版本，则需要设置`check_plugin_version`为`True`。

### 2.load torchscript pt文件

加载`horizon-plugin-pytorch`转换的定点模型时，一般需要`import horizon_plugin_pytorch`。使用HAT集成的`load_script_module`接口则可直接加载torchscript pt文件，使用示例如下：
```python
from hat.utils.checkpoint import load_script_module
pt_model = load_script_module(
    pt_path="your_model_path",
    map_location="cpu",
    _extra_files=None,
    check_plugin_version=True,
)
```
其中各个参数含义如下：

`pt_path`：表示要加载的模型参数地址

`map_location`：模型加载的目标设备

`_extra_files`：额外加载的信息

`check_plugin_version`：是否检查文件中的`horizon-plugin-pytorch`版本号，没有版本号或者版本号不一致会给出warning提示

其中，`pt_path`为必须配置的选项，如果需要检查`horizon-plugin-pytorch`的版本，则需要设置`check_plugin_version`为`True`。
