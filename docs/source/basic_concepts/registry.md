(hat-registry)=
# 注册机制
注册机制是辅助构建`config`的重要模块，也是`HAT`的重要组成部分。

本小节通过自定义模块的例子说明如何在注册机制下在增加新的模块并在`config`中正常使用。

## 自定义模块
以`backbone`为例，这里展示一下如何开发以`mobilenet`为例的新模块。

### 1. 定义一个新的backbone（如MobileNet）：

新建一个新文件：`hat/models/backbones/mobilenet.py`

```python
import torch.nn as nn
from hat.registry import OBJECT_REGISTRY

__all__ = ["MobileNet"]

@OBJECT_REGISTRY.register
class MobileNet(nn.Module):
    def __init__(self, args1, args2):
        pass
    def forward(self, x):
        pass
```

### 2. 导入新定义的模块

可以在`hat/models/backbones` 下新建一个 `__init__.py` 文件

### 3. 在`config`中使用新的`backbone`

```python
model = dict(
    ...
    backbone=dict(
        type="MobileNet",
        args1=xxx,
        args2=xxx,
    )
    ...
)
```
在 `build` 的时候会默认将所有的模块注册进去，以此类推，其他任何可注册的模块，都可以使用这种方法来完成开发和使用。
