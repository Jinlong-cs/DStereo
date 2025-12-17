# 自定义backends

在前序章节中，我们介绍了torchDynamo的核心功能为提取任意python函数为FX graph。进一步地，我们可以通过指定backends来完成编译优化。

torchDynamo已经提供了丰富的backends。其中不乏trt以外的优化手段，如`torchInductor`、`nvfuser`等等。
我们可以通过
```python
torchdynamo.list_backends()
```
获取支持的backends。

除了官方提供的若干backend外，我们还可以自己实现自定义backend。
我们提供了tensorRT通用化backends，实现存放在[hat/utils/compile_backends.py](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/blob/master/hat/utils/compile_backends.py)。

## 注册backend

backend输入为`torch.fx.GraphModule`和示例输入，用户完成各类对`GraphModule`的操作。
例如编译、优化、logging、debug等等。
```python
@create_backend
def my_custom_backend(gm, example_inputs):
    return gm.forward

def f(...):
    ...

f_opt = torchdynamo.optimize(f, backend=my_custom_backend)
```

更多使用示例，可以参考[custom backends](https://pytorch.org/docs/stable/dynamo/custom-backends.html)。