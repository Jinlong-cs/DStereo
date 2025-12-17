# FX 的 wrap 机制

fx 支持的 python 操作有限，因此在开发模型时，如需用到 fx 不支持的操作，需要做一些特殊处理。

为了处理 fx 不支持的操作，`torch.fx` 提供了 wrap 机制，可以将 function 注册为叶子结点，在 symbolic trace 时若遇到注册的 function，则将其作为一个整体原样保留，而不再关注 function 内部的逻辑。 下面举例进行说明

```python
from torch.fx import symbolic_trace

def unsupported_part(x):
    print("This is a print ignored by fx.")
    return x**2

def func_to_be_traed(x):
    return unsupported_part(x)

graph_model = symbolic_trace(func_to_be_traed)
graph_model.graph.print_tabular()
# opcode         name    target                   args      kwargs
# -------------  ------  -----------------------  --------  --------
# placeholder    x       x                        ()        {}
# call_function  pow_1   <built-in function pow>  (x, 2)    {}
# output         output  output                   (pow_1,)  {}
print(graph_model.code)
# def forward(self, x):
#     pow_1 = x ** 2;  x = None
#     return pow_1
```

可以看到，`unsupported_part` 中仅有 `pow` 操作被保留了下来，`print` 语句由于和 return value 无关而被丢弃了。

```python
from torch.fx import symbolic_trace, wrap

@wrap
def unsupported_part(x):
    print("This is a print ignored by fx.")
    return x**2

def func_to_be_traed(x):
    return unsupported_part(x)

graph_model = symbolic_trace(func_to_be_traed)
graph_model.graph.print_tabular()
# opcode         name              target                                         args                 kwargs
# -------------  ----------------  ---------------------------------------------  -------------------  --------
# placeholder    x                 x                                              ()                   {}
# call_function  unsupported_part  <function unsupported_part at 0x7f00a31d2ca0>  (x,)                 {}
# output         output            output                                         (unsupported_part,)  {}
print(graph_model.code)
# torch.fx._symbolic_trace.wrap("__main___unsupported_part")
#
# def forward(self, x):
#     unsupported_part = __main___unsupported_part(x);  x = None
#     return unsupported_part
```

可以看到，使用 `wrap` 将 `unsupported_part` 包装起来后，trace 时将其作为一个 `call_function` 结点保留了下来，代码执行时将直接调用 `unsupported_part`，因此内部任何逻辑都将被原样保留。

*为避免名字冲突，生成代码时会自动创建一个新的 NameSpace，`__main___unsupported_part` 即为 `unsupported_part` 在此 NameSpace 中的名字*

我们对 `torch.fx.wrap` 进行了扩展，支持更加灵活的使用方式，具体见 `hat.utils.model_helpers.fx_wrap` 的接口文档

***请不要使用 `torch.fx.wrap`，使用 `hat.utils.model_helpers.fx_wrap` 接口以避免有可能出现的兼容性问题***

在 FX 量化框架中，图优化相关操作（fuse）是依赖计算图的，被 wrap 之后的函数内部逻辑将不会生成计算图，因此也就无法执行图优化。

同时需要注意，浮点模型转到 QAT 模型、QAT 模型转到定点模型的过程通过算子替换的方式实现（不依赖计算图），因此算子是否量化仍取决于它的 `qconfig` （和 eager mode 一致），而不取决于算子调用是否被 wrap
