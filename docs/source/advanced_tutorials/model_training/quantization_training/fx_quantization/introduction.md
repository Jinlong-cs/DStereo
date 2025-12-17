# FX 量化简介

HAT 支持了基于 fx 的量化框架，在 config 中为使用到的 `Converter` 指定参数 `convert_mode = "fx"` 即可使用。

使用FX量化之前建议用户先看一下 [Eager量化训练](../eager_quantization/introduction.md)。

fx 是 torch 中用于模型编辑的纯 Python 实现的一套机制，采用 symbolic trace 的方式将模型中的操作保存为一个图表示，在图上对这些操作进行编辑后重新编译生成新的模型及 forward 代码。更多信息参见 [`torch.fx` 官方文档](https://pytorch.org/docs/1.10/fx.html)。

由于计算图的存在，基于 FX 的量化相比 eager mode 存在以下优势：
- 可以实现 fuse pattern 的自动化匹配，用户不再需要手动执行 fuse 过程
- 为其他基于计算图的优化提供了可能性

下面通过一个例子展示 symbolic trace 的一些特性：

```python
from torch import nn
import torch
from torch.nn import functional as F
from horizon_plugin_pytorch.quantization import QuantStub
from horizon_plugin_pytorch.quantization.quantize_fx import QuantizationTracer
from torch.quantization import DeQuantStub

from horizon_plugin_pytorch.quantization.fx.graph_module import GraphModuleWithAttr


class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)

        return x


class FxWrapExampleNet(nn.Module):
    def __init__(self):
        super(FxWrapExampleNet, self).__init__()
        self.quant = QuantStub()
        self.subnet = SubNet()
        self.dequant = DeQuantStub()

    def forward(self, input):
        x = self.quant(input)
        x = self.subnet(x)
        x = self.dequant(x)

        if self.training:
            return F.softmax(x, dim=1)
        else:
            return torch.argmax(x, dim=1)

model = FxWrapExampleNet()
tracer = QuantizationTracer([], [])
graph = tracer.trace(model)
graph.print_tabular()

# opcode         name         target                                args            kwargs
# -------------  -----------  ------------------------------------  --------------  -------------------------------------------
# placeholder    input_1      input                                 ()              {}
# call_module    subnet_conv  subnet.conv                           (input_1,)      {}
# call_module    subnet_bn    subnet.bn                             (subnet_conv,)  {}
# call_module    subnet_relu  subnet.relu                           (subnet_bn,)    {}
# call_function  softmax      <function softmax at 0x7fe1f566fc10>  (subnet_relu,)  {'dim': 1, '_stacklevel': 3, 'dtype': None}
# output         output       output                                (softmax,)      {}

graph_model = GraphModuleWithAttr(model, graph)
print(graph_model.code)

# def forward(self, input):
#     input_1 = input
#     subnet_conv = self.subnet.conv(input_1);  input_1 = None
#     subnet_bn = self.subnet.bn(subnet_conv);  subnet_conv = None
#     subnet_relu = self.subnet.relu(subnet_bn);  subnet_bn = None
#     softmax = torch.nn.functional.softmax(subnet_relu, dim = 1, _stacklevel = 3, dtype = None);  subnet_relu = None
#     return softmax
```

通过 fx 生成的 graph 和 forward 代码我们可以发现：
- 不同于 `torch.jit.trace`，symbolic trace 的粒度较粗，graph 中的结点可以是一次 module 的调用
- 条件语句被丢弃了，这个现象和一般意义上的 trace 一致，未执行到的操作将不会被记录下来
- forward 代码中直接调用了叶子节点（例如`self.subnet.conv`），非叶子节点将不会被调用，因此在非叶子节点上注册的一些 hook 也将不再生效

除以上特性外，fx 并未支持全部的 python 操作，若在模型中使用了不支持的操作，将会导致 symbolic trace 报错或产生非预期的行为。fx 支持的操作见 [symbolic trace 支持的操作说明](constraints) 小节

用户在开发支持 fx 量化的模型时，需要避免直接使用 fx 不支持的操作，若模型中确实需要这些操作（例如通过判断 `self.training` 决定计算 loss 还是执行后处理），fx 也提供了解决方案，具体见 [`fx.wrap`](fx_wrap) 相关内容
