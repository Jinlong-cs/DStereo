# Module Patch 介绍

## 背景与动机

1. 在`pytorch`中，模型间参数共享的实现方式通常是：将包含要共享参数的模块初始化，然后在不同的网络结构中重复调用。因此，我们无法在原生情况下做到*重复声明，共享使用*，这也使得在多任务情况下，如何（在共享参数的情况下）独立声明每个任务的网络结构成了一个问题。

2. 在`pytorch`中，每个`nn.Module`实例并不存在全局化的名称，而是在其每次被调用时，分别被赋予一个由变量名决定的局部命名。比如：

```python
conv = nn.Conv2d(1, 1, 1)

class Toy1(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = conv

class Toy2(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = conv

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = Toy1()
        self.t2 = Toy2()

m = Toy()
```
这是一个比较典型的多任务模型场景，这里可以把`m.t1`和`m.t2`看作两个任务对应的模型结构。

显然，`m.t1.a`和`m.t2.b`引用了相同的`nn.Conv2d`实例。这样的情况下，共享的module事实上在多任务模型中会重复存在，且由于名称完全取决于外部模型的具体实现，非常难以定位。需要一种机制，来赋予module实例一个全局名称，使得其更容易被追溯。

因此，我们实现了Module Patch，在语法上通过一个简单的context manager，做到了对重复声明和全局名称的支持。

示例

```python

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = Threshold(0.5, 0.5, inplace=True, node_name="thresh")

    def forward(self, data, label):
        out = self.loss(data, label)
        out = self.thresh(out)
        return out


class ModelShare(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = Threshold(0.5, 0.5, inplace=True, node_name="thresh")
        self.act = Identity(node_name="act")

    def forward(self, data, label):
        out = self.loss(data, label)
        out = self.act(self.thresh(out))
        return out


with TorchModulePatch():
    model1 = Model()
    model2 = ModelShare()

assert id(model1.loss) == id(model2.loss)
assert id(model1.thresh) == id(model2.thresh)

```

## 与HATBC workflow的结合

HATBC workflow是一个计算图引擎，提供了图、节点、op等分层的数据结构，用以表示计算图中的各种元素。

如上面的例子所示，在初始化参数加入`node_name`项后，即可将对应`module`声明为计算图中的一个节点。在随后通过`trace`方式生成的计算图中，在不同的地方声明的共享`module`只要输入相同，就会成为同一个节点。

## 基于HAT registry的声明

整体的使用方式可以参考[registry文档](../../../basic_concepts/registry.md)。加入`Module Patch`后，像正常使用`nn.Module`一样，只要在初始化参数中增加`node_name`项，即可达到相同的效果。


