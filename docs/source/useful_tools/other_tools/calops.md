# 如何计算模型计算量

## 计算量的定义

计算量是用来评估神经网络大小的常用工具。在常见的计算量工具中一般只统计两种操作的计算量，一个是卷积相关，另一个是全连接相关的。

以常见的`torch.nn.Conv2d`为例，输入数据的形状为`bs * c_in * w_in * h_in`, 输出数据的形状为`bs * c_out * w_out * h_out`，卷积核的大小为`f`。则该`Conv2d`的计算量为`2 * bs * f * f * c_in * c_out * w_out * h_out`。`2`表示加法计算量和乘法计算量各一半。

而`torch.nn.Linear`的情况里，输入数据的神经元个数为`c_in`,输出数据的神经元个数为`c_out`。其实这种全连接层可以作为一个特殊的卷积层，输入输出的大小均为`1x1`，卷积核大小也是`1x1`。则全连接层的计算量为`bs * (c_in * c_out + c_out)`，需要注意是这里乘法和加法的计算量并不完全一致。

>注意：量化模型和QAT模型和对应的浮点模型的计算量是完全一致的。

## 计算量工具的使用方法

目前计算量工具支持统计计算量的方法有两种，一种是利用`torch`中的`register_forward_hook`来统计计算量，支持的`op`有三种，分别为`torch.nn.Conv2d`、`torch.nn.Linear`和`torch.nn.ConvTranspose2d`。使用方式如下：

```bash
python3 tools/calops.py --config ${CONFIG_PATH}
```

另一种是利用`torch.fx`来统计计算量，支持的`op`有四种，分别为`torch.nn.Conv2d`、`torch.nn.Linear`、`torch.nn.ConvTranspose2d`和`matmul`。使用方式如下：

```bash
python3 tools/calops.py --config ${CONFIG_PATH} --method fx
```

其中`config`中影响计算量的主要key是`deploy_model`（或者`model`）,以及`deploy_inputs`。`model`相关的决定了计算量工具需要检查的模型，而`deploy_inputs`决定了输入的大小，除此之外，输入形状还可以通过`--input-shape B,C,H,W`的输入参数决定。

> 注意：有的模型并不支持使用`torch.fx`统计方法，不支持的情况可参考文档https://pytorch.org/docs/1.10/fx.html#limitations-of-symbolic-tracing。由于`register_forward_hook`统计方法支持的`op`较少，使用此种方法有可能导致统计量不准确。在模型支持`torch.fx`的情况下，建议优先使用`torch.fx`的统计方法。

## 常见分类模型的计算量（输入大小为`1x3x224x224`）

| network | OPS(G) |
| :----: | :---: |
| mobilenetv1 (alpha=1.0) | 0.57 |
| mobilenetv2 (alpha=1.0) | 0.31 |
| resnet18 | 1.81 |
| resnet50 | 3.86 |
| vargnetv2 | 0.36 |
| efficientnetb0 | 0.39 |
