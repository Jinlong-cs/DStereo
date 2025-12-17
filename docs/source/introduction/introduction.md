# 简介

HAT (Horizon Algorithm Toolkit) 是基于 Pytorch 和 Pytorch plugin 的接口开发的AI算法工具，旨在为地平线 BPU 提供高效且用户友好的 AI 算法工具包。

HAT 所依赖的 PyTorch 是一个针对深度学习, 并且使用 GPU 和 CPU 来优化的 tensor library (张量库)，是目前最受欢迎的深度学习框架之一。而 Pytorch plugin 是基于 Pytorch 开发的一套量化算法工具，专注于与芯片相贴近的量化功能实现，其量化算法与地平线芯片深度耦合，利用该工具训练得到的量化模型均可以正常编译和运行在地平线 BPU（BERNOULLI/BAYES）上。

HAT作为地平线开发的算法包基础框架，面向所有的算法用户开发和研究的用户，其量化训练与地平线芯片紧密相关，包含了 `浮点训练` --> `QAT 训练` --> `定点转化预测` --> `模型检查编译(针对地平线 BPU)` --> `上板精度仿真验证` 的整套流程。同时它还可以提供包含分类，检测，分割等常见的图像任务的SOTA(state-of-the-art)深度学习模型。



## 特性

- 基于 Pytorch 和 [horizon_plugin_pytorch](http://gitlab.hobot.cc/ptd/ap/dlp/horizon_plugin_pytorch)
- 包含从`浮点训练`到`上板精度仿真验证`的整套流程
- 包含分类、检测、分割等常见图像任务的 SOTA 模型，且所有示例都与地平线 BPU 兼容
## 示例模型
HAT 目前已包含以下深度学习模型。


**分类模型**

- [x]  MobileNet (V1 and V2)
- [x]  ResNet (18 and 50)
- [x]  EfficientNetB0
- [x]  VargNet_V2
- [x]  Swin-T
- [x]  MixvargeNet


**检测模型**

- [x]  FCOS

模型精度等信息可在 [model_zoo](../model_zoo/model_zoo.md) 获取。
