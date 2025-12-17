# 如何训练不同架构的qat模型

由于芯片架构不一样，能够在XJ3和J5上部署的模型会有一些差异，可以通过在config中设置March来实现得到不同架构的量化模型。

```python
from horizon_plugin_pytorch.march import March

march = March.BERNOULLI2  # BERNOULLI2 XJ3

march = March.BAYES     # BAYES J5
```

训练 `BERNOULLI2` 模型时，务必使用 [Calibration](../calibration/calibration.md)，不然精度可能会有风险。