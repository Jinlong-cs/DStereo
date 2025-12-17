# 如何不同模块设定不同学习率

## 根据不同的module name设定不同的学习率

模型中不同的module往往需要设置不同的学习率，可以通过如下方式根据名称设定不同的学习率

```python
xxx_trainer = dict(
    ...
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.SGD,
        model=model,
        optim_cfgs=dict(lr=0.1, momentum=0.9, weight_decay=1e-4),
        custom_param_mapper=dict(
            backbone={"lr": 0.001},
        ),
    ),
    ...,
)
```
如上所示，模型中backbone的学习率设置为0.001，其他的均为0.1

## 根据不同的module type设定不同的学习率

```python
xxx_trainer = dict(
    ...
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.SGD,
        model=model,
        optim_cfgs=dict(lr=0.1, momentum=0.9, weight_decay=1e-4),
        custom_param_mapper={
            nn.Conv2d: {"lr": 0.001},
        },
    ),
    ...,
)
```

如上所示，模型中Conv2d的学习率设置为0.001，其他的均为0.1