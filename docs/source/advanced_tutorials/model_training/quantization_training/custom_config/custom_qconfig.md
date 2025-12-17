# set_qconfig 书写规范和自定义 qconfig 介绍

## set_qconfig 方法书写规范

在对要量化的模型进行定义时，需要实现模型 `set_qconfig` 方法对量化方式进行配置。当前设置 QConfig 接口由 `hat.utils.qconfig_manager` 提供，`set_qconfig` 中调用 `hat.utils.qconfig_manager` 实现对模块 Qconfig 的设置，例如:

```python

# 注: 这份代码示例只是展示 set_qconfig 方法的实现规则，不是完整的量化模型代码

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.out_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self):
        ...
        
    def set_qconfig(self):
        # 若网络最后输出层为 conv，可以单独设置为 out_qconfig 得到更高精度的输出
        from hat.utils import qconfig_manager
        self.out_conv.qconfig = qconfig_manager.get_default_out_qconfig()

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        
    def forward(self):
        ...

    # 当前 Backbone 中没有特殊 layer，也没有需要设置 QConfig=None 的 layer 时，
    # 即都需要设置 default_qat_qconfig 时，可以不写 set_qconfig() 方法
    # def set_qconfig(self):

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = Backbone()
        self.head = Head()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self):
        ...
        
    # 需要父模块实现 set_qconfig 方法
    def set_qconfig(self):
        from hat.utils import qconfig_manager
        # 1. 首先指定父模块的 qconfig, 
        # 如果未对子模块设置 qconfig，子模块会自动使用父模块的 qconfig
        self.qconfig = qconfig_manager.get_default_qconfig()
        
        # 2. 如果有某子模块有特殊 layer，实现了 set_qconfig 方法，调用
        if self.head is not None:
            if hasattr(self.head, "set_qconfig"):
                self.head.set_qconfig()
                
        # 3. 如果有子模块不需要设置 Qconfig，需要设置 Qconfig 为 None
        if self.loss is not None:
            self.loss.qconfig = None

```

总结：从上面提供的代码事例可以看到，当前HAT下对需要量化的模块定义 `set_qconfig` 接口时遵循下面的逻辑:  
1、定义整个模块的 `qconfig` 参数:
```python
self.qconfig = qconfig_manager.get_default_qconfig()
```
这里 `qconfig_manager.get_default_qconfig()` 表示模块使用默认的量化配置，如果需要制定特殊的量化配置，可以参考下面第二节的内容。  
然后该 `qconfig` 参数配置会递归传递给该模块下的每个子模块中。 
 
2、对于该模块的子模块，如果需要定义不同的 `qconfig` 参数，可以对该子模块实现自己的 `set_qconfig` 接口，如果没有定义自己
的 `set_qconfig` 接口或者定义了没有使用，那么默认该子模块的  `qconfig` 参数和父模块保持一致:
```python
if self.head is not None:
    if hasattr(self.head, "set_qconfig"):
        self.head.set_qconfig()
```

3、对于不需要量化的子模块，需要手动设置它的 `qconfig` 为None，否则默认使用的是父模块的 `qconfig` 参数:
```python
if self.loss is not None:
    self.loss.qconfig = None
```

## 自定义 QAT QConfig 参数
HAT 支持 QAT 训练时使用自定义 QConfig，只需在 config 文件的 `qat_trainer` 中配置 `model_convert_pipeline` 的 `qconfig_params` 参数即可:

```python
qat_trainer = dict(
    ...
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1,
            ),
        ),
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    ...
)
```

`qconfig_params` 主要参数配置项有: `activation_fake_quant`, `weight_fake_quant`, `activation_qat_observer`, `weight_qat_observer`, `activation_calibration_observer`, `weight_calibration_observer`, `activation_qat_qkwargs`, `weight_qat_qkwargs`, `activation_calibration_qkwargs`, `weight_calibration_qkwargs`。

* `activation_fake_quant`: 指定 activation 的量化器，支持 `"fake_quant"`、`"lsq"`、`"pact"`, 缺省时使用默认值 `"fake_quant"`。
* `weight_fake_quant`: 指定 weight 的量化器。支持及使用方式同 `activation_fake_quant`。
* `activation_qat_observer`: 指定 qat 阶段 activation 的 observer，支持 `min_max`、`fixed_scale`、`clip`、`percentile`、`clip_std`、`mse`、`kl`、`mix`, 缺省时使用默认值 `min_max`。
* `weight_qat_observer`: 指定 qat 阶段 weight 的 observer，支持及使用方式同 `activation_qat_observer`。
* `activation_calibration_observer`: 指定 calibration 阶段 activation 的 observer，支持及使用方式同 `activation_qat_observer`, 缺省时使用默认值 `mse`。
* `weight_calibration_observer`: 指定 calibration 阶段 weight 的 observer，支持及使用方式同 `activation_qat_observer`。
* `activation_qat_qkwargs`: 指定 qat 阶段 activation 量化器的参数。
  * `activation_fake_quant` 是 `"fake_quant"` 时, `activation_qat_qkwargs` 可设置参数:
    ```python
    activation_qat_qkwargs=dict(
        averaging_constant=0.01,                # 设置 scale 的更新系数
    )
    ```
    
  * `activation_fake_quant` 是 `"lsq"` 时, `activation_qat_qkwargs` 可设置参数:
    ```python
    activation_qat_qkwargs=dict(
        scale=1.0,                              # 指定初始 scale，默认即可，一般不用设置
        zero_point=0.0,                         # 指定初始 zero_point，默认即可，一般不用设置
        use_grad_scaling=False,                 # 定义scale和 zero_point 的梯度是否由常数归一化，默认 False，默认即可，一般不用设置
    )
    ```
    
    * `activation_fake_quant` 是 `"pact"` 时, `activation_qat_qkwargs` 可设置参数:
    ```python
    activation_qat_qkwargs=dict(
        alpha=6.0,                              # 指定 activation 的 clip 参数，默认 6.0，一般不用设置
    )
    ```
* `weight_qat_qkwargs`: 指定 qat 阶段 weight 量化器的参数，参数配置及用法，同 `activation_qat_qkwargs`。
* `activation_calibration_qkwargs`: 指定 calibration 阶段 activation 量化器的参数，参数配置及用法，同 `activation_qat_qkwargs`。
* `weight_calibration_qkwargs`: 指定 calibration 阶段 weight 量化器的参数，参数配置及用法，同 `activation_qat_qkwargs`。

**注:** `activation_xxx_qkwargs` 和 `weight_xxx_qkwargs` 一般是不需要进行设置的，缺省使用默认配置即可。但当使用 calibration 后进行 QAT 训练时，可能需要修改 `averaging_constant`。
