# 数据校准

在量化训练(QAT)中，一个重要的步骤是确定量化参数 scale，一个合理的 scale 能够显著提升模型训练结果和加快模型的收敛速度。 Calibration 是通过用浮点模型在训练集上跑少数 batch 的数据（只跑 forward 过程，没有backward），通过一定方法去计算出 min_value 和 max_value，然后可以用这些 min_value 和 max_value 去获取 scale。在 QAT 的开始之前使用 calibration 做量化参数的微调，获取 scale，可以为qat提供更好的量化初始化参数，提升收敛速度和精度。


## 1. 如何定义 Calibration 模型

* **默认不需要对现有模型做任何修改**
  
  类似于定义量化模型时需要设置 QAT QConfig，Calibration 时也需要对模型设置 Calibration QConfig。除 observer 以外，Calibration QConfig 的设置应当与 QAT QConfig 保持一致。
  

* **自定义模型子模块 Calibration QConfig**

    在上文的默认情况下，会为模型的所有 Module(继承自 nn.Module) 设置 Calibration QConfig。因此，Calibration 时也就会对所有 Module 的特征分布进行统计。如果有特殊需求，可以模仿设置 QAT QConfig 的方式在模型内自定义实现 `set_qconfig` 方法，该方法为 QAT QConfig 和 Calibration QConfig 共用。定义时应调用 `qconfig_manager` 中的 `get_default_qconfig` 、 `get_default_out_qconfig` 和 `get_qconfig` 方法获取 Qconfig ，这三个方法会根据模型所处的阶段自动选择使用 Calibration QConfig 或 QAT QConfig 。
```python

class Classifier(nn.Module):
    def __init__(self,):
        ...
    
    def forward(self, x):
        ...
        
    # 自定义要做 Calibration 的模块
    def set_qconfig(self, ):
        from hat.utils import qconfig_manager
        self.qconfig = qconfig_manager.get_default_qconfig()
        self.out_conv.qconfig = qconfig_manager.get_default_out_qconfig()
        
        
        # 比如可以设置 Loss 的 qconfig 为 None，就会不再对 Loss 做 Calibration，
        # 可以一定程度减少统计量，提升 Calibration 速度，降低显存占用
        if self.loss is not None:
            self.loss.qconfig = None
```


## 2. 浮点模型做 Calibration
HAT 中集成了 Calibration 功能，由浮点模型做 Calibration 命令和正常训练相似，只需执行以下命令即可：

```bash
python3 tools/train.py --stage calibration ...

```

**需要注意的是 `config` 文件中 `calibration_trainer` 中的一些配置:**

```python

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop('sampler')  # Calibration do not support DDP or DP
calibration_batch_processor = copy.deepcopy(val_batch_processor)

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),  
            ),  
            dict(type="Float2Calibration"),
        ],  
    ),
    # 1. 设置 data_loader 和 batch_processor
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    # 2. 设置 calibration 迭代的 batch 数目
    num_steps=30,
    ...   
)
```

Calibration相关调参技巧见[这里](http://model.aidi.hobot.cc/api/docs/horizon_plugin_pytorch/latest/html/user_guide/calibration.html#id2)。

## 3. 自动搜索 Calibration 策略
自动校准（auto calibration）基于模型输出的 L2 距离逐层搜索最优的 calibration 策略。通过该接口，您只需提供期望搜索的 calibration 策略集合，便可自动化地完成 calibration，减少您手动调参的次数。相关概念见[这里](http://model.aidi.hobot.cc/api/docs/horizon_plugin_pytorch/latest/html/advanced_content/auto_calibration.html)。

```python

calibration_trainer = dict(
    type="Calibrator",
    ...
    num_steps=10, # 指定自动校准的 step 数，step 越多显存占用越多、耗时越长
    auto_calibration=True, # 开启自动校准
    auto_calibration_config=dict(
        observer_list=["mse", "kl", "percentile"],
        percentile_list=[99.9, 99.99, 99.999],
        preload_data=True,
    ),
)
```


您可以通过 `auto_calibration_config` 配置自动校准的行为，其可配置的参数有：

`observer_list`: 候选策略集合，默认为 ["percentile", "mse", "kl", "min_max"],

`percentile_list`: percentile 需要搜索的参数，默认为 [99.995,]。仅当 `observer_list` 中包含 "percentile" 时生效。

`preload_data`: 由于逐层搜索时需要频繁访问 Dataloader，该算法通常会遇到访存瓶颈而导致耗时过长。开启 `preload_data` 可将数据提前从 Dataloader 加载到 device 上，可大幅提升性能，但同时也会占用更多显存。默认为 `False`。如果开启 `preload_data` 后出现显存过载的情况，可尝试调小 `batch_size` 或 `num_steps` 以减少显存占用，或切换到标准 Calibration 流程。

## 4. 对 Calibration 模型做权值重构
权值重构（weight reconstruction）是一种基于梯度的校准策略，目前 HAT 已支持 Adaround 重构策略，相关概念见[这里](http://model.aidi.hobot.cc/api/docs/horizon_plugin_pytorch/latest/html/advanced_content/weight_reconstruction.html)。

通过在上述 calibration_trainer 中添加两个参数，即可对标准 Calibration 流程得到的 Calibration 模型（后续简称为标准 Calibration 模型）进行优化，进一步提升模型精度。

```python

calibration_trainer = dict(
    type="Calibrator",
    ...
    weight_reconstruction=True, # 开启权值重构
    weight_reconstruction_config=dict(preload_data=True), # 相关参数配置
)
```

您可以通过 `weight_reconstruction_config` 配置权值重构的行为，其可配置的参数有：

`preload_data`: 由于逐层优化时需要频繁访问 Dataloader，该算法通常会遇到访存瓶颈而导致耗时过长。开启 `preload_data` 可将数据提前从 Dataloader 加载到 device 上，可大幅提升性能，但同时也会占用更多显存。默认为 `False`。如果开启 `preload_data` 后出现显存过载的情况，可尝试调小 `batch_size` 以减少显存占用。

其他可配置参数及调参技巧见[这里](http://model.aidi.hobot.cc/api/docs/horizon_plugin_pytorch/latest/html/advanced_content/weight_reconstruction.html)。

* **`val_callback`的配置**

  权值重构会在 `Calibrator` 的 `on_loop_end` 被调用，而 `Validation` callback 的 `val_on_train_end` 默认为 `False`, 您需要手动设为 `True` 才能对重构后的模型做精度测试。

* **保存的 checkpoint**

  由于 `Checkpoint` callback 默认开启 `save_on_train_end`，因此您无需手动配置，重构后的模型会保存为 "calibration-checkpoint-last.pth.tar"，不会覆盖标准 Calibration 模型 "calibration-checkpoint-best.pth.tar"。

* **与 QAT 的兼容性**

  如果重构后的模型精度无法满足要求从而需要 QAT 训练，我们建议您在 QAT 时读取标准 Calibration 模型而不是重构模型，尽管流程上这样完全允许。尽管重构模型的精度往往比标准 Calibration 模型要好，理论上应该是一个更好的初始模型。但在我们的实验中，对于部分模型，以重构模型为起点的 QAT 模型精度并不一定更优。

## 5. 使用 Calibration 模型做 QAT 训练


```python
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0.0,  # 0.0 ~ 1.0 之间的浮点数
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1.0,  # 0.0 ~ 1.0 之间的浮点数
            ),
        ),
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.001,
        momentum=0.9,
    ),
    ...
)
```

* QAT 时 averaging_constant 参数设置: 
  
  量化时 scale 参数的更新规则是 `scale = (1 - averaging_constant) * scale + averaging_constant * current_scale`。
  
  在已有的一些实验中（主要是图像分类任务实验）发现，做完 calibration 后，把 activation 的 scale 固定住，不进行更新，即设置 activation 的`averaging_constant=0`, 并设置 weight 的`averaging_constant=1`，效果可能会相对略好一些。

    注：但这种设置不是适用于所有任务，在 lidar 任务中，固定 scale,精度也可能会变差。可根据实际情况调整。


接下来只需要执行正常的 QAT 训练命令，即可启动 QAT 训练:

```bash
python3 tools/train.py --stage qat ...
```
