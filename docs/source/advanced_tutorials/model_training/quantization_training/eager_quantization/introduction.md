# Eager量化训练流程简介

Eager量化训练跟平常所使用的浮点训练差不多，均为动态图，需要用户在添加自定义模型时实现 `fuse_model` 方法来完成模型融合，且实现 `set_qconfig` 方法对量化方式进行配置即可。通过下面一个例子来详细说明如何使用eager模式进行量化训练。

## Eager量化训练示例

### 添加自定义模型

```python
import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY


# 使用装饰器的方式将模型进行注册
@OBJECT_REGISTRY.register
class ExampleNet(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...

    def fuse_model(self):
        # 需要调用所有子模块的 fuse_model 方法
        if hasattr(self.submodule, "fuse_model"):
            self.submodule.fuse_model()

        # 具体 fuse 的接口见 horizon_plugin_pytorch 文档
        ...

    def set_qconfig(self):
        # 具体模型量化配置的接口见 horizon_plugin_pytorch 文档
        from hat.utils import qconfig_manager
        # 默认使用 qconfig_manager.get_default_qconfig() 得到的 QConfig
        self.qconfig = qconfig_manager.get_default_qconfig()
        # 对需要特殊处理的子模块，调用子模块的 set_qconfig，
        # 子模块的 set_qconfig  中只需实现对特殊 layer 的 QConfig 设置
        if hasattr(self.submodule, "set_qconfig"):
            self.submodule.set_qconfig()
        # 如果有特殊节点不需要设置QConfig，比如 loss，需要设置其QConfig 为 None
        if self.loss is Not None:
            self.loss.qconfig = None
        ...
```

### 添加 config 文件

```python
ckpt_dir = ...

model = dict(type="ExampleNet")

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=...,
    optimizer=...,
    batch_processor=...,
    num_epochs=...,
    device=None,
    callbacks=...,
    ...,
)

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
    data_loader=...,
    batch_processor=...,
    num_steps=...,
    device=None,
    callbacks=...,
    ...,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=...,
    optimizer=...,
    batch_processor=...,
    num_epochs=...,
    device=None,
    callbacks=...,
    ...,
)

val_callback = dict(
    type="Validation",
    data_loader=...,
    batch_processor=...,
    callbacks=[val_metric_updater, ...],
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

int_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    # int_trainer 中实际不包含训练流程
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    ################################
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
)
```

## 训练

只需在使用 `tools/train.py` 脚本时按顺序指定训练阶段即可，会自动根据训练阶段调用相应的 `trainer` 来执行训练过程：

```bash
python3 tools/train.py --stage float ...
python3 tools/train.py --stage calibration ...
python3 tools/train.py --stage qat ...
python3 tools/train.py --stage int_infer ...
```

- float：正常的浮点训练
- calibration：校准QAT模型，初始化一个浮点模型，将此模点模型转为 QAT 模型，统计并计算伪量化节点的 scale。
- qat：QAT训练（量化感知训练），初始化一个浮点模型，将此模点模型转为 QAT 模型，再加载calibration得到的权重进行训练。
- int_infer：定点转化预测，此阶段首先初始化一浮点模型，将此浮点模型先转为 QAT 模型并加载训练好的 QAT 模型权重，再将 QAT 模型转为定点模型。转出的定点模型无法进行训练，只能执行 validation 得到最终的定点模型精度

## 精度问题定位
在qat转quantized出现精度掉点时，使用者可以在converter中设置`preserve_qat_mode_dict`来控制部分op在定点模型中保持qat形式，来定位具体是哪个op在转定点之后导致的模型精度掉点。
```python
converters=[
    LoadCheckpoint(ckpt_name),
    QAT2Quantize(
        convert_mode=convert_mode,
        preserve_qat_mode_dict=dict(
            prefixes=("backbone",),
            types=(horizon.nn.qat.Conv2d,),
        )
    ),
],
```
在`QAT2Quantize`时可设置此参数。
- `prefixes`：指定要保持qat形式的 op 在模型中对应的 layer name（以 prefixes 开头的 layer）
- `types`：指定要保持qat形式的 op 的类型。目前需指定QAT op类型，后续会支持指定float op。
