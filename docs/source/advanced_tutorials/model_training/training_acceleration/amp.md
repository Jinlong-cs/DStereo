# 如何开启 AMP

AMP 全称为 Automatic Mixed Precision，即自动混合精度。AMP 开启后，pytorch 可以自动地在模型执行时将一些算子（如卷积和全连接）使用 `float16 ` 或者 `bfloat16` 进行计算，以达到提升计算速度、减少显存占用的效果。详见 [pytorch官方文档](https://pytorch.org/docs/stable/amp.html)。

HAT 中已经为 AMP 做好相关的工作，用户只需要在定义 config 文件中的 `batch_processor` 字段时将 `enable_amp` 参数设置为 `True` 即可。

默认的AMP方法使用 `float16`。如果需要设置AMP中的类型，设置enable_amp_dtype的类型即可，支持使用torch.float16和torch.bfloat16。

>注意：在模型验证时为得到准确的指标，一般是不需开启 AMP 的，在定义 `val_batch_processor` 字段时请将 `enable_amp` 参数设置为 `False` 。

```python
# configs/example.py

# 使用 BasicBatchProcessor
batch_processor = dict(
    type='BasicBatchProcessor',
    need_grad_update=...,
    batch_transforms=...,
    enable_amp=True,
    enable_amp_dtype=torch.float16,
)

# 使用 MultiBatchProcessor
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=...,
    batch_transforms=...,
    loss_collector=...,
    enable_amp=True,
    enable_amp_dtype=torch.float16,
)
```
