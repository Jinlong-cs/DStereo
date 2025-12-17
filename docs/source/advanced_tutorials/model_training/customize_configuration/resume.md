# 如何恢复训练

可以通过在 `config` 的 `{stage}_trainer` 中配置 `resume_optimizer` 和 `resume_epoch_or_step` 字段来恢复意外中断的训练，或仅恢复 optimizer 来进行 fine-tune。例如:

```python
xxx_trainer = dict(
    ...
    model_convert_pipeline=dict(
        ...
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path="checkpoint_path",  # 要 resume 的 checkpoint 路径。
            ),
        ],
    ),
    resume_optimizer=True,      # 恢复 optimizer
    resume_epoch_or_step=True,  # 恢复 epoch 或 step 
    ...,
)
```

恢复训练有三种使用场景:
1. 完全恢复: 该场景为恢复意外中断的训练，会恢复上一个checkpoint 的所有状态，包括 optimizer、LR、epoch、step 等。该场景需配置  `resume_optimizer=True`，并且需要配置`resume_epoch_or_step=True` 字段；
2. 恢复 optimizer 用于 fine-tune: 该场景只会恢复 optimizer 和 LR 的状态，但 epoch、step 都会从0开始，用于某些任务的 fine-tune。该场景需要配置 `resume_optimizer=True`，并且需要配置`resume_epoch_or_step=False`。
3. 只加载模型参数: 该场景只会加载模型参数，不会恢复其他任何状态(optimizer、epoch、step、LR)。该场景只需要在 `model_convert_pipeline` 中配置 `LoadCheckpoint` ，并且需要配置 `resume_optimizer=False` （默认）和 `resume_epoch_or_step=False`（默认）。