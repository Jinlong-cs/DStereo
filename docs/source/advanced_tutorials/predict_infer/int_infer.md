
# INT INFER GPU 加速

从 horizon-plugin-pytorch 1.6.3 版本开始，可以对 **J5 bayes** 模型的 INT INFER 使用 GPU 加速，在 HAT 中启用加速：

```python
int_infer_predictor = dict(
    type="Predictor",
    ...
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(
                type="QAT2Quantize", 
                convert_mode=convert_mode,
                fast_mode=True,         # 设置 fast-model 启动 GPU 加速， 默认 False
                use_cutlass=True,       # 使用 cutlass 对 conv 算子加速，默认 False
            ),
        ],
    ),
    ...
)

```

注意：
* `fast_mode` 需要 `horizon-plugin-pytorch>=1.6.3`；
* `use_cutlass` 需要 `horizon-plugin-pytorch>=1.10.1`；
* `fast_mode` 和 `use_cutlass` 可以同时使用；
* **该功能仅能用于推理加速，开启之后，INT 模型可以 trace，但不能使用 hbdk 编译**
* **当前仅支持 J5 bayes 量化模型**
