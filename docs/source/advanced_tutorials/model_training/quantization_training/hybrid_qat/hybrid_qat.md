# 异构QAT简介

异构QAT功能基于horizon_plugin_pytorch的FX功能开发，请先阅读plugin文档中的[Toturial-异构模型QAT](http://model.aidi.hobot.cc/api/docs/horizon_plugin_pytorch/latest/html/tutorials/hybrid_qat.html)。

## 异构模型与非异构模型的区别

异构模型是部署时一部分运行在BPU上，一部分运行在CPU上的模型，而非异构模型部署时则完全运行在BPU上。通常情况下，以下两类模型在部署时会成为异构模型：
1. 包含BPU不支持算子的模型。
2. 由于量化精度误差过大，用户指定某些算子运行在CPU上的模型。

## 如何训练异构QAT模型
异构模型的定义，qconfig设置等内容和FX模式下的非异构模型完全一致，只需要在QAT模型的converter内加上`hybrid`和`hybrid_dict`参数。`hybrid_dict`的设置请参考plugin文档中的[Toturial-异构模型QAT](http://model.aidi.hobot.cc/api/docs/horizon_plugin_pytorch/latest/html/tutorials/hybrid_qat.html#id3)。

```python
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
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            # 使用fx模式, 使用异构
            dict(
                type="Float2QAT",
                convert_mode="fx",
                hybrid=True,
                hybrid_dict={"module_name": ["conv2"]},
            )
        ],
    ),
    ...,
)
```

需要设置的converter包括`Float2QAT`，`Float2Calibration`。`QAT2Quantize`不需要指定 `hybrid`，可以根据QAT模型自动选择convert方式。

之后按照正常训练QAT模型的命令训练即可得到异构QAT模型。

## 如何导出QAT ONNX模型

异构QAT模型的定点化和编译依赖`hb_mapper` 工具实现，二者以onnx作为中间表示，所以得到异构QAT模型之后需要将其导出为ONNX格式。

导出onnx模型的配置也可以参考HAT中关于[export_onnx工具](../../../../useful_tools/deploy_tools/export_onnx.md)。


## 如何定点化和编译

在异构模式下，定点化和编译依赖`hb_mapper` 工具实现，HAT中的int_infer提供的数值仅供参考，不是必须的。在拿到qat onnx模型后，按照 [`hb_mapper`相关文档](http://j5.ddk.hobot.cc/cn/oe%26mapper/source/ptq/ptq_usage/check_model.html#hb-mapper-checker) 使用 `hb_mapper` 得到bin模型。
