# 如何将模型导出onnx

## 介绍

HAT集成了将模型导出onnx的功能。支持任意阶段的模型导出onnx。目前仅支持查看模型结构，不支持直接使用onnx进行推理。

## 使用步骤

按照如下流程进行配置使用导出onnx的工具
### 1.配置 config
在原有`config`文件的基础上增加`onnx_cfg`关键字
```python
onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
)
```
其中各个参数含义如下：
`model`：表示要导出onnx的模型
`inputs`：表示模型的输入
`stage`：指定要将哪一阶段的模型导出onnx
`model_convert_pipeline`：表示模型的convert过程，必须和`stage`参数对应
可根据情况配置的参数有：
`out_dir`：指定导出的onnx的存储路径。若不提供，则默认保存在config文件中定义的`ckpt_dir`路径或者当前路径下
`kwargs`：若用户需要传递其他export onnx的参数（参考`torch.onnx.export`），需添加的配置，在其中定义除`model`, `args`, `f`三个参数以外的关键字参数
```python
onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
    kwargs=dict(
        export_params=True,
        do_constant_folding=True,
        input_names=['xxx', 'yyy'],
    ),
)
```

### 2.运行脚本
运行tools/deploy/export_onnx.py脚本。
```shell
python3 tools/deploy/export_onnx.py -c config.py
```
脚本提供如下参数
- `-c, --config`: 必选参数，为config配置文件

若运行成功，默认在config文件中定义的`ckpt_dir`路径下生成`${stage}.onnx`文件；若未定义`ckpt_dir`，默认保存在当前路径下。如，上述示例命令会生成`int_infer.onnx`文件。
