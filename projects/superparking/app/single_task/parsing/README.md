# Parsing 单任务训练与测评

## 训练
在环境根目录下执行
```shell
python tools/train.py --config projects/superparking/configs/fisheye/single_task/parsing/fisheye_parsing.py stage float
python tools/train.py --config projects/superparking/configs/fisheye/single_task/parsing/fisheye_parsing.py stage qat
python tools/train.py --config projects/superparking/configs/fisheye/single_task/parsing/fisheye_parsing.py stage int_infer
```
## 预测、评测
由于int_infer阶段主要目的是生成部署使用的pt模型，最后结果不会包括后处理例如上采样等操作. 所以stage int就是用于量化模型的测评使用，对模型的输出做上采样等后处理操作以便测评能够进行
在环境根目录下执行
```shell
python tools/predict.py --config projects/superparking/configs/fisheye/single_task/parsing/eval_fisheye_parsing.py stage int
```
以qat_predictor为例子   
1. 如果想通过float获得qat_predictor
```python
qat_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_ckpt,
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=eval_callbacks,
    share_callbacks=False,
    log_interval=50,
)
```
2. 如果想直获得qat_predictor
```python
int_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=qat_ckpt,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=eval_callbacks,
    share_callbacks=False,
    log_interval=50,
)
```
3. 如果外部传入qat_ckpt获得qat_predictor
```python
int_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=qat_ckpt_0,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=eval_callbacks,
    share_callbacks=False,
    log_interval=50,
)
```
执行
```shell
python tools/predict.py --config projects/superparking/configs/fisheye/single_task/parsing/eval_fisheye_parsing.py stage qat --ckpt qat_ckpt_1
```
这种情况下qat_ckpt_0会被qat_ckpt_1覆盖