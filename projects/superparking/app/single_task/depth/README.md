# depth 单任务训练与测评

## 训练
在环境根目录下执行
```shell
python tools/train.py --config projects/superparking/configs/fisheye/single_task/parsing/depth_rec.py stage float
python tools/train.py --config projects/superparking/configs/fisheye/single_task/parsing/depth_rec.py stage qat
python tools/train.py --config projects/superparking/configs/fisheye/single_task/parsing/depth_rec.py stage int_infer
```
## 预测、评测
由于int_infer阶段主要目的是生成部署使用的pt模型，最后结果不会包括后处理例如深度值的解码. 所以stage int就是用于量化模型的测评使用，对模型的输出做后处理操作以便测评能够进行
在环境根目录下执行
```shell
python tools/predict.py --config projects/superparking/configs/fisheye/single_task/parsing/eval_depth_rec.py stage int
```


