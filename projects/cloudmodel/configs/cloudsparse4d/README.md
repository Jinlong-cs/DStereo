# CloudSparse4D
纯视觉BEV大模型

##  项目代码结构
- data
包括各项目评测数据集、训练数据列表、各类型数据加载逻辑(dataloaders, callbacks)

- evaluation
评测逻辑相关：包括collate_fn、各项目评测用到的reformat函数等

- utils
其他零散的util函数，包括aidi submit等

- model_configs
模型的configs

##  config结构

- config.py
包括大部分的超参数，比如：类别、views、输入尺寸、schedule等

- data.py
数据transform逻辑和train/eval数据版本信息

- model.py
模型相关超参数和模型结构定义

- entry.py
训练/评测主程序入口，仅包含entry逻辑，无需改动

## 运行命令

- 需要先编译sparse4d用到的deformable_func的so
```bash
cd hat/models/task_modules/sparse4d/ops
python3 setup.py develop
```

### 本地

- 评测
```bash
python tools/predict.py --config {cfg/entry.py} --ids ${device_ids} --stage "float"

# 示例
python tools/predict.py --config projects/cloudmodel/configs/cloudsparse4d/model_configs/v1.1/entry.py --ids "3" --stage "float"
```

- 训练
```bash
python tools/train.py --config {cfg/entry.py} --ids ${device_ids} --stage "float"

# 示例
python tools/train.py --config projects/cloudmodel/configs/cloudsparse4d/model_configs/v1.1/entry.py --ids "3" --stage "float"
```


### 集群

- 评测
```bash
aidi_eval=1 sh projects/cloudmodel/configs/cloudsparse4d/tools/submit.sh {cfg}

# 示例
aidi_eval=1 sh projects/cloudmodel/configs/cloudsparse4d/tools/submit.sh projects/cloudmodel/configs/cloudsparse4d/model_configs/v1.1
```

- 训练
```bash
sh projects/cloudmodel/configs/cloudsparse4d/tools/submit.sh {cfg}

# 示例
sh projects/cloudmodel/configs/cloudsparse4d/tools/submit.sh projects/cloudmodel/configs/cloudsparse4d/model_configs/v1.1
```