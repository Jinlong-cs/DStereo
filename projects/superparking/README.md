# Configs of models used in SuperParking

## 环境准备

1. 请参考[Installation Guide](../../docs/source/quick_start/installation.md)完成HAT基础开发环境的安装；
2. 若为developer，请按要求安装contributor环境；

## 运行

1. 首先将`GNUmakefile.template`复制到HAT根目录下，并改名为`GNUmakefile`；
2. 在各个命令中，都可以用`config={path_to_config}`来指定运行任务的入口config，当前可用的配置为：
    1. `projects/superparking/app/fisheye/multitask.py`(原图多任务配置)；
    2. `projects/superparking/app/ipm/multitask.py`(IPM多任务配置)；

随后即可使用其中预定义的命令来实现各种功能，如：

### 安装项目依赖库
1. 注意：mxnet库目前只支持cu111、cu102，安装本地环境时torch等环境需要安装对应的cu111、cu102版本;
- `make extra-packages`

### 训练

- `make train` (训练fp32模型) 
- `make qat` (训练qat模型) 

### 评测

- `make int_val` （提交int infer AIDI评测）

### 编译
- `make pipeline` (获取pt模型文件)
- `make compile` （编译模型，对于IPM任务，请执行`make compile-ipm`）

### 提交集群任务
- `make submit` （提交训练job到AIDI集群），可以用`cluster=xxx`来指定运行job的集群

## 提交合并请求

提交MR前请先修复pre-commit中的问题，确保代码符合基本规范；
