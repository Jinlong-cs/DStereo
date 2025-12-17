<!-- TOC -->
- [云端感知大模型](#云端感知大模型)
  - [环境准备](#环境准备)
  - [Quick Start](#quick-start)
    - [2D感知任务(检测、分割、分类)](#2d感知任务检测分割分类)
      - [训练](#训练)
      - [评测](#评测)
  - [ModelZoo](#modelzoo)
  - [模型推理服务](#模型推理服务)
  - [常见问题解答](#常见问题解答)
<!-- TOC -->

# 云端感知大模型

本项目主要用于开发云端感知大模型，支持云端大模型的训练、推理、评测等。当前支持的任务类型，包含物体检测、场景分割、物体实例分割等。同时，该项目已包含多个项目使用的基础数据，如Mono、Pilot项目。基于该框架，用户可以使用已有的大模型，也可开发自己定制的大模型。

## 环境准备

```shell
git clone git@gitlab.hobot.cc:ptd/algorithm/ai-platform-algorithm/HAT.git
make install run-env-cu116
```

## Quick Start

### 2D感知任务(检测、分割、分类)
#### 训练

```shell
# 本地pipeline test
python3 tools/train.py -c projects/cloudmodel/configs/perception_2d/cloudmodel_entry_trainval.py -s float -ids 0 --pipeline-test
# 提交集群测试
export PYTHONPATH=$(pwd)
cd plugins/k8s_submit
python3 submit.py --config ../../projects/cloudmodel/configs/perception_2d/submit_configs/k8s_config_cloudmodel.py --single-job
```

#### 评测

```shell
export PROJECT_ID=PD20230003
python3 tools/predict.py --config projects/cloudmodel/configs/perception_2d/cloudmodel_entry_aidieval.py --stage float -ids 0 --pipeline-test
```

## ModelZoo

在[ModelZoo](./model_service/ModelZoo.md)中, 收录了当前云端模型小组提供的所有模型。它们都可以使用下面的[模型推理服务](model_service/README.md)完成模型的推理。

    !需要注意，目前aidi提供的推理服务还不够完善，因此这里并没有对所有模型支持AIDI云端推理。
     如有需求，请在云端模型问答群中反馈。

## 模型推理服务

基于 aidi 提供的模型托管、模型推理服务能力，用户可以通过简单的 api 调用完成大模型推理部署。详见文档：

[模型推理服务](model_service/README.md)

## 常见问题解答

如果在使用中遇到任何问题，可以通过下面的问答群反馈。
包括但不限于：

1. 训练框架问题
2. 模型推理服务问题
3. 新模型需求

![img.png](_md_metas/img.png)