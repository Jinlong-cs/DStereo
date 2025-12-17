# Mono 标准化工具箱

## 基础工具
### Hatbc/HDFlow

```text
   Hatbc是感知算法的一个基础软件包。主要是帮用户完成静态/动态创建工作流程图(DAG)任务。
HDFlow作为算法中台，在Mono的工具中会大量使用。使用相关的工具之前，需要熟练掌握hdflow的使用。
  HDFlow/Hatbc是后面应用层工具的基础代码库。鉴于这两个软件包有一些入门成本，强烈建议在开始使用后面的工具前，仔细阅读并掌握hdflow/hatbc的使用方法。
```
文档、学习资料
  - HATBC
    - 文档
  http://model.aidi.hobot.cc/api/docs/HATBC/master-0.9.0-3d98415/html/index.html
  - HDFLOW
    - 文档
    http://model.aidi.hobot.cc/api/docs/HDFlow/master-0.4.0-3ce6e746d/html/index.html
    - 学习资料

  HDFlow 简介:

  [HDFlow介绍](https://horizonrobotics.feishu.cn/wiki/wikcn7Oz5R6hBInEITqE7dESebL)

  [hdflow-8月分享-v1.00.pptx](https://horizonrobotics.feishu.cn/wiki/wikcnDMsinqg6eeD9AvUoUtwMwd)

---

## [数据挖掘](data_mining)
```text
   这里的 `数据挖掘` 是指从 海量采集数据中 挖掘出对模型有较大增益效果的 数据并送去标注。由于标注成本较高，所以数据挖掘 是必不可少的。
```
项目文档，参考:
 https://horizonrobotics.feishu.cn/wiki/wikcnP6xWiyNuXXCXuM8RmBaXcF

 ---

 ## [数据部署](data_management_and_deploy)
 ```text
    量产环境中，标注好的数据可能会用作训练(数据打包)、评测(数据创建测试集)或其他用途。这种对标注数据的使用，统称为数据部署。
    数据在转化为新的结构后，需要对源数据可跟踪。这也是数据部署工具需要解决的问题。在Mono项目中，目前需要解决的是训练、评测数据集的规范化管理。
    本文介绍的数据部署工具，包含了标注数据的管理以及后续的评测集创建与数据打包功能的实现。
 ```
 详见文档:
 https://horizonrobotics.feishu.cn/wiki/wikcngVYMRQMWLxQsJbchB46dVh

---
 ## [Badcase处理V1.0](issue_auto_regress)

    基于Jira的badcase处理工具。目前已停止使用。请使用 Badcase处理v2

 ## [Badcase处理v2](badcase_process_v2)
 ```text
    在算法迭代过程中，需要对测试同学反馈的问题做出分析、处理、以及新模型对这些badcase的解决情况。Mono的badcase处理工具就是为了帮助用户快速对应这些badcase。降低算法同学的负担。
 ```
使用文档：

工具使用： https://horizonrobotics.feishu.cn/wiki/wikcnVYk7ckzE909vhnzW6yl2hj

Badcase分析工作流：https://horizonrobotics.feishu.cn/wiki/wikcnrDOobyjrueGsYt2Fq8MPpb



---

 ## [模型发版](model_release)
Mono模型在完成模型评测生成report过后，使用此工具可以一键完成后续的发版动作,目前包括拉取关键指标、模型编译、badcase回归、生成发版报告。
使用文档: https://horizonrobotics.feishu.cn/wiki/wikcnRqxjzIFizxLPh3fwcyFdTf

 ---
 ## [real3D](real3d)

 包括Mono real3D常规GT生成脚本及评测集生成脚本，具体说明见[Mono Real3D 说明文档](https://horizonrobotics.feishu.cn/wiki/wikcnHgojFmfwGh154IRhSBICCc)。

 另外3d及脱敏多任务Jira可视化流程已经接入badcase2.0流程中，使用说明见[Badcase(Jira)Pack批量回归可视化infer操作文档](https://horizonrobotics.feishu.cn/docx/TCc2dV74foh78HxgvPGct8bfn9f)。

---
## [数据脱敏](data_anonymization)
    在对外交付时，需要对数据中的敏感信息进行处理。比如人脸、车牌。数据脱敏工具可以帮助用户完成数据脱敏。


# 各工具负责人

|   badcase   | data_anonymization  | data_management_and_deploy | data_mining | model_release | real3d
|   ----      | ----                | ----                       | ----        | ----          | ----     |
|   hao.chen  | hao.chen            | hao.chen                   | guowei.hao  | kedi01.liu    | jin.yang |
