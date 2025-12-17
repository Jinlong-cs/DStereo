# Configs of models used in Mono

## 环境准备

首先请参考[Installation Guide](../../docs/source/quick_start/installation.md)完成HAT基础开发环境的安装

随后，请在**本目录**执行

```shell
pip3 install --use-deprecated=legacy-resolver -r requirements.patch -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
```

若为developer，请按要求安装contributor环境；

## 运行

### 训练

首先将`GNUmakefile.template`复制到HAT根目录下，并改名为`GNUmakefile`.

随后即可使用其中预定义的命令来实现各种功能，如：

- `make train` (训练fp32模型) 
- `make qat` (训练qat模型) 
- `make val` (评测模型)
- `make submit` （提交训练job到AIDI集群），可以用`cluster=xxx`来指定运行job的集群

在各个命令中，都可以用`config={path_to_config}`来指定运行任务的入口config。其默认值为`projects/pilot/configs/multitask_dev/multitask.py`。


## 提交合并请求

提交MR前请先执行以下步骤：
1. 请先在本地执行`make pipeline`，确认无block bug；
2. 修复pre-commit中的问题，确保代码符合基本规范；