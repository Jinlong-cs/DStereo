# Configs of models used in Pilot

## 环境准备

### Pilot3&5

[镜像管理](https://horizonrobotics.feishu.cn/wiki/wikcnsicemSV1DByUBMpJ0gevMh)


首先请参考[Installation Guide](../../docs/source/quick_start/installation.md)完成HAT基础开发环境的安装

随后，请在**本目录**执行

```shell
pip3 install -U hbdk-internal -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

pip3 install horizon-plugin-pytorch -U -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu${cuda_version}/torch${torch_version} -i http://pypi.hobot.cc/simple --extra-index-url http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc --trusted-host art-internal.hobot.cc
pip3 install horizon-plugin-profiler -U -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

pip3 install mxnet_horizon_cu{cuda_version} -U -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip3 install --use-deprecated=legacy-resolver -r requirements.txt -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
```
安装pilot相关模型训练所特别需要的专用依赖项。

其中，`cuda_version`可以是111或者102，`torch_version`建议使用1102，取决于环境中torch及其对应的cuda版本。

### Pilot5.1

[镜像管理](https://horizonrobotics.feishu.cn/wiki/QPwmwOs9uimPX7kZwDIcM80unHb)

```shell
cd HAT
sh projects/pilot/dev/bev_env.sh
```

## 模型训练

请参考[相关文档](tools/train/README.md)中的方式启动多任务模型的训练。

## 模型评测

请参考[相关文档](tools/eval/README.md)中的方式启动多任务模型的预测、评测。

## 模型PackInfer

请参考[相关文档](tools/pack_infer/README.md)中的方式启动多任务模型的预测、评测。

## 模型编译、发布

请参考[相关文档](dev/publish/README.md)中的方式启动模型的编译、发布

## ISSUE回归&回灌评测

请参考[相关文档](tools/fillback_eval/README.md)中的方式启动模型的issue回归与回灌评测
