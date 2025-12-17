nlu解析模型pipeline
## 环境准备
python 3.6/3.8 （后续hat陆续停止对3.6支持，推荐使用3.8）

如果需要使用hbdk3.37.1版本，需基于python3.6环境。

### 搭建流程参考
以conda环境为例。
#### python3.8
```
conda create -n hat_py38 python=3.8
conda activate hat_py38

#HAT
git clone git@gitlab.hobot.cc:ptd/algorithm/ai-platform-algorithm/HAT.git
cd HAT
HAT_ROOT_DIR_PATH=$(pwd)
export PYTHONPATH=${HAT_ROOT_DIR_PATH}:$PYTHONPATH

make run-env-cu111

#如果需要提交代码，需要准备代码提交环境
make env

pip install -U pycocotools
pip install setuptools==63.4.1
pip install pytest==6.2.5
pip install packaging==21.3
pip install numpy==1.20.3
pip install onnx

#horizon_plugin_pytorch
pip install -U horizon-plugin-pytorch==1.0.2 -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu111/torch1102 -i http://pypi.hobot.cc/simple --extra-index-url http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc --trusted-host art-internal.hobot.cc
pip install -U horizon-plugin-profiler -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

#工具链 verifier与hbdk版本号保持一致（说明：3.8环境下需要使用高版本工具链，不支持低版本的hbdk）
pip install hbdk==3.41.5 hbdk-model-verifier==3.41.5 horizon-tc-ui==1.13.5 horizon_nn==0.15.5 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple

#HAT 依赖中默认安装的click版本会导致hbmapper checker报错
pip install click==7.1.2

```

#### python3.6
```
conda create -n hat_nlu python=3.6.8
conda activate hat_nlu
pip install torch==1.10.2+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple
pip install torchvision==0.11.3+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple
pip install torchaudio==0.10.2+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple
pip install pytorch-crf==0.7.2
pip install onnx==1.6.0

#HAT
git clone git@gitlab.hobot.cc:ptd/algorithm/ai-platform-algorithm/HAT.git
cd HAT
HAT_ROOT_DIR_PATH=$(pwd)
export PYTHONPATH=${HAT_ROOT_DIR_PATH}:$PYTHONPATH

#如果需要提交代码，需要准备代码提交环境
make env

#安装HAT项目中的其他依赖
pip install --use-deprecated=legacy-resolver -r requirements.txt -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

#安装NLU独有的依赖
pip install --use-deprecated=legacy-resolver -r projects/halo/nlu/requirements.txt -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

#horizon_plugin_pytorch
pip install -U horizon-plugin-pytorch==1.0.2 -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu111/torch1102 -i http://pypi.hobot.cc/simple --extra-index-url http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc --trusted-host art-internal.hobot.cc
pip install -U horizon-plugin-profiler -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

#工具链 verifier与hbdk版本号保持一致
pip install hbdk==3.37.1 hbdk-model-verifier==3.37.1 horizon-tc-ui==1.9.6 horizon_nn==0.14.1 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple

#HAT 依赖中默认安装的click版本会导致hbmapper checker报错
pip install click==7.1.2

```

### 环境测试
在当前目录下，基于demo数据测试。

PTQ:
```
./run_ptq_demo.sh
./deploy_ptq.sh configs/tcn_ptq_demo.py
```

QAT:
```
./run_qat_demo.sh
./deploy_qat.sh configs/tcn_qat_demo.py
```

## PTQ流程
说明：训练、评估、部署相关参数在config文件中进行配置。

1.模型训练及模型评估
```
#正式流程
./run_ptq_official.sh
```
```
#demo流程,仅用于流程验证
./run_ptq_demo.sh
```

2.模型部署流程
注意：运行部署脚本前需确认上传路径和版本号符合预期，避免误删之前已发的版本。（运行部署脚本时程序会进行二次询问。）
```
#正式流程
./deploy_ptq.sh configs/tcn_ptq_official.py
```
```
#demo流程,仅用于流程验证
./deploy_ptq.sh configs/tcn_ptq_demo.py
```

## QAT流程
说明：训练、评估、部署相关参数在config文件中进行配置。

1.模型训练及模型评估
```
#正式流程
./run_qat_official.sh
```
```
#demo流程,仅用于流程验证
./run_qat_demo.sh
```

2.模型部署流程
注意：运行部署脚本前需确认上传路径和版本号符合预期，避免误删之前已发的版本。（运行部署脚本时程序会进行二次询问。）
```
#正式流程
./deploy_qat.sh configs/tcn_qat_official.py
```
```
#demo流程,仅用于流程验证
./deploy_qat.sh configs/tcn_qat_demo.py
```

## 浮点模型推理

```
./inference.sh
```
