# AIDI云端模型部署工具使用文档
该文档记录了AIDI云端模型部署工具使用方式，方便用户快速将模型部署到aidi平台，并完成自测。

## 使用方式
### 工具入口
通过命令行的方式执行
```shell
python3 plugins/aidi_inference/aidi_deploy.py   --config plugins/aidi_inference/configs/aidi_config.py \
                                                --publish-version v0.0.1 \
                                                --model-params xxx \
                                                --overwrite
```
参数说明：
- config：必填参数，提供部署config，下面部署config提供更详细的说明。
- publish-version：必填参数，部署模型版本号
- model-params: 选填参数，部署模型权重文件；若不提供，默认使用config中params_path参数
- overwrite：控制项，若指定版本模型已存在，是否覆盖
### 部署config
我们一个完整的模型部署需要提供以下必要信息：
- model_name：模型名
- model_version：模型版本
- model_tags：模型标签
- param_file：模型checkpoint地址
- model_config：模型构建config文件路径，需要提供模型的初始化、前后处理实现方式。关于该config，我们在模型构建config中详细说明。
- docker_image：云端运行基础docker
- py_deps：云端运行所需py依赖。 注意：通过该方式上传至云端的py库，会将库整个打包并上传并放入PYTHONPATH中，但部分需要编译的库使用该方法会无法正常使用，请重新制作docker！

我们以resnet18分类模型为参考，[基础config example](./configs/aidi_config.py)

以及如下可选配置项：
- model_cls: 云端模型基础类(继承`aidisdk.model.Model`)，默认使用HatModel，用户仅需考虑 模型初始化、模型前后处理 的实现。但若不满足需求，可以自定义model_cls
- example_input：模型输入示例。为了方便模型使用者理解模型输入，我们建议开发者提供每个模型的输入示例(真实数据最佳)；此外，我们提供了部署模型基础测试方法，需要依赖它来进行测试。

### 模型构建config
在模型构建config中，需要提供模型初始化、前后处理实现方式。
若使用默认的model_cls HatModel, 需要提供 Inference封装的推理模型。在云端模型init时，会通过build_from_register构建该实例。

> 这里需要说明的是，HatModel中也存在preprocess、postprocess实现方法，但并没有做任何操作。这是因为我们已经使用了这里的`Inference`进行封装，并调用该实例的forward方法，其中包含了pre/postprocess，不需要在HatModel中再做任何修改。因此，HatModel作为默认的model_cls，大部分情况下开发者提供前后处理即可，不需要重复实现model_cls。

例如：
```Python
model = dict(
    type="xxx"
)
preprocess= dict(
    type="xxx"
)
postprocess= dict(
    type="xxx"
)
# 也可以通过非build的方式直接指定处理函数，如
# def func()
#     xxx
#     return xxx
# postprocess=func

inference = dict(  # noqa
    type="Inference",
    model=model,
    device=[0],
    march="bayes",
    pre_processors=[preprocess],
    post_processors=[postprocess],
    model_convert_pipeline=None,
)
```
若模型需要进行转化(例如float -> qat)，可以自定义model_convert_pipeline。

> **关于模型输入输出格式，我们强烈建议**：
使用hatbc.message封装并返回，why & how[参考](http://model.aidi.hobot.cc/api/docs/HATBC/latest/html/build/tutorials/basic_data_class/2_perception_data.html)。通常我们在本地将输入数据(如图像)进行message封装，并在preprocess中完成解析并送入模型网络；在postprocess中对网络输出结果进行message封装并返回请求端。

### 部署模型测试
通过上面的config，我们可以很容易地完成一个云端模型的部署流程(或者云端模型的版本迭代更新)。
在这之后，我们往往还需要对部署模型进行测试，我们提供了一个基础的[测试类](./tests/base.py)。对于自己的模型，
用户需要额外实现推理结果compare函数来进行本地&云端结果比对。

基于example_input，测试包含：
1. 本地 & 云端跑通测试
2. 本地 & 云端运行测速
3. 本地 & 云端一致性比对

参考示例
```Python
# 针对plugins/aidi_inference/configs/aidi_config.py resnet18分类模型
pytest -s -x plugins/aidi_inference/tests/test_example.py
```
并输出本地 & 云端 FPS信息、diff信息

### 部署模型运行
除了通过测试来执行云端aidi模型推理外，我们也提供了一个简单的运行脚本`run_predict.py`，示例：
```Shell
python3 run_predict.py --config configs/aidi_config.py
```
可以通过增加`--local`来执行本地推理。

### 本地模拟云端环境
在接入aidi云端推理的过程中，大家经常会遇到本地能够跑通，但云端总是会出现奇奇怪怪的问题。为了方便调试，我们可以通过在
本地开发机加载docker镜像模拟云端推理环境，方法如下：
> 拉取模型保存时设置的依赖镜像
>
> docker pull docker.hobot.cc/auto/hdflow-runtime-cu100:just_a_test
>
> 启动容器，并挂载开发机上的gpu卡
>
> docker run -it --rm --gpus=all docker.hobot.cc/auto/hdflow-runtime-cu100:just_a_test
>
> 将推理文件复制到云端，并云端执行
> 
> docker cp run_predict.py  mycontainer_id:/root/run_predict.py
>
> (docker) python3 run_predict.py --config configs/aidi_config.py --local

**注意本地开发机使用docker可能需要权限，请联系开发机管理员！**

# FAQ
Q：有没有aidisdk示例

A：参考 https://gitlab.hobot.cc/ptd/ap/aidi/ogremagi/aidisdk/-/tree/master/example/model_serving, aidi官方用户手册 https://horizonrobotics.feishu.cn/wiki/wikcn9cAS9hG9tekD888pT1j5ce

Q：集群不知道为啥报错，没有错误信息？

A：可以先尝试本地local predict看看有没有问题，若没问题，进一步可能是：1. 本地&集群环境存在差异 2. 集群机器问题，找振风

Q：明明上传了python库，为什么云端还是报错，提示缺乏 ModuleNotFoundError?

A：这种情况，往往出现在部分需要编译的库上，例如mmcv，通过py_deps仅能将本地pkgs文件夹打包上传，若该库需要编译，那么往往通过py_deps上传是不够的，此时应该重新制作docker，构建完整的库环境。

Q：为什么云端推理看上去成功执行，但返回结果报错？例如
```
ts = time.time()
for in range(num infer):
    result = inference.request(test input)

plugins/aidi inference/tests/base.py:73:
-----------------------------------------------------------------------------
../../../miniconda3/envs/py3.8/1ib/python3.8/site-packages/typeguard/ init.py:1033: in wrappelsetval tncarg../../aidisdk/aidisdk/aidisdk/model/inference.py:220: in request
    return self.load data(
../../aidisdk/aidisdk/aidisdk/model/inference.py:201: in load data
    return dill.loads(base64.b64decode(data.encode())
../../../.local/lib/python3.8/site-packages/dill/_dill.py:286:in loads
    return load(file,ignore.**kwds)
../../../.local/lib/python3.8/site-packages/dill/_dill.pv:272: in load
    return Unpickler(file.ignore=ignore, **kwds).load()
.../../../.local/lib/python3.8/site-packages/dill/_dill.py:419 in load
    obj = stockUnpickler.load(self)
ssage.py:362: insetstate
../../../miniconda3/envs/py3.8/lib/python3.8/site
    self.post_init()
../../../miniconda3/envs/py3.8/lib/python3.8/site-packagssage/frame.py:43: in _post_init
    self._generate_get_method(key)
-----------------------------------------------------------------------------
self = <AttributeEror('velocity') raised in repr()] Frame object at 0x7f5e5f9eb00), attr = 'attributes', customized filter = Mone, alias = None
```

A：如你所见，云端接口返回了序列化对象，但在本地反序列化时失败。造成这种错误的常见原因是该对象依赖的本地pkg与云端pkg存在版本差异，实现上可能存在差异，请务必确保在传递复杂数据结构时本地&云端相关依赖库版本一致！！！