# HAT Docker 镜像使用说明

## 简介
本文档介绍 HAT docker 的版本发布规则，并记录当前可用 docker 版本。

## 发布及命名规则
HAT 当前 docker 统一命名格式为：

**docker.hobot.cc/dlp/hat:runtime-`{python版本}`-`{torch版本}`-`{device}`-`{后缀}`**

其中，
- `python 版本`：与公司主推 python 版本一致，当前支持 py3.8；
- `torch 版本`：已适配的 torch 版本，比如: torch2.0.1 、torch1.13.0；
- `device`: cpu 或 cuda 版本（cuda 与上述 torch 版本有一定对应关系），比如 cpu、cu118、cu116；
- `后缀`:  `hat.__version__` 结尾或 `commit-id` 结尾；

依据 后缀 的不同，又分为了两种版本 ：

1. 以 `hat.__version__ `结尾，格式如下：

    **docker.hobot.cc/dlp/hat:runtime-{python 版本}-{torch版本}-{device}-`{hat.__version__}`**

    比如：**docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-`2.0.1`**

    > 注意：该版本的 docker 会随着 HAT 的 daily ci 而每日更新，直至 hat.__version__ 发生变化

2. 以 `commit_id`(前 7 位) 结尾，格式如下：

    **docker.hobot.cc/dlp/hat:runtime-{python 版本}-{torch版本}-{device}-`{commit_id[:7]}`**

    比如：**docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-`c67816b`**
    
    > 注意：该版本的 docker 在 HAT daily ci 中被创建出来之后，就不会再进行更新。
    > docker 在 daily ci 中构建，并非 HAT repo 的每个 commit-id 都对应一个 docker

当前  `{python版本}`、 `{torch版本}`、`{device}` 之间的对应关系如下：

<table style="text-align:center;">
    <tr>
        <td> python 版本 </td>
        <td> torch 版本 </td>
        <td> device 版本 </td>
    </tr>
    <tr>
        <td rowspan="7"> py3.8 </td>
        <td rowspan="2"> torch2.0.1 </td>
        <td> cpu </td>
    </tr>
    <tr>
        <td> cu118 </td>
    </tr>
    <tr>
        <td rowspan="2"> torch1.13.0 </td>
        <td> cpu </td>
    </tr>
    <tr>
        <td> cu116 </td>
    </tr>
    <tr>
        <td rowspan="3"> torch1.10.2 </td>
        <td> cpu </td>
    </tr>
    <tr>
        <td> cu111 </td>
    </tr>
    <tr>
        <td> cu1102 </td>
    </tr>
</table>



## GPU docker 使用

GPU docker 可用于 GPU 上的分布式训练、推理评测任务等。

### 集群任务中使用

在集群任务中使用 docker，可参照 {ref}`k8s/submit_k8s_job`，在 `k8s_config` 中设置 docker 参数即可，这里不再赘述。

### 本地机器使用

以 `docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-2.0.1` 为例，实际操作中换成自己的 docker 镜像

1. 首先需要从镜像仓库中拉取 docker 至本地，格式为 `docker pull xxx`，比如：
```bash
docker pull docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-2.0.1
```

2. 启动 docker：
docker 运行的基本命令为：
```bash
docker run -it --entrypoint /bin/bash docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-2.0.1
```
需要注意的是，上述命令未挂载 GPU，当我们在使用 GPU docker 时，需要在 `docker run` 命令中指定要使用的 gpu，加上 `--gpu` 参数即可，比如：
```bash 
docker run -it --gpus all --entrypoint /bin/bash docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-2.0.1
```
其中，`--gpus all` 即指定要使用当前本地机器上的所有 GPU。若要使用部分 GPU，也可以通过 `--gpus` 参数指定：
```bash
docker run -it --gpus '"device=0,2"' --entrypoint /bin/bash docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-2.0.1
```
该命令即指定了 docker 启动时使用本地机器上的第 1 和第 3 块 GPU。

其他更多 docker 用法见 [docker 文档](https://docs.docker.com/engine/reference/commandline/run/)。


## CPU docker 使用

CPU docker **不能用于训练任务，只能用于 cpu 上的推理评测、编译等任务。**

| 功能 | 脚本 |是否支持 | 备注 |
| :-----: | :----: | :----: | :----: |
| 训练 | `train.py` | 不支持 |  |
| 评测 | `predict.py` | 支持 | 需设置 `backend=GLOO`|
| 编译 | `deploy/compile_perf.py`</br> `deploy/compile_standalone.py`</br>`deploy/pack_hbm.py` 等 | 支持 | 编译相关脚本都可在 cpu docker 中使用 |
| 转 onnx | `deploy/export_onnx.py` | 支持 |  |
| 转 tensorrt | `deploy/export_tensorrt.py` | 不支持 |  |

**Note:**

使用 HAT 中的 `tools/predict.py` 在 CPU 上进行推理评测时，需注意两点：
1. config 中 `device-ids` 需设置为 `None`:
```python
device-ids = None
```

1. 需要给 `tools/predict.py` 指定 `backend=GLOO`:
```bash
python3 tools/predict.py --config xxxx.py --backend GLOO --stage ...
```

### 集群任务中使用

同上，在集群任务中使用 docker，可参照 {ref}`k8s/submit_k8s_job`，在 `k8s_config` 中设置 docker 参数即可，这里不再赘述。

### 本地机器使用

CPU docker 使用起来较为简单，参照上述 GPU 使用方法，也分为两步：

1. 首先拉取 docker 至本地，比如：
```bash
docker pull docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-2.0.1
```

2. 启动 docker，CPU docker 无需挂载 GPU，使用基本的 `docker run` 命令即可，比如：
```bash
docker run -it --entrypoint /bin/bash docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-2.0.1
```

其他更多 docker 用法见 [docker 文档](https://docs.docker.com/engine/reference/commandline/run/)。
