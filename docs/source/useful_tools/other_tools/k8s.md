(k8s/submit_k8s_job)=
# 如何提交到集群

## 集群介绍

集群平台是面向AI产品算法研发的一站式算法开发平台。打通了数据管理、模型训练、模型评测、模型编译和发版流程，实现了对复杂数据和训练反馈环的管理，是一套稳定高效的算法开发平台。详细的细节可以看[AIDI 平台](http://user-manual.aidi.hobot.cc/docs/aidi-model/chapter_2)的介绍。

这里我们主要介绍在`HAT`中如何使用集群的模型训练功能。与本地的开发机相比，集群拥有独立的训练资源，更加方便用户可以有效完成训练任务。

## 集群脚本提交

`HAT`中提供集群提交的插件，可以方便用户直接使用。提交脚本利用`aidisdk`的接口向集群提交任务，不依赖于``traincli``和 ``hitc job``命令行工具，不依赖yaml文件，可直接在命令行使用`--queue`指定使用的集群队列名称，appid和appkey等信息则不需要配置，使用示例如下：

```bash
cd plugins/k8s_submit/
python3 submit.py --config k8s_config.py --queue share-3090-idc 
```

另外，`submit.py`提供了任务名和机器资源等相关配置的接口：
* `--dag-name`可指定 DAG 任务名称
* `--job-name`可指定 job 任务名称
* `--single-job` 指定以 aidisdk `single_job` 接口提交训练任务（AIDI前端可"复制"实验）
* `--num-machines`可指定任务使用集群的机器个数
* `--num-gpus-per-machine`可指定每台机器使用的GPU卡数
* `--job-type`可指定任务类型

**注意：** 由于 AIDI中的限制，`dag-name` 和 `job-name` 格式需要符合 “大小写字母/数字/下划线的组合，且需要以字母开头”。

此外，为包含更丰富的信息，还可以使用参数

* `--dag-desc` 上报DAG的描述信息，仅在非`single_job`模式下有效果
* `--job-desc` 上报Job的描述信息

两个desc的格式限制一致，都是长度小于128的字符串。

`submit.py`使用了config文件，即`k8s_config.py`。config文件中包含了用户经常需要修改的集群配置。

## k8s_config的详细介绍

`k8s_config.py`提供了用户自定义命令的设置接口，目前用户自定义命令的关键字有`prefix_cmds_on_master`、`suffix_cmds_on_master`、`custom_cmds_before_job_list`、`custom_cmds_after_job_list`和`job_list`。

这些命令执行的顺序依次为`prefix_cmds_on_master`->`custom_cmds_before_job_list`->`job_list`->`custom_cmds_after_job_list`->`suffix_cmds_on_master`。下面为上述命令的说明。

`prefix_cmds_on_master`关键字含义为：用户自定义命令，在master机器上执行，不进行多机分发，执行顺序排在首位；
`custom_cmds_before_job_list`关键字含义为：用户自定义命令，目前采用`torchrun`的方式进行多机分发，执行顺序在第二位；
`job_list`关键字含义为：集群运行的所有脚本，目前采用`torchrun`的方式进行多机分发，执行顺序排在第三位；
`custom_cmds_after_job_list`关键字含义为：用户自定义命令，目前采用`torchrun`的方式进行多机分发，执行顺序在第四位；
`suffix_cmds_on_master`关键字含义为：用户自定义命令，在master机器上执行，不进行多机分发，执行顺序排在末位。

此外，`dag_desc`和`job_desc`信息也可以在config中指定，执行过程中，在命令行参数没有提供时，会使用config参数。

`k8s_config.py`支持用户自定义环境变量，环境变量可在各种cmds里设置。

`k8s_config.py`的使用示例如下所示：

```python
import hat

job_name = "torch-hat-k8s-example"  # 提交job的运行名
job_password = "newk8s666"  # 集群提交文件的压缩密码

num_machines = 1  # 使用集群的机器个数
num_gpus_per_machine = 4  # 每台机器的GPU卡数

# 集群需要的基本统计数据
framework = "pytorch"  # 使用深度学习框架
task_label = "HAT"  # 使用算法包
project_id = "AM2018-R72"  # 项目号

priority = 5  # 集群排队的优先级，可选范围[1,2,3,4,5]
version = hat.__version__.split(".dev")[0]

docker_image = (  # 集群运行使用的DOCKER镜像
    # CUDA 11.8
    "docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-%s" % version
    # CUDA 11.6
    # "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-%s" % version
)
max_jobtime = 10000  # 单位:分钟，任务最长运行时间：如7200 = 5days

# launcher only for multi-machines
launcher = "torch"  # DDPTrainer的启动方式

# upload folder
upload_folder_name = "k8s_job"  # 提交集群使用的目录
folder_list = [  # 提交集群使用的HAT目录
    "../../hat",
    "../../tools",
    "../../examples",
    "ssh_launcher.py",  # 使用 `torch` launcher 时需要该文件
]
job_list = [  # 集群运行的所有脚本
    "python3 tools/train.py --config examples/classification/resnet18.py --stage float",
    "python3 tools/train.py --config examples/classification/resnet18.py --stage qat",
    "python3 tools/train.py --config examples/classification/resnet18.py --stage int_infer",
]
```

用户可以通过修改`k8s_config.py`来达到修改集群提交的基础配置的目的，而训练任务的具体配置需要通过修改训练的`config`来完成，比如这里的`examples/classification/resnet18.py`。

> 注意：集群训练完成之后，被调度机器上的资源会被全部回收。因此在任务结束之前，一定注意提前将数据或者模型转移到永久保存的介质上（如`HDFS`，`GPFS`等）。

## generate_submit_files的详细介绍

`generate_submit_files.py`主要用来实现生成代码路径、生成任务脚本命令的功能。

生成代码路径的功能介绍：脚本读取`k8s_config.py`中设置的`folder_list`，将`folder_list`中的文件统一复制到上传目录中。后续`submit.py`调用脚本，获取生成的上传目录，通过调用`aidisdk`的接口上传至集群中。

生成任务脚本命令的功能介绍：脚本读取`k8s_config.py`中设置的各种用户自定义命令，同时自动添加集群模型训练必须的环境变量，生成任务脚本，即`job.sh`。后续`submit.py`调用脚本，获取`job.sh`，通过调用`aidisdk`的接口上传至集群中。

###  aidisdk

`aidisdk`提供了 DAG 模块，用以支持 DAG 任务的提交。`submit.py`调用了`dag.new_dag` 和 `dag.new_job` 方法来创建和提交任务，不再依赖其他命令行工具和辅助yaml文件，整体使用方法更简洁和直接。

参考资料如下：
http://model.aidi.hobot.cc/api/docs/AIDISDK/latest/html/tutorials/compute/dagpipeline.html#dag

## 多机训练

多机训练是集群提供的，可以简单有效加速训练的方法。用户可以并行调度多台机器来加速训练。

`HAT`算法包中，用户可以通过简单配置`num_machines`的大小来申请多台机器同时训练。

> 注意：目前多机训练的`launcher`支持`mpi`和`torch`两种方式。
