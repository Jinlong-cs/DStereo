# 启动方式

tools+configs是HAT目前最基础的训练方式。但是很多情况下，我们需要处理多卡或者分布式的环境。这些环境需要依赖一些第三方库的方式，才能够在多个环境中把基础的训练方式有效的组织起来。

在多卡或者分布式的环境里，常用的启动方式有mpirun，torchrun等，这里我们从命令行的形式简单讲一下这些启动方式的区别。其他情况如自动化集群提交脚本里，其实也是利用一些自动化手段最终生成这些启动方式需要的命令。

下面以renset18的float训练为例，把所有目前HAT支持的启动方式都列出来，作为参考。

## 最简单的模式

这种最简单的模式，支持的范围包括单机单卡和单机多卡两种模式，不支持多机多卡，配置的方式只需要修改configs中的device_ids的索引即可。

需要说明的是，这里单机多卡的支持方式，其实是借助torch.multiprocess来把单机内部的所有进程有效的管理起来，当然这也是这种模式不支持多机的原因。

```shell
# 单机单卡
python3 tools/train.py --config examples/classification/resnet18.py --stage float

# 单机多卡 （设置config中的device_ids）
python3 tools/train.py --config examples/classification/resnet18.py --stage float
```

## torchrun

首先需要注意的是，torchrun要求torch的版本大于等于`1.10.0`。对于小于`1.10.0`使用的torch.distributed.launch在HAT中则没有单独支持，也不建议使用了。

torchrun是torch框架为了支持用户方便快捷的处理分布式环境里面的各种环境变量而提供的启动工具。

torchrun的细节可以参考[社区文档](https://pytorch.org/docs/1.10/elastic/run.html?highlight=torchrun)。

```shell
# 单机单卡（不建议使用torchrun）
torchrun --nproc_per_node=1 tools/train.py --config examples/classification/resnet18.py --stage float --launcher torch

# 单机多卡（nproc_per_node和config中device_ids卡数保持一致）
# 如果需要指定卡训练需要单独加上CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=“0,1,2,3” torchrun --nproc_per_node=4 tools/train.py --config examples/classification/resnet18.py --stage float --launcher torch

# 多机多卡
# node1:
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=8888 --rdzv_backend=c10d --rdzv_endpoint=hostip1 tools/train.py --config examples/classification/resnet18.py --stage float --launcher torch
# node2(rdzv_id需要和node1完全一致):
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=8888 --rdzv_backend=c10d --rdzv_endpoint=hostip1 tools/train.py --config examples/classification/resnet18.py --stage float --launcher torch

# 借助于ssh_launcher自动处理多个节点：
python3 ssh_launcher.py -n 2 -H /job_data/mpi_hosts 'torchrun --nodes=2 --nproce_per_node=4 --rdzv_id=8888 --rdzv_backend=c10d tools/train.py --config examples/classification/resnet18.py --step float --launcher torch'
```

## mpirun

mpirun是用来处理分布式多进程的进程管理常用的工具。mpirun最大的优势是对分布式和单机多进程有统一的抽象。因此mpirun也可以同时处理单机和多机的任何情况。

```shell
# 单机单卡
mpirun -n 1 -ppn 1 python3 tools/train.py --config examples/classification/resnet18.py --step float --launcher mpi

# 单机多卡（ppn和configs中的device_ids保持一致）
mpirun -n 4 -ppn 4 python3 tools/train.py --config examples/classification/resnet18.py --step float --launcher mpi

# 多机多卡
mpirun -n 8 -ppn 4 python3 tools/train.py --config examples/classification/resnet18.py --step float --dist-url hostip --launcher mpi
```

最后，需要说明的是，不管是python多进程，torchrun，还是mpirun其实都是进程管理程序，而进程间的通信方法依赖于torch内部process group的初始化方法，因此不同的管理程序对于训练效率没有影响。

而不同的管理程序最大的区别其实就是进程管理方法的区别：比如异常退出的时候，是否能在主进程端拿到所有节点的报错信息。单个进程异常时，能否保证所有进程完整退出。其他的像内部开发模式等其实整体上区别也不大。

