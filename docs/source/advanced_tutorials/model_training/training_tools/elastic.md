# HAT 弹性(容错)训练使用介绍

## 引言

在集群训练任务中，特别是在拥有大量节点的场景下，由于节点故障而导致任务失败的情况时有发生。这种情况下，任务中断会增加意外故障的概率。每当任务中断时，需要人工介入来重新提交和恢复任务，这会导致一定的延迟。针对这种情况，弹性（容错）训练的需求应运而生。
目前，HAT 中已接入 AIDI 容错训练功能，本文档主要从用户角度，介绍用户如何在 HAT 中使用容错功能。

## 容错

容错是指当节点环境故障导致任务中断后，以相同计算资源，自动完成任务重启和resume。


### 使用方法

1. 首先是在任务提交 config (即 k8s_config) 中修改配置，以开启容错功能，包括以下内容：

```python

# 1. 设置 launcher 为 torchrun，指定为 torchrun 时，才会使用容错功能
launcher = "torchrun" 

# 2. job_max_restarts 设置最大重启次数，job_max_restarts=0 时，不会重启
job_max_restarts = 3

# 3. 设置任务重启模式
# * 若设置为 "always", 则只要任务挂掉，无论遇到任何错误，都会重启任务
job_restart_mode = "always"  # or ""

# * 若设置为 ""，则只有匹配到 elastic_pattern_files 中的错误时才会重启，其他错误则结束任务
elastic_pattern_files = ["default_elastic_patterns.yaml"]


# 4. 上传文件中，加上 watch_dog、elastic_launcher、default_elastic_patterns 文件
folder_list = [
    ...
    "watch_dog.py",
    "elastic_launcher.py",
    "default_elastic_patterns.yaml",
]
```

**说明：**
对于容错重启策略，目前支持两种可配选项 "Always" 和 ""(默认) ，其中：
- "Always"：表示无论遇到任何错误，只要任务中途挂掉，都会重启任务。
- ""：表示只有匹配到 elastic_pattern_files 中的错误时才会重启，其他错误则会结束退出任务，旨在做更精准的容错。
当前，HAT 默认重启错误可在 default_elastic_patterns.yaml 中看到：
```yaml
restart:
  patterns:
    - '.*RuntimeError: DataLoader worker \(pid \d+\) is killed by signal: Killed.*'
    - '.*RuntimeError: DataLoader worker \(pid \d+\) is killed by signal: Bus error.*'
    - '.*RuntimeError: NCCL communicator was aborted on rank \d+.*'
    - '.*OSError: \[Errno \d+\] No space left on device:.*'
  exitcode:
    - 210
    - 211
    - 220
    - 230


exit:
  patterns:
    - '^/pytorch/aten/src/ATen/native/cuda/Loss\.cu.*Assertion.*failed.'
```
当前 `default_elastic_patterns.yaml` 中错误类型是根据 SD 项目中整理而来，可能不能完全覆盖所有的意外情况，若用户有其他明确已知节点故障而导致任务挂掉的错误，请及时联系 HAT 团队更新规则。

2. 提交集群任务
增加以上配置后，以 `single_job` 方式提交集群任务，例如:
```bash
python3 submit.py --config k8s_config --queue xxx --single-job
```
提交集群任务的详细介绍，请参照 {ref}`k8s/submit_k8s_job`。



**说明:**
这次支持容错功能的同时，HAT 中增加支持了 DataLoader 的 resume 功能，但仍有以下限制：
* Dataloader 的 resume 仅支持 dataloader 使用了 `torch.utils.data.DistributedSampler` 的情况，且仅在 `by_step` 训练时生效；

**注:** 在 resume 时，无论是模型参数，还是 dataloader 的恢复，都依赖于训练中保存的 checkpoint 文件，因此，请根据自身情况合理设置 checkpoint 文件保存间隔。

## 弹性
(暂未支持)
