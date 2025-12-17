# 在 HAT 框架中使用 DeepSpeed 优化训练

## 引言

DeepSpeed是一个深度学习优化库，使分布式训练和推理变得简单、高效。

目前，HAT 已支持部分 DeepSpeed 特性，本文档主要从用户角度，介绍用户如何在 HAT 中使用 DeepSpeed 的一些特性。

## 环境准备

安装 DeepSpeed

```shell
pip3 install deepspeed

```

## 使用方法

1. 添加 DeepSpeed 配置

目前可以通过使用 json 文件配置或者是直接在 HAT config 文件进行配置来使用 DeepSpeed 的特性。

下面提供一个使用 json 文件来配置使用 ZeRO(Zero Redundancy Optimizer) 特性的例子：

```json
{
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 10000,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        }
    },
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },
    "zero_allow_untested_optimizer": true,
}
```

如果是直接在 config 文件中配置，直接将上述 json 文件的内容完全拷贝到 config 文件中即可。

ZeRO利用数据并行的聚合计算和内存资源来减少用于模型训练的每个设备（GPU）的内存和计算需求。ZeRO通过在分布式训练硬件中的可用设备（GPU和CPU）之间划分各种模型训练状态（权重、梯度和优化器状态）来减少每个GPU的内存消耗

2. 修改 HAT 中的 config 文件

修改 trainer:

配置 trainer type.
```python
float_trainer["type"] = "DeepSpeedTrainer" # 或者是 "deepspeed_trainer"
```

配置 deepspeed 特性方式一:使用 json 文件来配置。
```python
float_trainer["config_json"] = "deep_speed.json"
```

配置 deepspeed 特性 方式二:直接在 config 文件中使用 dict 来配置。
```python
config_params = {
    "train_micro_batch_size_per_gpu": 16,
    ...
    ...
    "zero_allow_untested_optimizer": true,
}
float_trainer["config_params"] = config_params
```

上述配置修改完成后，就可以按照常规启动 HAT 训练，DeepSpeed 将会生效。
使用 DeepSpeed 单机或者是多机的启动方式与当前启动方式没有区别，单机仍然是 'python tools/train.py', 多机是 'torchrun' 加相关的启动脚本。具体可参考本文档中 'launcher' 相关章节。
当然用户也可以通过 deepspeed 来启动训练。

单机示例：
```shell
deepspeed --num_nodes 1 --num_gpus 2 tools/train.py --config examples/classification/mobilenetv1.py --stage float --device-ids "2,3"
```

多机示例：
```shell
deepspeed --hostfile hostfile tools/train.py --config examples/classification/mobilenetv1.py --stage float --device-ids "0,1,2,3"
```

hostfile 是由 ip 地址加 slots 数量组成的文件，示例如下
```
#hostfile
10.10.112.94 slots=4
10.10.112.93 slots=4
```
目前提交至集群多机使用 deepspeed 启动的自动化脚本还在完善中，用户可优先选择 torchrun 在集群启动多机训练，或手动配置相关的启动脚本及 hostfile.

在完成训练之后会自动保存 DeepSpeed 的 checkpoint, 里面包含模型本身和 DeepSpeed 相关的训练参数，可以通过以下配置 load:

```python
float_trainer["checkpoint_dir"] = "tmp_models/ds-last-ckpt"
```
与 torch 原生 save checkpoint 会保存一个 pt 文件不同，deepspeed 每次 save 的 checkpoint 是一个目录，内部包含两个 pt. 因此用户在 load 时仅需指定该目录即可，而不是特定的 pt 文件。
如果想要个性化地 load 模型参数，可以自定义一个 custom_load_fn 来完成，custom_load_fn 接收 src 和 dst 两个参数，这两个参数分别代表特定的参数字典和当前模型，示例如下：

```python
def custom_load_fn(src, dst):
    ...
    ...

# 修改配置文件使用 custom_load_fn

float_trainer["custom_load_fn"] = custom_load_fn
```

其余参数配置方式与当前 DDP 大致相同。

**注意**
目前暂不支持 delay_sync/enable_apex 和 DeepSpeed 搭配使用。

## DeepSpeed 训练之后如何做 Predict

DeepSpeed 训练之后做 Predict 的方式和其他 Trainer 的方法没有区别，都是使用 HAT 中的 'Predictor' 构建模型和 tools/predict.py 来启动。唯一需要注意的是 LoadCheckpoint 所使用的 pt 文件是目录下两个 pt 文件中以 'model_states.pt' 结尾的那个。

## 更多参考资料

deepspeed api: https://deepspeed.readthedocs.io/en/latest/index.html
github link: https://github.com/microsoft/DeepSpeed
