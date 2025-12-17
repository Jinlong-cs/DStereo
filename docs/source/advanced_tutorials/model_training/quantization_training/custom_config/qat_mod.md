# qat_mode 介绍

## 作用

qat_mode 用于设置 QAT 阶段是否带 BN 进行量化训练，配合海图提供的 FuseBN 接口还可以控制量化训练全程带 BN 或是中途逐步吸收 BN.

## 可选项定义

qat_mode 可选的设置有如下三种：

```python
class QATMode(object):

    FuseBN = "fuse_bn"
    WithBN = "with_bn"
    WithBNReverseFold = "with_bn_reverse_fold"
```

## 原理介绍

### fuse_bn

QAT 阶段没有 BN，海图默认的量化训练方式。通过将 qat_mode 设置为 fuse_bn，在浮点模型 op 融合的过程中 BN 的 weight 和 bias 均被吸收到 Conv 的 weight 和 bias 中，原来的 Conv + BN 的组合将只剩下 Conv，这一吸收过程理论上是没有误差的。

### with_bn

QAT 阶段带 BN 进行训练。
通过设置 qat_mode 为 with_bn，浮点模型转为 QAT 模型的时候 BN 不会吸收进 Conv，而是在 QAT 阶段以 “Conv + BN + 输出量化节点” 的形式作为一个被融合的量化 op 存在于量化模型中。最终在量化训练结束转为 quantized(也称 int infer) 模型的步骤中，BN 的 weight 和 bias 将自动吸收进 conv 的量化参数中，吸收之后得到的 quantized op 和原来的QAT op 计算结果保持一致。
在这一模式下，用户还可以选择在 QAT 中途将 BN 吸收进 Conv。用户手动吸收 BN 前后 QAT 模型 的 forward 结果不一致，原因是 BN weight 吸收至 Conv weight 之后，在之前量化训练中统计出来的量化参数 conv_weight_scale 不再适用于当前的 conv_weight，在对 conv_weight 的量化中将产生较大误差，需要继续进行量化训练调整量化参数。

### with_bn_reverse_fold

QAT 阶段带 BN 进行训练。
本模式与 with_bn 的不同之处在于在 BN 吸收之前，量化训练阶段计算 conv_weight_scale 时会考虑 BN 的 weight(具体的计算方式不在此详述)，目的是为了吸收 BN weight 之后 conv_weight_scale 仍然适用于新的 conv_weight。该模式用意是为分步吸收 BN 提供一种无损的吸收方式：在量化训练中途吸收 BN，吸收前后模型 forward 结果理论上完全一致，用户可以在量化训练结束前逐步吸收模型中所有的 BN 并且保证每次吸收之后 loss 不会有太大的波动。
在该模式下如果有 BN 在量化训练结束时仍未被吸收，在 QAT 模型转 quantized 模型的过程中剩余的 BN 将自动被吸收，这一吸收操作理论上是无损的。

## 用法

### 设置 qat_mode

用户只需要在 `model_convert_pipeline` 中设置 `qat_mode` 即可
Example:

```python
model_convert_pipeline=dict(
    type="ModelConvertPipeline",
    qat_mode="with_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
        ),
    ],
)
```

### 查看当前 qat_mode

```python
from horizon_plugin_pytorch.qat_mode import get_qat_mode
qat_mod = get_qat_mode()
```

### 设置逐步吸收 BN

在 with_bn 和 with_bn_reverse_fold 两种模式下，用户可以将 FuseBN 设置为回调函数用于在指定的 epoch 或是 step 吸收指定 module 中的 BN.
FuseBN 定义:

```python
class FuseBN(OnlineModelTrick):
    Args:
        module: sub model names to fuse BN.
        step_or_epoch: when to fuseBN, same length as module.
        update_by: by step or by epoch.
        inplace: if fuse BN inplace
    def __init__(
        self,
        modules: List[List[str]],
        step_or_epoch: List[int],
        update_by: str,
        inplace: bool = False,
    )
```

在 config 文件中使用 FuseBN Example:

```python
from hat.callbacks.online_model_trick import FuseBN

# 定义回调函数
# 命名为 backbone 的 module 中的 BN 将在第 1000 个 step 被吸收
# 命名为 neck 的 module 中的 BN 将在第 1500 个 step 被吸
fuse_bn_callback = FuseBN(
   modules=[['backbone'], ['neck']],
   step_or_epoch=[1000, 1500],
   update_by='step',
)

# 将回调函数加入到 trainer 中
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=...,
    optimizer=...,
    batch_processor=...,
    num_epochs=...,
    device=None,
    callbacks=[
        callbacks0,
        ..., 
        fuse_bn_callback,
        callbacks99
    ],
    ...,
)
```

## qat_mode 总结

| qat_mode              | BN 何时被吸收                        |  如何吸收BN                                | 理论上吸收后模型 forward 结果是否有变化 |
|-----------------------|--------------------------------------|--------------------------------------------|-----------------------------------------|
| fuse_bn               | 一定在浮点模型 op 融合过程           | 执行 fuse_module 之后吸收完成              |        无                               |
| with_bn               | 可以在量化训练中途                   | 通过设置回调函数在指定 epoch 或 batch 吸收 |        有                               |
| with_bn               | 可以在 QAT 模型转 quantized 模型过程 | 随 QAT 转 quantized 自动完成               |        无                               |
| with_bn_reverse_fold  | 可以在量化训练中途                   | 通过设置回调函数在指定 epoch 或 batch 吸收 |        无                               |
| with_bn_reverse_fold  | 可以在 QAT 模型转 quantized 模型过程 | 随 QAT 转 quantized 自动完成               |        无                               |

一般训练流程是浮点训练到理想精度然后量化训练，该流程只需要使用 fuse_bn 即可。如果是没有浮点训练一开始就是量化训练，为了确保模型能收敛，才需要使用带 BN 的量化训练模式。

注：本文中之所以说“理论上吸收前后无损“或”无变化“，是由于在实际计算中吸收前后两次浮点计算的结果有较低的概率会在小数点较靠后的数位上不一致，微小的变化加上量化操作导致吸收 BN 后 Conv 的输出相比吸收前 Conv + BN 的输出在部分数值上可能会产生一个输出 scale 的绝对误差。
