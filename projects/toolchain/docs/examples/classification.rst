MobileNetV1分类模型训练
=======================

这篇教程主要是告诉大家如何利用 `HAT` 在 `ImageNet` 上训练一个\
state-of-art的浮点和定点模型。
`ImageNet` 是图像分类里用的最多的数据集，很多最先进的图像分类研究\
都会优先基于这个数据集做好验证。虽然有很多方法在社区或者其他途径里可\
以获取到 state-of-art 的分类模型，但从头训一个 state-of-art 的分类\
模型仍然不是一个简单的任务，本篇教程将会重点讲叙从数据集准备开始如何\
在 `ImageNet` 上训练出一个 state-of-art 的模型，包括浮点、量化和\
定点三种模式。

其中 `ImageNet` 数据集可以从这里下载：http://image-net.org/。\
下载之后的数据集格式为：

    .. code-block:: bash

        ILSVRC2012_img_train.tar
        ILSVRC2012_img_val.tar
        ILSVRC2012_devkit_t12.tar.gz


这里我们使用 `MobileNetV1` 的例子来详细介绍整个分类的流程。


1. 训练流程
-----------

如果你只是想简单的使用 `HAT` 的接口来进行简单的实验，那么首先阅读一下\
这一小节的内容是个不错的选择。对于所有的训练和评测的任务， `HAT` 统一\
采用 `tools + config` 的形式来完成。在准备好原始数据集之后，通过以下\
的流程，我们可以方便的完成整个训练的流程。


数据集准备
>>>>>>>>>>>>

首先是数据集打包，打包数据集与原始数据集在处理速度上有明显的优势，这里\
我们选择与 `PyTorch` 一脉相承的 `LMDB` 的打包方法，当然由于 `HAT` 在\
处理 `dataset` 上的灵活性，其他形式的数据集打包和读取形式，如 `MXRecord` \
也是可以独立支持的。

在 `tools/datasets` 目录下提供了 `cityscapes` 、 `imagenet` 、 `voc`\、
`mscoco` 这些常见数据集的打包脚本。例如 `imagenet2lmdb` 的脚本，可以利用\
`torchvision` 提供的默认公开数据集处理方法直接将原始的公开 `ImageNet` 数据\
集转成 `Numpy` 或者 `Tensor` 的格式，最后将得到的数据统一用 `msgpack`\
的方法压缩到 `LMDB` 的文件中。

这个过程可以很方便通过下面的脚本完成数据集打包:

    .. code-block:: bash

       python3 tools/datasets/imagenet_cls/imagenet2lmdb.py --src-data-dir ${src-data-dir} --target-data-dir ${target-data-dir} --split-name train --num-workers 10

       python3 tools/datasets/imagenet_cls/imagenet2lmdb.py --src-data-dir ${src-data-dir} --target-data-dir ${target-data-dir} --split-name val --num-workers 10


在完成数据集打包之后，可以得到含有 `ImageNet` 的 `LMDB` 数据集。下一步就\
可以开始训练。

模型训练
>>>>>>>>>>>>

准备打包好数据集之后，便可以开始训练模型。只需要运行下面的命令就可以启动训练：

    .. code-block:: bash

       python3 tools/train.py --step "float" --config configs/classification/mobilenetv1.py
       

       
由于 `HAT` 算法包使用了注册机制，使得每一个训练任务都可以按照这种\
`train.py` 加上 `config` 配置文件的形式启动。`train.py` 是统一\
的训练脚本，与任务无关，我们需要训练什么样的任务、使用什么样的数据集\
以及训练相关的超参数设置都在指定的 `config` 配置文件里面。上面的命\
令中 `--step` 后面的参数可以是 `"float"` 、 `"qat"` 、`"int_infer"` ，\
分别可以完成浮点模型、量化模型的训练以及量化模型到定点模型的转化，\
其中量化模型的训练依赖于上一步浮点训练产出的浮点模型，定点模型的转化\
依赖于量化训练产生的量化模型，具体内容见上一节量化的细节。


模型验证
>>>>>>>>>>>>

在完成训练之后，可以得到训练完成的浮点、量化或定点模型。和训练方法类似，\
我们可以用相同方法来对训好的模型做指标验证，得到为 `Float` 、 `QAT` \
和 `Quantized` 的指标，分别为浮点、量化和完全定点的指标，详细说明见量化细节。

    .. code-block:: bash

        python3 tools/train.py --step "float" --val-only --val-ckpt ${float-checkpoint-path} --config configs/classification/mobilenetv1.py

        python3 tools/train.py --step "qat" --val-only --val-ckpt ${qat-checkpoint-path} --config configs/classification/mobilenetv1.py

        python3 tools/train.py --step "int_infer" --val-only --val-ckpt ${int8-checkpoint-path} --config configs/classification/mobilenetv1.py

和训练模型时类似，`--step` 后面的参数为 `"float"` 、 `"qat"` 、 `"int_infer"`\
时，分别可以完成对训练好的浮点模型、量化模型、定点模型的验证。

模型推理
>>>>>>>>>>>>

 `HAT` 提供了 `infer.py` 脚本对各阶段训练好的模型的推理结果进行可视化展示:

    .. code-block:: bash

        python3 tools/infer.py --config configs/classification/mobilenetv1.py --step {float/qat/int_infer} --visual-type Imagenet --input-size 224x224 --img ${img-path}



仿真上板精度验证
>>>>>>>>>>>>>>>>>

除了上述模型验证之外，我们还提供和上板完全一致的精度验证方法，可以通过下面的方式完成：

    .. code-block:: bash

       python3 tools/align_bpu_validation.py --config configs/classification/mobilenetv1.py


定点模型编译
>>>>>>>>>>>>

在 `HAT` 中集成的量化训练工具链主要是为了地平线的芯片准备的，因此，对于量化模型的\
检查和编译是必须的。我们在训练脚本中提供可以对模型检查的接口，可以让用户定义好量化模\
型之后，先检查能否在 `BPU` 上正常运行,目前该检查在启动 `train.py` 脚本训练 `quantize=True` \
的模型时会默认打开。

在模型训练完成后，可以通过 `compile_perf` 脚本将量化模型编译成可以上板运行\
的 `hbm` 文件，同时该工具也能预估在 `BPU` 上 的运行性能：

    .. code-block:: bash

       python3 tools/compile_perf.py --config configs/classification/mobilenetv1.py


针对地平线不同架构的BPU，可以在 `configs/classification/mobilenetv1.py` 中设置 `march = March.BAYES` 或 `march = March.BERNOULLI2`:


以上就是从数据准备到生成量化可部署模型的全过程。


2. 训练细节
-----------

在这个说明中，我们仍使用 `MobileNetV1` 作为示例，对模型训练需要注意的\
一些事项进行说明，主要为 `config` 的一些相关设置。

训练条件
>>>>>>>>>

在超过100w张图片的 `ImageNet` 上训练一个深度学习模型是非常消耗资源的，\
主要的瓶颈在于矩阵计算和数据读取。

对于矩阵计算，这里非常建议使用一个高性能的 GPU 来替代 CPU 作为训练工具，\
另外同时使用多个 GPU 可以有效的降低训练时间。

对于数据读取，推荐使用更好的 CPU 和 SSD 存储。多线程的 CPU 加速和更好的\
SSD 存储对于数据读取有很大帮助。需要注意的是，整个 `ImageNet` 大概会\
占用 300G 的存储资源，所以 SSD 存储至少得有 300G 的存储空间。


网络结构
>>>>>>>>

`HAT` 或者其他一些社区都可以找到 `MobileNetV1` 丰富的实现方法，因此，\
`MobileNetV1` 的具体实现在这里并不赘述。

在 `HAT` 的 `config` 中，我们可以直接用下面的 dict 就可以构建一个浮点\
的 `MobileNetV1` 分类模型。用户可以直接修改 `backbone` 中的配置参数\
来达到修改模型目的。

    .. code-block:: python

       model = dict(
            type="Classifier",
            backbone=dict(
                type="MobileNetV1",
                num_classes=1000,
                bn_kwargs={},
            ),
            losses=dict(type="CEWithLabelSmooth")
       )
       

模型除了 `backbone` 之外，还有 losses 模块，在常见的分类模型中，我们一般\
直接采用 `Cross-Entropy` 作为训练的 loss，但是越来越多的实验证明，在分类\
的 loss 中加上 `Label-Smooth` 对训练的结果有帮助，尤其是在结合 `Cosine` \
的 lr 更新方法上。

通常在定义好一个模型之后，尤其是一些公版模型，我们会有检查计算量的需求。 `HAT` \
通过 `calops` 的工具来计算模型的计算量，执行方法如下：

    .. code-block:: bash
    
       python3 tools/calops.py --config configs/classification/mobilenetv1.py

这种计算量的统计工具，是可以同时支持浮点和定点模型的。


数据增强
>>>>>>>>

关于 `ImageNet` 训练的数据增强已经逐渐形成一种共识，我们以 `torchvision` 提供\
的数据增强作为基础，构建分类训练的数据增强，包括 `RandomResizedCrop` ，\
`RandomHorizontalFlip`， `ColorJitter` 。

因为最终跑在 `BPU` 上的模型使用的是 `YUV444` 的图像输入，而一般的训练图像输\
入都采用 `RGB` 的形式，所以 `HAT` 提供 `BgrToYuv444` 的数据增强来将 `RGB` \
转到 `YUV444` 的格式。为了优化训练过程，`HAT` 使用了 `batch_processor`，\
可将一些增强处理放在 `batch_processor` 中优化训练：

    .. code-block:: python

        dataset=dict(
            type="ImageNetFromLMDB",
            data_path="./tmp_data/imagenet/train_lmdb/",
            transforms=[
                dict(
                    type=torchvision.transforms.RandomResizedCrop,
                    size=224,
                    scale=(0.08, 1.0),
                    ratio=(3. / 4., 4. / 3.),
                ),
            ],
            num_samples=1281167
        )

对应的 `batch_processor` 部分
        
    .. code-block:: python
    
        batch_processor = dict(
            type='BasicBatchProcessor',
            need_grad_update=True,
            batch_transforms=[
                dict(type=torchvision.transforms.RandomHorizontalFlip),
                dict(
                    type=torchvision.transforms.ColorJitter,
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                ),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type=torchvision.transforms.Normalize, mean=128.0, std=128.0),
            ],
        )

验证集的数据转换相对简单很多，最主要的区别是做短边 `Resize` 到256，和 `CenterCrop` 。\
其他的颜色空间转换和训练集是一样的。

    .. code-block:: python

        dataset=dict(
            type="ImageNetFromLMDB",
            data_path="./tmp_data/imagenet/val_lmdb/",
            transforms=[
                dict(type=torchvision.transforms.Resize, size=256),
                dict(type=torchvision.transforms.CenterCrop, size=224),
            ],
            num_samples=50000,
        ),
        
        val_batch_processor = dict(
            type='BasicBatchProcessor',
            need_grad_update=False,
            batch_transforms=[
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type=torchvision.transforms.Normalize, mean=128.0, std=128.0),
            ],
        )


训练策略
>>>>>>>>

在 `ImageNet` 上训练不同分类模型的训练策略大体上一致，但也有微小的区别。\
这里我们主要介绍有效果提升的细节。

`Cosine` 的学习策略配合 `Warmup` 与普通的 `StepLr` 有一些提升效果。适当延长\
epoch的训练长度对于小模型也有提升。

另外，只对 `weight` 的参数施加 L2 norm 也是推荐的训练策略。\
`configs/classification/mobilenetv1.py` 文件中的 `float_trainer`，\
`qat_trainer`，`int_trainer` 分别对应浮点、量化、定点模型的训练策略。\
下面以 `float_trainer` 训练策略示例：

    .. code-block:: python
    
       float_trainer = dict(
            type='distributed_data_parallel_trainer',
            model=model,
            data_loader=data_loader,
            optimizer=dict(
                type=torch.optim.SGD,
                params={"weight": dict(weight_decay=4e-5)},
                lr=0.4,
                momentum=0.9,
            ),
            batch_processor=batch_processor,
            num_epochs=240,
            device=None,
            callbacks=[
                stat_callback,
                dict(
                    type="CosLrUpdater",
                    warmup_by='epoch',
                    warmup_len=5,
                    step_log_interval=1000,
                ),
                metric_updater,
                val_callback,
                ckpt_callback,
            ],
            train_metrics=[ dict(type="Accuracy"), dict(type="TopKAccuracy", top_k=5)],
            val_metrics=[ dict(type="Accuracy"), dict(type="TopKAccuracy", top_k=5)],
       )


量化训练
>>>>>>>>

关于量化训练中的关键步骤，比如准备浮点模型、算子替换、插入量化和反量化节点、设置量化参数以及算子的融合等，请阅读 `《Horizon Plugin PyTorch》` 手册中的
`浮点模型准备` 和 `算子融合` 两节中的内容。这里主要讲一下 `HAT` \
的分类中如何定义和使用量化模型。

在模型准备的好情况下，包括量化已有的一些模块完成之后， `HAT` 在训练脚本\
中统一使用下面的脚本将浮点模型映射到定点模型上来。

    .. code-block:: python

        model.fuse_model()
        model.set_qconfig()
        horizon.quantization.prepare_qat(model, inplace=True)


量化训练的策略并不统一，这里简单描述分类模型训练中的常见策略。

量化训练的整体策略可以直接沿用浮点训练的策略，但学习率和训练长度需要\
适当调整。因为有浮点预训练模型，所以量化训练的学习率 `Lr` 可以很小，\
一般可以从 0.001 或 0.0001 开始，并可以搭配 `StepLrUpdater` 做 1-2 次\
`scale=0.1` 的 `Lr` 调整；同时训练的长度不用很长。此外 `weight decay` \
也会对训练结果有一定影响。


模型检查编译和仿真上板精度验证
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

对于 `HAT` 来说，量化模型的意义在于可以在 `BPU` 上直接运行。因此，\
对于量化模型的检查和编译是必须的。前文提到的 `compile_perf` 脚本也\
可以让用户定义好量化模型之后，先检查能否在 `BPU` 上正常运行，并可通过\
`align_bpu_validation` 脚本获取模型上板精度。用法同前文。


3. 预训练模型
-------------

`HAT` 已经提供了丰富的在 `ImageNet` 上的预训练模型，可以参照 `modelzoo` \
的内容，所有模型都在发布包中。
