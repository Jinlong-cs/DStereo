UNet分割模型训练
================

作为 `HAT` 分割任务的示例，这篇教程主要向大家展示如何使用 `HAT` 在 `Cityscapes` \
数据集上训练一个 state-of-the-art 的浮点和定点模型。

`Cityscapes` 是一个城市驾驶场景的图像数据集，包含了5000张具有像素级标注的图像，图像\
中的物体被分为19个类别。分割任务相对复杂，对模型能力的要求较高，使用小模型在分割任务\
上取得较好的指标并不是很轻易的事。本教程将从零开始，详细描述如何使用 `HAT` 在\
`Cityscapes` 数据集上训练一个 state-of-the-art 的分割模型，并在浮点模型基础上进行\
量化训练，最终得到一个定点模型。

1. 训练流程
-----------

数据集下载
>>>>>>>>>>

要下载 `Cityscapes` 数据集，首先需要在 \
`官方网站 <https://www.cityscapes-dataset.com/register/>`_ 注册一个账号。

之后便可在 `下载页面 <https://www.cityscapes-dataset.com/downloads/>`_ 下载需要\
的数据集文件，这里我们只需要 `gtFine_trainvaltest.zip` 和 \
`leftImg8bit_trainvaltest.zip` 两个文件。

同时， `Cityscapes` 数据集官方还提供了一个脚本用于数据的下载和处理，见 \
`Github链接 <https://github.com/mcordts/cityscapesScripts>`_ 。\
首先使用如下命令安装官方工具：

.. code-block:: bash

    python3 -m pip install cityscapesscripts

然后使用官方工具下载所需数据集文件（注意，使用此工具下载仍需要登录上面注册的账号）。

.. code-block:: bash

    csDownload -d ${data-dir} gtFine_trainvaltest.zip
    csDownload -d ${data-dir} leftImg8bit_trainvaltest.zip

最后对下载好的文件进行解包即可（可选）：

.. code-block:: bash

    cd ${data-dir}
    unzip gtFine_trainvaltest.zip
    unzip leftImg8bit_trainvaltest.zip



数据集打包
>>>>>>>>>>

为了高效地读取数据，我们推荐预先将数据集打包为 `LMDB` 格式。 `HAT` 提供了\
``cityscapes2lmdb.py`` 脚本来方便地将数据集从原始公开的格式转换为 ``numpy.ndarray``\
或 ``torch.Tensor`` ，使用 ``msgpack`` 对数据进行封装，并最终打包为 `LMDB` 文件。

数据集打包的命令如下：

.. code-block:: bash

    python3 tools/datasets/cityscapes_seg/cityscapes2lmdb.py --src-data-dir ${data-dir} --split-name train
    python3 tools/datasets/cityscapes_seg/cityscapes2lmdb.py --src-data-dir ${data-dir} --split-name val

生成的 lmdb 文件保存在 ${data-dir}/train_lmdb 和 ${data-dir}/val_lmdb 路径下。

模型训练
>>>>>>>>

将数据集打包为 `LMDB` 文件后，就可以开始模型的训练了。 `HAT` 提供了 ``train.py``\
训练脚本来方便地配合 `config` 文件实现模型训练。

模型训练的命令如下，开始训练之前请确保将 ``unet.py`` 配置文件中的数据集路径\
（``data_rootdir``）设置为数据集 `LMDB` 文件所在位置。

.. code-block:: bash

    python3 tools/train.py --step float --config configs/segmentation/unet.py
    python3 tools/train.py --step qat --config configs/segmentation/unet.py

以上两个命令分别完成浮点模型和定点模型的训练，其中定点模型的训练需要以训练好的\
浮点模型为基础，具体内容请阅读 `《Horizon Plugin PyTorch》` 手册中的
`浮点模型准备` 和 `算子融合` 两节中的内容。

指标验证
>>>>>>>>

训练完成后，脚本将会自动地对训练好的模型指标进行验证，除此之外， `HAT` 还提供了脚本\
来对训练好的浮点模型和定点模型进行指标验证。此脚本同样使用 `config` 文件来进行配置。

.. code-block:: bash

    python3 tools/train.py --step float --val-only --config configs/segmentation/unet.py
    python3 tools/train.py --step qat --val-only --config configs/segmentation/unet.py
    python3 tools/train.py --step int_infer --val-only --config configs/segmentation/unet.py

仿真上板精度验证
>>>>>>>>>>>>>>>>>

除了上述模型验证之外，我们还提供和上板完全一致的精度验证方法，可以
通过下面的方式完成：

.. code-block:: bash

    python3 tools/align_bpu_validation.py --config configs/segmentation/unet.py

模型推理
>>>>>>>>

`HAT` 提供了 ``infer.py`` 脚本对各阶段训练好的模型的推理结果进行可视化展示。

.. code-block:: bash

    python3 tools/infer.py -c configs/segmentation/unet.py --step {float/qat/int_infer} --visual-type Cityscapes --input-size 1024x2048


模型编译
>>>>>>>>

`HAT` 集成的量化训练框架所使用的量化算法专为地平线的芯片设计，因此使用 `HAT` 训练\
好的定点模型可以使用 `HBDK` 提供的工具进行编译，生成可在芯片上运行的定点模型。我们\
提供了 ``compile_perf.py`` 脚本来方便地完成编译。

.. code-block:: bash

    python3 tools/compile_perf.py -c configs/segmentation/unet.py --opt 2

至此，我们便从零开始得到了一个可在地平线芯片上运行的分割任务定点模型。

2. 训练细节
-----------

模型结构
>>>>>>>>

.. code-block:: python

    model = dict(
        type="Segmentor",
        backbone=dict(
            type="MobileNetV1",
            num_classes=-1,
            bn_kwargs=dict(eps=2e-5, momentum=0.1),
            alpha=0.25,
            dw_with_relu=True,
            include_top=False,
            flat_output=False,
        ),
        neck=dict(
            type="DwUnet",
            base_channels=8,
            bn_kwargs=dict(eps=2e-5, momentum=0.1),
            act_type=nn.ReLU,
            use_deconv=False,
            dw_with_relu=True,
            output_scales=(4, 8, 16, 32, 64),
        ),
        head=dict(
            type="SegHead",
            num_classes=19,
            in_strides=(4, 8, 16, 32, 64),
            out_strides=(4, 8, 16, 32, 64),
            stride2channels={
                4: 32,
                8: 64,
                16: 128,
                32: 256,
                64: 512
            },
            feat_channels=(32, 64, 128, 256, 512),
            stacked_convs=0,
            argmax_output=False,
            dequant_output=True,
            int8_output=True,
            upscale=False,
            output_with_bn=True,
            bn_kwargs=dict(eps=2e-5, momentum=0.1),
        ),
        losses=dict(
            type="ClassificationFocalLoss",
            loss_name="Focal",
            num_classes=19,
            weight=tuple(np.array((256, 128, 64, 32, 16)) / 19),
            reduction="mean",
        ),
    )

分割模型主要由 `backbone`， `neck` ， `head` 三部分组成，这里我们使用了\
`MobileNetV1_0.25` 作为 `backbone` ，`MobileNet` 是一种轻量高效的网络结构。\
`neck` 则使用了 `Unet` 结构，这种结构可以综合各个尺度上的 `featuremap`，保留精细的\
空间信息。 `head` 为一个卷积层，负责输出最终的分割结果。

我们使用了 `FocalLoss` 作为损失函数。 `FocalLoss` 可视为一种动态权重的交叉熵损失\
函数，可以较好地解决类别不平衡带来的训练困难的问题。

`Unet` 的层次化结构和 `FPN` 的思想一致，很适合采用相同的训练方式，即在 `Unet` 的\
每个 `scale` 上都构造一个输出，将此输出和对应尺寸的 `ground truth` 构造损失函数进行\
训练，通过对网络的各个尺度都进行监督，为网络训练提供更加丰富的参考信息，降低训练难度，\
提高训练速度和最终精度。同时考虑到我们需要的最终结果是最大尺寸网络输出(scale=4)，为了\
避免其他尺寸的梯度过大影响到最大尺寸输出的精度，我们按照 `scale` 为损失函数增加了相应\
的权重， `scale` 越大的层权重越小。

通常在定义好一个模型之后，尤其是一些公版模型，我们会有检查计算量的需求。\
`HAT` 提供了 ``calops.py`` 来计算模型的计算量，命令如下：

    .. code-block:: bash

       python3 tools/calops.py --config configs/segmentation/unet.py

这种计算量的统计工具，是可以同时支持浮点和定点模型的。


数据预处理
>>>>>>>>>>

首先，我们使用 ``LabelRemap`` 将数据标签重映射至 [0, 18] 的区间。

对于训练集， ``SegRandomAffine`` 可以对图片进行随机仿射变换来进行数据增强，我们只\
配置了随机的缩放，没有做任何旋转操作。

由于训练采用了类 `FPN` 的方式，我们需要将标签缩放至不同的大小，用于模型不同 `scale` \
的训练。

因为最终跑在 `BPU` 上的模型使用的是 `YUV444` 的图像输入，而一般的训练图像输\
入都采用 `RGB` 的格式，所以 `HAT` 提供 ``ImageBgrToYuv444`` 的数据增强来\
将 `RGB` 数据转到 `YUV444` 的格式。

最后，归一化对于深度学习模型训练是必须的。

值得注意的是，这里我们使用了 ``BasicBatchProcessor`` 进行任务的训练，\
此 `Processor` 支持在 `GPU` 上以 `batch` 为单位对数据做预处理。由于分割任务的数据\
预处理过程相对复杂，若使用 `CPU` 来做，会出现处理瓶颈。使用 `GPU` 预处理数据会提高\
显存使用量，导致最大 `batch size` 降低。但即使如此，最终的训练速度也有可观的提升。

.. code-block:: python

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="CityscapesFromLMDB",
            data_path=os.path.join(data_rootdir, "train_lmdb"),
            transforms=[
                dict(type="PILToTensor"),
            ],
            num_samples=2975,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=True,
    )
    batch_processor = dict(
        type="BasicBatchProcessor",
        need_grad_update=True,
        batch_transforms=[
            dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
            dict(type="SegOneHot", num_classes=num_classes),
            dict(type="SegResize", size=data_shape[1:]),
            dict(
                type="SegRandomAffine",
                degrees=0,
                scale=(0.5, 2.0),
                interpolation=InterpolationMode.BILINEAR,
                label_fill_value=0,
            ),
            dict(type="ImageBgrToYuv444", rgb_input=True),
            dict(type="ImageNormalize", mean=128.0, std=128.0),
            dict(
                type="Scale",
                scales=tuple(1 / np.array(train_scales)),
                mode="bilinear"
            ),
        ],
        loss_collector=loss_collector,
    )


验证集的数据预处理与训练集相比不需要做随机的仿射变换和多尺度缩放，\
其他步骤一致，不再赘述。

训练策略
>>>>>>>>

分割任务可以看作是像素级的分类任务，因此训练策略也和分类任务高度相似，首先在保证可以\
收敛的前提下尽可能调大学习率可以提高训练速度；在使用某一学习率训练至精度不再增长时，\
适当将学习率调小，模型便可继续收敛，精度进一步提升；最终训练完成后对测试集精度和\
训练集精度进行比较，若训练集精度高于测试集过多，则可以认为模型出现了过拟合，此时适当\
增大 `weight decay` 可以增强模型的泛化能力，减少过拟合，从而获得更高的测试集精度。

.. code-block:: python

    float_trainer = dict(
        type="distributed_data_parallel_trainer",
        model=model,
        data_loader=data_loader,
        optimizer=dict(
            type=torch.optim.SGD,
            params={"weight": dict(weight_decay=5e-3)},
            lr=0.01,
            momentum=0.9,
        ),
        batch_processor=batch_processor,
        device=None,
        num_epochs=train_epochs,
        callbacks=[
            stat_callback,
            dict(
                type="StepDecayLrUpdater",
                lr_decay_id=[200, 240, 280],
                lr_decay_factor=0.1,
            ),
            metric_updater,
            tb_callback,
            aidi_tb_callback,
            tb_loss_callback,
            aidi_tb_loss_callback,
            val_callback,
            ckpt_callback,
        ],
        sync_bn=True,
    )

量化训练策略
>>>>>>>>>>>>

量化训练的目的是在训练好的浮点模型基础上对数据进行模拟量化，来模拟定点计算的过程，\
使得经过量化训练的模型转为定点模型时的精度损失降到最低。

由于浮点模型经过了充分的训练已经收敛到了较优的状态，量化训练通常只需要对模型进行微调\
即可，学习率不能设置过大，可以从 1e-4 的数量级开始尝试，其他参数一般和浮点训练一致\
即可。同样地，微调模型所需的训练量较低，一般训练几十个 `epoch` 就可以了。

由于量化训练之前模型已经有一个不错的精度，精度提升的区间较小，且对数据进行了模拟量化，\
会导致训练过程波动较大。此时需要一些耐心，认真观察，从波动中看出趋势，适量调整参数，\
才能取得最好的结果。

模型检查和编译
>>>>>>>>>>>>>>

对于`HAT`来说，量化模型的意义在于可以在`BPU`上直接运行。因此，对于量化模型的检查\
和编译是必须的。

`HAT` 提供的 ``compile_perf.py`` 脚本首先会对模型进行检查，确保其可以正常\
运行在`BPU`上，之后对模型进行编译，生成可在`BPU`上运行的模型，在编译完成后，还会对\
编译出的定点模型进行测试，预估其在 `BPU` 上的运行性能，命令如下：

.. code-block:: bash

    python3 tools/compile_perf.py -c configs/segmentation/unet.py --opt 2


3. 预训练模型
-------------

HAT已经提供了此例子的预训练模型，所有模型都在发布包中。
