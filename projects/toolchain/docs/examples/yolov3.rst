YOLOV3检测模型训练
===================

这篇教程以YOLOv3-MobileNetv1为例，告诉大家如何在`Pascal-VOC`上完成浮点\
和定点模型的训练。

`PascalVOC`是图像检测，图像分割等比赛中最常用的基础数据集。这里我们\
使用2007和2012的两届的训练数据集作为实验的训练集，使用2007的验证集\
作为实验的验证集。\
原始数据集可以从这个地址下载：https://pjreddie.com/projects/pascal-voc-dataset-mirror/。获取的数据集格式如下：

    .. code-block:: bash

        VOCtrainval_06-Nov-2007.tar
        VOCtrainval_11-May-2012.tar
        VOCtest_06-Nov-2007.tar


YOLO系列的目标检测算法已经成为目标检测历史中重要的组成部分。YOLOv3\
是YOLO系列目标检测算法的第三版，与之前的算法相比，尤其是针对小目标，\
精度有显著的提升。

这里我们使用YOLOv3-MobileNetv1的例子来详细介绍在PascalVOC上训练目标检测的整个流程。


1. 训练流程
-----------

如果你只是想简单的把YOLOv3的内容训练起来，那么可以首先阅读一下这一章的内容。和其他任务一样，对于所有的训练，评测任务，`HAT`统一采用\
`tools + config`的形式来完成。在准备好原始数据集之后，可以通过下面的流程，\
方便的完成整个训练的流程。

在`datasets/voc2lmdb`的脚本中，利用`torchvision`提供的默认公开数据集处理
方\
法直接将原始的公开数据集转成`Numpy`或者`Tensor`的格式，最后将得到的\
数据统一用`msgpack`相关的方法压缩到`LMDB`的文件中。

这个过程可以很方便通过下面的脚本完成数据集打包:

    .. code-block:: bash

       python3 tools/datasets/voc_det/voc2lmdb.py --src-data-dir ${data-dir} --split-name trainval --num-workers 10

       python3 tools/datasets/voc_det/voc2lmdb.py --src-data-dir ${data-dir} --split-name test --num-workers 10

在完成数据集打包之后，可以得到含有`PascalVOC`的压缩数据集。

通常在定义好一个模型之后，尤其是一些公版模型，我们会有检查计算量的需求。 `HAT` \
通过 `calops` 的工具来计算模型的计算量，执行方法如下：

    .. code-block:: bash

       python3 tools/calops.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py

这种计算量的统计工具，是可以同时支持浮点和定点模型的。

下一步就可以开始训练。训练也可以通过下面的脚本来完成，在训练之前需\
要确认配置中数据集路径是否已经切换到已经打包好的数据集路径，同时一般\
训练检测需要提供在ImageNet上训练好的backbone模型，检查对应的路径是否\
设置正确即可。

训练可以直接通过如下的脚本来完成：

    .. code-block:: bash

       python3 tools/train.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py --step float

       python3 tools/train.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py --step qat

       python3 tools/train.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py --step int_infer

这两个脚本分别完成浮点模型和定点模型的训练，其中定点模型的训练依赖于\
上一步浮点训练产出的浮点模型。

在完成训练之后，可以得到训练完成的浮点模型或定点模型。和训练方法类似，\
我们可以用相同方法来对训好的模型做指标验证。同样是复用train脚本中val-only的功能来完成。

    .. code-block:: bash

       python3 tools/train.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py --step float --val-only

       python3 tools/train.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py --step qat --val-only

       python3 tools/train.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py --step int_infer --val-only


另外，我们还提供和上板完全一致的精度验证方法，通过下面的方式完成:

    .. code-block:: bash

       python3 tools/align_bpu_validation.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py

与其他的`HAT`提供的量化模型类似，YOLOV3也可以通过编译部署在地平线的芯片上。同样这个流程也可以用compile脚本来完成。

    .. code-block:: bash

       python3 tools/compile_perf.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py

最后，我们可以利用训练好模型做预测和可视化，这个流程可以通过infer脚本来完成。

    .. code-block:: bash

       python3 tools/infer.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py --step float --visual-type PascalVoc --img ${img-path}

其他step的模型infer情况可以依次类推。


以上就是从数据准备到生成量化可部署模型的YOLOv3的全过程。


2. 训练细节
-----------

网络结构
>>>>>>>>>

在`HAT`的`YOLOv3`的模型搭建中，我们可以直接使用下面的dict来构建浮\
点的检测模型，用户可以方便的修改其中的参数或者key类型，来达到配置网络的目的。

YOLOv3的网络模型一共被分为5个部分，包括backbone，neck，head，\
postprocess，losses。其中postprocess主要作用于验证阶段的后处理，\
而losses主要是作为训练阶段的损失函数。

    .. code-block:: python

        data_shape = (3, 416, 416)
        num_classes = 20
        anchors = [
            [(116, 90), (156, 198), (373, 326)],
            [(30, 61), (62, 45), (59, 119)],
            [(10, 13), (16, 30), (33, 23)],
        ]

        model = dict(
            type="YOLOV3",
            backbone=dict(
                type="MobileNetV1",
                alpha=1.0,
                include_top=False,
            ),
            neck=dict(
                type="YOLOv3Neck",
                backbone_idx=[-1, -2, -3],
                in_channels_list=[1024, 512, 256],
                out_channels_list=[512, 256, 128],
            ),
            head=dict(
                type="YOLOv3Head",
                neck_idx=[-3, -2, -1],
                in_channels_list=[1024, 512, 256],
                num_classes=num_classes,
                anchors=anchors,
            ),
            loss=dict(
                type="YOLOV3Loss",
                num_classes=num_classes,
                anchors=anchors,
                img_size=data_shape,
                ignore_thresh=0.7,
                loss_xy=dict(type=torch.nn.BCELoss, size_average=False),
                loss_wh=dict(type=torch.nn.MSELoss, size_average=False),
                loss_conf=dict(type=torch.nn.BCELoss, size_average=False),
                loss_cls=dict(type=torch.nn.BCELoss, size_average=False),
                lambda_loss=[2.5, 2.5, 1.0, 1.0],
            ),
            postprocess=dict(
                type="YOLOV3PostProcess",
                anchors=anchors,
                num_classes=num_classes,
                img_size=data_shape,
                score_thresh=0.01,
                nms_thresh=0.45,
                topK=400,
            ),
        )

网络的主体部分由`MobileNetv1`的backbone，`YOLOv3-FPN`的neck，以及多个\
分辨率的head组成。训练阶段的损失函数包含训练Target的生成和4个Loss函数组成，\
而验证阶段的后处理是常见的按照类别做NMS的操作。

数据增强
>>>>>>>>>

和其他网络的训练过程类似，跑在`BPU`上的模型使用的是`YUV444`的图像输入，而一般的训练图像输\
入都采用`RGB`的形式，所以`HAT`提供`BgrToYuv444`的数据增强来将`RGB`转到\
`YUV444`的格式。归一化对于深度学习模型训练是必须的。

除此之外，常见的检测数据增强这里也会使用，如使用`DetColorAug`来做颜色\
上的数据增强。同时`DetExpand`会对图片和Labels做随机的扩充，而`MinIoURadomCrop`\
可以对图片和Labels做随机的crop。`FlipLeftRight`可以做随机的翻转再\
配合`Resize`统一调整到训练的尺寸上。

    .. code-block:: python

        dataset=dict(
            type="PascalVOCFromLMDB",
            data_path="./data/voc/trainval_lmdb",
            num_samples=16551,
            transforms=[
                dict(type="DetColorAug"),
                dict(type="DetExpand", ratio_range=(1, 4)),
                dict(
                    type="MinIoURandomCrop",
                    min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
                ),
                dict(type="DetFlipLeftRight"),
                dict(type="DetResize", size=data_shape[1:]),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type="Normalize", mean=128.0, std=128.0),
            ],
        )

验证集上的数据增强则简单很多，主要是`DetReize`和颜色空间的转换。

    .. code-block:: python

        dataset=dict(
            type="PascalVOCFromLMDB",
            data_path="./data/voc/test_lmdb",
            num_samples=4952,
            transforms=[
                dict(type="DetResize", size=data_shape[1:]),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type="Normalize", mean=128.0, std=128.0),
            ]
        )

数据增强是现在深度学习模型训练中必不可少的方法，以上就是训练VOC检测\
的过程中，YOLOv3采用的数据增强方法。

训练策略
>>>>>>>>>

检测模型训练过程中，使用预训练的backbone模型已经成为一种通用的方法。\
这里我们使用`Checkpoint`的callback来做预训练模型的加载，注意这里的\
allow_miss和ignore_extra一定设置为False，False可以确保模型能够差异化的加载。

这里使用`Step`的学习策略，适当延长epoch的训练长度对模型有会有微弱的提升。

    .. code-block:: python

        float_trainer = dict(
            type='distributed_data_parallel_trainer',
            model=model,
            data_loader=data_loader,
            optimizer=dict(
                type=torch.optim.SGD,
                params={"weight": dict(weight_decay=5e-4)},
                lr=0.001,
                momentum=0.9,
            ),
            batch_processor=batch_processor,
            num_epochs=200,
            device=None,
            callbacks=[
                stat_callback,
                dict(
                    type="StepDecayLrUpdater",
                    warmup_by='epoch',
                    warmup_len=4,
                    step_log_interval=100,
                    lr_decay_id=[160, 180],
                    lr_decay_factor=0.1,
                ),
                val_callback,
                ckpt_callback,
            ],
            sync_bn=True,
            train_metrics=None,
            val_metrics=[
                dict(type="VOC07MApMetric", num_classes=num_classes),
            ],
        )

        float_solver = dict(
            trainer=float_trainer,
            quantize=False,
            resume_checkpoint=(
                "tmp_models/mobilenetv1_cls/float-checkpoint-best.pth.tar"
            ),
            resume_optimizer=False,
            resume_epoch_or_step=False,
            allow_miss=True,
            ignore_extra=True,
        )


量化训练
>>>>>>>>

关于量化训练中的关键步骤，比如准备浮点模型、算子替换、插入量化和反量化节点、设置量化参数以及算子的融合等，请阅读 `《Horizon Plugin PyTorch》` 手册中的
`浮点模型准备` 和 `算子融合` 两节中的内容。这里主要讲一下`HAT`\
的检测中如何定义和使用量化模型。

在模型准备的好情况下，包括量化已有的一些模块完成之后，`HAT`在训练脚本\
中统一使用下面的脚本将浮点模型映射到定点模型上来。

    .. code-block:: python

        model.fuse_model()
        model.set_qconfig()
        horizon.quantization.prepare_qat(model, inplace=True)

`YOLOv3`的量化训练策略除了学习率和epoch训练长度之外，其他的和浮点训练\
基本上完全对齐。`YOLOv3`的学习率可以很小，同时训练长度也可以不用很长。


模型检查和编译
>>>>>>>>>>>>>>

对于`HAT`来说，量化模型的意义在于可以在`BPU`上直接运行。因此，\
对于量化模型的检查和编译是必须的。我们在训练脚本中提供可以对模型\
检查的接口，可以让用户定义好量化模型之后，先检查能否在`BPU`上正常\
运行,目前该检查在启动train.py脚本训练quantize=True的模型时会默认打开。

在训练完成之后，`compile`的工具用来将量化模型编译成可以上板运行的\
`hbm`文件，同时该工具也能预估在BPU上的运行性能，可以采用以下脚本：

    .. code-block:: bash

       python3 tools/compile_perf.py --config configs/detection/yolov3/pascalvoc_mobilenetv1.py
