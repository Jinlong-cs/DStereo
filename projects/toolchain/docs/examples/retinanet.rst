RetinaNet检测模型训练
=====================


这篇教程以retinanet-vargnetv2为例，告诉大家如何使用HAT算法包训练一个定点的检测模型。在开始量化感知训练，也就是定点模型训练\
之前，首先需要训练一个精度较高的纯浮点模型，然后基于这个纯浮点模型做finetune，就可以快速的训练出定点模型。所以我们从训练一个纯\
浮点的retinanet-vargnetv2模型开始讲起。


数据集准备
-------------
在开始训练模型之前，第一步是需要准备好数据集，这里我们下载\
MSCOCO的 `train2017.zip <http://images.cocodataset.org/zips/train2017.zip>`_ 和\
`val2017.zip <http://images.cocodataset.org/zips/val2017.zip>`_ 做为网络的训练集和验证集,同时需要下载相应的标签数据\
`annotations_trainval2017.zip <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>`_ , 解压缩之后\
数据目录结构如下所示：

  .. code-block:: bash

    tmp_data
    |-- mscoco
      |-- annotations_trainval2017.zip
      |-- train2017.zip
      |-- val2017.zip
      |-- annotations
      |-- train2017
      |-- val2017

同时，为了提升训练的速度，我们对原始的jpg格式的数据集做了一个打包，将其转换成lmdb格式的数据集。只需要运行下面的脚本，就可以成功\
实现转换：

  .. code-block:: python

    python3 tools/datasets/coco_det/mscoco2lmdb.py --src-data-dir ./tmp_data/mscoco/ --target-data-dir ./tmp_data/mscoco --split-name train
    python3 tools/datasets/coco_det/mscoco2lmdb.py --src-data-dir ./tmp_data/mscoco/ --target-data-dir ./tmp_data/mscoco --split-name val

上面这两条命令分别对应着转换训练数据集和验证数据集，打包完成之后，data目录下的文件结构应该如下所示：

  .. code-block:: bash

    tmp_data
    |-- mscoco
      |-- annotations
      |-- train2017
      |-- train_lmdb
      |-- val2017
      |-- val_lmdb

train_lmdb和val_lmdb就是打包之后的训练数据集和验证数据集，也是网络最终读取的数据集。


浮点模型训练
-------------
数据集准备好之后，就可以开始训练浮点型的retinanet-vargnetv2检测网络了。在网络训练开始之前，你可以使用以下命令先测试一下网络的计算量\
和参数数量：

  .. code-block:: python

    python3 tools/calops.py --config configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py --input-shape "1,3,1024,1024"

如果你只是单纯的想启动这样的训练任务，只需要运行下面的命令就可以：

  .. code-block:: python

    python3 tools/train.py --step float --config configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py

由于HAT算法包使用了一种巧妙的注册机制，使得每一个训练任务都可以按照这种train.py加上config配置文件的形式启动。train.py是\
统一的训练脚本，与任务无关，我们需要训练什么样的任务、使用什么样的数据集以及训练相关的超参数设置都在指定的config配置文件里面。config文件
里面提供了模型构建、数据读取等关键的dict。


模型构建
>>>>>>>>>
retinanet的网络结构可以参考
`论文 <https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf>`_ ，这里\
不做详细介绍。我们通过在config配置文件中定义`model`这样的一个dict型变量，就可以方便的实现对模型的定义和修改。

  .. code-block:: python

    model = dict(
        type="RetinaNet",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            bn_kwargs={},
            include_top=False,
        ),
        neck=dict(
            type="RetinaNetFPN",
            strides=[8, 16, 32],
            in_channels=[32, 32, 64, 128, 256],
            out_channels=256,
            num_outs=5,
        ),
        head=dict(
            type="RetinaNetHead",
            num_classes=num_classes,
            num_anchors=len(scales) * len(ratios),
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
        ),
        anchors=dict(
            type="RetinaNetAnchor",
            scales=scales,
            ratios=ratios,
            levels=[3, 4, 5, 6, 7],
        ),
        targets=dict(
            type="RetinaNetTarget",
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
        ),
        post_process=dict(
            type="RetinaNetPostProcess",
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=300,
            topk_candidates=1000,
        ),
        loss_cls=dict(
            type="FocalLoss",
            loss_name="FocalLoss",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            eps=1e-12,
            reduction="mean",
        ),
        loss_reg=dict(
            type="SmoothL1Loss",
            loss_name="SmoothL1Loss",
            beta=1.0 / 9.0,
            reduction='mean',
            loss_weight=1.0
        ),
    )


其中，`model`下面的`type`表示定义的模型名称，剩余的变量表示模型的其他组成部分。这样定义模型的好处在于我们可以很方便的替换我们想要的结构。例如，如果我们想训练一个backbone为resnet50的模型，只需要将`model`下面的`backbone`替换掉就可以。
训练脚本在启动之后，会调用`build_model`接口，将这样一个dict类型的model变成类型为`torch.nn.Module`类型的model。

  .. code-block:: python

    def build_model(cfg, default_args=None):
        if cfg is None:
            return None
        assert "type" in cfg, "type is need in model"
        return build_from_cfg(cfg, MODELS)


数据增强
>>>>>>>>>
跟`model`的定义一样，数据增强的流程是通过在config配置文件中定义`data_loader`和`val_data_loader`这两个dict来实现的，分别对应着\
训练集和验证集的处理流程。以`data_loader`为例，

  .. code-block:: python

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="CocoFromLMDB",
            data_path="./tmp_data/mscoco/train_lmdb/",
            transforms=[
                dict(
                    type="RandomFlip",
                    px=0.5,
                    py=0,
                ),
                dict(
                    type="Resize",
                    img_scale=(800, 1024),
                    keep_ratio=True,
                ),
                dict(
                    type="Pad",
                    size=(1024, 1024),
                ),
                dict(
                    type="ToTensor",
                    to_yuv=True,
                ),
                dict(
                    type="Normalize",
                    mean=128.0,
                    std=128.0,
                ),
            ],
            num_samples=118287,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dict(type="Collate2D"),
    )

其中type直接用的pytorch自带的接口torch.utils.data.DataLoader，表示的是将`batch_size`大小的图片组合到一起。
这里面唯一需要关注的可能是`dataset`这个变量，`CocoFromLMDB`表示\
从lmdb数据集中读取图片，路径也就是我们在第一部分数据集准备中提到的路径。`transforms`下面包含着一系列的数据增强。`val_data_loader`中除了图片\
翻转(RandomFlip），其他的数据变换和`data_loader`一致。你也可以通过在`transforms`中插入新的dict实现自己希望的数据增强操作。


训练策略
>>>>>>>>
为了训练一个精度高的模型，好的训练策略是必不可少的。对于每一个训练任务而言，相应的训练策略同样都定义在其中的config文件中，从`float_trainer`这个变量\
就可以看出来。

  .. code-block:: python

    float_trainer = dict(
        type='distributed_data_parallel_trainer',
        model=model,
        data_loader=data_loader,
        optimizer=dict(
            type=torch.optim.SGD,
            params={"weight": dict(weight_decay=5e-5)},
            lr=0.01,
            momentum=0.9,
        ),
        batch_processor=batch_processor,
        num_epochs=12,
        device=None,
        callbacks=[
            stat_callback,
            dict(
                type="StepDecayLrUpdater",
                warmup_len=0.3,
                lr_decay_id=[8, 11],
                step_log_interval=10,
            ),
            val_callback,
            ckpt_callback,
        ],
        sync_bn=True,
        val_metrics=dict(
            type="COCODetectionMetric",
            ann_file="./tmp_data/mscoco/instances_val2017.json",
        )
    )

    float_solver = dict(
        trainer=float_trainer,
        resume_checkpoint="./models/vargnetv2_cls/float-checkpoint-best.pth.tar",
        allow_miss=True,
        ignore_extra=True,
        resume_optimizer=None,
        resume_epoch_or_step=False,
        quantize=False,
    )

`float_trainer`从大局上定义了我们的训练方式，包括使用多卡分布式训练（distributed_data_parallel_trainer），模型训练的epoch次数，以及优化器的选择。\
同时`callbacks`中体现了模型在训练过程中使用到的小策略以及用户想实现的操作，包括学习率的变换方式(WarmupStepLrUpdater)，在训练过程中验证模型的\
指标(Validation)，以及保存(Checkpoint)模型的操作。当然，如果你有自己希望模型在训练过程中实现的操作，也可以按照这种dict的方式添加。
`float_solver`负责将整个训练的逻辑给串联起来，其中也会负责模型的pretrain。

通过上面的介绍，你应该对config文件的功能有了一个比较清楚的认识。然后通过前面提到的训练脚本，就可以训练一个高精度的纯浮点的检测模型。当然训练一个\
好的检测模型不是我们最终的目的，它只是做为一个pretrain为我们后面训练定点模型服务的。


量化模型训练
------------
当我们有了纯浮点模型之后，就可以开始训练相应的定点模型了。和浮点训练的方式一样，我们只需要通过运行下面的脚本就可以训练定点模型了：

  .. code-block:: python

    python3 tools/train.py --step qat --config configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py

可以看到，我们的配置文件没有改变，只改变了`step`的类型。此时我们使用的训练策略来自于config文件中的qat_trainer

  .. code-block:: python

    qat_trainer = dict(
        type='distributed_data_parallel_trainer',
        model=model,
        data_loader=data_loader,
        optimizer=dict(
            type=torch.optim.SGD,
            params={"weight": dict(weight_decay=5e-5)},
            lr=0.0005,
            momentum=0.9,
        ),
        batch_processor=batch_processor,
        num_epochs=5,
        device=None,
        callbacks=[
            stat_callback,
            dict(
                type="StepDecayLrUpdater",
                warmup_len=0.03,
                lr_decay_id=[4],
                step_log_interval=10,
            ),
            val_callback,
            ckpt_callback,
        ],
        val_metrics=dict(
            type="COCODetectionMetric",
            ann_file="./tmp_data/mscoco/instances_val2017.json",
        )
    )
    qat_solver = dict(
        trainer=qat_trainer,
        quantize=True,
        check_quantize_model=True,
        pre_step='float',
        pre_step_checkpoint=os.path.join(
            ckpt_dir, 'float-checkpoint-best.pth.tar'
        ),
        strict_match=True,
    )


1、quantize参数的值不同
>>>>>>>>>>>>>>>>>>>>>>>>>

当我们训练量化模型的时候，需要设置quantize=True，此时相应的浮点模型会被转换成量化模型，相关代码如下：

  .. code-block:: python

    model.fuse_model()
    model.set_qconfig()
    horizon.quantization.prepare_qat(model, inplace=True)

关于量化训练中的关键步骤，比如准备浮点模型、算子替换、插入量化和反量化节点、设置量化参数以及算子的融合等，请阅读 `《Horizon Plugin PyTorch》` 手册中的
`浮点模型准备` 和 `算子融合` 两节中的内容。

2、训练策略不同
>>>>>>>>>>>>>>>

正如我们之前所说，量化训练其实是在纯浮点训练基础上的finetue。因此量化训练的时候，我们的初始学习率设置为浮点训练的十分之一，训练的epoch次数也\
大大减少，最重要的是`model`定义的时候，我们的`pretrained`需要设置成已经训练出来的纯浮点模型的地址。

做完这些简单的调整之后，就可以开始训练我们的量化模型了。


模型验证
>>>>>>>>>>>>>
模型训练完成之后，我们还可以验证训练出来的模型性能。由于我们提供了float和qat两阶段的训练过程，相应的我们可以验证这两个阶段训练出来的模型性能，\
只需要相应的运行以下两条命令即可：

  .. code-block:: python

    python3 tools/train.py --step float --config configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py --val-only
    python3 tools/train.py --step qat --config configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py --val-only

同时，我们还提供了quantization模型的性能测试，只需要运行以下命令：

  .. code-block:: python

    python3 tools/train.py --step int_infer configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py --val-only

这个显示出来的精度才是最终的int8模型的真正精度，当然这个精度和qat验证阶段的精度应该是保持十分接近的。

仿真上板精度验证
>>>>>>>>>>>>>>

除了上述模型验证之外，我们还提供和上板完全一致的精度验证方法，可以
通过下面的方式完成：

    .. code-block:: bash

       python3 tools/align_bpu_validation.py --config configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py

结果可视化
>>>>>>>>>>>>>>
如果你希望可以看到训练出来的模型对于单张图片的检测效果，我们的tools文件夹下面同样提供了单张图片预测及可视化的脚本，你只需要运行以下脚本即可：

  .. code-block:: python

    python3 tools/infer.py --config configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py --step int_infer --visual-type Mscoco
当然，你也可以把命令中的int_infer换成float和qat，分别测试float阶段和qat阶段训练出来的模型对于单张图片的预测效果。



模型检查和编译
>>>>>>>>>>>>>>

在训练完成之后，可以使用`compile`的工具用来将量化模型编译成可以上板运行的\
`hbm`文件，同时该工具也能预估在BPU上的运行性能，可以采用以下脚本：

    .. code-block:: bash

       python3 tools/compile_perf.py --config configs/detection/retinanet/retinanet_vargnetv2_fpn_mscoco.py --out-dir ./ --opt 2

