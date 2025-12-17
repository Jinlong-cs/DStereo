Bev多任务模型训练
=====================


BEV参考算法基于Horizon Torch Samples（地平线自研深度学习框架）开发，关于Horizon Torch Samples的使用介绍可以参考Horizon Torch Samples使用文档。BEV参考算法的训练config位于HAT/configs/bev/路径下。
下文以HAT/configs/bev/bev_mt_ipm.py为例介绍如何配置并训练BEV参考算法。


数据集准备
-------------
这里以nuscense数据集为例, 可以从https://www.nuscenes.org/nuscenes下载数据集。
同时，为了提升训练的速度，我们对原始的jpg格式的数据集做了一个打包，将其转换成lmdb格式的数据集。只需要运行下面的脚本，就可以成功\
实现转换：


.. code-block:: python
  
    python36 tools/datasets/nuscenes_packer.py --src-data-dir WORKSAPCE/datasets/nuscenes/ --pack-type lmdb --target-data-dir . --version v1.0-trainval --split-name val
    python36 tools/datasets/nuscenes_packer.py --src-data-dir WORKSAPCE/datasets/nuscenes/ --pack-type lmdb --target-data-dir . --version v1.0-trainval --split-name train


上面这两条命令分别对应着转换训练数据集和验证数据集，打包完成之后，data目录下的文件结构应该如下所示：

  .. code-block:: bash

    tmp_data
    |-- nuscenes
      |-- metas
      |-- v1.0-trainval 
        |-- train_lmdb
        |-- val_lmdb

train_lmdb和val_lmdb就是打包之后的训练数据集和验证数据集，也是网络最终读取的数据集。metas中为分割模型需要的地图信息。


浮点模型训练
-------------
数据集准备好之后，就可以开始训练浮点型的bev多任务网络了。

如果你只是单纯的想启动这样的训练任务，只需要运行下面的命令就可以：

  .. code-block:: python

    python3 tools/train.py --stage float --config configs/bev/bev_mt_imp.py

由于Horizon Torch Samples算法包使用了一种巧妙的注册机制，使得每一个训练任务都可以按照这种train.py加上config配置文件的形式启动。train.py是\
统一的训练脚本，与任务无关，我们需要训练什么样的任务、使用什么样的数据集以及训练相关的超参数设置都在指定的config配置文件里面。config文件
里面提供了模型构建、数据读取等关键的dict。


模型构建
>>>>>>>>>

.. code-block:: python

    model = dict(
        type="BevStructure",
        bev_feat_index=-1,
        backbone=dict(
            type="efficientnet",
            bn_kwargs=bn_kwargs,
            model_type="b0",
            num_classes=1000,
            include_top=False,
            activation="relu",
            use_se_block=False,
        ),
        neck=dict(
            type="FastSCNNNeck",
            in_channels=[40, 320],
            feat_channels=[64, 64],
            indexes=[-3, -1],
            bn_kwargs=bn_kwargs,
        ),
        view_transformer=dict(
            type="WrappingTransformer",
            bev_upscale=2,
            bev_size=bev_size,
            num_views=6,
            drop_prob=0.1,
            grid_quant_scale=grid_quant_scale,
        ),
        bev_transforms=[
            dict(
                type="BevRotate",
                bev_size=bev_size,
                rot=(-0.3925, 0.3925),
            ),
            dict(type="BevFlip", prob_x=0.5, prob_y=0.5, bev_size=bev_size),
        ],
        bev_encoder=dict(
            type="BevEncoder",
            backbone=dict(
                type="efficientnet",
                bn_kwargs=bn_kwargs,
                model_type="b0",
                num_classes=1000,
                include_top=False,
                activation="relu",
                use_se_block=False,
                in_channels=64,
            ),
            neck=dict(
                type="BiFPN",
                in_strides=[2, 4, 8, 16, 32],
                out_strides=[2, 4, 8, 16, 32],
                stride2channels=dict({2: 16, 4: 24, 8: 40, 16: 112, 32: 320}),
                out_channels=48,
                num_outs=5,
                stack=3,
                start_level=0,
                end_level=-1,
                fpn_name="bifpn_sum",
            ),
        ),
        bev_decoders=[
            dict(
                type="BevSegDecoder",
                name="bev_seg",
                use_bce=use_bce,
                task_weight=10.0,
                bev_size=bev_size,
                task_size=task_map_size,
                head=dict(
                    type="DepthwiseSeparableFCNHead",
                    input_index=0,
                    in_channels=48,
                    feat_channels=48,
                    num_classes=seg_classes,
                    dropout_ratio=0.1,
                    num_convs=2,
                    bn_kwargs=bn_kwargs,
                ),
                target=dict(
                    type="FCNTarget",
                ),
                loss=dict(
                    type="CrossEntropyLossV2",
                    loss_name="decode",
                    reduction="mean",
                    ignore_index=-1,
                    use_sigmoid=use_bce,
                    class_weight=2.0 if use_bce else [1.0, 5.0, 5.0, 5.0],
                ),
                decoder=dict(
                    type="FCNDecoder",
                    upsample_output_scale=1,
                    use_bce=use_bce,
                    bg_cls=-1,
                ),
            ),
            dict(
                type="BevDetDecoder",
                name="bev_det",
                task_weight=1.0,
                head=dict(
                     type="CenterPoint3dHead",
                     in_channels=48,
                     tasks=tasks,
                     share_conv_channels=48,
                     share_conv_num=1,
                     common_heads=dict(
                          reg=(2, 2),
                          height=(1, 2),
                          dim=(3, 2),
                          rot=(2, 2),
                          vel=(2, 2),
                     ),
                     head_conv_channels=48,
                     num_heatmap_convs=2,
                     final_kernel=3,
                 ),
                target=dict(
                     type="CenterPoint3dTarget",
                     class_names=NuscenesDataset.CLASSES,
                     tasks=tasks,
                     gaussian_overlap=0.1,
                     min_radius=2,
                     out_size_factor=1,
                     norm_bbox=True,
                     max_num=500,
                     bbox_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                ),
                loss_cls=dict(type="GaussianFocalLoss", loss_weight=1.0),
                loss_reg=dict(
                      type="L1Loss",
                      loss_weight=0.25,
                  ),
                decoder=dict(
                     type="CenterPoint3dDecoder",
                     class_names=NuscenesDataset.CLASSES,
                     tasks=tasks,
                     bev_size=bev_size,
                     out_size_factor=1,
                     use_max_pool=True,
                     max_pool_kernel=3,
                     score_threshold=0.1,
                     nms_type=[
                         "rotate",
                         "rotate",
                         "rotate",
                         "circle",
                         "rotate",
                         "rotate",
                     ],
                     min_radius=[4, 12, 10, 1, 0.85, 0.175],
                     nms_threshold=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
                    decode_to_ego=True,
                 ),
             ),
         ]
     )
       
           
其中，`model`下面的`type`表示定义的模型名称，剩余的变量表示模型的其他组成部分。这样定义模型的好处在于我们可以很方便的替换我们想要的结构。例如，如果我们想训练一个backbone为resnet50的模型，只需要将`model`下面的`backbone`替换掉就可以。


数据增强
>>>>>>>>>
跟`model`的定义一样，数据增强的流程是通过在config配置文件中定义`data_loader`和`val_data_loader`这两个dict来实现的，分别对应着\
训练集和验证集的处理流程。以`data_loader`为例，

  .. code-block:: python

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="NuscenesDataset",
            data_path=os.path.join(data_rootdir, "train_lmdb"),
            transforms=[
                dict(type="BevImgResize", scales=(0.6, 0.8)),
                dict(type="BevImgCrop", size=(512, 960), random=True),
                dict(type="BevImgFlip", prob=0.5),
                dict(type="BevImgRotate", rot=(-5.4, 5.4)),
                dict(
                    type="BevImgTransformWrapper",
                    transforms=[
                        dict(type="PILToTensor"),
                        dict(type="BgrToYuv444", rgb_input=True),
                        dict(type="Normalize", mean=128.0, std=128.0),
                    ],
                ),
             ],
             bev_size=bev_size,
             map_size=map_size,
             map_path=meta_rootdir,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=True,
        collate_fn=collate_nuscenes,
    )

    val_data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="NuscenesDataset",
            data_path=os.path.join(data_rootdir, "val_lmdb"),
            transforms=[
                dict(type="BevImgResize", size=(540, 960)),
                dict(type="BevImgCrop", size=(512, 960)),
                dict(
                     type="BevImgTransformWrapper",
                     transforms=[
                         dict(type="PILToTensor"),
                         dict(type="BgrToYuv444", rgb_input=True),
                         dict(type="Normalize", mean=128.0, std=128.0),
                     ],
                ),
            ],
            bev_size=bev_size,
            map_size=map_size,
            map_path=meta_rootdir,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=True,
        collate_fn=collate_nuscenes,
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
        type="distributed_data_parallel_trainer",
        model=model,
        data_loader=data_loader,
        optimizer=dict(
            type=torch.optim.AdamW,
            params={"weight": dict(weight_decay=weight_decay)},
            lr=start_lr,
        ),
        batch_processor=batch_processor,
        device=None,
        num_epochs=train_epochs,
        callbacks=[
            stat_callback,
            loss_show_update,
            dict(
                type="CosLrUpdater",
                step_log_interval=500,
                warmup_by="epoch",
                warmup_len=0,
            ),
            val_callback,
            ckpt_callback,
        ],
        sync_bn=True,
        train_metrics=dict(
            type="LossShow",
        ),
    )

    float_solver = dict(
        trainer=float_trainer,
        pretrain_checkpoint=(
            "./tmp_pretrained_models/efficientnet_cls/float-checkpoint-best.pth.tar"
        ),
        quantize=False,
        allow_not_init=True,
        allow_miss=True,
        ignore_extra=True,
        strict_match=True,
    )
 

`float_trainer`从大局上定义了我们的训练方式，包括使用多卡分布式训练（distributed_data_parallel_trainer），模型训练的epoch次数，以及优化器的选择。\
同时`callbacks`中体现了模型在训练过程中使用到的小策略以及用户想实现的操作，包括学习率的变换方式(WarmupStepLrUpdater)，在训练过程中验证模型的\
指标(Validation)，以及保存(Checkpoint)模型的操作。当然，如果你有自己希望模型在训练过程中实现的操作，也可以按照这种dict的方式添加。
`float_solver`负责将整个训练的逻辑给串联起来，其中也会负责模型的pretrain。

  .. note::

      如果需要复现精度，config中的训练策略最好不要修改。否则可能会有意外的训练情况出现。

通过上面的介绍，你应该对config文件的功能有了一个比较清楚的认识。然后通过前面提到的训练脚本，就可以训练一个高精度的纯浮点的检测模型。当然训练一个\
好的检测模型不是我们最终的目的，它只是做为一个pretrain为我们后面训练定点模型服务的。


量化模型训练
------------
当我们有了纯浮点模型之后，就可以开始训练相应的定点模型了。和浮点训练的方式一样，我们只需要通过运行下面的脚本就可以训练定点模型了：

  .. code-block:: python

    python3 tools/train.py --stage qat --config configs/bev/bev_mt_ipm.py

可以看到，我们的配置文件没有改变，只改变了`stage`的类型。此时我们使用的训练策略来自于config文件中的qat_trainer

  .. code-block:: python
  
    qat_trainer = dict(
        type="distributed_data_parallel_trainer",
        model=model,
        data_loader=data_loader,
        optimizer=dict(
            type=torch.optim.AdamW,
            params={"weight": dict(weight_decay=weight_decay)},
            lr=qat_lr,
        ),
        batch_processor=batch_processor,
        device=None,
        num_epochs=qat_train_epochs,
        callbacks=[
            stat_callback,
            loss_show_update,
            dict(
                type="StepDecayLrUpdater",
                lr_decay_id=[6],
                lr_decay_factor=0.1,
            ),
            val_callback,
            ckpt_callback,
        ],
        sync_bn=True,
        train_metrics=dict(
            type="LossShow",
        ),
    )

    qat_solver = dict(
        trainer=qat_trainer,
        quantize=True,
        check_quantize_model=True,
        pre_step="float",
        pre_step_checkpoint=os.path.join(
            ckpt_dir, "float-checkpoint-best.pth.tar"
        ),
        strict_match=True,
        ignore_extra=True,
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

    python3 tools/predict.py -c configs/bev/bev_mt_ipm.py --stage float
    python3 tools/predict.py -c configs/bev/bev_mt_ipm.py --stage qat

同时，我们还提供了quantization模型的性能测试，只需要运行以下命令：

  .. code-block:: python

    python3 tools/predict.py -c configs/bev/bev_mt_ipm.py --stage int_infer

这个显示出来的精度才是最终的int8模型的真正精度，当然这个精度和qat验证阶段的精度应该是保持十分接近的。（模型精度可能会因为环境依赖的不同会有些浮动）


除了上述模型验证之外，我们还提供和上板完全一致的精度验证方法，可以
通过下面的方式完成：

  .. code-block:: python

       python3 tools/bev_align_bpu_validation.py --config configs/bev/bev_mt_ipm.py --dataset nuscenes


结果可视化
>>>>>>>>>>>>>>
如果你希望可以看到训练出来的模型对于单帧的检测效果，我们的tools文件夹下面同样提供了单帧预测及可视化的脚本，\
你只需要按照infer_bev.py中的格式给出每张图片的大小以及单应矩阵， 然后运行以下脚本即可：

  .. code-block:: python

    python3 tools/infer_bev.py --config configs/bev/bev_mt_ipm.py --dataset nuscenes --inputs ${img_list} --homo ${homography.npy}  --input-format yuv --is-plot



模型检查和编译
>>>>>>>>>>>>>>

在训练完成之后，可以使用`compile`的工具用来将量化模型编译成可以上板运行的\
`hbm`文件，同时该工具也能预估在BPU上的运行性能，可以采用以下脚本：

    .. code-block:: bash

       python3 tools/compile_perf.py --config configs/bev/bev_mt_ipm.py --out-dir ./ --opt 2

