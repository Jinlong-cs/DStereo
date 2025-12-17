PointPillars检测模型训练
=====================


这篇教程主要是告诉大家如何利用 `HAT` 在雷达点云数据集 `KITTI-3DObject` 上从头开始训练一个\
`PointPillars` 模型，包括浮点、量化和定点模型。

数据集准备
-------------
在开始训练模型之前，第一步是需要准备好数据集，我们在KITTI官网下载 `3DObject据集 <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>` ，
包括4个文件: 
    * 1. `left color images of object data set`, 
    * 2.  `velodyne point clouds`, 
    * 3. `camera calibration matrices of object data set`, 
    * 4. `taining labels of object data set` ，

下载上述4个文件后，解压并按照如下方式组织文件夹结构：

  .. code-block:: bash

    ├── tmp_data
    │   ├── kitti3d
    │   │   ├── testing
    │   │   │   ├── calib
    │   │   │   ├── image_2
    │   │   │   ├── velodyne
    │   │   ├── training
    │   │   │   ├── calib
    │   │   │   ├── image_2
    │   │   │   ├── label_2
    │   │   │   ├── velodyne

为了创建 KITTI 点云数据，首先需要加载原始的点云数据并生成相关的包含目标标签和标注框的数据标注文件，\
同时还需要为 KITTI 数据集生成每个单独的训练目标的点云数据，并将其存储在 `data/kitti/gt_database` 的 `.bin`` 格式的文件中，\
此外，需要为训练数据或者验证数据生成 .pkl 格式的包含数据信息的文件。随后，通过运行下面的命令来创建 KITTI 数据:
 
  .. code-block:: bash

    mkdir ./tmp_data/kitti3d/ImageSets
    # 从社区下载数据集划分文件
    wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./tmp_data/kitti3d/ImageSets/test.txt
    wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./tmp_data/kitti3d/ImageSets/train.txt
    wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./tmp_data/kitti3d/ImageSets/val.txt
    wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./tmp_data/kitti3d/ImageSets/trainval.txt
    python3 tools/create_data.py --dataset "kitti3d" --root-dir "./tmp_data/kitti3d"

执行上述命令后，生成的文件目录如下：

  .. code-block:: bash

    ├── tmp_data
    │   ├──── kitti3d
    │   │   ├── ImageSets
    │   │   │   ├── test.txt
    │   │   │   ├── train.txt
    │   │   │   ├── trainval.txt
    │   │   │   ├── val.txt
    │   │   ├── testing
    │   │   │   ├── calib
    │   │   │   ├── image_2
    │   │   │   ├── velodyne
    │   │   │   ├── velodyne_reduced        # 新生成的 velodyne_reduced
    │   │   ├── training
    │   │   │   ├── calib
    │   │   │   ├── image_2
    │   │   │   ├── label_2
    │   │   │   ├── velodyne
    │   │   │   ├── velodyne_reduced        # 新生成的 velodyne_reduced
    │   │   ├── kitti3d_gt_database           # 新生成的 kitti_gt_database
    │   │   │   ├── xxxxx.bin
    │   │   ├── kitti3d_infos_train.pkl       # 新生成的 kitti_infos_train.pkl
    │   │   ├── kitti3d_infos_val.pkl         # 新生成的 kitti_infos_val.pkl
    │   │   ├── kitti3d_dbinfos_train.pkl     # 新生成的 kitti_dbinfos_train.pkl
    │   │   ├── kitti3d_infos_test.pkl        # 新生成的 kitti_infos_test.pkl
    │   │   ├── kitti3d_infos_trainval.pkl    # 新生成的 kitti_infos_trainval.pkl


同时，为了提升训练的速度，我们对数据信息文件做了一个打包，将其转换成lmdb格式的数据集。只需要运行下面的脚本，就可以成功\
实现转换：

  .. code-block:: python

    python3 tools/datasets/kitti3d_packer.py --src-data-dir ./tmp_data/kitti3d/ --target-data-dir ./tmp_data/kitti3d --split-name train --pack-type lmdb
    python3 tools/datasets/kitti3d_packer.py --src-data-dir ./tmp_data/kitti3d/ --target-data-dir ./tmp_data/kitti3d --split-name val --pack-type lmdb

上面这两条命令分别对应着转换训练数据集和验证数据集，打包完成之后，data目录下的文件结构应该如下所示：

  .. code-block:: bash

    ├── tmp_data
    │   ├──── kitti3d
    │   │   ├── pack_data       # 新生成的 lmdb
    │   │   │   ├── train
    │   │   │   ├── val
    │   │   ├── ImageSets
    │   │   │   ├── test.txt
    │   │   │   ├── train.txt
    │   │   │   ├── trainval.txt
    │   │   │   ├── val.txt
    │   │   ├── testing
    │   │   │   ├── calib
    │   │   │   ├── image_2
    │   │   │   ├── velodyne
    │   │   │   ├── velodyne_reduced
    │   │   ├── training
    │   │   │   ├── calib
    │   │   │   ├── image_2
    │   │   │   ├── label_2
    │   │   │   ├── velodyne
    │   │   │   ├── velodyne_reduced
    │   │   ├── kitti3d_gt_database
    │   │   │   ├── xxxxx.bin
    │   │   ├── kitti3d_infos_train.pkl
    │   │   ├── kitti3d_infos_val.pkl
    │   │   ├── kitti3d_dbinfos_train.pkl
    │   │   ├── kitti3d_infos_test.pkl
    │   │   ├── kitti3d_infos_trainval.pkl

`train_lmdb` 和 `val_lmdb` 就是打包之后的训练数据集和验证数据集，也是网络最终读取的数据集，
 `kitti3d_gt_database` 和 `kitti3d_dbinfos_train.pkl` 是训练是用于采样的样本。


浮点模型训练
-------------
数据集准备好之后，就可以开始训练浮点型的 PointPillars 检测网络了。在网络训练开始之前，你可以使用以下命令先测试一下网络的计算量\
和参数数量：

  .. code-block:: python

    python3 tools/calops.py --config configs/detection/pointpillars/pointpillars_kitti_car.py

如果你只是单纯的想启动这样的训练任务，只需要运行下面的命令就可以：

  .. code-block:: python

    python3 tools/train.py --stage float --config configs/detection/pointpillars/pointpillars_kitti_car.py

由于HAT算法包使用了一种巧妙的注册机制，使得每一个训练任务都可以按照这种train.py加上config配置文件的形式启动。train.py是\
统一的训练脚本，与任务无关，我们需要训练什么样的任务、使用什么样的数据集以及训练相关的超参数设置都在指定的config配置文件里面。config文件
里面提供了模型构建、数据读取等关键的dict。


模型构建
>>>>>>>>>
PointPillars 的网络结构可以参考
`论文 <https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf>`_ ，这里\
不做详细介绍。我们通过在config配置文件中定义`model`这样的一个dict型变量，就可以方便的实现对模型的定义和修改。

  .. code-block:: python

    model = dict(
        type="PointPillarsDetector",
        feature_map_shape=get_feature_map_size(pc_range, voxel_size),
        pre_process=dict(
            type="PointPillarsPreProcess",
            pc_range=pc_range,
            voxel_size=voxel_size,
            max_voxels_num=max_voxels_num,
            max_points_in_voxel=max_points_in_voxel,
        ),
        reader=dict(
            type="PillarFeatureNet",
            num_input_features=4,
            num_filters=(64,),
            with_distance=False,
            pool_size=(1, max_points_in_voxel),
            voxel_size=voxel_size,
            pc_range=pc_range,
            bn_kwargs=norm_cfg,
            quantize=True,
            use_4dim=True,
        ),
        backbone=dict(
            type="PointPillarScatter",
            num_input_features=64,
            use_horizon_pillar_scatter=True,
            quantize=True,
        ),
        neck=dict(
            type="SequentialBottleNeck",
            layer_nums=[3, 5, 5],
            ds_layer_strides=[2, 2, 2],
            ds_num_filters=[64, 128, 256],
            us_layer_strides=[1, 2, 4],
            us_num_filters=[128, 128, 128],
            num_input_features=64,
            bn_kwargs=norm_cfg,
            use_tconv=True,
            use_secnet=True,
            quantize=True,
        ),
        head=dict(
            type="PointPillarHead",
            num_classes=len(class_names),
            in_channels=sum([128, 128, 128]),
            use_direction_classifier=True,
        ),
        anchor_generator=dict(
            type="Anchor3DGeneratorStride",
            anchor_sizes=[[1.6, 3.9, 1.56]],
            anchor_strides=[[0.32, 0.32, 0.0]],
            anchor_offsets=[[0.16, -39.52, -1.78]],
            rotations=[[0, 1.57]],
            class_names=class_names,
            match_thresholds=[0.6],
            unmatch_thresholds=[0.45],
        ),
        targets=dict(
            type="LidarTargetAssigner",
            box_coder=dict(
                type="GroundBox3dCoderTorch",
                n_dim=7,
            ),
            class_names=class_names,
            positive_fraction=-1,
            region_similarity_calculator=dict(type="NearestIouSimilarity"),
        ),
        loss=dict(
            type="PointPillarsLoss",
            num_classes=len(class_names),
            loss_cls=dict(
                type="SigmoidFocalLoss",
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
            ),
            loss_bbox=dict(
                type="WeightedSmoothL1Loss",
                sigma=3.0,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                codewise=True,
                loss_weight=2.0,
            ),
            loss_dir=dict(
                type="WeightedSoftmaxClassificationLoss",
                name="direction_classifier",
                loss_weight=0.2,
            ),
        ),
        postprocess=dict(
            type="PointPillarsPostProcess",
            num_classes=len(class_names),
            box_coder=dict(
                type="GroundBox3dCoderTorch",
                n_dim=7,
            ),
            use_direction_classifier=True,
            num_direction_bins=2,
            # test_cfg
            use_rotate_nms=False,
            nms_pre_max_size=1000,
            nms_post_max_size=300,
            nms_iou_threshold=0.5,
            score_threshold=0.4,
            post_center_limit_range=[0, -39.68, -5, 69.12, 39.68, 5],
            max_per_img=100,
        ),
    )

其中，`model`下面的`type`表示定义的模型名称，剩余的变量表示模型的其他组成部分。这样定义模型的好处在于我们可以很方便的替换我们想要的结构。
训练脚本在启动之后，会调用`build_model`接口，将这样一个dict类型的model变成类型为`torch.nn.Module`类型的model。


数据增强
>>>>>>>>>
跟`model`的定义一样，数据增强的流程是通过在config配置文件中定义`data_loader`和`val_data_loader`这两个dict来实现的，分别对应着\
训练集和验证集的处理流程。以`data_loader`为例，

  .. code-block:: python

    dataset = dict(
      type="Kitti3D",
      data_path="./tmp_data/kitti3d/train_lmdb",
      transforms=[
          dict(
              type="ObjectSample",
              class_names=class_names,
              remove_points_after_sample=False,
              db_sampler=db_sampler,
          ),
          dict(
              type="ObjectNoise",
              gt_rotation_noise=[-0.15707963267, 0.15707963267],
              gt_loc_noise_std=[0.25, 0.25, 0.25],
              global_random_rot_range=[0, 0],
              num_try=100,
              class_names=class_names,
          ),
          dict(
              type="PointRandomFlip",
              probability=0.5,
          ),
          dict(
              type="PointGlobalRotation",
              rotation=[-0.78539816, 0.78539816],
          ),
          dict(
              type="PointGlobalScaling",
              min_scale=0.95,
              max_scale=1.05,
          ),
          dict(
              type="ShufflePoints",
          ),
          dict(
              type="ObjectRangeFilter",
              point_cloud_range=pc_range,
          ),
          dict(type="Reformat"),
      ],
    )
    
    data_loader = dict(
      type=torch.utils.data.DataLoader,
      dataset=dataset,
      sampler=dict(type=torch.utils.data.DistributedSampler),
      batch_size=batch_size_per_gpu,
      shuffle=False,
      num_workers=1,
      pin_memory=True,
      collate_fn=collate_kitti3d,
    )


其中type直接用的pytorch自带的接口torch.utils.data.DataLoader，表示的是将`batch_size`大小的图片组合到一起。
这里面唯一需要关注的可能是`dataset`这个变量， `data_path` 路径也就是我们在第一部分数据集准备中提到的路径。
`transforms`下面包含着一系列的数据增强。`val_data_loader`中只有除了点云Pillar化(Voxelization）和 Reformat。你也可以通过在`transforms`中插入新的dict实现自己希望的数据增强操作。


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
                betas=(0.95, 0.99),
                lr=5e-4,
                weight_decay=0.1,
            ),
            batch_processor=batch_processor,
            num_epochs=160,
            device=None,
            callbacks=[
                stat_callback,
                loss_show_update,
                dict(
                    type="CyclicLrUpdater",
                    target_ratio=(10, 1e-4),
                    cyclic_times=1,
                    step_ratio_up=0.4,
                    step_log_interval=50,
                ),
                dict(
                    type="CyclicOptimParamUpdater",
                    param_name="betas",
                    target_ratio=(0.85 / 0.95, 1),
                    cyclic_times=1,
                    step_ratio_up=0.4,
                    step_log_interval=50,
                ),
                dict(
                    type="GradClip",
                    max_norm=35,
                    norm_type=2,
                ),
                val_callback,
                ckpt_callback,
            ],
            sync_bn=True,
            train_metrics=dict(
                type="LossShow",
            ),
            val_metrics=dict(
                type="Kitti3DMetricDet",
                compute_aos=True,
                current_classes=class_names,
                difficultys=[0, 1, 2],
            ),
        )


`float_trainer`从大局上定义了我们的训练方式，包括使用多卡分布式训练（distributed_data_parallel_trainer），模型训练的epoch次数，以及优化器的选择。\
同时`callbacks`中体现了模型在训练过程中使用到的小策略以及用户想实现的操作，包括学习率的变换方式(OneCycleLrUpdater)，在训练过程中验证模型的\
指标(Validation)，以及保存(Checkpoint)模型的操作。当然，如果你有自己希望模型在训练过程中实现的操作，也可以按照这种dict的方式添加。
`float_trainer`负责将整个训练的逻辑给串联起来，其中也会负责模型的pretrain。

  .. note::

      如果需要复现精度，config中的训练策略最好不要修改。否则可能会有意外的训练情况出现。

通过上面的介绍，你应该对config文件的功能有了一个比较清楚的认识。然后通过前面提到的训练脚本，就可以训练一个高精度的纯浮点的检测模型。当然训练一个\
好的检测模型不是我们最终的目的，它只是做为一个pretrain为我们后面训练定点模型服务的。


量化模型训练
------------
当我们有了纯浮点模型之后，就可以开始训练相应的定点模型了。和浮点训练的方式一样，我们只需要通过运行下面的脚本就可以训练定点模型了：

  .. code-block:: python

    python3 tools/train.py --stage qat --config configs/detection/pointpillars/pointpillars_kitti_car.py

可以看到，我们的配置文件没有改变，只改变了`step`的类型。此时我们使用的训练策略来自于config文件中的qat_trainer

  .. code-block:: python
    
        qat_trainer = dict(
            type="distributed_data_parallel_trainer",
            model=model,
            model_convert_pipeline=dict(
                type="ModelConvertPipeline",
                qat_mode="fuse_bn",
                qconfig_params=dict(
                    activation_qat_qkwargs=dict(
                        averaging_constant=0,
                    ),
                    weight_qat_qkwargs=dict(
                        averaging_constant=1,
                    ),
                ),
                converters=[
                    dict(type="Float2QAT", convert_mode=convert_mode),
                    dict(
                        type="LoadCheckpoint",
                        checkpoint_path=os.path.join(
                            ckpt_dir, "calibration-checkpoint-best.pth.tar"
                        ),
                    ),
                ],
            ),
            data_loader=data_loader,
            optimizer=dict(
                type=torch.optim.SGD,
                params={"weight": dict(weight_decay=0.0)},
                lr=2e-4,
                momentum=0.9,
            ),
            batch_processor=batch_processor,
            num_epochs=50,
            device=None,
            callbacks=[
                stat_callback,
                loss_show_update,
                dict(
                    type="CyclicLrUpdater",
                    target_ratio=(10, 1e-4),
                    cyclic_times=1,
                    step_ratio_up=0.4,
                    step_log_interval=50,
                ),
                val_callback,
                ckpt_callback,
            ],
            train_metrics=dict(
                type="LossShow",
            ),
            val_metrics=dict(
                type="Kitti3DMetricDet",
                compute_aos=True,
                current_classes=class_names,
                difficultys=[0, 1, 2],
            ),
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

    python3 tools/predict.py --stage float --config configs/detection/pointpillars/pointpillars_kitti_car.py
    python3 tools/predict.py --stage qat --config configs/detection/pointpillars/pointpillars_kitti_car.py

同时，我们还提供了quantization模型的性能测试，只需要运行以下命令：

  .. code-block:: python

    python3 tools/predict.py --stage int_infer --config configs/detection/pointpillars/pointpillars_kitti_car.py

这个显示出来的精度才是最终的int8模型的真正精度，当然这个精度和qat验证阶段的精度应该是保持十分接近的。

仿真上板精度验证
>>>>>>>>>>>>>>

除了上述模型验证之外，我们还提供和上板完全一致的精度验证方法，可以
通过下面的方式完成：

    .. code-block:: bash

       python3 tools/align_bpu_validation.py --config configs/detection/pointpillars/pointpillars_kitti_car.py --dataset kitti3d


结果可视化
>>>>>>>>>>>>>>
如果你希望可以看到训练出来的模型对于单帧雷达点云的检测效果，我们的tools文件夹下面同样提供了点云预测及可视化的脚本，你只需要运行以下脚本即可：

  .. code-block:: python

    python3 tools/infer_lidar.py --config configs/detection/pointpillars/pointpillars_kitti_car.py --dataset kitti3d --input-points ${lidar-pointcloud-path} --is-plot


模型检查和编译
>>>>>>>>>>>>>>

在训练完成之后，可以使用`compile`的工具用来将量化模型编译成可以上板运行的\
`hbm`文件，同时该工具也能预估在BPU上的运行性能，可以采用以下脚本：

    .. code-block:: bash

       python3 tools/compile_perf.py --config configs/detection/pointpillars/pointpillars_kitti_car.py --out-dir ./ --opt 3

