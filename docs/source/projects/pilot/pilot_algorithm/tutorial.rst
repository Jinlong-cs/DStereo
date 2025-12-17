.. _pilot-config-develop:

Tutorial：增加一个任务
==================================

Pilot周视模型训练框架在提供已有多任务和单任务的相应实现以外，还支持用户增加自定义的训练任务。如何在训练框架中添加自定义的训练任务呢？下面的实际操作教程应该能够给您答案。

训练框架中增加一个任务存在两种方式：

* 一种是直接完整定义一个单任务的方式， ``projects/pilot/configs/single_task`` 目录下存放了许多已实现的例子，比如depth estimation等，这类任务单独进行模型训练。

* 一种是通过在多任务模型训练中增加一个单任务的方式，类似 ``projects/pilot/configs/resize_2`` 目录下实现的众多单任务例子，比如vehicle wheel detection等，这类单任务与其他单任务一起进行多任务的模型训练。

总之，增加的方式需要根据任务的特点以及该任务的模型结构特点去进行选择。比如，新增任务适合使用二阶段的模型结构，即 ``TwoStageDetector`` 模型结构的，就可以考虑在多任务模型训练中添加新任务。

.. GENERATED FROM PYTHON SOURCE LINES 17-34

.. _build_config:

config搭建
--------------------------------------------------------

.. _注册机制: ../tutorials/registry.html

在训练框架中添加新的任务进行训练，需要添加新任务的config文件，以此来定义该任务的训练配置。但在搭建config前，需要确保以下步骤已经完成：

* 模块设计和实现：在添加新任务时，您需要进行该任务具体模块的设计和实现，搭建的config中会使用到这些模块。``hat`` 目录下提供了丰富的各类基础模块，在熟悉
  ``hat`` 目录下的各层文件夹的分类后，您在模块实现时可以进行参考或使用。

* 模块注册：确保新增加的任务中使用到的各个模块都已在 ``hat`` 目录下实现后，需要通过HAT中的注册机制进行注册，具体的注册模块操作参考 `注册机制`_ 进行操作。

至于如何搭建config，在您根据前面提到的两种增加方式进行选择后，可以参考已有的实现进行类似的配置，包括定义新任务的模型结构，数据部分，以及其他模块。当然，为了让您更好地
了解如何搭建新任务的config，下面将会举例说明。


.. GENERATED FROM PYTHON SOURCE LINES 36-41

方式一：构建一个单任务模型训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以参考 ``projects/pilot/configs/single_task`` 目录下已实现的单任务config来搭建您自己的任务config。


.. GENERATED FROM PYTHON SOURCE LINES 43-262

方式二：在多任务模型训练中添加一个新任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在多任务模型训练中添加新任务，需要参考多任务模型训练框架中单任务的定义规则去定义新增任务的关键模块，包括：定义模型结构的接口 ``get_mode()`` 、定义数据加载器模块的 ``dataloader``、
定义fake input模块的 ``inputs`` 、以及定义log信息输出的 ``metric_updater`` , 从而通过这些模块的定义来组成新增任务的config。

下面以在多任务模型训练中添加汽车车轮关键点检测 ``vehicle_wheel_kps`` 任务为例，去介绍如何搭建新任务的config。

模型结构的定义
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

定义模型结构时，需要根据新任务的功能来定义该任务的模型结构以及模型结构中的各个模块。

多任务模型训练中每个子任务模型结构的具体定义是通过该任务config中 ``get_model()`` 实现的，在 ``model.py`` 中将对这些 ``get_model()`` 进行调用，从而获得每个子任务的模型结构配置。

下面我们简单介绍两种典型模型结构的定义方法。

检测类任务的模型定义方法
```````````````````````````````````````````````````

:py:class:`~hat.models.structures.detectors.two_stage.TwoStageDetector` 类模型结构用于对输入图像进行二阶段的感知，第一个阶段模块 :py:class:`~hat.models.task_modules.anchor_module.AnchorModule` 输出一定数量的ROI（region of interest），然后将这些ROI传入
到第二阶段模块 :py:class:`~hat.models.task_modules.roi_module.RoIModule` 中进行具体目标的预测。

下面展示了一个二阶段任务的模型结构定义示例：

.. code-block:: python
    :emphasize-lines: 0

    def get_model(mode):

        return dict(
            type="TwoStageDetector",
            backbone=xxx,
            neck=xxx,
            rpn_module=dict(
                type="AnchorModule",
                args1=xxx,
                args2=xxx,
                ...
            ), # 第一阶段模块定义
            roi_module=dict(
                type="RoIModule",
                args1=xxx,
                args2=xxx,
                ...
            ), # 第二阶段模块定义
        )

``vehicle_wheel_kps`` 任务为二阶段的关键点检测任务，显然也应该用这种方式定义模型结构。

* 首先，我们可以从 ``common.py`` 中引用共享的backbone和neck进行相应模块的定义。

* 第一阶段模块，考虑到车辆关键点检测是在车辆检测的基础上进行的，因此可以从 ``vehicle_detection`` 任务的 ``AnchorModule`` 中引用定义好的子模块，实现这些子模块在两个任务之间的共享。

* 第二阶段模块 ``RoIModule`` 则是需要根据 ``vehicle_wheel_kps`` 任务的特点去进行设计和定义，因此 ``RoIModule`` 中会存在一部分任务特定的子模块（比如针对关键点label编码的 ``target`` 等）是不与其他任务共享的。
  如果需要添加其他与车轮相关的子任务，如 ``vehicle_wheel_det`` ，则可将 ``RoIModule`` 中子模块，如 ``head`` 队列的第一个head （即 ``roi_head`` ）进行复用，以更充分地实现任务之间的模块共享。

同时，值得注意的是，``vehicle_wheel_kps`` 任务在定义 ``AnchorModule`` 时， ``target`` 和 ``loss`` 模块设置为None。
这样的定义方式意味着：在训练阶段， ``vehicle_wheel_kps`` 不对RPN部分做任何参数调整。


分割类模型结构的定义
`````````````````````````````````````````````

训练框架提供了针对分割任务的一阶段模型结构 :py:class:`~hat.models.structures.segmentor.SegmentorV2`，该类模型结构的定义示例如下所示：

.. code-block:: python
    :emphasize-lines: 0

    def get_model(mode):
        return dict(
            type="SegmentorV2",
            backbone=xxx,
            neck=xxx,
            head=xxx,
            loss=xxx if mode is not "train" else None,
            desc=xxx if mode is not "train" else None, # 给预测输出增加description
        )

例如 ``lane_segmentation`` 任务为车道线分割任务，在训练框架中采用的是 :py:class:`~hat.models.structures.segmentor.SegmentorV2` 类模型结构。

在具体的模块定义中，我们同样可以从 ``common.py`` 中引用共享的backbone和neck进行相应模块的定义，值得注意的是分割任务的neck定义与
二阶段的检测任务稍有不同，需要使用分割任务的neck子模块。

紧接着则是定义各单任务私有的模块：用于预测的 ``head`` 、训练时的 ``loss`` 以及描述预测输出的 ``desc`` 。

本小节我们结合示例介绍了两种模型结构的定义方式，您还可以根据算法包中的其他示例进行模型结构的定义。

数据部分的定义
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本小节介绍数据部分的定义。

定义好模型模块之后，您可以从 ``common.py`` 中引用 ``datapaths`` ，根据任务名获取该单任务的数据配置，从而获得了该任务的所有训练数据地址（rec_paths）、标注地址（anno_paths）和数据集的采样权重。

当您定义任务的 ``data_loader`` 时，可以使用提供的数据加载器类进行数据集 ``dataset`` 、 数据预处理 ``transfrom`` 和
一些数据采集方式的定义。该任务的dataloader定义可以参照如下代码：


.. code-block:: python
    :emphasize-lines: 0

    ds = datapaths.vehicle_wheel_kps
    data_paths = [d["data_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        num_workers=0,
        batch_size=batch_size,
        dataset=dict(
            type="DistributedComposeRandomDataset",
            sample_weights=sample_weights,
            datasets=[
                dict(
                    type="DetSeg2DAnnoDataset",
                    idx_path=os.path.join(path, "idx"),
                    img_path=os.path.join(path, "img"),
                    anno_path=os.path.join(path, "anno"),
                    transforms=[
                        dict(
                            type="KPSIterableDetRoITransform",
                            kps_num=2,
                            # roi transform
                            target_wh=resize_hw[::-1],
                            resize_wh=None
                            if resize_hw is None
                            else resize_hw[::-1],
                            img_scale_range=(0.5, 2.0),
                            roi_scale_range=(0.7, 1.0 / 0.7),
                            min_sample_num=1,
                            max_sample_num=1,
                            center_aligned=False,
                            inter_method=inter_method,
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            min_valid_area=100,
                            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
                            min_edge_size=10,
                            rand_translation_ratio=rand_translation_ratio,
                            rand_aspect_ratio=0.0,
                            rand_rotation_angle=0,
                            flip_prob=0.5,
                            clip_bbox=False,
                            pixel_center_aligned=pixel_center_aligned,
                            min_kps_distance=4,
                            keep_aspect_ratio=True,
                        ),
                        dict(
                            type="PadDetData",
                            max_gt_boxes_num=200,
                            max_ig_regions_num=100,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )


其他部分的定义
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本节为您介绍任务模型的其他两个部分：

* ``inputs`` ：在模型编译时用来trace模型计算图的fake input，定义的内容可以是用'0'去构成模型的一次输入。具体定义的示例如下：

.. code-block:: python
    :emphasize-lines: 0

    inputs = dict(
        train=dict(
            input_1=torch.zeros((x, x, x)),
            input_2=torch.zeros(x),
            ...
        ),
        val=dict(),
        test=dict(),
    )

* ``metric_updater`` ：该模块可以使得该任务训练时的loss log信息按照指定间隔进行输出，通过定义 ``MetricUpdater`` 类及该类的参数来实现（由于功能上和指标计算类似，需要一定间隔触发）。具体定义的示例如下：

.. code-block:: python
    :emphasize-lines: 0

    metric_updater = dict(
        type="MetricUpdater",
        metrics=[
            dict(type="LossShow", name="loss_1"),
            dict(type="LossShow", name="loss_2"),
        ],
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=[  # 与 metrics 参数对应
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}_loss_1$",
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}_loss_2$",
                ),
            ]
        ),
        step_log_freq=xxx,
        epoch_log_freq=xxx,
        log_prefix=task_name,
        reset_metrics_by="log",
    )


添加新任务索引
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

到这里，您已经完成了新增任务 ``vehicle_wheel_kps`` 的config搭建，此外还需要在训练框架中增加新增任务的索引。

具体操作是在 ``common.py`` 的 ``tasks`` 列表中增加该任务的任务名，任务名即是该任务config文件名前缀。


.. GENERATED FROM PYTHON SOURCE LINES 265-275

开始训练
--------------------------------------------------------

完成新任务的config增加后，使用工具 ``tools/train.py`` 开始训练，该工具有相应的参数 ``--conifg`` 接收训练任务的config文件：

* 对于完整单任务的训练，可以直接将新增config传入到训练工具中。

* 对于多任务模型的训练，可以将相应种类目录下的 ``multitask.py`` 传入到训练工具中，根据多任务训练中config的层级结构关系，
  训练框架获得多任务模型的模型结构、训练数据和其他部分的配置。请参考：:ref:`train-operations-of-multitask`。


开始推理、评测
--------------------------------------------------------

完成训练后，您可以使用模型推理和模型评测工具来评价模型的训练效果，请参考 :ref:`infer-operations-of-multitask` 和 :ref:`eval-operations-of-multitask` 进行模型的推理和评测操作。
