Bev Multi-task Model Training
=====================================

The BEV reference algorithm is developed based on Horizon Torch Samples (Horizon's own deep learning framework), and you can refer to the Horizon Torch Samples usage documentation for an introduction to the use of Horizon Torch Samples.
The training config for the BEV reference algorithm is located under the HAT/configs/bev/ path.
The following part takes HAT/configs/bev/bev_mt_ipm.py as an example to describe how to configure and train the BEV reference algorithm.

Dataset Preparation
-------------------------

Here is an example of the nuscense dataset, which can be downloaded from https://www.nuscenes.org/nuscenes.
Also, in order to improve the speed of training, we have done a packing of the original jpg format dataset to convert it to lmdb format. 
Just run the following script and it will be successful to achieve the conversion.

.. code-block:: python
  
    python36 tools/datasets/nuscenes_packer.py --src-data-dir WORKSAPCE/datasets/nuscenes/ --pack-type lmdb --target-data-dir . --version v1.0-trainval --split-name val
    python36 tools/datasets/nuscenes_packer.py --src-data-dir WORKSAPCE/datasets/nuscenes/ --pack-type lmdb --target-data-dir . --version v1.0-trainval --split-name train

The above two commands correspond to transforming the training dataset and the validation dataset, respectively. After the packing is completed, the file structure in the data directory should look as follows.

  .. code-block:: bash

    tmp_data
    |-- nuscenes
      |-- metas
      |-- v1.0-trainval
        |-- train_lmdb
        |-- val_lmdb

The train_lmdb and val_lmdb are the training and validation datasets after packaging, and are the datasets that the network will eventually read. metas is the map information needed for the segmentation model.

Floating-point Model Training
----------------------------------

Once the dataset is ready, you can start training the floating-point bev multitasking network. 

If you simply want to start such a training task, just run the following command.

  .. code-block:: python

    python3 tools/train.py --stage float --config configs/bev/bev_mt_imp.py

Because the Horizon Torch Samples algorithm package uses a clever registration mechanism, each training task can be started in the form of the train.py plus a config configuration file. 
The train.py is a unified task-independent training script, the kind of task we need to train, the kind of dataset we use, and the hyperparameter settings related to training are all in the specified config configuration file. 
The config file provides the key dict such as model building, data reading, and so on.

Model Construction
>>>>>>>>>>>>>>>>>>>>>>>>>

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
       
           
Where `type` under `model` indicates the name of the defined model, and the remaining variables indicate the other components of the model. 
The advantage of defining the model this way is that we can easily replace the structure we want. For example, if we want to train a model with a backbone of resnet50, we just need to replace `backbone` under `model`.


Data Augmentation
>>>>>>>>>>>>>>>>>>>>>

Like the definition of `model`, the data enhancement process is implemented by defining two dicts `data_loader` and `val_data_loader` in the config configuration file, corresponding to 
training set and the processing flow of the validation set. Take `data_loader` as an example.

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

Where type directly uses the interface torch.utils.data.DataLoader that comes with pytorch, which represents the combination of `batch_size` size images together.
The only thing to be concerned about here is probably the `dataset` variable, `CocoFromLMDB` means reads the image from the lmdb dataset, and the path is the same path we mentioned in the first part of the dataset preparation.
 `transforms` contains a series of data enhancements underneath. Except for the image flip (RandomFlip), the other data transformations of the `val_data_loader` are the same as `data_loader`. 
You can also achieve the data augmentation you want by inserting a new dict in `transforms`.

Training Strategies
>>>>>>>>>>>>>>>>>>>>>>>>>

In order to train a model with high accuracy, a good training strategy is essential.
For each training task, the corresponding training strategy is defined in the config file as well, as can be seen from the variable `float_trainer`.

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
            "./tmp_pretrained_models/effcientnet_cls/float-checkpoint-best.pth.tar"
        ),
        quantize=False,
        allow_not_init=True,
        allow_miss=True,
        ignore_extra=True,
        strict_match=True,
    )

The `float_trainer` defines our training approach in the big picture, including the use of distributed_data_parallel_trainer, the number of epochs for model training, and the choice of optimizer. 
Also, the `callbacks` reflect the small strategies used by the model during training and the operations that the user wants to implement, including the way to transform the learning rate (WarmupStepLrUpdater), the metrics to validate the model during training (Validation), and the operations to save (Checkpoint) the model. 
Of course, if you have operations that you want the model to implement during training, you can also add them in this dict way.
The `float_solver` is responsible for stringing together the entire training logic, which will also be responsible for the pretrain of the model.

  .. note::

      If reproducibility accuracy is needed, the training strategy in config is best not modified. Otherwise, unexpected training situations may occur.

With the above introduction, you should have a clearer understanding of the functions of the config file. Then you can train a high-precision pure floating-point detection model by the training script mentioned earlier.
Of course, training a good detection model is not our ultimate goal, it is only used as a pretrain for us to train a fixed-point model later.

Quantized Model Training
------------------------------

Once we have a pure floating-point model, we can start training the corresponding fixed-point model. In the same way as with floating-point training, we can train the fixed-point model simply by running the following script.

  .. code-block:: python

    python3 tools/train.py --stage qat --config configs/bev/bev_mt_ipm.py

As you can see, our config file has not changed, only the type of `stage` has been changed. The training strategy we use at this point comes from qat_trainer in the config file.

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
        check_quantize_model=False,
        pre_step="float",
        pre_step_checkpoint=os.path.join(
            ckpt_dir, "float-checkpoint-best.pth.tar"
        ),
        strict_match=True,
    )

1. The value of quantize parameter is different
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

When we train the quantized model, we need to set quantize=True, at this time the corresponding floating point model will be converted into a quantized model, the relevant code is as follows.

  .. code-block:: python

    model.fuse_model()
    model.set_qconfig()
    horizon.quantization.prepare_qat(model, inplace=True)

For key steps in quantization training, such as preparing the floating-point model, operator substitution, inserting quantization and inverse quantization nodes, setting quantization parameters, 
and fusing operators, please read the sections ``Floating-point Model Preparation'' and ``Operator Fusion'' in the ``Horizon Plugin PyTorch'' manual.

2. Different training strategies
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

As we said before, quantization training is in fact finetune based on pure floating-point training, so when quantization training, our initial learning rate is set to one-tenth of the floating-point training, 
the number of epochs for training is largely reduced, most importantly, when defining the `model`, our `pretrained` needs to be set to the address of a pure floating-point model that has already been trained.

After making these simple adjustments, we can start training our quantitative model.

Model Verification
>>>>>>>>>>>>>>>>>>>>>>>>

After the model is trained, we can also verify the performance of the trained model. Since we provide two stages of training process, float and qat, so we can verify the performance of the model trained in these two stages.
It is only necessary to run the following two commands accordingly.

  .. code-block:: python

    python3 tools/predict.py -c configs/bev/bev_mt_ipm.py --stage float
    python3 tools/predict.py -c configs/bev/bev_mt_ipm.py --stage qat

Also, we provide performance tests for the quantization model. Simply run the following command.

  .. code-block:: python

    python3 tools/predict.py -c configs/bev/bev_mt_ipm.py --stage int_infer

This displayed accuracy is the real accuracy of the final int8 model, which of course should be very close to the accuracy of the qat verification stage.(Model accuracy may fluctuate somewhat depending on the environment dependency) 

Simulation of On-board Accuracy Validation
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
In addition to the above model validation, we also provide the exact same accuracy validation method simulating the on-board conditions, as below:

  .. code-block:: python

        python3 tools/bev_align_bpu_validation.py --config configs/bev/bev_mt_ipm.py --dataset nuscenes       

Results Visualization
>>>>>>>>>>>>>>>>>>>>>>>>>

If you want to see the results of a single-frame detection of the trained model, we also provide scripts for single-frame prediction and visualization under our tools folder. 
You just need to give the size of each image and the single response matrix according to the format in infer_bev.py, and then run the following script.

  .. code-block:: python

    python3 tools/infer_bev.py --config configs/bev/bev_mt_ipm.py --dataset nuscenes --inputs ${img_list} --homo ${homography.npy}  --input-format yuv --is-plot

Model Checking And Compilation
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

After training, the `compile` tool can be used to compile the quantization model into an `hbm` file that can be run on the board.
Also, this tool can predict the performance of the run on BPU, by using the following script.

    .. code-block:: bash

       python3 tools/compile_perf.py --config configs/bev/bev_mt_ipm.py --out-dir ./ --opt 2

