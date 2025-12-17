import copy
import os

import torch
from hatbc.filestream.bucket.client import get_bucket_client
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
bucket_client = get_bucket_client()
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "fatigue-eye-status"
local = True
if local:
    root_url = ""  # noqa
    root_dir = bucket_client.url_to_local(root_url)
    # root_dir = "/home/users/han.tang/workspace/model_result"
    # job_id = "0103-mix05-multitask-large-fc"
    job_id = "finetune-all-0307-faceid-more-50-epoch-decay025-mix05-large-fc-0423_binocular_difference_wholedata_balance"  # noqa
    # temp_qat_job_id = "0103-use-0103-mix05-multitask-freeze-bk"
    pretrain_url = "dmpv2://MultiMode_3/jiaqi.quan/fatigue/model/eye-status/mixvargenet/finetune-all-0307-faceid-more-50-epoch-decay025-mix05-large-fc-0423_binocular_difference_wholedata_balance"  # noqa
    # root_dir = "/home/users/han.tang/workspace/model_result/fatigue"
    # job_id = os.environ.get("JOB_ID", "eye_status_baseline_reproduce")
    num_workers = 5
else:
    # root_dir = "/job_data/"
    root_url = "dmpv2://MultiMode_3/jiaqi.quan/fatigue/model/eye-status/mixvargenet/mix_binary_classification"  # noqa
    root_dir = bucket_client.url_to_local(root_url)
    job_id = "finetune-all-0307-faceid-more-50-epoch-decay025-mix05-large-fc-0423_binocular_difference_wholedata_balance"  # noqa
    num_workers = 12
    pretrain_url = "dmpv2://MultiMode_3/jiaqi.quan/fatigue/model/eye-status/mixvargenet/finetune-all-0307-faceid-more-50-epoch-decay025-mix05-large-fc-0423_binocular_difference_wholedata_balance"  # noqa

batch_size_per_gpu = 256
device_ids = [0, 1, 2, 3]

ckpt_dir = os.path.join(root_dir, job_id)
# temp_ckpt_dir = os.path.join(root_dir, temp_qat_job_id)
# ckpt_dir = temp_ckpt_dir
# temp_ckpt_dir = ckpt_dir
pretrain_dir = bucket_client.url_to_local(pretrain_url)

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
num_ldmk = 16
input_shape = (96, 160, 3)

bn_kwargs = dict(eps=1e-3, momentum=0.01)

in_channels = [32, 32, 32, 64, 128]
out_channels = [32, 32, 64, 128, 256]
alpha = 0.5
in_channels = [int(x * alpha) for x in in_channels]
out_channels = [int(x * alpha) for x in out_channels]
ldmk_in_channels = out_channels[1:]

net_config = [
    [
        MixVarGENetConfig(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

cls_net_config = [
    [
        MixVarGENetConfig(
            in_channels=out_channels[-1],
            out_channels=int(320 * alpha),
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],
]

mixvargenet_cls_head = dict(
    type="EyeMixVarGEClsHead",
    net_config=cls_net_config,
    num_classes=5,
    in_channels=int(320 * alpha),
    out_channels=1280,
    bn_kwargs=bn_kwargs,
    pool_size=(3, 5),
    include_top=True,
)

# deploy_cls_head
deploy_mixvargenet_cls_head = dict(
    type="EyeMixVarGEClsHead",
    net_config=cls_net_config,
    num_classes=5,
    in_channels=int(320 * alpha),
    out_channels=1280,
    bn_kwargs=bn_kwargs,
    pool_size=(3, 5),
    include_top=True,
    flat_output=False,
)


irrs_head = dict(
    type="IrisMutilBranchHead",
    classfier_num=5,
    bn_kwargs=bn_kwargs,
    alpha=0.5,
)
cls_head = dict(
    type="EyeMultiBranchHead",
    l_cls_head=copy.deepcopy(mixvargenet_cls_head),
    r_cls_head=copy.deepcopy(mixvargenet_cls_head),
    # r_cls_head=mixvargenet_cls_head,
    # l_cls_head=mixvargenet_cls_head,
)

bin_cls_head = dict(
    type="EyeMixVarGEBinClsHead",
    bn_kwargs=bn_kwargs,
    pool_size=(3, 5),
    in_channels=int(256 * alpha),
    out_channels=320,
    bias=False,
    dropout_rate=0.2,
)

# deploy_bin_cls_head
deploy_bin_cls_head = dict(
    type="EyeMixVarGEBinClsHead",
    bn_kwargs=bn_kwargs,
    pool_size=(3, 5),
    in_channels=int(256 * alpha),
    out_channels=320,
    bias=False,
    flat_output=False,
    dropout_rate=0.2,
)

multi_bin_head = dict(
    type="EyeMultiBinBranchHead",
    l_cls_head=copy.deepcopy(bin_cls_head),
    r_cls_head=copy.deepcopy(bin_cls_head),
)


# cls_head = irrs_head
deploy_cls_head = dict(
    type="EyeMultiBranchHead",
    l_cls_head=copy.deepcopy(deploy_mixvargenet_cls_head),
    r_cls_head=copy.deepcopy(deploy_mixvargenet_cls_head),
)
deploy_multi_bin_head = dict(
    type="EyeMultiBinBranchHead",
    l_cls_head=copy.deepcopy(deploy_bin_cls_head),
    r_cls_head=copy.deepcopy(deploy_bin_cls_head),
)

# Train Model
model = dict(
    type="EyeStatusMultiTaskClassifier",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    ldmk_neck=dict(
        type="FPEM_FFM",
        bn_kwargs=bn_kwargs,
        in_channels_list=ldmk_in_channels,
        out_channels=128,
        # in_channels_list=[24, 48, 96, 192],
        # out_channels=192,
        fpem_repeat=1,
        act_type="relu",
    ),
    cls_head=cls_head,
    bin_cls_head=multi_bin_head,
    ldmk_head=dict(
        type="EyeLdmkHead",
        heatmap_head=dict(
            type="EyeLdmkHeatmapHead",
            in_channels=16,
            num_ldmk=num_ldmk,
        ),
        vector_head=dict(
            type="EyeLdmkVectorHead",
            num_ldmk=num_ldmk,
            in_channels=16,
        ),
    ),
    left_cls_losses=dict(
        type="SoftTargetCrossEntropy",
    ),
    right_cls_losses=dict(
        type="SoftTargetCrossEntropy",
    ),
    left_bin_cls_losses=dict(
        type=torch.nn.BCEWithLogitsLoss,
        reduction="mean",
    ),
    right_bin_cls_losses=dict(
        type=torch.nn.BCEWithLogitsLoss,
        reduction="mean",
    ),
    ldmk_losses=dict(
        type="SmoothL1Loss",
        beta=1.0,
    ),
    heatmap_losses=dict(
        type="SmoothL1Loss",
        beta=1.0,
    ),
    num_classes=5,
)

# Val Model
val_model = dict(
    type="EyeStatusMultiTaskClassifier",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    cls_head=cls_head,
    bin_cls_head=multi_bin_head,
)


# Deploy Model
deploy_model = dict(
    type="EyeStatusMultiTaskClassifier",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    cls_head=deploy_cls_head,
    bin_cls_head=deploy_multi_bin_head,
)
deploy_inputs = dict(
    img=torch.randn(1, 3, 96, 160),
    # eye_cls_labels=torch.randn(1, 10),
    # gt_vector=torch.randn(1, num_ldmk, 64),
    # gt_vector_weight=torch.randn(1, num_ldmk, 96),
    # gt_heatmap=torch.randn(1, num_ldmk, 24, 40),
    # gt_heatmap_weight=torch.randn(1, num_ldmk, 24, 40),
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)


# sampler_url = (
#     "dmpv2://MultiMode/han.tang/data/train_data/fatigue/sampler/reproduce"
# )
# sampler_data = bucket_client.url_to_local(sampler_url)
# filelist_path= "adam-finetune-all-1213-balance_sampler_list_50.txt"
# filelist_path = "adam-finetune-all-1121-balance_sampler_list_50.txt"
# filelist_path= "adam-finetune-all-0103-balance_sampler_list_50.txt"
# filelist_path = os.path.join(sampler_data, filelist_path)

if local:
    # data_root = "/horizon-bucket/MultiMode/mm_algorithms_data/eye_status/data/cleaned_20230223_test/"
    # data_root = "/horizon-bucket/MultiMode_3/fatigue/train-data/0824-old/"
    # data_root = "/horizon-bucket/MultiMode_3/jiaqi.quan/fatigue/train-data/20230321/"
    # data_root = "/horizon-bucket/MultiMode_3/jiaqi.quan/fatigue/test-data/20230331/"
    data_root = (
        "/horizon-bucket/MultiMode_3/jiaqi.quan/fatigue/test-data/20230321/"
    )
else:
    data_root = "/cluster_home/HobotDataset/fatigue/20220824"

rec_paths = [
    "eye_status_da_0617_clean_20220614_train.rec",
    "eye_status_da_0624_clean_20220614_train.rec",
    "eye_status_da_20200702_clean_20220614_train.rec",
    "eye_status_da_20200703_clean_20220614_train.rec",
    "eye_status_h9_20210609_clean_20220614_train.rec",
    "eye_status_cd569_0204_clean_20220614_train.rec",
    "eye_status_e300_20220905_train.rec",
    "eye_status_s202_1_20220905_train.rec",
    "eye_status_s202_2_20220905_train.rec",
    "eye_status_s202_3_20220905_train.rec",
    "eye_status_20220329_train.rec",
    "eye_status_20220309_train.rec",
    "eye_status_20220104_train.rec",
    "eye_status_phone_smoke_20221228_train.rec",
    "eye_status_other_20221228_train.rec",
]
less_rec_paths = [
    "eye_status_da_0617_clean_20220614_train.rec",
    "eye_status_da_0624_clean_20220614_train.rec",
    "eye_status_da_20200702_clean_20220614_train.rec",
    "eye_status_da_20200703_clean_20220614_train.rec",
    "eye_status_h9_20210609_clean_20220614_train.rec",
    "eye_status_cd569_0204_clean_20220614_train.rec",
    "eye_status_e300_20220905_train.rec",
    "eye_status_s202_1_20220905_train.rec",
    "eye_status_s202_2_20220905_train.rec",
    "eye_status_s202_3_20220905_train.rec",
]


rec_paths = [os.path.join(data_root, rec_path) for rec_path in rec_paths]
# rec_paths = [os.path.join(data_root, rec_path) for rec_path in less_rec_paths]


# dataloader
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="EyeStatusDataset",
        rec_paths=rec_paths,
        transforms=[
            dict(
                # type="FlipEyeStatus",
                # px=0.5,
                type="RandomFlip",
                px=0.5,
            ),
            dict(
                type="RandomShiftRotateScale",
                rotate_prob=0.4,
                max_rotate_angle=10.0,
                shift_prob=0.4,
                resize=True,
                max_shift_range=(0.05, 0.05),
            ),
            dict(
                # type="ScaleCenterCrop",
                # target_shape=input_shape,
                # roi_scale=0.5,
                # scale=(0.9, 1.1),
                type="CropRecROI",
                crop_type="scale",
                target_shape=(96, 160, 3),
                # base_roi=[48, 80, 144, 240],
                base_roi=[80, 48, 240, 144],
                crop_jitter_range=0.1,
                center_shift_range=0,
            ),
            dict(
                type="GenerateGaussianHeatmap",
                num_ldmk=16,
                feat_stride=4,
                heatmap_shape=[24, 40],
            ),
            dict(
                type="GenerateGaussianVector",
                num_ldmk=16,
                feat_stride=4,
                vector_size=[40, 24],
            ),
            dict(
                type="ToTensor",
            ),
        ],
    ),
    sampler=dict(
        # type="DistributedEyeStatusFilelistSampler",
        type="DistributedEyeStatusBalanceSampler",
        # filelist_path="/cluster_home/HobotDataset/fatigue/sampler/train-sampler/balance_sampler_list.txt",
        # filelist_path=filelist_path,
        # init_epoch=0,
        drop_last=True,
        # shuffle=False,
    ),
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    pin_memory=True,
)

val_rec_paths = [
    "eye_status_da_0701_clean_20220614_train.rec",
    # "binocular_difference_20230331_test.rec",
    # "cleaned_20230223_test.rec"
    # "eye_status_small_eye_test_20221017_train.rec"
    # "narrow_easy.rec",
    # "eye_status_test_steer_1000_20221025_train.rec",
]
# val_rec_paths = [os.path.join(data_root, rec_path) for rec_path in rec_paths]
val_rec_paths = [
    os.path.join(data_root, rec_path) for rec_path in val_rec_paths
]

# dataloader
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="EyeStatusDataset",
        rec_paths=val_rec_paths,
        transforms=[
            dict(
                # type="ScaleCenterCrop",
                # target_shape=input_shape,
                # roi_scale=0.5,
                # scale=(1.0, 1.0),
                type="CropRecROI",
                crop_type="center",
                target_shape=(96, 160, 3),
                # base_roi=[48, 80, 144, 240],
                base_roi=[80, 48, 240, 144],
                # base_roi = [112, 186, 208, 346],
                crop_jitter_range=0.0,
            ),
            dict(
                type="GenerateGaussianHeatmap",
                num_ldmk=num_ldmk,
                feat_stride=4,
                heatmap_shape=[24, 40],
            ),
            dict(
                type="GenerateGaussianVector",
                num_ldmk=num_ldmk,
                feat_stride=4,
                vector_size=[40, 24],
            ),
            dict(
                type="ToTensor",
            ),
        ],
    ),
    sampler=dict(
        type=torch.utils.data.DistributedSampler,
        shuffle=False,
    ),
    batch_size=batch_size_per_gpu,
    num_workers=1,
)


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="ColorJitter",
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(1),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)


def update_metric(metrics, batch, model_outs):
    _, losses = model_outs
    for metric in metrics:
        metric.update(losses)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)

close_cls_ind = 0


def update_eval_metric(metrics, batch, model_outs):
    cls_target = batch["gt_eye_cls_labels"]
    l_cls_target, r_cls_target = torch.split(cls_target, 5, dim=1)
    cls_target = torch.cat([l_cls_target, r_cls_target], dim=0)

    # pred, loss = model_outs
    # l_feat = pred["l_feat"]
    # r_feat = pred["r_feat"]

    # l_bin_feat = pred["l_bin_feat"]
    # r_bin_feat = pred["r_bin_feat"]
    l_feat, r_feat, l_bin_feat, r_bin_feat = model_outs

    batch_size = l_feat.size()[0]
    pred = torch.cat([l_feat, r_feat], dim=0)
    pred = pred.view(batch_size * 2, -1)

    bin_pred = torch.cat([l_bin_feat, r_bin_feat], dim=0)
    bin_pred = bin_pred.view(batch_size * 2, -1)

    for metric in metrics:
        if isinstance(metric.name, list):
            if metric.name[0].find("eye_status_metrics") == 0:
                metric.update(cls_target, pred)
            elif metric.name[0].find("eye_status_mix_metrics") == 0:
                metric.update(cls_target, pred, bin_pred)

            elif metric.name[0].find("eye_status_bin_metrics") == 0:
                metric.update(cls_target, bin_pred)

        else:
            if metric.name.find("left_eye_acc") == 0:
                valid_inds = torch.sum(l_cls_target, axis=-1) != 0
                label_cls = torch.argmax(l_cls_target, -1).view(-1)[valid_inds]
                pred_cls = torch.argmax(
                    pred[
                        :batch_size,
                    ],
                    -1,
                )
                pred_cls = pred_cls.view(-1)[valid_inds]
                metric.update(label_cls, pred_cls)
            elif metric.name.find("right_eye_acc") == 0:
                valid_inds = torch.sum(r_cls_target, axis=-1) != 0
                label_cls = torch.argmax(r_cls_target, -1).view(-1)[valid_inds]
                pred_cls = torch.argmax(
                    pred[
                        batch_size:,
                    ],
                    -1,
                )
                pred_cls = pred_cls.view(-1)[valid_inds]
                metric.update(label_cls, pred_cls)
            else:
                metric.update(cls_target, pred)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_eval_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix="Validation" + task_name,
)
val_metric_updater["log_prefix"] = "Validation " + task_name

# predictor
float_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    "float-checkpoint-last.pth.tar",
                    # "float-checkpoint-step-499.pth.tar",
                ),
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="Accuracy",
            name="left_eye_acc",
        ),
        dict(
            type="Accuracy",
            name="right_eye_acc",
        ),
        dict(
            type="EyeStatusMetrics",
            num_classes=5,
            name="eye_status_metrics",
        ),
        dict(
            type="BinEyeStatusMetrics",
            bin_inds=[0, 3],
            sigmoid_thresh=0.5,
            name="eye_status_bin_metrics",
        ),
        dict(
            type="MixEyeStatusMetrics",
            num_classes=5,
            name="eye_status_mix_metrics",
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    "qat-checkpoint-last.pth.tar"
                    # "qat-checkpoint-step-999.pth.tar"
                    # ckpt_dir, "float-checkpoint-step-5999.pth.tar"
                ),
                ignore_extra=True,
                check_hash=False,
                # verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="Accuracy",
            name="left_eye_acc",
        ),
        dict(
            type="Accuracy",
            name="right_eye_acc",
        ),
        dict(
            type="EyeStatusMetrics",
            num_classes=5,
            name="eye_status_metrics",
        ),
        dict(
            type="BinEyeStatusMetrics",
            bin_inds=[0, 3],
            sigmoid_thresh=0.5,
            name="eye_status_bin_metrics",
        ),
        dict(
            type="MixEyeStatusMetrics",
            num_classes=5,
            narrow_thresh=0.5,
            close_thresh=0.5,
            open_thresh=0.5,
            close_thresh_close=0.5,
            name="eye_status_mix_metrics",
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
)

freeze_modules = [["backbone"], ["cls_head"], ["ldmk_neck"], ["ldmk_head"]]

freeze_callback = dict(
    type="FreezeModule",
    modules=freeze_modules,
    step_or_epoch=[0, 0, 0, 0],
    update_by="epoch",
    only_batchnorm=False,
)


ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    interval_by="step",
    save_interval=500,
    name_prefix=training_step + "-",
    strict_match=True,
    save_hash=False,
)
trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

# epoch_step = 153
epoch_step = 202
decay_epoch = [30, 35, 40, 45, 50]
lr_decay_id = [epoch_id * epoch_step for epoch_id in decay_epoch]

qat_epoch_step = 202
qat_decay_epoch = [10, 13, 15, 16, 17]
qat_lr_decay_id = [epoch_id * qat_epoch_step for epoch_id in qat_decay_epoch]


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    pretrain_dir,
                    "float-checkpoint-last.pth.tar",
                ),
                verbose=True,
                allow_miss=True,
            ),
        ],
    ),
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        # params={
        #     "backbone": dict(weight_decay=2e-4),
        #     "ldmk_neck": dict(weight_decay=0),
        #     "cls_head": dict(weight_decay=2e-4),
        #     "heatmap_head": dict(weight_decay=0),
        # },
        lr=0.015,
    ),
    batch_processor=batch_processor,
    find_unused_parameters=True,
    num_epochs=50,
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=500,
            update_by="step",
            warmup_by="step",
            warmup_len=epoch_step,
            lr_decay_id=lr_decay_id,
            lr_decay_factor=0.25,
        ),
        metric_updater,
        ckpt_callback,
        freeze_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="Loss"),
    ],
    val_metrics=[],
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                verbose=True,
                ignore_extra=True,
            ),
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    pretrain_dir, "qat-checkpoint-step-3999.pth.tar"
                ),
                ignore_extra=True,
                check_hash=False,
                allow_miss=True,
            ),
        ],
    ),
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=5e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=50,
    num_steps=1600,
    stop_by="epoch",
    device=None,
    find_unused_parameters=True,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=100,
            lr_decay_id=qat_lr_decay_id,
            lr_decay_factor=0.1,
        ),
        metric_updater,
        ckpt_callback,
        freeze_callback,
        # trace_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="Loss"),
    ],
    val_metrics=[],
)


int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    "qat-checkpoint-last.pth.tar",
                    # "qat-checkpoint-step-2999.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
)
