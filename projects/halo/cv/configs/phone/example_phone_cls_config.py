import copy
import os

import torch
from horizon_plugin_pytorch.march import March
from torch import nn

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

enable_model_tracking = True

task_name = "phone_classification"
num_classes = 3
batch_size_per_gpu = 32
device_ids = [0, 1, 2, 3]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
qat_mode = "fuse_bn"
input_size = 128
num_epochs = 40
lr = 0.01
warmup_len = 2
num_machines = 1
num_gpus_per_machine = len(device_ids)
max_jobtime = 16000


job_list = [
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/phone/example_phone_cls_config.py --stage float",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/phone/example_phone_cls_config.py --stage qat",  # noqa E501
    "python3 -W ignore tools/train.py --config projects/halo/cv/configs/phone/example_phone_cls_config.py --stage int_infer",  # noqa E501
]


model = dict(
    type="PhoneClassifier",
    backbone=dict(
        type="TinyVargNetV2",
        bn_kwargs={},
        num_classes=num_classes,
        alpha=1.0,
        group_base=4,
        flat_output=False,
        include_top=False,
    ),
    losses=nn.CrossEntropyLoss(),
    num_classes=num_classes,
)
deploy_model = dict(
    type="PhoneClassifier",
    backbone=dict(
        type="TinyVargNetV2",
        bn_kwargs={},
        num_classes=num_classes,
        alpha=1.0,
        group_base=4,
        flat_output=False,
        include_top=False,
    ),
    losses=None,
    num_classes=num_classes,
)
deploy_inputs = dict(img=torch.randn((1, 3, input_size, input_size)))

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)


train_rec_list = [
    "phone_pilot_positive_0628.rec",
    "phone_wechat_pilot_0701.rec",
    "phone_wechat_pilot_0702.rec",
    "phone_wechat_pilot_0703.rec",
    "phone_positive_negative_0723.rec",
    "phone_positive_negative_0724.rec",
    "phone_positive_negative_0725.rec",
    "noaugv1_changan11.rec",
    "touchshoulder.rec",
    "2020092122_small_image_train.rec",
    "cd569_0129_train.rec",
    "touch_face.rec",
    "2021_0521_pos_pack.rec",
    "h9_4w_batch2_labeled.rec",
    "random2w_batch1_labeled.rec",
    "a29_backhand.rec",
    "guangqi_train_data_batch0_labeled.rec",
]
val_rec_list = ["phone_pilot_positive_0628"]

all_rec_info_list = [
    "anno_phone_refresh_inter_newRule_recInfo_0605.json",
    "anno_phone_recInfo_0530_newRule.json",
]
all_rec_imgIdx_list = [
    "anno_phone_refresh_inter_newRule_summary_0605.json",
    "anno_phone_oldVal_summary_0524.json",
    "anno_phone_imgIdx_0530_newRule.json",
]

data_root_path = "/horizon-bucket/MultiMode_2/mm_algorithms_data/bing.li/backup/data-1/phone/datasets/phone_rec_0903"

all_rec_info_list = [
    os.path.join(data_root_path, i) for i in all_rec_info_list
]
all_rec_imgIdx_list = [
    os.path.join(data_root_path, i) for i in all_rec_imgIdx_list
]
train_rec_list = [os.path.join(data_root_path, i) for i in train_rec_list]
val_rec_list = [os.path.join(data_root_path, i) for i in val_rec_list]

mean = 128
std = 128

train_transforms = [
    dict(
        type="RandomRotateCrop",
        net_input_size=(128, 128),
        rot_prob=1.0,
        rot_angle_range=30,
        center_shift_prob=1.0,
        center_shift_range=0.01,
        norm_ratio=1.25,
        norm_method="longside_square",
        norm_jitter_range=0.25,
        net_target_size=(128, 128),
        base_len=1.0,
    ),
    dict(
        type="OneFromMultiple",
        transforms=[
            dict(
                type="RandomNoise",
                prob=1,
                min=-5,
                max=5,
            ),
            dict(
                type="GaussianNoise",
                prob=1,
                mean=0,
                sigma=1,
            ),
            dict(
                type="SaltPepperNoise",
                prob=1,
                s_ratio=0.05,
                p_ratio=0.05,
            ),
        ],
        probs=[1, 0.0, 0.0],
    ),
    dict(
        type="RandomFlip",
        px=1,
        py=0,
    ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=(-0.2, 0.2),
        contrast_limit=(-0.2, 0.2),
        brightness_by_max=True,
        p=0.5,
    ),
    dict(
        type="HueSaturationValue",
        hue_range=(-20, 20),
        sat_range=(-30, 30),
        val_range=(-20, 20),
        p=0.5,
    ),
    dict(type="ConvertDataType", convert_map={"img": "float32"}),
    dict(type="ToTensor", to_yuv=False),
]

val_transforms = [
    dict(
        type="RandomRotateCrop",
        net_input_size=(128, 128),
        rot_prob=0,
        rot_angle_range=0,
        center_shift_prob=0,
        center_shift_range=0,
        norm_ratio=1.25,
        norm_method="longside_square",
        norm_jitter_range=0,
        net_target_size=(128, 128),
        base_len=1.0,
    ),
    dict(
        type="ToTensor",
        to_yuv=True,
        use_yuv_v2=True,
    ),
]


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PhoneDataset",
        rec_list=train_rec_list,
        all_rec_info_list=all_rec_info_list,
        all_rec_imgIdx_list=all_rec_imgIdx_list,
        transforms=train_transforms,
        data_root_path=data_root_path,
        mode="train",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PhoneDataset",
        rec_list=val_rec_list,
        all_rec_info_list=all_rec_info_list,
        all_rec_imgIdx_list=all_rec_imgIdx_list,
        transforms=val_transforms,
        data_root_path=data_root_path,
        mode="val",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        # dict(type="TorchVisionAdapter", interface="RandomHorizontalFlip"),
        dict(
            type="TorchVisionAdapter",
            interface="ColorJitter",
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
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
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    target = batch["labels"]
    preds, losses = model_outs
    for metric in metrics:
        metric.update(target, preds)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=10,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

SavePhoneResult = dict(
    type="SavePhoneResult",
    output_dir=ckpt_dir,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[
        val_metric_updater,
        SavePhoneResult,
    ],
    val_model=None,
)

aidi_exp_model_callback = dict(
    type="AIDIExpModel",
    model_name=task_name,
    task_type="classification",
    save_model="best",
    platforms=["J5"],
)


find_unused_parameters = False
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-6)},
        lr=lr,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=warmup_len,
            step_log_interval=1000,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="Accuracy"),
        dict(type="PrecisionThresh", cls_id=1, thresh=0.7),
        dict(type="RecallThresh", cls_id=1, thresh=0.7),
        dict(type="PrecisionThresh", cls_id=2, thresh=0.7),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="PrecisionThresh", cls_id=1, thresh=0.7),
        dict(type="RecallThresh", cls_id=1, thresh=0.7),
        dict(type="PrecisionThresh", cls_id=2, thresh=0.7),
    ],
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-6)},
        lr=0.00001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[15, 25],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=2),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        # dict(type="TopKAccuracy", top_k=2),
        dict(type="PrecisionThresh", cls_id=1, thresh=0.7),
        dict(type="RecallThresh", cls_id=1, thresh=2),
        dict(type="PrecisionThresh", cls_id=2, thresh=0.7),
    ],
)

# just for saving int_infer pth and pt
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
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
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

# predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="PrecisionThresh", cls_id=1, thresh=0.7),
        dict(type="RecallThresh", cls_id=1, thresh=2),
        dict(type="PrecisionThresh", cls_id=2, thresh=0.7),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
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
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="PrecisionThresh", cls_id=1, thresh=0.7),
        dict(type="RecallThresh", cls_id=1, thresh=2),
        dict(type="PrecisionThresh", cls_id=2, thresh=0.7),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

int_infer_predictor = dict(
    type="Predictor",
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
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="PrecisionThresh", cls_id=1, thresh=0.7),
        dict(type="RecallThresh", cls_id=1, thresh=2),
        dict(type="PrecisionThresh", cls_id=2, thresh=0.7),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

k8s_config = dict(
    job_name=task_name,
    job_password="phone_123456",
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework="pytorch",
    task_label="halo_multitask",
    project_id="PDT2021005",
    input_bucket="MultiMode_2,MultiMode",
    output_bucket="MultiMode_2,MultiMode",
    priority=5,
    docker_image="docker.hobot.cc/imagesys/hat:halo-runtime-py38-cu111-torch1-102",  # with ablu and trackeval  # noqa
    # default 7200 = 5days
    max_jobtime=max_jobtime,
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        "../../hat",
        "../../plugins",
        "../../tools",
        "../../projects/halo/cv/",
        "url2IP.py",
        "ssh_launcher.py",
    ],
    job_list=job_list,
)
