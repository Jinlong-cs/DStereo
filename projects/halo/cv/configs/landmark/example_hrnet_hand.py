import copy
import os
import warnings

import torch
import yaml
from datasets.hand_landmark import get_datasets
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex

warnings.filterwarnings("ignore")
DEBUG = True
NUM_LDMK = 21
LDMK_PAIRS = []


training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "hrnet-w32_hand_ldmk_heatmap_id0"

if DEBUG:
    batch_size_per_gpu = 40
    device_ids = [0, 1]
else:
    batch_size_per_gpu = 32
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
if not os.path.isdir("tmp_models"):
    os.makedirs("./tmp_models")
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BERNOULLI2

path = os.path.dirname(os.path.abspath(__file__))
hrnet_config = dict(yaml.load(open(f"{path}/hrnet/hrnet.yaml", "r")))

# Train Model
model = dict(
    type="LdmkModel",
    backbone=dict(type="HRNet", config=hrnet_config["hrnet_w64"]),
    mode="train",
    decoder=None,
    vector_head=None,
    coords_head=None,
    feat_stride=4,
    heatmap_head=dict(
        type="LdmkHeatmapHead",
        in_channels=960,
        num_ldmk=NUM_LDMK,
        is_train=True,
        loss_func=dict(type="LdmkLoss", loss_type="l2"),
    ),
    cls_head=None,
    loss_weights={"heatmap": 1.0},
)

# Val Model
val_model = dict(
    type="LdmkModel",
    backbone=dict(type="HRNet", config=hrnet_config["hrnet_w64"]),
    mode="val",
    decoder=None,
    vector_head=None,
    coords_head=None,
    feat_stride=4,
    heatmap_head=dict(
        type="LdmkHeatmapHead",
        in_channels=960,
        num_ldmk=NUM_LDMK,
        is_train=False,
    ),
    cls_head=None,
)

# Deploy Model
deploy_model = copy.deepcopy(val_model)
deploy_model["mode"] = "deploy"

# transform
train_transforms = [
    dict(type="RandomFlip", px=0.5),
    dict(
        type="RandomRotateCrop",
        net_input_size=(256, 256),
        rot_prob=0.6,
        rot_angle_range=90,
        center_shift_prob=0,
        center_shift_range=0,
        norm_ratio=1.25,
        norm_method="longside_square",
        norm_jitter_range=0.2,
    ),
    dict(
        type="GaussianNoise",
        prob=0.2,
        mean=0,
        sigma=2,
    ),
    dict(
        type="RandomNoise",
        prob=0.2,
        min=-5,
        max=5,
    ),
    dict(
        type="SaltPepperNoise",
        prob=0.2,
        s_ratio=0.05,
        p_ratio=0.05,
    ),
    dict(
        type="GaussianBlur",
        p=0.2,
        kernel_size_min=3,
        kernel_size_max=9,
        sigma_min=1.0,
        sigma_max=5.0,
    ),
    dict(
        type="MotionBlur",
        p=0.2,
        length_min=2,
        length_max=10,
        angle_min=1,
        angle_max=359,
    ),
    dict(
        type="RandomGray",
        p=0.3,
    ),
    dict(
        type="RandomOcclusion",
        prob=0.5,
        occ_type="whole",
        occ_num=1,
        size_ratio=0.7,
    ),
    dict(
        type="GenerateGaussianHeatmap",
        num_ldmk=NUM_LDMK,
        feat_stride=4,
        heatmap_shape=(64, 64),
        sigma=2,
        encoding_method="unbiased",
    ),
    dict(type="ToTensor"),
]
val_transforms = [
    dict(
        type="RandomRotateCrop",
        net_input_size=(256, 256),
        rot_prob=0.0,
        rot_angle_range=0,
        center_shift_prob=0,
        center_shift_range=0,
        norm_ratio=1.25,
        norm_method="longside_square",
        norm_jitter_range=0.0,
    ),
    dict(type="ToTensor"),
]


# datasets
train_datasets = get_datasets(
    [
        # external data
        "01_data_shujutang_v2",
        "03_data_shujutang",
        "03_data_shujutang_v2",
        # stereo in car
        "606579_1_stereo_incar_01",
        "606580_1_stereo_incar_01",
        "610592_1_stereo_incar_02",
        "610593_1_stereo_incar_02",
        "611358_1_stereo_incar_train_05",
        "611359_1_stereo_incar_train_06",
        "611360_1_stereo_incar_train_07",
        "611361_1_stereo_incar_train_08",
        "611028_1_stereo_incar_train_09",
        "611029_1_stereo_incar_train_10",
        "611030_1_stereo_incar_train_11",
        "611031_1_stereo_incar_train_12",
        "611032_1_stereo_incar_train_13",
        "611033_1_stereo_incar_train_14",
        "hand_kps_multi_view_in_car_train_cam_15",
        "hand_kps_multi_view_in_car_train_cam_16",
        "hand_kps_multi_view_in_car_train_cam_17",
        "hand_kps_multi_view_in_car_train_cam_18",
        "hand_kps_multi_view_in_car_train_cam_19",
        "hand_kps_multi_view_in_car_train_cam_20",
        "hand_kps_multi_view_in_car_train_cam_21",
        "hand_kps_multi_view_in_car_train_cam_22",
        # stereo_incar_train_01 & stereo_incar_train_02 与上面的重复
        # public
        "cmu_hand_kps_train",
        "coco_wholebody_train_v1.0",
        "cvpr_2019_augments_coco_train",
        "FreiHand_train",
        "FreiHand_train_v2",
        "GANeratedHands_train",
        "RHD_train",
        # AIOT
        "hand_kps_aiot_badcase_train_01",
        "hand_kps_x3_train_01",
        "hand_kps_x3_train_child",
        # C281
        "hand_kps_C281_common_gesture_train_01",
        "hand_kps_C281_common_gesture_train_02",
        "hand_kps_C281_common_gesture_train_03",
        "hand_kps_C281_common_gesture_train_04",
        "hand_kps_C281_common_gesture_train_05",
        "hand_kps_C281_common_gesture_train_06",
        "hand_kps_C281_common_gesture_train_07",
        "hand_kps_C281_common_gesture_train_08",
        "hand_kps_C281_finger-rot_gesture_train_01",
        "hand_kps_C281_special_gesture_train_01",
        "hand_kps_C281_special_gesture_train_02",
        "hand_kps_C281_special_gesture_train_03",
        "hand_kps_C281_special_gesture_train_04",
        # CD569
        "hand_kps_CA_supplement_train",
        "hand_kps_cd569_badcase_01",
        "hand_kps_cd569_badcase_train_02",
        "hand_kps_cd569_badcase_train_03",
        "hand_kps_cd569_normal_01",
        "hand_kps_in_car_rgb_train_18k",
        "hand_kps_in_car_rgb_train_part1",
        "hand_kps_in_car_rgb_train_part2",
        # E300
        "hand_kps_E300_common_gesture_train_01",
        "hand_kps_E300_common_gesture_train_02",
        # guangqi
        "hand_kps_guangqi_train_01",
        # H9
        "hand_kps_h9_ir_train_01",
        "hand_kps_h9_ir_train_02",
        "hand_kps_h9_ir_train_03",
        "hand_kps_h9_ir_train_04",
        "hand_kps_h9_ir_train_05",
        # T18
        "hand_kps_T18_common_gesture_train_01",
        "hand_kps_T18_common_gesture_train_02",
        "hand_kps_T18_common_gesture_train_03",
        "hand_kps_T18_common_gesture_train_04",
        # ToF
        "hand_kps_tof_ir_train_01",
        # multi-view in lab
        "multi_cam_matting_train_01",
        "multi_cam_matting_train_02",
        "multi_cam_matting_train_03",
        "multi_cam_matting_train_04",
        "multi_cam_matting_train_05",
        "multi_cam_matting_train_06",
        "multi_cam_matting_train_07",
        "multi_cam_matting_train_08",
        # realsense
        "realsense_rgbd_matting_train_02",
    ]
)

val_datasets = get_datasets(["hand_kps_in_car_rgb_test"])

# dataloader
deploy_inputs = dict(img=torch.randn((1, 3, 256, 256)))
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="LdmkDataset",
        image_lmdb_list=train_datasets["image_lmdb_list"],
        anno_lmdb_list=train_datasets["anno_lmdb_list"],
        num_ldmk=NUM_LDMK,
        task_type="hand",
        data_type="lmdb",
        ldmk_pairs=LDMK_PAIRS,
        transforms=train_transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=5,
    pin_memory=False,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="LdmkDataset",
        image_lmdb_list=val_datasets["image_lmdb_list"],
        anno_lmdb_list=val_datasets["anno_lmdb_list"],
        num_ldmk=NUM_LDMK,
        task_type="hand",
        data_type="lmdb",
        ldmk_pairs=LDMK_PAIRS,
        transforms=val_transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=False,
)


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("total_loss"),
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
    for metric, key in zip(metrics, model_outs):
        metric.update(model_outs[key])


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix="Validation" + task_name,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    save_hash=False,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=val_model,
    val_on_train_end=False,
)


float_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0.00001)},
        lr=1e-3,
    ),
    batch_processor=batch_processor,
    num_epochs=80,
    num_steps=20,
    stop_by="epoch" if DEBUG else "epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[60, 70],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        # val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="LdmkLoss"),
        dict(type="LossShow", name="Loss"),
    ],
    val_metrics=[],
)
float_solver = dict(
    trainer=float_trainer,
    quantize=False,
    allow_not_init=True,
    strict_match=True,
    pretrain_checkpoint="hrnetv2_w32_imagenet_pretrained.pth",
    allow_miss=True,
    ignore_extra=True,
)
