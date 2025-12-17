import copy
import os
import warnings

import torch
from data_hub import (
    get_anno_path,
    get_anno_path_list,
    get_image_path,
    get_image_path_list,
)
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

is_local_train = not os.path.exists("/running_package")

VERSION = ConfigVersion.v2
enable_model_tracking = True
warnings.filterwarnings("ignore")
bn_kwargs = dict(eps=2e-5, momentum=0.1)
focal_length = 5000.0


training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "spin"
# mixvargenet 1.25
# pretrain_path = "http://fm-weixiang-yue.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3207039_mixvargenet125-imagenet-pretrain-20230628-123929/output/models/mixvargenet125_cls/float-checkpoint-best-6dfc25f9.pth.tar" # noqa
# mixvargenet 1.5
pretrain_path = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/weixiang.yue/plat_gpu/hobot-dag-3226110_mixvargenet150-imagenet-pretrain-20230629-220344/output/models/mixvargenet125_cls/float-checkpoint-best-b09fa351.pth.tar"  # noqa
# resnet50
# pretrain_path = "HAT-resnet50.pth.tar"

if is_local_train:
    batch_size_per_gpu = 16
    device_ids = [0, 1, 2, 3]
    ckpt_dir = "./tmp_models/%s" % task_name
    if not os.path.isdir("tmp_models"):
        os.makedirs("./tmp_models")
else:
    batch_size_per_gpu = 128
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    ckpt_dir = "/job_data/models/%s" % task_name
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)

log_freq = 25
num_epoch = 60
qat_num_epoch = 10
LR = 1e-4
QAT_LR = 1e-6
INPUT_RES = 224
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
qat_mode = "fuse_bn"
sync_bn = False
LDMK_PAIRS = [
    [0, 5],
    [1, 4],
    [2, 3],
    [6, 11],
    [7, 10],
    [8, 9],
    [20, 21],
    [22, 23],
]

SMPL_POSE_PAIRS = [
    [1, 2],
    [4, 5],
    [7, 8],
    [10, 11],
    [13, 14],
    [16, 17],
    [18, 19],
    [20, 21],
    [22, 23],
]

test_inputs = dict(img=torch.randn((1, 3, INPUT_RES, INPUT_RES)))
# public dataset list
public_dataset_list = [
    "mpi-inf-3dhp",
    "mpii",
    "lsp-orig",
    "lspet",
    "h36m",
    "coco",
]
# cockpit dataset list
cockpit_dataset_list = [
    "h9_train_01",
    "changan_202003_RGB",
    "MMGestureBatch_01",
    "MMGestureBatch_02",
    "MMGestureBatch_03",
    "MMGestureBatch_04",
    "TrainValsetCD569_trainset",
    "A11_RGB_IMS_trainset_01",
    "A11_RGB_IMS_trainset_02",
    "A11_RGB_IMS_trainset_03",
    "A11_RGB_IMS_trainset_04",
    "A11_RGB_IMS_trainset_05",
    "A11_RGB_IMS_trainset_06",
    "A11_RGB_IMS_trainset_07",
    "CD569_nonproduct_4ways_trainset_01",
    "CD569_nonproduct_4ways_trainset_02",
    "CD569_nonproduct_4ways_trainset_03",
    "CD569_nonproduct_4ways_trainset_04",
    "CD569_RGB_trainset_01",
    "A11_RGB_RMS_trainset_01",
    "A11_RGB_RMS_trainset_02",
    "A11_RGB_RMS_trainset_03",
    "A11_RGB_RMS_trainset_04",
    "A11_RGB_RMS_trainset_05",
    "A11_RGB_RMS_trainset_06",
    "A11_RGB_RMS_trainset_07",
    "C281_IMS_trainset_01",
    "C281_gesture_cockpit_trainset_01",
    "C281_gesture_cockpit_trainset_02",
    "C281_gesture_cockpit_trainset_03",
    "T18_gesture_cockpit_trainset_01",
    "T18_gesture_cockpit_trainset_02",
    "S311_gesture_cockpit_trainset_01",
    # child
    "chengdu_s311",
    # aiot
    # "xiaodu",
    # "lingkang",
]

workshop_dataset_list = [
    "20230629",
    "20230626",
    "20230621",
    "20230620",
    "20230619",
    "20230613",
    "20230609",
    "20230608",
    "20230531",
    "20230530",
    "20230522",
    "20230518",
    "20230516",
    "20230515",
    "20230512",
    "20230511",
    "20230510",
    "20230509",
    "20230508",
    "20230506",
    "20230505",
    "20230504",
]
# train dataset list
# train_dataset_list = public_dataset_list + cockpit_dataset_list
# train_dataset_list = cockpit_dataset_list + workshop_dataset_list
# train_dataset_list = cockpit_dataset_list
train_dataset_list = workshop_dataset_list
val_dataset_list = [
    # "h36m_valid_protocol1",
    # "h36m_valid_protocol2",
    # "mpi_inf_3dhp_valid",
    # "3dpw_test",
    # "C281_gesture_cockpit_valset_01",
    # "C281_gesture_cockpit_valset_02",
    # "CD569_nonproduct_4ways_valset",
    # "MMGestureBatch_val",
    # "chengdu_s311_testset_04",
    "20230630",
]
val_desc_list = [
    # "h36m_p1",
    # "h36m_p2",
    # "mpi_inf_3dhp",
    # "3dpw",
    # "C281_gesture_cockpit_valset_01",
    # "C281_gesture_cockpit_valset_02",
    # "CD569_nonproduct_4ways_valset",
    # "MMGestureBatch_val",
    # "chengdu_s311_testset_04",
    "20230630",
]

# transform
train_transforms = [
    dict(
        type="CropRecROI",
        crop_type="scale",
        target_shape=(INPUT_RES, INPUT_RES, 3),
        base_roi=[48, 48, 192, 192],
        crop_jitter_range=0.25,
        random_type="uniform",
    ),
    dict(
        type="RandomShiftRotateScale",
        max_rotate_angle=30,
        rotate_prob=0.4,
        resize=True,
    ),
    dict(
        type="RandomFlip",
        px=0.5,
    ),
    dict(
        type="GaussianNoise",
        prob=0.2,
        mean=0,
        sigma=2,
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
        type="RandomNoise",
        prob=0.2,
        min=-5,
        max=5,
    ),
    dict(
        type="NormalizeLdmk",
        norm_scale=INPUT_RES,
    ),
    dict(type="ToTensor"),
]

val_transforms = [
    dict(
        type="CropRecROI",
        crop_type="center",
        target_shape=(INPUT_RES, INPUT_RES, 3),
        base_roi=[48, 48, 192, 192],
        crop_jitter_range=0,
        random_type="uniform",
    ),
    dict(
        type="NormalizeLdmk",
        norm_scale=INPUT_RES,
    ),
    dict(type="ToTensor"),
]

# datasets
anno_label = "pseudo_anno2_lmdb"
train_dataset = dict(
    type="Human3dMixedDataset",
    dataset_list=train_dataset_list,
    img_path_list=get_image_path_list(train_dataset_list),
    anno_path_list=get_anno_path_list(train_dataset_list, anno_label),
    transforms=train_transforms,
    ldmk_pairs=LDMK_PAIRS,
    smpl_pose_pairs=SMPL_POSE_PAIRS,
    ignore_smpl=False,
    ignore_3d=False,
)

val_datasets = [
    dict(
        type="Human3dDataset",
        dataset=ds_name,
        image_path=get_image_path(ds_name),
        anno_path=get_anno_path(ds_name, anno_label),
        ldmk_pairs=LDMK_PAIRS,
        smpl_pose_pairs=SMPL_POSE_PAIRS,
        transforms=val_transforms,
        ignore_smpl=False,
        ignore_3d=True,
    )
    for ds_name in val_dataset_list
]

# fits dict
FITS_DIR = "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/static_fits_kid_template"  # noqa
fits_dict = dict(
    type="FitsDict",
    load_dir=FITS_DIR,
    checkpoint_dir=ckpt_dir,
    dataset_list=train_dataset_list,
)

smpl_model = dict(
    type="ExtentedSMPL",
    model_path="spin_params/data/smpl",
    batch_size=batch_size_per_gpu,
    create_transl=False,
    age="kid",
    kid_template_path="spin_params/data/smpl_kid_template.npy",
    joint_regressor_train_extra="spin_params/data/J_regressor_extra.npy",
)

mixvargenet_075_config = [
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=24,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=24,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=24,
            out_channels=48,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=48,
            out_channels=64,
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
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=128,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

mixvargenet_10_config = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
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
            in_channels=32,
            out_channels=32,
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
            in_channels=32,
            out_channels=64,
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
            in_channels=64,
            out_channels=96,
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
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

mixvargenet_125_config = [
    [
        MixVarGENetConfig(
            in_channels=40,
            out_channels=40,
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
            in_channels=40,
            out_channels=40,
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
            in_channels=40,
            out_channels=96,
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
            in_channels=96,
            out_channels=160,
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
            in_channels=160,
            out_channels=192,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

mixvargenet_150_config = [
    [
        MixVarGENetConfig(
            in_channels=48,
            out_channels=48,
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
            in_channels=48,
            out_channels=48,
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
            in_channels=48,
            out_channels=128,
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
            in_channels=128,
            out_channels=192,
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
            in_channels=192,
            out_channels=240,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]
mixvargenet_model = dict(
    type="HMR",
    backbone=dict(
        type="MixVarGENet",
        net_config=mixvargenet_150_config,
        input_channels=3,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    head=dict(
        type="HMRHead",
        smpl_mean_params="spin_params/data/smpl_mean_params.npz",
        n_iter=3,
        in_channels=240,
    ),
)

resnet_model = dict(
    type="HMR",
    backbone=dict(
        type="ResNet50",
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        bias=False,
        include_top=False,
    ),
    head=dict(
        type="HMRHead",
        smpl_mean_params="spin_params/data/smpl_mean_params.npz",
        n_iter=3,
    ),
)

# Train Model
model = dict(
    type="SPIN",
    mode="train",
    model=mixvargenet_model,
    smpl=smpl_model,
    human3d_loss=dict(
        type="Human3dLoss",
        loss_weights={
            "shape_loss_weight": 0 * 60,
            "keypoint_loss_weight": 5 * 60,
            "keypoint3d_loss_weight": 5 * 60,
            "pose_loss_weight": 1.0 * 60,
            "beta_loss_weight": 0.1 * 60,
            "cam_loss_weight": 1.0 * 60,
        },
        disable_hips=False,
    ),
    smplify=dict(
        type="SMPLify",
        camera_step_size=1e-2,
        body_step_size=1e-2,
        prior_folder="spin_params/data",
        camera_num_iters=100,
        body_num_iters=100,
        smpl_model=smpl_model,
        focal_length=focal_length,
        loss_weights={
            "pose_prior_weight": 4.78,
            "shape_prior_weight": 5,
            "angle_prior_weight": 15.2,
        },
    ),
    img_res=INPUT_RES,
    focal_length=focal_length,
    run_smplify=False,
    use_weakproj=False,
    gt_train_weight=1.0,
    fits_dict=fits_dict,
)

# Val Model
val_model = dict(
    type="SPIN",
    mode="val",
    model=mixvargenet_model,
    smpl=None,
    human3d_loss=None,
    smplify=None,
    img_res=INPUT_RES,
    focal_length=focal_length,
    run_smplify=False,
    use_weakproj=False,
    gt_train_weight=1.0,
    fits_dict=None,
)

deploy_model = copy.deepcopy(val_model)
deploy_model["mode"] = "deploy"

deploy_inputs = dict(img=torch.randn((1, 3, INPUT_RES, INPUT_RES)))

# dataloader
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


val_data_loaders = [
    dict(
        type=torch.utils.data.DataLoader,
        dataset=val_dataset,
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    for val_dataset in val_datasets
]

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
    loss_collector=collect_loss_by_regex("loss"),
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
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(batch, model_outs)


val_metric_updater_list = []
for val_desc in val_desc_list:
    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=[
            dict(
                type="Human3dMetric",
                name="PA-MPJPE",
                dataset_name=val_desc,
                smpl_model_dir="spin_params",
                use_pa=True,
            ),  # noqa
            dict(
                type="Human3dMetric",
                name="MPJPE",
                dataset_name=val_desc,
                smpl_model_dir="spin_params",
                use_pa=False,
            ),  # noqa
        ],
        metric_update_func=update_val_metric,
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix="Validation " + val_desc + " " + task_name,
    )
    val_metric_updater_list.append(val_metric_updater)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_callback = dict(
    type="StepDecayLrUpdater",
    lr_decay_id=[25, 45],
    lr_decay_factor=0.1,
    step_log_interval=100,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    interval_by="epoch",
    strict_match=False,
    best_refer_metric=None,
    save_on_train_end=True,
    # mode="min",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loaders,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updater_list,
    interval_by="epoch",
    val_interval=1,
    val_model=val_model,
    val_on_train_end=True,
    log_interval=100,
)

qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback.update(
    dict(
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=[
                dict(type="Float2QAT"),
            ],
        )
    )
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)


def state_dict_update(state_dict):
    key_list = [key for key in state_dict.keys()]  # noqa
    for key in key_list:
        if "smpl" in key or "init" in key:
            state_dict.pop(key)
    return state_dict


def pretrain_state_dict_update(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict["model." + k] = v
    return new_state_dict


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrain_path,
                state_dict_update_func=pretrain_state_dict_update,  # use for mixvargenet pretrain
                allow_miss=True,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=0)},
        lr=LR,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epoch,
    stop_by="epoch",
    device=None,
    sync_bn=sync_bn,
    callbacks=[
        stat_callback,
        metric_updater,
        ckpt_callback,
        lr_callback,
        # val_callback,
        # trace_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="keypoint_loss"),
        dict(type="LossShow", name="keypoint_3d_loss"),
        dict(type="LossShow", name="regr_pose_loss"),
        dict(type="LossShow", name="regr_betas_loss"),
        dict(type="LossShow", name="shape_loss"),
        dict(type="LossShow", name="regr_cam_loss"),
    ],
)


# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_data_loader["dataset"]["transforms"] = train_transforms
calibration_batch_processor = copy.deepcopy(batch_processor)
calibration_step = 400


calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_calibration_observer="min_max",
        ),
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                state_dict_update_func=state_dict_update,
                allow_miss=True,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
            dict(type="Float2Calibration"),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        qat_val_callback,
        ckpt_callback,
    ],
    log_interval=calibration_step / 5,
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0.0,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1.0,
            ),
        ),
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
                state_dict_update_func=state_dict_update,
                allow_miss=True,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=0)},
        lr=QAT_LR,
    ),
    batch_processor=batch_processor,
    num_epochs=qat_num_epoch,
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[5, 8],
            lr_decay_factor=0.1,
            step_log_interval=100,
        ),
        metric_updater,
        # qat_val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="keypoint_loss"),
        dict(type="LossShow", name="keypoint_3d_loss"),
        dict(type="LossShow", name="regr_pose_loss"),
        dict(type="LossShow", name="regr_betas_loss"),
        dict(type="LossShow", name="shape_loss"),
        dict(type="LossShow", name="regr_cam_loss"),
    ],
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
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
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
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                state_dict_update_func=state_dict_update,
                allow_miss=True,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=val_data_loaders,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updater_list,
    log_interval=100,
    share_callbacks=False,
)

qat_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(val_model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
                state_dict_update_func=state_dict_update,
                allow_miss=True,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=val_data_loaders,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updater_list,
    log_interval=100,
    share_callbacks=False,
)
