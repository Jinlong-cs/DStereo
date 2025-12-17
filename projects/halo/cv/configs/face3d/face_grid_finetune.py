import copy
import os

import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from projects.halo.cv.configs.face3d.face3d_datahub import get_dataset

try:
    import lpips
except ImportError:
    raise ImportError("Please install lpips")
import warnings

warnings.filterwarnings("ignore")

DEBUG = True  # True #False

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "tmp_debug"
if DEBUG:
    batch_size_per_gpu = 16
    device_ids = [
        0,
    ]  # 2, 3]
else:
    batch_size_per_gpu = 48
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]


load_pth_path = "tmp_models/xxx/float-checkpoint-last.pth.tar"

ckpt_dir = "./tmp_models/%s" % task_name
log_dir = "./tmp_models/%s/log" % task_name
log_freq = 50
enable_tensorboard = True
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
# march = March.BERNOULLI2
march = March.BAYES

undistort = False


train_datasets = ["kinect120"]
val_datasets = ["kinect120"]

train_data = get_dataset(train_datasets)


# Model for training
model = dict(
    type="Face3dModel",
    modeltype="pp",
    ldmk_norm=True,
    undistort=undistort,
    use_flame=True,
    use_grid_sample=True,
    backbone=dict(
        type="VargNetV2",
        input_channels=3,
        num_classes=1000,
        include_top=False,
        disable_quanti_input=True,
        bn_kwargs={},
    ),
    head=dict(
        type="Face3dHead",
        kernel_size=4,
        in_channels=256,
    ),
    flame=dict(
        type="FLAME",
        flame_model_path="face3d_data/new_generic_model.pkl",
        flame_lmk_embedding_path="face3d_data/fined_landmark_embedding.npy",
    ),
    flame_tex=dict(
        type="FLAMETex",
        tex_path="face3d_data/FLAME_albedo_from_BFM.npz",
    ),
    renderer=dict(
        type="NVRenderer",
        obj_filename="face3d_data/head_template_mesh.obj",
        render_size=(256, 256),
    ),
    lpips=lpips.LPIPS(
        pretrained=True,
        pnet_rand=True,
        model_path="face3d_data/lpips.pth",
        verbose=True,
        net="vgg",
    ),
    loss_weights={
        "ldmk": 512,  # 256.0,
        "photo": 10,  # 10.0,
        "lpips": 30,  # 30.0,
        "shape_reg": 3e-2,
        "exp_reg": 1e-2,
        "tex_reg": 1e-2,
        "flame_transl_xy_reg": 128,  # 128,
        "flame_transl_z_reg": 640,  # 640,
    },
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=train_data["image_list"],
        mask_path_list=train_data["mask_list"],
        anno_path_list=train_data["anno_list"],
        transforms=[
            dict(
                type="SimpleNormGenGridMap",
                norm_ratio=1.2,
                expand_crop_hw=640,
                norm_method="longside_square",
            ),
            dict(type="SimpleNormPositionEncoding"),
            dict(type="ToTensor"),
        ],
        stage="finetune",
        modeltype="pp",
        is_gray=True,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=train_data["image_list"],
        mask_path_list=train_data["mask_list"],
        anno_path_list=train_data["anno_list"],
        transforms=[
            dict(
                type="SimpleNormGenGridMap",
                norm_ratio=1.2,
                expand_crop_hw=640,
                norm_method="longside_square",
            ),
            dict(type="SimpleNormPositionEncoding"),
            dict(type="ToTensor"),
        ],
        stage="finetune",
        modeltype="pp",
        is_gray=True,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=1,
    pin_memory=False,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        # dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(0),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        # dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(0),
)


def update_metric(metrics, batch, model_outs):
    for metric, loss in zip(metrics, model_outs):
        metric.update(loss)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=50,
)


ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)


# NOTE: use pretrain params from pretrain stage
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    # model_convert_pipeline=dict(
    #     type="ModelConvertPipeline",
    #     converters=[
    #         dict(
    #             type="LoadCheckpoint",
    #             checkpoint_path=load_pth_path,
    #             allow_miss=True,
    #             ignore_extra=True,
    #         ),
    #     ],
    # ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-4,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="epoch" if DEBUG else "epoch",
    num_epochs=1,
    num_steps=300,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            # warmup_by="epoch",
            # warmup_len=3,
            step_log_interval=100,
            lr_decay_id=[20, 35],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        ckpt_callback,
    ],
    sync_bn=False,
    train_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
    val_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
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
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-5,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="epoch",
    num_epochs=1,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=1,
            step_log_interval=100,
            lr_decay_id=[20, 35],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        ckpt_callback,
    ],
    sync_bn=False,
    train_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
    val_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
)


# Deploy Model for saving checkpoint
deploy_model = copy.deepcopy(model)
deploy_model["deploy"] = True

deploy_inputs = dict(
    img=torch.randn((1, 1, 640, 640)),
    grid_map=torch.randn((1, 2, 128, 128)),
    position_map=torch.randn((1, 2, 128, 128)),
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,  # deploy_trace_inputs,
)

int_model = copy.deepcopy(deploy_model)
int_model["cvrt2int"] = True
# just for saving int_infer pth and pt
int_infer_trainer = dict(
    type="Trainer",
    model=int_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
                ignore_extra=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=data_loader,
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
    name="dms_face_3d_pose_pp",
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "dms_face_3d_pose_pp.hbm"),
    layer_details=True,
    input_source=["ddr", "pyramid", "ddr"],
)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        if isinstance(metric.name, list) and len(metric.name) == 4:
            metric.update(batch, model_outs["pose"])
        elif "Eye3dMae_x_left" in metric.name:
            metric.update(batch, model_outs["left_eye_pred"])
        elif "Eye3dMae_x_right" in metric.name:
            metric.update(batch, model_outs["right_eye_pred"])
        else:
            print(metric.name)
            assert 0


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)
val_metric = [
    dict(type="PoseMAE", name=["Roll", "Pitch", "Yaw", "Mae"], modeltype="pp"),
    dict(
        type="Eye3dMAE",
        name=["Eye3dMae_x_left", "Eye3dMae_y_left", "Eye3dMae_z_left"],
    ),
    dict(
        type="Eye3dMAE",
        name=["Eye3dMae_x_right", "Eye3dMae_y_right", "Eye3dMae_z_right"],
    ),
]
float_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(deploy_model),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metric,
    log_interval=log_freq,
)

qat_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(deploy_model),
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
    callbacks=[val_metric_updater],
    metrics=val_metric,
    log_interval=log_freq,
)


int_infer_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(deploy_model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metric,
    log_interval=log_freq,
)
