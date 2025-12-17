import copy
import os

import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index

try:
    import lpips
except ImportError:
    raise ImportError("Please install lpips")
import warnings

warnings.filterwarnings("ignore")

DEBUG = True

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "vargnetv2_face3d_finetune_stage_id01"

if DEBUG:
    batch_size_per_gpu = 4
    device_ids = [1]
else:
    batch_size_per_gpu = 32
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = "./tmp_models/%s" % task_name
log_dir = "./tmp_models/%s/log" % task_name
log_freq = 50
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
# march = March.BERNOULLI2
march = March.BAYES

train_lmdb_list = [
    ["face_ldmk_lmdb/image_lmdb", "FacePublic_lmdb/image_lmdb"],
    ["face_ldmk_lmdb/mask_lmdb", "FacePublic_lmdb/mask_lmdb"],
    ["face_ldmk_lmdb/anno_lmdb", "FacePublic_lmdb/anno_lmdb"],
]

# Model for training
model = dict(
    type="Face3dModel",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        include_top=False,
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
        flame_lmk_embedding_path="face3d_data/new_landmark_embedding.npy",
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
        "ldmk": 256.0,
        "photo": 10.0,
        "lpips": 30.0,
        "shape_reg": 1e-3,
        "exp_reg": 1e-3,
        "tex_reg": 1e-3,
    },
)

# Deploy Model for saving checkpoint
deploy_model = copy.deepcopy(model)
deploy_model["deploy"] = True

deploy_inputs = dict(img=torch.randn((1, 3, 128, 128)))

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=train_lmdb_list[0],
        mask_path_list=train_lmdb_list[1],
        anno_path_list=train_lmdb_list[2],
        transforms=[
            dict(type="RandomFlip", px=0.0, py=0.0),
            dict(
                type="RandomRotateCrop",
                rot_prob=1.0,
                rot_angle_range=30.0,
                center_shift_prob=1.0,
                center_shift_range=0.025,
                norm_ratio=1.2,
                norm_method="longside_square",
                norm_jitter_range=0.1,
                net_input_size=(128, 128),
                net_target_size=(256, 256),
                base_len=200,
            ),
            dict(type="ToTensor"),
        ],
        stage="finetune",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=False,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=["val_lmdb/image_lmdb"],
        mask_path_list=["val_lmdb/mask_lmdb"],
        anno_path_list=["val_lmdb/anno_lmdb"],
        transforms=[
            dict(
                type="RandomRotateCrop",
                rot_prob=0.0,
                rot_angle_range=0.0,
                center_shift_prob=0.0,
                center_shift_range=0.0,
                norm_ratio=1.2,
                norm_method="longside_square",
                norm_jitter_range=0.0,
                net_input_size=(128, 128),
                net_target_size=(256, 256),
                base_len=200,
            ),
            dict(type="ToTensor"),
        ],
    ),
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
    loss_collector=collect_loss_by_index(0),
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
    loss_collector=collect_loss_by_index(0),
)


def update_metric(metrics, batch, model_outs):
    for metric, loss in zip(metrics, model_outs):
        metric.update(loss)

    # for metric in metrics:
    # metric.update(target, preds)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
)

# note: val_model couldn't set as None
# Use train `model` as val model will use copy.deepcopy
# can't pickle nvdiffrast_plugin.RasterizeGLStateWrapper objects
val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=model,
    val_on_train_end=False,
    log_interval=100,
)

loss_list = ["Loss", "Lmk", "Shape", "Exp", "Tex"]


def tb_update(writer, model_outs, global_step_id, **kwargs):
    loss = [ll.item() for ll in model_outs.values()]
    for t, loss in zip(loss_list, loss):
        writer.add_scalar(
            tag=t,
            scalar_value=float(loss),
            global_step=global_step_id,
        )


# NOTE: use pretrain params from pretrain stage
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir.replace("finetune", "pretrain"),
                    "float-checkpoint-last.pth.tar",
                ),
                allow_miss=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-3,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="step" if DEBUG else "epoch",
    num_epochs=40,
    num_steps=300,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=100,
            lr_decay_id=[15, 25],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
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
    ],
    val_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
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
        lr=1e-3,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="step" if DEBUG else "epoch",
    num_epochs=30,
    num_steps=300,
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
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
    ],
    val_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
    ],
    find_unused_parameters=False,
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
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
    callbacks=[ckpt_callback, trace_callback],
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
