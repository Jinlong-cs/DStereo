import os

import numpy as np
import torch
import yaml

try:
    from mmcv.cnn.bricks.transformer import MultiheadAttention
except ImportError:
    MultiheadAttention = None
from common import backbone
from horizon_plugin_pytorch.march import March

from hat.data.collates.collates import default_collate_v2
from hat.engine.processors.loss_collector import collect_loss_by_regex

if os.environ.get("CLUSTER"):
    ckpt_dir = "/job_data/work_dir"
    local_train = False
else:
    ckpt_dir = "./work_dir"
    local_train = True

log_dir = os.path.join(ckpt_dir, "logs")
trainer_type = "distributed_data_parallel_trainer"

checkpoint_path = "http://fm-jin-yang.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3545859_sparse4d-only-front-real3d-id040-20230728-143620/output/work_dir/checkpoint-last-64ad4148.pth.tar"  # noqa

remote_debug = False
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES
convert_mode = "fx"
job_name = "sparse4d_mixvargnet_real3d_front"
enable_amp = False

num_gpus = 8 if not remote_debug else 2
device_ids = list(range(num_gpus))
batch_size = 4 if not local_train else 2
num_iters_per_epoch = 5000
num_epochs = 10
checkpoint_epoch_interval = 1

num_cams = 3
num_classes = 1
embed_dims = 256
num_groups = 8
num_single_frame_decoder = 1
num_decoder = 6
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
model = dict(
    type="Sparse4D",
    use_deformable_func=use_deformable_func,
    backbone=backbone,
    neck=dict(
        type="Unet",
        in_strides=[2, 4, 8, 16, 32, 64],
        out_strides=strides,
        stride2channels={2: 32, 4: 32, 8: 64, 16: 96, 32: 160, 64: 320},
        out_stride2channels={
            4: embed_dims,
            8: embed_dims,
            16: embed_dims,
            32: embed_dims,
            64: embed_dims,
        },
        group_base=32,
        factor=2,
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="/horizon-bucket/mono_3d_data/jin.yang/sparse4d/real3d_kmeans900_300_batch.npy",  # noqa E501
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            # num_temp_instances=600,
            # confidence_decay=0.6,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            embed_dims=embed_dims,
            vel_dims=-1,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ]
        * num_single_frame_decoder
        + [
            "temp_interaction",
            "interaction",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ]
        * (num_decoder - num_single_frame_decoder),
        temp_instance_interaction=dict(
            type=MultiheadAttention,
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            dropout=0.1,
        ),
        instance_interaction=dict(
            type=MultiheadAttention,
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            dropout=0.1,
        ),
        norm_layer=dict(type=torch.nn.LayerNorm, normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            activate=dict(type="ReLU", inplace=True),
            embed_dims=embed_dims,
            num_fcs=2,
            ffn_drop=0.1,
            pre_norm=True,
            in_channels=embed_dims * 2,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            use_deformable_func=use_deformable_func,
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,
            attn_drop=0.15,
            residual_mode="cat",
            use_camera_embed=True,
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=0,
                fix_scale=[
                    [0, -0.5, 0],
                    [0.45, -0.5, 0],
                    [-0.45, -0.5, 0],
                    [0, -0.95, 0],
                    [0, 0, 0],
                    [0, -0.5, 0.45],
                    [0, -0.5, -0.45],
                    [0.25, -0.5, 0],
                    [-0.25, -0.5, 0],
                    [0, -0.75, 0],
                    [0, -0.25, 0],
                    [0, -0.5, 0.25],
                    [0, -0.5, -0.25],
                ],
                embed_dims=embed_dims,
                rot_axis=1,
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            output_dim=8,
        ),
        target=dict(
            type="SparseBox3DTarget",
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 2,
        ),
        loss_cls=dict(
            type="FocalLoss",
            loss_name="loss_cls",
            num_classes=num_classes + 1,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(type="L1Loss", loss_weight=0.25),
        gt_cls_key="gt_labels_3d",
        gt_reg_key="gt_bboxes_3d",
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 5,
    ),
)

yaml_path = os.path.join(os.path.dirname(__file__), "dataset.yaml")
dataset_dict = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
data_paths = dataset_dict["real3d"]
data_paths = list(map(lambda x: f"/horizon-bucket/{x}", data_paths))
if local_train:
    import random

    random.shuffle(data_paths)
    data_paths = data_paths[:1]

num_classes = 1
num_dist = 8
max_objs = 100
select_sample = False
track_params = [0, 0, 0, 1]
view = "front"

transforms = [
    dict(
        type="Real3DDatasetAdaptor",
        num_classes=num_classes,
    ),
    dict(
        type="RepeatKeys",
        keys=(
            "imgs",
            "identity_trans_matrix",
            "cam_intrinsic",
            "cam_distcoeffs",
        ),
        repeat_times=num_cams,
    ),
    dict(
        type="ResizeCropFlipImage",
        transform_matrix_key=["cam_intrinsic"],
        data_aug_config_list=[
            dict(resize=0.25, crop=(0, 0, 960, 512)),
            dict(resize=0.5, crop=(448, 189, 448 + 960, 189 + 512)),
            dict(resize=1.0, crop=(1408, 827, 1408 + 960, 827 + 512)),
        ],
    ),
    dict(type="MVT4DImgFormat"),
    dict(type="BgrToYuv444", rgb_input=False),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
    dict(
        type="Sparse4DAdaptor",
        projection_key="identity_trans_matrix",
    ),
    dict(
        type="FixLengthPad",
        keys=("gt_bboxes_3d", "gt_labels_3d"),
        lengths=300,
        constant_values=-99,
    ),
    dict(
        type="ConvertDataType",
        convert_map=dict(
            gt_bboxes_3d=np.float32,
            gt_labels_3d=np.int64,
        ),
    ),
    dict(
        type="MultiViewCollect3D",
        keep_keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_labels_3d",
            "gt_bboxes_3d",
            "img_metas",
            "gt_depth",
            "focal",
            "cam_intrinsic",
            "cam_distcoeffs",
        ],
        img_metas_keys=["timestamp", "T_global", "T_global_inv"],
    ),
]

dataset = dict(
    type="Real3DDatasetRec",
    paths=data_paths,
    num_classes=num_classes,
    transforms=transforms,
    select_sample=select_sample,
    num_dist=num_dist,
    view=view,
    track_params=track_params,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    num_workers=min(batch_size, 12),
    batch_size=batch_size,
    sampler=dict(
        type=torch.utils.data.distributed.DistributedSampler,
        shuffle=True,
    ),
    dataset=dataset,
    collate_fn=default_collate_v2,
)

lr = 3e-4
optimizer = dict(
    type=torch.optim.AdamW,
    lr=lr,
    weight_decay=0.001,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[],
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
    grad_scaler=dict(
        type=torch.cuda.amp.GradScaler,
        init_scale=32.0,
        growth_interval=int(1e8),
    )
    if enable_amp
    else None,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=200,
    batch_size=batch_size,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    interval_by="step",
    save_interval=num_iters_per_epoch * checkpoint_epoch_interval,
    save_on_train_end=True,
)

lr_callback = dict(
    type="CosLrUpdater",
    warmup_by="step",
    warmup_len=500,
    warmup_begin_lr=lr / 3,
    step_log_interval=200,
    max_steps=num_iters_per_epoch * num_epochs,
    stop_lr=lr * 1e-3,
)

grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[("backbone", 0.5)],
    clip_grad_norm=25.0,
    clip_norm_type=2,
)
tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir if local_train else "/job_tboard/",  # noqa
    update_freq=50,
)

loss_names = []
for decoder_idx in range(num_decoder):
    loss_names.extend([f"loss_cls_{decoder_idx}", f"loss_reg_{decoder_idx}"])
train_metrics = [dict(type="LossShow", name=loss_names)]


def train_update_loss(metrics, batch, model_outs):
    losses = dict()
    for loss_name in loss_names:
        losses[loss_name] = model_outs[loss_name]
    for metric in metrics:
        metric.update(losses)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=train_update_loss,
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix=job_name,
    reset_metrics_by="log",
)


callbacks = [
    metric_updater,
    stat_callback,
    lr_callback,
    ckpt_callback,
    grad_scale_callback,
    tensorboard_callback,
]

float_trainer = dict(
    type=trainer_type,
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    data_loader=data_loader,
    stop_by="step",
    num_steps=num_iters_per_epoch * num_epochs,
    device=None,
    callbacks=callbacks,
    train_metrics=train_metrics,
    model_convert_pipeline=dict(
        type="FloatQatConvertPipeline",
        qat_mode="fuse_bn",
        enable_qat=False,
        checkpoint_mode="pre_stage",  # "resume"
        checkpoint_configs=dict(
            checkpoint_path=checkpoint_path,
            state_dict_update_func=None,
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
        qconfig_params=None,
    ),
    find_unused_parameters=False,
    # loss_scale=32.0,
    sync_bn=True,
)

# ========================= submit ==========================

num_machines = 1
num_gpus_per_machine = num_gpus
job_password = "6150"

framework = "pytorch"
task_label = "HAT"
project_id = "PDT20220004"
input_bucket = "matrix,mono_3d_data,SD_Algorithm"

priority = 5
docker_image = "docker.hobot.cc/imagesys/hat:mono-runtime-cu111-torch1102-py38-mmcv142-hat211-plugin1106"  # noqa
max_jobtime = 10000 if not remote_debug else 120  # default 7200 = 5days

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
curr_root = os.path.realpath(os.path.dirname(__file__))
work_dir = os.path.realpath(os.getcwd())
config_file = os.path.join(
    os.path.relpath(curr_root, work_dir),
    os.path.basename(os.path.basename(__file__)),
)
folder_list = [
    f"{work_dir}/hat",
    f"{work_dir}/tools",
    f"{work_dir}/projects",
    f"{work_dir}/plugins/k8s_submit/url2IP.py",
]
job_list = [
    f"python3 -W ignore tools/train.py --config {config_file} --stage float",
]


model_name = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
model_version = os.path.basename(os.path.dirname(__file__))
job_name = "sparse4d_" + model_name + "-" + model_version
