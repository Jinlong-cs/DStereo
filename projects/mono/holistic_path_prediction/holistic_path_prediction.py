import copy
import json
import os

import torch
from datasets import train_data_loader, val_data_loader
from horizon_plugin_pytorch.quantization import March

from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BERNOULLI2
# march = March.BAYES

task_name = "hpp"
local_train = not os.path.exists("/running_package")
if local_train:
    ckpt_dir = "./tmp_models_num0/%s" % task_name
    vis_img_saved_path = "./tmp_models_num0/vis/%s" % task_name
else:
    ckpt_dir = "/job_data/model/hpp"
    vis_img_saved_path = "/job_data/visual_img"

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

float_lr = 0.001
train_epochs = 120
qat_train_epochs = 15
quanti_lr = 0.00001
weight_decay = 0.001

data_shape = (3, 256, 512)
train_scales = (4, 8, 16, 32, 64)
alpha = 0.5  # 0.25
bn_kwargs = dict(eps=2e-5, momentum=0.1)


model_type = "hpp"
# ------------------------
# model
# ------------------------
channel_list_ = [32, 32, 64, 128, 256]
out_channels_ = int(channel_list_[2] * alpha)
model = dict(
    type="HppModel",
    out_indices=2,
    backbone=dict(
        type="VargNetV2",
        input_channels=3,
        input_sequence_length=1,
        num_classes=1000,
        factor=2,
        alpha=alpha,
        bias=True,
        bn_kwargs=bn_kwargs,
        group_base=8,
        include_top=False,
        head_factor=2,
    ),
    decode_head=dict(
        type="HPPDecodeHead",
        block_num=4,
        in_channels=out_channels_,
        out_channels=out_channels_,
    ),
    losses=dict(
        type="HppLoss",
        ins_embedding_channel=4,
        weight_offset=0.2,
        weight_exist=1.0,
        weight_nonexist=1.0,
        weight_attention=0.1,
        weight_sisc=0.5,
    ),
)

if training_step in ["int_infer", "qat"]:
    test_inputs = {"img": torch.randn((1, 3, 256, 512))}
else:
    test_inputs = {
        "img": torch.randn((1,) + data_shape),
        "img_name": "xxx.jpg",
        "pred": [],
    }

test_model = dict(
    type="HppModel",
    out_indices=2,
    backbone=dict(
        type="VargNetV2",
        input_channels=3,
        input_sequence_length=1,
        num_classes=1000,
        factor=2,
        alpha=alpha,
        bias=True,
        bn_kwargs=bn_kwargs,
        group_base=8,
        include_top=False,
        head_factor=2,
    ),
    decode_head=dict(
        type="HPPDecodeHead",
        block_num=4,
        in_channels=out_channels_,
        out_channels=out_channels_,
    ),
    losses=None,
    decode=dict(
        type="HPPDecoder",
        feat_stride=8,
        img_shape=(512, 256),
        point_thresh=(0.7),
    ),
)

log_freq = 100


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


# ------------------------
# callback
# ------------------------
train_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)

val_batch_processor = dict(
    type="MultiBatchProcessor", need_grad_update=False, loss_collector=None
)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="hpp_loss"),
    ],
    metric_update_func=update_loss,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="validation_" + task_name,
)

stats_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_update_callback = dict(
    type="CosLrUpdater",
    warmup_len=3000,
    step_log_interval=log_freq,
)

qat_lr_update_callback = dict(
    type="CosLrUpdater",
    warmup_by="epoch",
    warmup_len=0,
    step_log_interval=log_freq,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode=None,
)

tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")
# tb data saved to aidi platform is not permanent, save a copy to model dir
tb_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_log_path, training_step, "loss"),
    loss_name_reg="^.*loss.*",
    update_freq=100,
    update_by="step",
)

# job on aidi platform will have TENSORBOARD_LOG_PATH env,
# where the saved tb can be shown through aidi web UI
aidi_tb_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
        "loss",
    ),
    loss_name_reg="^.*loss.*",
    update_freq=100,
    update_by="step",
)

# ------------------------
# float train
# ------------------------
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=float_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=train_batch_processor,
    num_epochs=train_epochs,
    callbacks=[
        loss_show_updater,
        lr_update_callback,
        stats_callback,
        tb_loss_callback,
        aidi_tb_loss_callback,
        # val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="float-",
            strict_match=True,
            mode=None,
            save_hash=False,
        ),
    ],
    sync_bn=True,
)


def update_metric_hpp(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(batch, model_outs)


# ------------------------
# float predict
# ------------------------
val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric_hpp,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="validation_" + task_name,
)
hpp_metric = dict(
    type="HPPMetric",
    img_saved_path=vis_img_saved_path,
    visualized=False,
    poly_fit=False,
)
float_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    log_interval=50,
    callbacks=[
        val_metric_updater,
    ],
    metrics=[hpp_metric],
)

# ------------------------
# qat train
# ------------------------
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
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=quanti_lr,
        weight_decay=0.01,
    ),
    batch_processor=train_batch_processor,
    num_epochs=qat_train_epochs,
    device=None,
    callbacks=[
        loss_show_updater,
        qat_lr_update_callback,
        stats_callback,
        tb_loss_callback,
        aidi_tb_loss_callback,
        # qat_val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="qat-",
            strict_match=True,
            mode=None,
            save_hash=False,
        ),
    ],
)

# ------------------------
# qat predict
# ------------------------
qat_predictor = dict(
    type="Predictor",
    model=test_model,
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
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    log_interval=50,
    callbacks=[
        val_metric_updater,
    ],
    metrics=[hpp_metric],
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=test_inputs,
)

# ------------------------
# int_infer train
# ------------------------
deploy_model = copy.deepcopy(test_model)
params_desc = dict(
    image_h=data_shape[1],
    image_w=data_shape[2],
    ego_near_length=10,
    ego_far_length=70,
    ego_left=8,
    ego_right=8,
    up_pixel=10,
    ipm_roi=[-8, 70, 16, 60],
)
add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task=f"{task_name}_traj_pred_prob",
                score_threshold=0.6,
                **params_desc,
            )
        ),
        json.dumps(
            dict(
                task=f"{task_name}_traj_pred_offset",
                **params_desc,
            )
        ),
    ],
)
deploy_model["post_process"] = add_desc_pp
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

int_infer_predictor = dict(
    type="Predictor",
    model=test_model,
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
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    log_interval=50,
    callbacks=[
        val_metric_updater,
    ],
    metrics=[hpp_metric],
)
# ------------------------
# compile
# ------------------------
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="hpp_path_predict",
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    opt="O2",
)
