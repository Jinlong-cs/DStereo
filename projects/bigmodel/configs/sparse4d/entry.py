import os

import torch
from common import (
    ckpt_dir,
    do_experiment,
    docker_image,
    folder_list,
    framework,
    input_bucket,
    job_list,
    job_name,
    job_password,
    launcher,
    max_jobtime,
    num_gpus_per_machine,
    num_machines,
    priority,
    project_id,
    task_label,
    train_batch_size,
    upload_folder_name,
)
from data_loaders import train_dataloader
from horizon_plugin_pytorch.march import March
from schedule import (  # num_epochs,
    interval_by,
    lr,
    num_steps,
    save_interval,
    stat_log_freq,
    step_log_interval,
    warmup_steps,
)
from sparse4d_dynamic_3d_detection import (
    get_sparse_model,
    sparse4d_metric_updater,
    train_metrics,
)

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

device_ids = list(range(num_gpus_per_machine))

if os.environ.get("CLUSTER"):
    redirect_config_logging_path = f"/job_data/config_log_{training_step}.log"
else:
    redirect_config_logging_path = os.path.join(
        f"./tmp_models/config_log_{training_step}.log"
    )

seed = 1280
cudnn_benchmark = False
log_rank_zero_only = True
march = March.BAYES

model = get_sparse_model()

if do_experiment:
    num_steps = num_steps // 4

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[],
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    grad_scaler=dict(
        type=torch.cuda.amp.GradScaler,
        init_scale=32.0,
        growth_interval=int(1e8),
    ),
    enable_amp=False,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=stat_log_freq,
    batch_size=train_batch_size,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    save_interval=save_interval,
    name_prefix=training_step + "-",
    mode="max",
    strict_match=False,
    interval_by=interval_by,
    save_on_train_end=True,
)

lr_callback = dict(
    type="CosLrUpdater",
    warmup_by="step",
    warmup_len=warmup_steps,
    step_log_interval=step_log_interval,
    max_steps=num_steps - warmup_steps,
)


grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[
        ("backbone", 0.5),
    ],
    clip_grad_norm=35.0,
    clip_norm_type=2,
)

callbacks = [
    sparse4d_metric_updater,
    stat_callback,
    lr_callback,
    ckpt_callback,
    grad_scale_callback,
]


optimizer = dict(
    type=torch.optim.AdamW,
    params={"weight": dict(weight_decay=4e-5)},
    lr=lr,
    weight_decay=0.01,
)

# pretrain_checkpoint = "dmpv2://matrix/users/zixiang.pei/workspace/pretrained_models/small_256_5cars_bev_float-checkpoint-last-b1f5d9f4-formattedv2.pth.tar"
# pretrain_checkpoint = "dmpv2://adas/big_model/model_zoo/small_galaxy_c385_float-checkpoint-last-2520071f-formatted.pth.tar"
pretrain_checkpoint = "http://fm-zixiang-pei.train.hogpu.cc/plat_gpu/hobot-dag-3589138_hat-job-sparse4d-x3c-1F-20230802-145247/output/models/sparse4d_x3c_1F/float-checkpoint-last-54356184.pth.tar"

trainer_type = "distributed_data_parallel_trainer"
float_trainer = dict(
    type=trainer_type,
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    data_loader=train_dataloader,
    num_steps=num_steps,
    stop_by="step",
    device=None,
    callbacks=callbacks,
    train_metrics=train_metrics,
    val_metrics=train_metrics,
    model_convert_pipeline=dict(
        type="FloatQatConvertPipeline",
        qat_mode="fuse_bn",
        enable_qat=False,
        checkpoint_mode="pre_stage",  # "resume"
        checkpoint_configs=dict(
            checkpoint_path=pretrain_checkpoint,
            state_dict_update_func=None,
            allow_miss=True,
            ignore_extra=True,
            ignore_tensor_shape=True,
            verbose=True,
        ),
        qconfig_params=None,
    ),
    find_unused_parameters=False,
    sync_bn=True,
)


k8s_config = dict(
    job_name=job_name,
    job_password=job_password,
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework=framework,
    task_label=task_label,
    project_id=project_id,
    input_bucket=input_bucket,
    priority=priority,
    docker_image=docker_image,
    max_jobtime=max_jobtime,
    launcher=launcher,
    upload_folder_name=upload_folder_name,
    folder_list=folder_list,
    job_list=job_list,
)
