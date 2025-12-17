from common import val_transforms
from models import val_model
from vehicle_roi_3d import val_data_loader, val_metric_updater

from hat.core.task_sampler import TaskSampler

device_ids = [1]

val_task_config = dict(vehicle=dict(sampling_factor=1))
val_dataloaders = dict(vehicle=val_data_loader)

val_task_sampler = TaskSampler(
    task_config=val_task_config, method="sample_all"
)

val_data_loader = dict(
    type="MultitaskLoader",
    loaders=val_dataloaders,
    task_sampler=val_task_sampler,
    mode="validation",
    return_task=True,
    custom_length=None,
)


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    batch_transforms=val_transforms,
)

callbacks = [
    val_metric_updater,
]

predict_solver = dict(
    predictor=dict(
        type="Predictor",
        model=val_model,
        data_loader=val_data_loader,
        batch_processor=batch_processor,
        callbacks=callbacks,
        num_epochs=1,
    ),
    train_step="sparse_3d_freeze_bn_2",
    checkpoint="http://fm-xinjie-wang.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize2_cc02_x3c_day_v4.0-20220513-214951/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-b3874d59.pth.tar",
    allow_miss=False,
    ignore_extra=True,
    ckpt_step="sparse_3d_freeze_bn_2",
)
