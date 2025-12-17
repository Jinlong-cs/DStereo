from copy import deepcopy

from common import model_checkpoint, val_transforms  # is_with_pe,; pe_config,
from models import march, val_model  # noqa: F401
from schedule import get_fuse_patterns_by_stage, train_stages
from vehicle_roi_3d import val_data_loader, val_metric_updater

from hat.core.task_sampler import TaskSampler

# from vehicle_detection import val_data_loader, val_metric_updater


device_ids = [0]

# Uncomment this part if vehicle detection validation is used.
# if is_with_pe:
#     pe_generator = dict(
#                         type="PEGenerator",
#                         pe_config=deepcopy(pe_config),
#                     )

#     for dataset in val_data_loader["dataset"]["datasets"]:
#         dataset["transforms"].insert( 1, pe_generator)

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


base_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=val_data_loader,
    batch_processor=batch_processor,
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)


for stage in train_stages:
    pre, cur = get_fuse_patterns_by_stage(stage)

    predictor = deepcopy(base_predictor)
    predictor.update(
        model_convert_pipeline=dict(
            type="QATFuseBNConvertPipeline",
            qat_mode="with_bn_reverse_fold",
            pre_stage_fuse_patterns=pre,
            cur_stage_fuse_patterns=cur,
            fuse_part_configs=dict(
                fuse_method="fuse_norm",
                regex=True,
                strict=False,
            ),
            checkpoint_mode="resume",
            checkpoint_configs=dict(
                checkpoint_path=model_checkpoint,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            qconfig_params=None,
        )
    )
    globals()[f"{stage}_predictor"] = predictor
