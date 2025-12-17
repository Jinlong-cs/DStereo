import os

import torch
from image_fail_segmentation import (  # noqa
    J5_ltc,
    desc_id,
    enable_model_tracking,
    eval_data_setting,
    input_hw,
    model_checkpoint,
    model_name,
    model_version,
    tasks,
    val_model,
)

from hat.core.data_struct.app_struct import reformat_seg_to_aidi_eval
from hat.core.proj_spec.lane_parsing import get_colormap
from hat.utils import Config
from hat.utils.bucket import url_to_local_path

suffix = os.getenv("HAT_PILOT_PREDICTION_NAME_SUFFIX", "")
prediction_name = f"{model_name}_{suffix}" if suffix else model_name


device_ids = [0]
log_rank_zero_only = True

eval_tags = ["resize", "all_region"]

# data
ds_path = os.path.join(
    os.path.dirname(__file__),
    "../datasets/image_fail_seg_eval_datasets.py",
)

evalset_config = Config.fromfile(ds_path)
dataset_ids = evalset_config.get(eval_data_setting)

pred_batch_size = 64

all_callbacks = []
all_data_loaders = []

local_datapath = url_to_local_path(
    "dmpv2://auto_eval/adas_eval/eval_platform/fs/"
)

task_names = [task["name"] for task in tasks]

roi_region = (0, 0, input_hw[1], input_hw[0])

# 评测
if any(ltc in eval_data_setting for ltc in J5_ltc):
    march = "bayes"
else:
    march = "bernoulli2"

val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(
        type="BPUPyramidResizer",
        scale_wh=(0.25, 0.25),
        pyramid_type="ips",
        pyramid_idx=1,
    ),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

for _, all_datasets in dataset_ids.items():
    for task_cfg, datasets in all_datasets.items():

        if isinstance(task_cfg, tuple):
            extra_task_key, task_key = task_cfg
        else:
            extra_task_key = None
            task_key = task_cfg

        # skip datasets of tasks not included in current model
        if extra_task_key is not None:
            if extra_task_key not in task_names:
                continue
        elif task_key not in task_names:
            continue
        obj_key = task_names[0]
        reformat_output_fn = reformat_seg_to_aidi_eval
        reformat_out_fn_kwargs = dict(
            obj_key=obj_key,
        )

        for ds in datasets:
            data_path = os.path.join(local_datapath, str(ds), "datasets")

            eval_callback = dict(
                type="AIDIEval",
                aidi_eval_dataset_id=ds,
                output_root=f"./eval_res/{ds}",
                prediction_name=model_name,
                prediction_tags=eval_tags,
                project_id=os.getenv("PROJECT_ID"),
                reformat_output_fn=reformat_output_fn,
                reformat_out_fn_kwargs=reformat_out_fn_kwargs,
            )

            visualize_callback = dict(
                type="ComposeVisualize",
                callbacks=[
                    dict(
                        type="DetMultitaskVisualize",
                        out_keys=[obj_key],
                        output_dir=f"./tmp_viz_imgs/{model_name}/{ds}",
                        vis_configs=dict(
                            image_fail_parsing=dict(
                                colormap=get_colormap(task_names[0], desc_id),
                                alpha=0.7,
                            ),
                        ),
                        save_viz_imgs=False,
                    ),
                ],
            )
            callbacks = [
                visualize_callback,
                eval_callback,
                dict(
                    type="StatsMonitor",
                    log_freq=5,
                ),
            ]

            all_callbacks.append(callbacks)

            data_loader = dict(
                type=torch.utils.data.DataLoader,
                sampler=dict(
                    type=torch.utils.data.DistributedSampler,
                    shuffle=False,
                ),
                dataset=dict(
                    type="ModelEvalRawDataset",
                    data_path=data_path,
                    to_rgb=True,
                    buf_only=True,
                    return_orig_img=True,
                    transforms=val_transforms,
                ),
                batch_size=pred_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )
            all_data_loaders.append(data_loader)

qat_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
        inverse_transforms=dict(
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_checkpoint
                if model_checkpoint
                else f"aidi_artifact://{model_name}/qat/{model_version}/qat-checkpoint-last.pth.tar",
                enable_tracking=enable_model_tracking,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
        ],
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)
int_infer_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
        inverse_transforms=dict(
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_checkpoint
                if model_checkpoint
                else f"aidi_artifact://{model_name}/qat/{model_version}/qat-checkpoint-last.pth.tar",
                enable_tracking=enable_model_tracking,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)
