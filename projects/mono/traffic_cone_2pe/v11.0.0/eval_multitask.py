import os

import torch
from auto2d_base import test_batch_size_per_gpu
from cone_bollard_classification import test_transforms
from entry import test_model
from eval_utils import (
    download_hat_model_from_aidi,
    reformat_tc2pe_to_aidi_eval,
)
from horizon_plugin_pytorch.quantization import March

from hat.data.collates.collates import collate_2d_2pe
from hat.utils import Config

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"

aidi_model_name = "Traffic_Cone_Exp_2pe_6cls"
aidi_model_version = "v11.0.00"
aidi_model_stage = "qat-bernoulli2"

model_ckpt = download_hat_model_from_aidi(
    aidi_model_name, aidi_model_version, aidi_model_stage
)
prediction_name = f"{aidi_model_name}_{aidi_model_version}_{aidi_model_stage}"
device_ids = [2]
log_rank_zero_only = True
# If eval_part_graph_model_only == True, forward only part graph model
# for diff task datasets, it usually speeds eval up
eval_part_graph_model_only = False

eval_tags = ["hat", "traffic_cone", "2pe"]

march = os.environ.get("HAT_MARCH", March.BAYES)
assert march in [March.BERNOULLI, March.BERNOULLI2, March.BAYES]

all_callbacks = []
all_data_loaders = []

# data
ds_path = os.path.join(
    os.path.dirname(__file__),
    "eval_datasets.py",
)
dataset_ids = Config.fromfile(ds_path).dataset_ids

local_datapath = os.path.join(
    bucket_root, "auto_eval/adas_eval/eval_platform/fs/"
)

tasks = [
    dict(name="cone_bollard_classification"),
]
task_names = [task["name"] for task in tasks]


for task_type, all_datasets in dataset_ids.items():
    for dataset, task_cfgs in all_datasets.items():

        data_path = os.path.join(local_datapath, str(dataset), "datasets")
        data_loader = dict(
            type=torch.utils.data.DataLoader,
            sampler=dict(
                type=torch.utils.data.DistributedSampler,
                shuffle=False,
            ),
            batch_size=test_batch_size_per_gpu,
            shuffle=False,
            num_workers=16,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_2d_2pe,
            dataset=dict(
                type="MtlEvalRaw2PEDataset",
                data_path=data_path,
                task_type=task_type,
                to_rgb=False,
                buf_only=True,
                return_orig_img=False,
                transforms=test_transforms,
                instance_key="common_box",
            ),
        )
        all_data_loaders.append(data_loader)

        reformat_output_fn = reformat_tc2pe_to_aidi_eval
        reformat_out_fn_kwargs = dict(
            task_cfgs=task_cfgs,
        )
        prediction_tags = {
            "cone_bollard_classification": "cone_bollard_classification"
        }
        eval_callback = dict(
            type="AIDIEval",
            aidi_eval_dataset_id=dataset,
            output_root=f"./eval_res/{dataset}",  # noqa
            prediction_name=prediction_name,
            prediction_tags=prediction_tags,
            project_id="PDT20220004",
            reformat_output_fn=reformat_output_fn,
            reformat_out_fn_kwargs=reformat_out_fn_kwargs,
            task_cfgs=task_cfgs,
        )

        callbacks = [
            eval_callback,
            dict(
                type="StatsMonitor",
                log_freq=5,
            ),
        ]

        all_callbacks.append(callbacks)


qat_predictor = dict(
    type="Predictor",
    model=test_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="MultiBatchProcessor"
        if eval_part_graph_model_only
        else "BasicBatchProcessor",  # noqa
        batch_transforms=[
            dict(type="BgrToYuv444V2", rgb_input=False),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
        need_grad_update=False,
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=None,
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_ckpt,
                allow_miss=False,
                ignore_extra=False,
                check_hash=False,
            ),
        ],
    ),
)

float_predictor = dict(
    type="Predictor",
    model=test_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="MultiBatchProcessor"
        if eval_part_graph_model_only
        else "BasicBatchProcessor",  # noqa
        batch_transforms=[
            dict(type="BgrToYuv444V2", rgb_input=False),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
        need_grad_update=False,
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        # qat_mode="fuse_bn",
        qconfig_params=None,
        converters=[
            # dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_ckpt,
                allow_miss=False,
                ignore_extra=False,
                check_hash=False,
            ),
        ],
    ),
)
