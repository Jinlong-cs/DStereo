import os

import torch
import torchvision
from horizon_plugin_pytorch.march import March

from hat.data.transforms.detection import Normalize, ToTensor
from hat.data.transforms.gaze import Clip
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
CHIP = "J5"  # only supported by J5

task_name = "gaze_grid_sample"
batch_size_per_gpu = 2
device_ids = [3]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES  # only supported by J5(BAYES)

image_size = (512, 330)
grid_size = (320, 150)

bn_kwargs = {
    "eps": 2e-05,
    "momentum": 0.1,
}

model = dict(
    type="SampleModel",
    output_size=grid_size,
    deploy=False,
)

deploy_model = dict(
    type="SampleModel",
    output_size=grid_size,
    deploy=True,
)

deploy_inputs = dict(
    img=torch.rand((1, 1, image_size[1], image_size[0])) * 2 - 1,
    grid=torch.rand((1, 2, grid_size[1], grid_size[0])) * 2 - 1,
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SampleModelDataset",
        image_size=image_size,
        grid_size=grid_size,
        num=10,
        transforms=torchvision.transforms.Compose(
            [
                Clip(),
                ToTensor(),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SampleModelDataset",
        image_size=image_size,
        grid_size=grid_size,
        num=10,
        transforms=torchvision.transforms.Compose(
            [
                Clip(),
                ToTensor(),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_index(1),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=None,
    loss_collector=None,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode=None,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=0.0)},
        lr=5e-3,
    ),
    batch_processor=batch_processor,
    num_epochs=1,
    device=None,
    callbacks=[
        stat_callback,
        ckpt_callback,
    ],
    train_metrics=[],
    val_metrics=[],
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
        lr=1e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=1,
    device=None,
    callbacks=[
        stat_callback,
        ckpt_callback,
    ],
    train_metrics=[],
    val_metrics=[],
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
    input_source=["ddr", "pyramid"],
)
