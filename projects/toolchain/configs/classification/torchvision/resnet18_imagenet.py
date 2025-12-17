import copy
import os

import torch
from horizon_plugin_pytorch.march import March
from PIL import Image

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "qat")

task_name = "torchvision_resnet18_imagenet"
num_classes = 1000
batch_size_per_gpu = 128
device_ids = [0, 1, 2, 3]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES

model = dict(
    type="Classifier",
    backbone=dict(
        type="resnet18",
        num_classes=1000,
    ),
    losses=dict(type="CEWithLabelSmooth"),
)
deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="resnet18",
        num_classes=1000,
        flat_output=False,
    ),
    losses=None,
)
deploy_inputs = dict(img=torch.randn((1, 3, 224, 224)))

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
        type="ImageNet",
        data_path="./tmp_data/imagenet/train_lmdb/",
        transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="RandomResizedCrop",
                size=224,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path="./tmp_data/imagenet/val_lmdb/",
        transforms=[
            dict(type="TorchVisionAdapter", interface="Resize", size=256),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="TorchVisionAdapter", interface="RandomHorizontalFlip"),
        dict(
            type="TorchVisionAdapter",
            interface="ConvertImageDtype",
            dtype=torch.float32,
        ),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ],
    loss_collector=collect_loss_by_index(1),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="ConvertImageDtype",
            dtype=torch.float32,
        ),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ],
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    target = batch["labels"]
    preds, losses = model_outs
    for metric in metrics:
        metric.update(target, preds)


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    metric_update_func=update_metric,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)


def update_state_dict_func(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = "backbone." + k if "backbone." not in k else k
        if k == "fc.weight":
            v = v.data.view(v.data.shape + (1, 1))
        new_state_dict[new_k] = v
    return new_state_dict


ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
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
                    ckpt_dir, "resnet18-5c106cde.pth"
                ),
                state_dict_update_func=update_state_dict_func,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=1e-4)},
        lr=0.0001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=30,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[15, 25],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        ckpt_callback,
        val_callback,
    ],
    train_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
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
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
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
    input_source=["ddr"],
)

# predictor
qat_predictor = dict(
    type="Predictor",
    model=model,
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
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

int_infer_predictor = dict(
    type="Predictor",
    model=model,
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
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

infer_ckpt = int_infer_trainer["model_convert_pipeline"]["converters"][1][
    "checkpoint_path"
]


infer_transforms = [
    dict(type="TorchVisionAdapter", interface="Resize", size=256),
    dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
    dict(type="TorchVisionAdapter", interface="PILToTensor"),
    dict(
        type="TorchVisionAdapter",
        interface="ConvertImageDtype",
        dtype=torch.float32,
    ),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
]

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNetFromImage",
        root="./tmp_orig_data/imagenet/val",
        split="val",
        transforms=infer_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


align_bpu_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode="eager"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=infer_ckpt,
            ),
            dict(type="QAT2Quantize", convert_mode="eager"),
        ],
    ),
    data_loader=align_bpu_data_loader,
    metrics=[
        dict(type="Accuracy"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


def process_inputs(infer_inputs, transforms=None):
    ori_img = Image.open(infer_inputs["imgs"]).convert("RGB")
    model_input = {
        "img": ori_img,
    }
    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)
    return model_input, ori_img


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs
    preds = viz_func(vis_inputs, preds)
    return f"The result is: {int(preds)}"


infer_cfg = dict(
    model=model,
    infer_inputs=dict(
        imgs="./tmp_orig_data/imagenet/val/val/n01440764/ILSVRC2012_val_00000293.JPEG",
    ),
    process_inputs=process_inputs,
    viz_func=dict(type="ClsViz", is_plot=True),
    process_outputs=process_outputs,
    transforms=infer_transforms,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode="eager"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=infer_ckpt,
            ),
            dict(type="QAT2Quantize", convert_mode="eager"),
        ],
    ),
)

onnx_cfg = dict(
    model=deploy_model,
    stage="qat",
    inputs=deploy_inputs,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode="eager"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=infer_ckpt,
            ),
        ],
    ),
)
