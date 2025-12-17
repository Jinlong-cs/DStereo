import copy
import os
import os.path as osp

import torch
from horizon_plugin_pytorch.quantization import March
from read_yaml import load_data_info, load_yaml

from hat.callbacks.aidi_eval import AIDIEvalTaskType
from hat.data.collates.collates import collate_2d
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.utils import Config

cfg_dir = osp.dirname(osp.dirname(__file__))


cudnn_benchmark = True
march = March.BAYES
seed = None
log_rank_zero_only = True
enable_tensorboard = True
val_only = False

step = "float"  # float | qat | int_infer

if step not in ["float", "qat", "int_infer"]:
    raise NotImplementedError("step type %s is not supported." % step)

job_name = osp.basename(__file__).replace(".py", "")
job_password = "nopassword123"  # enter your password

job_list = [
    "python3 tools/train.py --stage %s --config projects/isp/app/crop/%s.py"
    % (step, job_name),  # your running command
]

local_train = not os.path.exists("/running_package")

datas = load_yaml(cfg_dir + "/datasets/lx179_af817_x8b/person_v00.yaml")
(
    train_data_paths,
    val_data_paths,
    eval_data_paths,
    val_anno_data_path,
    train_num_samples,
    val_num_samples,
    eval_num_samples,
    camera_info,
) = load_data_info(datas["person"])

# resume config
checkpoint_path = "/horizon-bucket/NeuralISP/user/xue01.xiao/ckpt/J3_person_det_crop_raw_day_10w_baseline_padself_min_iou_alpha05/float-checkpoint-epoch-0016.pth.tar"
check_hash = False

if local_train:
    device_ids = [0]
    num_machines = 1
    num_gpus_per_machine = len(device_ids)
    num_gpus = num_machines * num_gpus_per_machine
    log_freq = 10
    batch_size_per_gpu = 1  # [add]  4
    batch_size = batch_size_per_gpu * num_gpus
    val_interval = 1
    ckpt_dir = (
        "checkpoints/det/%s" % job_name
    )  # replace with your own ckpt dir.
    log_dir = "logs/%s" % (osp.basename(__file__).replace(".py", ""))
    train_num_workers, val_num_workers = 8, 4
    train_num_samples = [2]
    val_num_samples = [2]
    json_save_prefix = (
        "results/det/%s" % job_name
    )  # replace with your own log dir.
    json_save_prefix_adas = "results_adas/det/%s" % job_name

# k8s setiings, for aidi platform
k8s_config_file = osp.join(cfg_dir, "k8s_config.py")
base_k8s_config = Config.fromfile(k8s_config_file)
k8s_config = dict()
k8s_config.update(base_k8s_config._cfg_dict)
k8s_config["job_name"] = job_name
k8s_config["job_password"] = job_password
k8s_config["num_machines"] = 1
k8s_config["num_gpus_per_machine"] = 8
k8s_config["job_list"] = job_list

if not local_train:
    device_ids = [i for i in range(k8s_config["num_gpus_per_machine"])]
    num_machines = k8s_config["num_machines"]
    num_gpus_per_machine = k8s_config["num_gpus_per_machine"]
    val_interval = 4
    log_freq = 30
    batch_size_per_gpu = 20  # TODO For 3090, 16 is better.  8 titanxp
    num_gpus = num_machines * num_gpus_per_machine
    batch_size = batch_size_per_gpu * num_gpus
    task_name = "person_det"
    ckpt_dir = "/job_data/models/%s" % task_name
    log_dir = "/job_tboard/"
    train_num_workers, val_num_workers = 4, 2
    json_save_prefix = "/job_data/results/det/%s" % job_name
    json_save_prefix_adas = "/job_data/results_adas/det/%s" % job_name


# task settings
input_size = (1216, 448)
input_size_val = (384, 1024)
dynamic_roi_params = {"w": 1024, "h": 384, "fp_x": 512, "fp_y": 253}

resize_hw_for_val = (1080, 1920)
# infer_model_type = "crop_with_raw_pack"
infer_model_type = "crop_wo_resize"

lr = 2e-3
weight_decay = 4e-5


if step == "float":
    lr = 4e-3  # 2e-3 4e-3
    epoch = 70
elif step == "qat":
    lr = 1e-3
    epoch = 20


# model
bn_kwargs = dict(eps=1e-5, momentum=0.1)
alpha = 0.5
feat_channels = 64
num_classes = 1
img_scale = input_size[::-1]
size_divisor = 64


model = dict(
    type="FCOS",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        input_channels=3,
        alpha=alpha,
        bn_kwargs=bn_kwargs,
        group_base=8,
        include_top=False,
        model_type="VargNetV2",
        factor=2,
        bias=True,
        extend_features=False,
        disable_quanti_input=False,
        flat_output=False,
        input_sequence_length=1,
        head_factor=1,
    ),
    neck=dict(
        type="BiFPN",
        fpn_name="bifpn_sum",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[4, 8, 16, 32, 64],
        stride2channels=get_vargnetv2_stride2channels(alpha),
        out_channels=feat_channels,
        stack=2,
        start_level=1,
        end_level=-1,
        num_outs=5,
    ),
    head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=(4, 8, 16, 32, 64),
        out_strides=(4, 8, 16, 32, 64),
        stride2channels={
            4: feat_channels,
            8: feat_channels,
            16: feat_channels,
            32: feat_channels,
            64: feat_channels,
        },
        feat_channels=feat_channels,
        upscale_bbox_pred=True,
        stacked_convs=2,
        int8_output=True,
        dequant_output=True,
    ),
    targets=dict(
        type="DynamicFcosTarget",
        strides=[4, 8, 16, 32, 64],
        cls_out_channels=num_classes,
        background_label=num_classes,
        topK=10,
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            reduction="none",
        ),
        loss_reg=dict(
            type="GIoULoss", loss_name="reg", loss_weight=2.0, reduction="none"
        ),
    ),
    post_process=dict(
        type="FCOSDecoder",
        num_classes=num_classes,
        strides=[4, 8, 16, 32, 64],
        nms_use_centerness=True,
        nms_sqrt=False,
        transforms=[
            dict(
                type="FixedCrop",
                dynamic_roi_params=dynamic_roi_params,
            ),
        ],
        inverse_transform_key=["scale_factor", "crop_offset"],
        test_cfg=dict(
            nms_pre=1000,
            nms=dict(name="nms", iou_threshold=0.5, max_per_img=100),
            score_thr=0.4,
            min_bbox_size=0,
        ),
        truncate_bbox=False,
        filter_score_mul_centerness=True,
    ),
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=num_classes + 1,
        alpha=0.25,
        gamma=2.0,
        loss_weight=1.0,
    ),
    loss_centerness=dict(
        type="CrossEntropyLoss", loss_name="centerness", use_sigmoid=True
    ),
    loss_reg=dict(
        type="GIoULoss",
        loss_name="reg",
        loss_weight=1.0,
    ),
)

train_dataset = [
    dict(
        type="Auto2dFromLMDB",
        data_path=train_data_path,
        num_samples=train_num_sample,
        to_rgb=False,
        transforms=[
            dict(
                type="RawPad",
                # method="padself",
                method="padzero",
            ),
            dict(
                type="RandomFlip",  # 先flip再crop，和fsd顺序不一样
                px=0.5,
                py=0,
            ),
            dict(
                type="RandomCrop",
                size=img_scale,
                center_crop_prob=0.5,
                min_iou=0.5,
                truncate_gt=False,
            ),
            dict(type="Normalize", raw_norm=True, mean=0.0, std=4096.0),
            dict(
                type="Pad",
                size=img_scale,
            ),
            dict(type="ToTensor", to_yuv=False),
        ],
        infer_model_type=infer_model_type,
    )
    for train_data_path, train_num_sample in zip(
        train_data_paths, train_num_samples
    )
]

train_dataset = dict(
    type="ConcatDataset",
    datasets=train_dataset,
)

train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_2d,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=train_num_workers,
    pin_memory=True,
)

val_transforms = [  # [add]
    dict(type="Resize", img_scale=resize_hw_for_val, keep_ratio=True),
    dict(
        type="RawPad",
        # method="padself",
        method="padzero",
    ),
    dict(
        type="FixedCrop",
        dynamic_roi_params=dynamic_roi_params,
    ),
    dict(type="Normalize", raw_norm=True, mean=0.0, std=4096.0),
    dict(
        type="Pad",
        size=input_size_val,
    ),
    dict(type="ToTensor", to_yuv=False),
]
if infer_model_type == "crop_wo_resize" or "crop_with_raw_pack":
    val_transforms.pop(0)

val_dataset = [
    dict(
        type="Auto2dFromLMDB",
        data_path=val_data_path,
        num_samples=val_num_sample,
        to_rgb=False,
        transforms=val_transforms,
        infer_model_type=infer_model_type,
        camera_info=camera_info,
    )
    for val_data_path, val_num_sample in zip(val_data_paths, val_num_samples)
]

val_dataset = dict(
    type="ConcatDataset",
    datasets=val_dataset,
)

val_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    batch_size=batch_size_per_gpu,
    collate_fn=collate_2d,
    sampler=dict(
        type=torch.utils.data.DistributedSampler,
        shuffle=False,
        drop_last=False,
    ),
    num_workers=val_num_workers,
    pin_memory=True,
)

# set keep_ratio=False in eval_dataloader
eval_transforms = copy.deepcopy(val_transforms)
if eval_transforms[0]["type"] == "Resize":
    eval_transforms[0]["keep_ratio"] = False

eval_dataset = [
    dict(
        type="Auto2dFromLMDB",
        data_path=eval_data_path,
        num_samples=eval_num_sample,
        to_rgb=False,
        transforms=eval_transforms,
        infer_model_type=infer_model_type,
        camera_info=camera_info,
    )
    for eval_data_path, eval_num_sample in zip(
        eval_data_paths, eval_num_samples
    )
]

eval_dataset = dict(
    type="ConcatDataset",
    datasets=eval_dataset,
)

eval_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=eval_dataset,
    batch_size=batch_size_per_gpu,
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=val_num_workers,
    pin_memory=True,
    drop_last=False,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="loss_" + job_name,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=5000,
    epoch_log_freq=val_interval,
    log_prefix="Validation " + job_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

deploy_inputs = dict(img=torch.randn((1, 3, 512, 960)))

if val_only:
    ckpt_callback = dict(
        type="Checkpoint",
        save_dir=ckpt_dir,
        name_prefix="float-",
        save_interval=1,
        # deploy_model=deploy_model,
        # deploy_inputs=deploy_inputs,
        strict_match=True,
        mode="max",
        monitor_metric_key="mAP",
    )
else:
    ckpt_callback = dict(
        type="Checkpoint",
        save_dir=ckpt_dir,
        name_prefix="%s-" % step,
        save_interval=1,
        # deploy_model=None,
        # deploy_inputs=None,
        strict_match=True,
        mode="max",
        monitor_metric_key="mAP",
        save_hash=False,
    )

val_callback = dict(
    type="Validation",
    data_loader=val_dataloader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=val_interval,
    val_on_train_end=True,
)


def tb_update(writer, model_outs, global_step_id, **kwargs):
    cls_loss = float(kwargs.get("losses")[0])
    reg_loss = float(kwargs.get("losses")[1])
    centerness_loss = float(kwargs.get("losses")[2])
    writer.add_scalar(
        tag="cls_loss",
        scalar_value=cls_loss,
        global_step=global_step_id,
    )
    writer.add_scalar(
        tag="reg_loss",
        scalar_value=reg_loss,
        global_step=global_step_id,
    )
    writer.add_scalar(
        tag="centerness_loss",
        scalar_value=centerness_loss,
        global_step=global_step_id,
    )


tb_update_funcs = None
if enable_tensorboard:
    tb_update_funcs = [tb_update]

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir,  # noqa
    update_freq=log_freq,
    tb_update_funcs=tb_update_funcs,
)

if checkpoint_path is not None:
    converters = [
        dict(
            type="LoadCheckpoint",
            checkpoint_path=checkpoint_path,
            check_hash=check_hash,
        )
    ]
else:
    converters = []

model_convert_pipeline = dict(
    type="ModelConvertPipeline", converters=converters
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    # model_convert_pipeline=model_convert_pipeline,
    data_loader=train_dataloader,
    optimizer=dict(type=torch.optim.Adam, lr=lr, weight_decay=weight_decay),
    batch_processor=batch_processor,
    num_epochs=epoch,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            warmup_begin_lr=lr,
            warmup_by="epoch",
            step_log_interval=log_freq,
            lr_decay_id=[int(6 / 12.0 * epoch), int(10 / 12.0 * epoch)],
        ),
        val_callback,
        ckpt_callback,
        tensorboard_callback,
    ],
    train_metrics=[
        dict(
            type="LossShow",
        ),
    ],
    sync_bn=True,
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="%s/val.json" % val_anno_data_path,
        save_prefix=json_save_prefix,
    ),
)

float_solver = dict(
    trainer=float_trainer,
    allow_miss=True,
    ignore_extra=True,
    # resume_optimizer=True,
    # resume_epoch_or_step=True,
    resume_optimizer=False,
    resume_epoch=None,
    resume_step=None,
    quantize=False,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=train_dataloader,
    optimizer=dict(type=torch.optim.Adam, lr=lr, weight_decay=weight_decay),
    batch_processor=batch_processor,
    num_epochs=epoch,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            warmup_begin_lr=lr,
            warmup_by="epoch",
            step_log_interval=log_freq,
            lr_decay_id=[int(8 / 12.0 * epoch), int(11 / 12.0 * epoch)],
        ),
        val_callback,
        ckpt_callback,
        tensorboard_callback,
    ],
    train_metrics=[
        dict(
            type="LossShow",
        ),
    ],
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="%s/val.json" % val_anno_data_path,
        save_prefix=json_save_prefix,
    ),
)

qat_solver = dict(
    trainer=qat_trainer,
    quantize=True,
    check_quantize_model=True,
    pre_step="float",
    pre_step_checkpoint=os.path.join(
        ckpt_dir, "float-checkpoint-best.pth.tar"
    ),
    resume_optimizer=None,
    resume_epoch=None,
    resume_step=None,
    allow_miss=True,
    strict_match=True,
    ignore_extra=True,
)


int_trainer = dict(
    type="Trainer",
    model=model,
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[val_callback],
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="%s/val.json" % val_anno_data_path,
        save_prefix=json_save_prefix,
    ),
)

int_solver = dict(
    trainer=int_trainer,
    quantize=True,
    check_quantize_model=True,
    pre_step="qat",
    pre_step_checkpoint=os.path.join(ckpt_dir, "qat-checkpoint-best.pth.tar"),
    strict_match=True,
    resume_optimizer=False,
    resume_epoch=False,
    resume_step=False,
    allow_miss=True,
    ignore_extra=True,
)

step2solver = dict(float=float_solver, qat=qat_solver, int_infer=int_solver)


def tensor_to_ndarray(output):
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            output[k] = v.cpu().numpy()
        else:
            output[k] = v
    return output


def reformat_aidi_eval_out(batch, output, task_name):
    output = tensor_to_ndarray(output)
    img_data = []
    assert isinstance(batch["img_name"], list)
    for i in range(len(batch["img_name"])):
        one_img_data = {}
        img_name = batch["img_name"][i]
        if img_name[:-4] == ".raw":
            one_img_data["image_key"] = img_name[:-4] + ".jpg"
        else:
            one_img_data["image_key"] = img_name
        one_img_data[task_name] = []

        bboxes = output["pred_bboxes"][i][:, :4]
        bboxes_scores = output["pred_bboxes"][i][:, 4]
        crop_roi = batch["crop_roi"][i]
        one_img_data["crop_roi"] = crop_roi
        one_img_data["valid_roi"] = crop_roi
        for j in range(bboxes.shape[0]):
            one_bbox = {}
            one_bbox["bbox"] = bboxes[j].tolist()
            one_bbox["bbox_score"] = float(bboxes_scores[j])
            one_img_data[task_name].append(one_bbox)
        img_data.append(one_img_data)
    return img_data


aidi_eval_callback = dict(
    type="AIDIEval",
    output_root=json_save_prefix_adas,
    project_id="TD20220002",
    prediction_name=job_name,
    prediction_tags=["crop", "baseline"],
    aidi_eval_dataset_id=[6036745],
    aidi_eval_task_type=AIDIEvalTaskType.Detection_2D,
    aidi_eval_host="http://model.aidi.hobot.cc",
    reformat_output_fn=reformat_aidi_eval_out,
    reformat_out_fn_kwargs={"task_name": "person"},
)

predictor = dict(
    type="Predictor",
    model=model,
    data_loader=eval_dataloader,
    batch_processor=val_batch_processor,
    num_epochs=1,
    device=None,
    callbacks=[
        stat_callback,
        aidi_eval_callback,
    ],
    share_callbacks=False,
)

predict_solver = dict(
    predictor=predictor,
    train_step="float",
    checkpoint=os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar"),
    ckpt_step="float",
    ignore_extra=True,
)
