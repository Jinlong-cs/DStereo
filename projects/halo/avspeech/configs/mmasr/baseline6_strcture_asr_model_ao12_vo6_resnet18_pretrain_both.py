# Copyright (c) Horizon Robotics, All rights reserved.

# flake8: noqa

import os
import pathlib
from copy import deepcopy

import torch
import torchvision
import yaml
from aidisdk.utils import running_in_cluster
from horizon_plugin_pytorch.march import March
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hat.callbacks.checkpoint import FlexibleStateDict
from hat.callbacks.lr_updater import NoamLrUpdater
from hat.callbacks.tensorboard import TensorBoard
from hat.data.collates.collates import CocktailCollate
from hat.data.datasets.avspeech.audio import (
    HDF5WaveformReader,
    TendisWaveformReader,
)
from hat.data.datasets.avspeech.basic_data_info import TendisBasicInfoReader
from hat.data.datasets.avspeech.image import TendisVideoImageReader
from hat.data.datasets.avspeech.info import MMASRInfoList
from hat.data.datasets.avspeech.symbol_table import SymbolTable
from hat.data.datasets.avspeech_dataset import CocktailDatasetV0
from hat.data.samplers.dist_bucket_sampler import DistRandomBucketSampler
from hat.data.transforms.avspeech.align import ForceAlignV2
from hat.data.transforms.avspeech.audio import (
    ChannelSelect,
    FBank,
    ReSample,
    SpecAug,
)
from hat.data.transforms.avspeech.augment import OnLineAugmentPipeline
from hat.data.transforms.avspeech.images import (
    CoarseDropout,
    FancyPCA,
    HueSaturationValue,
    ImageListNormalize,
    ImageListRandomCrop,
    ImageListRandomFlip,
    ImageListStack,
    ImageListToYUV444,
    RandomBrightnessContrast,
    ToGray,
)
from hat.data.transforms.avspeech.text import LabelToTensor, Tokenize
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.metrics.avspeech_metric import WER
from hat.metrics.loss_show import LossShow
from hat.models.structures.carp_structure_sC import CocktailE2EStructureC
from hat.models.task_modules.avspeech.carp_encoder import AVConformerCnnEncoder
from hat.models.task_modules.avspeech.cmvn import GlobalCMVN, load_cmvn
from hat.models.task_modules.avspeech.ctc_greedy import CTCGreedyDecoder
from hat.models.task_modules.avspeech.decoder_cnn import (
    CTC,
    BiTransformerCnnDecoder,
)
from hat.models.task_modules.avspeech.embed import Embedding
from hat.models.task_modules.avspeech.mask import ChunkMaskV2, TargetMask
from hat.models.task_modules.avspeech.speech_decoder import CTCDecoder
from hat.utils.root_helper import RootHelper
from projects.avspeech.utils.callbacks.error_analysis import DumpBadCase

join = os.path.join

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

# ! HAT ENGINE 配置
# 使用配置文件的名字作为 task_name
# 可以手动设置名字
task_name = pathlib.Path(__file__).stem
# 保存模型的前缀名称
model_prefix = "MMASR-" + task_name.split("_")[0]

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES
qat_mode = "fuse_bn"


# ! 训练相关的配置
CARP_ROOT = RootHelper.get_project_root("halo/avspeech")
data_root = os.path.join(CARP_ROOT, "data")
symbol_table_path = (
    "/horizon-bucket/J2MM/speech/audio_only/bigdata/data/dict/lang_char.txt"
)
bpe_model_path = (
    "/horizon-bucket/J2MM/speech/audio_only/bigdata/data/train/bpe_model.model"
)

is_json_cmvn = True
cmvn_file = (
    "/horizon-bucket/J2MM/speech/audio_only/bigdata/data/train/global_cmvn"
)
if cmvn_file is not None:
    mean, istd = load_cmvn(cmvn_file, is_json_cmvn)
    global_cmvn = GlobalCMVN(
        torch.from_numpy(mean).float(), torch.from_numpy(istd).float()
    )
else:
    global_cmvn = None

cluster_device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
# cluster_device_ids = [0, 1]
local_device_ids = [0]


# 各种 path 列表
gpu_rir_path = "/horizon-bucket/J2MM/speech/hdf5/gpu_rirs/gpu_rirs.hdf5"
noise_info = "/horizon-bucket/J2MM/speech/hdf5/noise_info.txt"
train_info_txt = "/horizon-bucket/J2MM/speech/hdf5/mmasr_info/train/train_fyz_j2mm_1000id.info"
train_hdf5_audio_path = "/horizon-bucket/J2MM/speech/hdf5/audio.hdf5"
val_hdf5_audio_path = {
    "FYZ": "/horizon-bucket/J2MM/speech/hdf5/mmasr_val/FYZ_val_mixed_no_hisf",
    "QR": "/horizon-bucket/J2MM/speech/hdf5/mmasr_val/QR_val_mixed_no_hisf",
    "Qianren": "/horizon-bucket/J2MM/speech/hdf5/mmasr_val/Qianren_val_mixed_no_hisf",
}
test_hdf5_audio_path = {
    "FYZ": "/horizon-bucket/J2MM/speech/hdf5/mmasr_val/FYZ_val_mixed_no_hisf",
    "QR": "/horizon-bucket/J2MM/speech/hdf5/mmasr_test/QR_test_mixed_no_hisf",
    "Qianren": "/horizon-bucket/J2MM/speech/hdf5/mmasr_test/Qianren_test_mixed_no_hisf",
}
val_info_txt = {
    "FYZ": "/horizon-bucket/J2MM/speech/hdf5/mmasr_info/val/val.info",
    "QR": "/horizon-bucket/J2MM/speech/hdf5/mmasr_info/val/qr_choose_500_for_val.info",
    "Qianren": "/horizon-bucket/J2MM/speech/hdf5/mmasr_info/val/Qianren_batch01_mmasr_hy_val_source_proc.info",
}
test_info_txt = {
    "FYZ": "/horizon-bucket/J2MM/speech/hdf5/mmasr_info/val/val.info",
    "QR": "/horizon-bucket/J2MM/speech/hdf5/mmasr_info/test/QR_batch01_mmasr_test_alpha_proc_check.info",
    "Qianren": "/horizon-bucket/J2MM/speech/hdf5/mmasr_info/test/Qianren_batch01_mmasr_test_no_alpha_proc_check.info",
}
noise_hdf5_file = "/horizon-bucket/J2MM/speech/hdf5/noise.hdf5"

if running_in_cluster():
    # 集群训练的一些配置
    device_ids = cluster_device_ids
    batch_size = 8
    num_worker = 4
    prefetch_factor = 4
    ckpt_dir = f"/job_data/models/{task_name}/"
    epoch_log_freq = 1
    step_log_freq = 100
    sampler_bucket_size = 10000
else:
    # 本地训练的一些配置
    device_ids = local_device_ids
    batch_size = 8
    num_worker = 2
    prefetch_factor = 2
    hdf5_audio_path = "/horizon-bucket/J2MM/speech/hdf5/audio.hdf5"  # local
    val_hdf5_audio_path = {
        "FYZ": "/horizon-bucket/J2MM/speech/hdf5/mmasr_val/FYZ_val_mixed_no_hisf",
        "QR": "/horizon-bucket/J2MM/speech/hdf5/mmasr_val/QR_val_mixed_no_hisf",
    }
    test_hdf5_audio_path = {
        "FYZ": "/horizon-bucket/J2MM/speech/hdf5/mmasr_val/FYZ_val_mixed_no_hisf",
        "QR": "/horizon-bucket/J2MM/speech/hdf5/mmasr_test/QR_test_mixed_no_hisf",
        "Qianren": "/horizon-bucket/J2MM/speech/hdf5/mmasr_test/Qianren_test_mixed_no_hisf",
    }
    # test_hdf5_audio_path = "/horizon-bucket/J2MM/speech/hdf5/mmasr_test"
    noise_hdf5_file = "/horizon-bucket/J2MM/speech/hdf5/noise.hdf5"
    ckpt_dir = f"./job_data/models/{model_prefix}/"
    epoch_log_freq = 1
    step_log_freq = 100
    sampler_bucket_size = 10000


ao_checkpoint_path = (
    "/horizon-bucket/J2MM/speech/hdf5/mmasr_pretrain_model/avg_30_xingchen.pt"
)
vo_checkpoint_path = "/horizon-bucket/J2MM/speech/hdf5/mmasr_pretrain_model/exp0202_strcture_sc_vo6_resnet18_float-checkpoint-average-min_wer-20-6c1b15e7.pth.tar"
gpu_rir_path = "/horizon-bucket/J2MM/speech/hdf5/gpu_rirs/gpu_rirs.hdf5"
noise_info = "/horizon-bucket/J2MM/speech/hdf5/noise_add_babble_info.txt"

tb_log_path = os.path.join(ckpt_dir, "tensorboard")
tb_log_path = os.getenv("TENSORBOARD_LOG_PATH", tb_log_path)
tendis_kwargs = dict(
    host="aidi-kv-cluster-01.hogpu.cc",
    port=7616,  # 原来是 7617
    db=0,
    socket_connect_timeout=5000,
    password="pB9cA2eN1eD1lJ0",
    socket_keepalive=False,
)

# symbol_table
symbol_table = SymbolTable(symbol_table_path)
# non_lang_syms = read_non_lang_symbols(None)
# input output
fbank_size = 80
vocab_size = len(symbol_table)
num_epochs = 1000

# # update model config
# wenet_config_file = "projects/carp/config/tasks/mmasr/train_xingchen.yaml"
# with open(wenet_config_file) as fr:
#     wenet_config = yaml.load(fr, Loader=yaml.FullLoader)
# wenet_config['input_dim'] = fbank_size
# wenet_config['output_dim'] = vocab_size
# wenet_config['cmvn_file'] = cmvn_file
# wenet_config['is_json_cmvn'] = True

# 网络结构相关配置
# model = init_asr_model(wenet_config)
encoder_output_size = 256
bn_kwargs = dict(eps=1e-5, momentum=0.1)
model = dict(
    type=CocktailE2EStructureC,
    vocab_size=vocab_size,
    encoder=dict(
        type=AVConformerCnnEncoder,
        global_cmvn=global_cmvn,
        input_size=80,
        activation_type="relu",
        attention_dropout_rate=0.1,
        attention_heads=4,
        causal=True,
        cnn_inner_channel=512,
        cnn_module_kernel=7,
        dropout_rate=0.1,
        final_norm="layer_norm",
        audio_frontend="conv2d8",
        video_frontend="resnet18",
        linear_units=2048,
        audio_num_blocks=12,
        video_num_blocks=6,
        fusion_num_blocks=6,
        fusion_type="cat",
        output_size=encoder_output_size,
        static_chunk_size=0,
        use_dynamic_chunk=True,
        use_dynamic_left_chunk=True,
    ),
    decoder=dict(
        type=BiTransformerCnnDecoder,
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        attention_heads=4,
        dropout_rate=0.1,
        linear_units=2048,
        num_blocks=3,
        positional_dropout_rate=0.1,
        r_num_blocks=3,
        self_attention_dropout_rate=0.1,
        src_attention_dropout_rate=0.1,
    ),
    ctc=dict(
        type=CTC,
        odim=vocab_size,
        encoder_output_size=encoder_output_size,
    ),
    ctc_decoder=dict(
        type=CTCGreedyDecoder,
        vocab_path=symbol_table_path,
        blank_id=0,
        keep_blank=True,
    ),
    ctc_weight=0.3,
    ignore_id=-1,
    reverse_weight=0.3,
    lsm_weight=0.1,
)

# 数据增强的配置
augment_config = os.path.join(
    CARP_ROOT, "resources/exp_mmasr_augment_linearly_no_norm_low_snr.yaml"
)
with open(augment_config) as fr:
    aug_config = yaml.load(fr, Loader=yaml.FullLoader)

train_dataset = dict(
    type=CocktailDatasetV0,
    audio_reader=dict(
        type=TendisWaveformReader,
        ret_dtype="float32",
        channel_first=True,
        tendis_kwargs=tendis_kwargs,
    ),
    image_reader=dict(
        type=TendisVideoImageReader,
        fps=25,
        origin_fps=30,
        zfill_num=6,
        missing_image_complete_mode="zero",
        filter_rules={"missing_ratio": {"threshold": 0.3}},
        tendis_kwargs=tendis_kwargs,
    ),
    info_list=dict(
        type=MMASRInfoList, path=train_info_txt, left_limit=0.3, right_limit=10
    ),
    basic_info_reader=dict(
        type=TendisBasicInfoReader, tendis_kwargs=tendis_kwargs
    ),
    mode="train",
    vad_expand=0.2,
    transforms=dict(
        type=torchvision.transforms.Compose,
        transforms=[
            # 音频处理
            dict(type=ReSample, new_freq=16000),
            dict(
                type=OnLineAugmentPipeline,
                config=aug_config,
                noise_hdf5_file=noise_hdf5_file,
                ret_dtype="float32",
                gpu_rir_path=gpu_rir_path,
                noise_info=noise_info,
                p=0.5,
            ),
            dict(type=ChannelSelect, channel=0),
            dict(
                type=FBank,
                num_mel_bins=fbank_size,
                frame_shift=10,
                frame_length=25,
                dither=1.0,
            ),
            dict(
                type=SpecAug,
                num_t_mask=2,
                num_f_mask=2,
                max_t=50,
                max_f=10,
                p=1.0,
            ),
            # 强制对齐
            dict(type=ForceAlignV2, audio_left_extend=7),
            # 图片处理
            dict(
                type=RandomBrightnessContrast,
                brightness_limit=0.4,
                contrast_limit=0.4,
                brightness_by_max=False,
                p=0.3,
            ),
            dict(
                type=CoarseDropout,
                max_holes=1,
                max_height=96,
                max_width=30,
                fill_value=[183, 197, 248],
                p=0.3,
            ),
            dict(
                type=HueSaturationValue,
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=0,
                p=0.2,
            ),
            dict(
                type=FancyPCA,
                alpha=0.8,
                p=0.3,
            ),
            dict(
                type=ToGray,
                p=0.3,
                rgb_data=False,
            ),
            dict(type=ImageListStack, hwc2chw=True),
            dict(type=ImageListRandomFlip, p=0.5),
            dict(
                type=ImageListRandomCrop,
                scale=(0.8, 1.0),
                ratio=(1.0, 1.0),
                p=0.5,
            ),
            dict(type=ImageListToYUV444),
            dict(
                type=ImageListNormalize,
                mean=[128.0, 128.0, 128.0],
                std=[128.0, 128.0, 128.0],
            ),
            # 标签处理
            dict(
                type=Tokenize,
                symbol_table=symbol_table,
                bpe_model_path=None,
                non_lang_syms=None,
                split_with_space=False,
            ),
            dict(type=LabelToTensor),
        ],
    ),
)

train_dataloader = dict(
    type=DataLoader,
    dataset=train_dataset,
    batch_size=batch_size,
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=-1),
    num_workers=num_worker,
    sampler=dict(
        type=DistRandomBucketSampler,
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        bucket_size=sampler_bucket_size,
    ),
    prefetch_factor=prefetch_factor,
)

FYZ_val_db_levels = ["FYZ 0db", "FYZ m5db", "FYZ clean"]
QR_val_db_levels = ["QR 0db", "QR m5db", "QR clean"]

val_db_levels = []
val_db_levels.extend(FYZ_val_db_levels)
val_db_levels.extend(QR_val_db_levels)

data_loaders = []
for each in val_db_levels:
    data_type = each.split()[0]  # FYZ QR
    db_level = each.split()[1]  # "25db"
    db_val_hdf5_audio_path = os.path.join(
        val_hdf5_audio_path[data_type],
        f"{data_type}_mmasr_val_{db_level}.hdf5",
    )
    db_val_info_txt = val_info_txt[data_type]

    val_dataset = dict(
        type=CocktailDatasetV0,
        audio_reader=dict(
            type=HDF5WaveformReader,
            hdf5_file=db_val_hdf5_audio_path,
            ret_dtype="float32",
            channel_first=True,
        ),
        image_reader=dict(
            type=TendisVideoImageReader,
            fps=25,
            origin_fps=30,
            zfill_num=6,
            missing_image_complete_mode="zero",
            filter_rules=None,
            tendis_kwargs=tendis_kwargs,
        ),
        info_list=dict(type=MMASRInfoList, path=db_val_info_txt),
        basic_info_reader=dict(
            type=TendisBasicInfoReader,
            tendis_kwargs=tendis_kwargs,
        ),
        mode="val",
        vad_expand=0.2,
        transforms=dict(
            type=torchvision.transforms.Compose,
            transforms=[
                # 音频处理
                dict(type=ReSample, new_freq=16000),
                # dict(
                #     type=OnLineAugmentPipeline,
                #     config=aug_config,
                #     noise_hdf5_file=noise_hdf5_file,
                #     ret_dtype="float32",
                #     gpu_rir_path=gpu_rir_path,
                # ),
                dict(type=ChannelSelect, channel=0),
                dict(
                    type=FBank,
                    num_mel_bins=fbank_size,
                    frame_shift=10,
                    frame_length=25,
                    dither=0.0,
                ),
                # 强制对齐
                dict(type=ForceAlignV2, audio_left_extend=7),
                # 图片处理
                dict(type=ImageListStack, hwc2chw=True),
                dict(type=ImageListToYUV444),
                dict(
                    type=ImageListNormalize,
                    mean=[128.0, 128.0, 128.0],
                    std=[128.0, 128.0, 128.0],
                ),
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=symbol_table,
                    bpe_model_path=None,
                    non_lang_syms=None,
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    val_dataloader = dict(
        type=DataLoader,
        dataset=val_dataset,
        batch_size=batch_size,
        pin_memory=False,
        collate_fn=dict(type=CocktailCollate, ignore_id=-1),
        num_workers=2,
        sampler=dict(
            type=DistributedSampler, dataset=val_dataset, shuffle=False
        ),
    )

    data_loaders.append(val_dataloader)

# 任务相关的配置
batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_index(0),
    enable_amp=False,
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=None,
    loss_collector=None,
    enable_amp=False,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=step_log_freq,
    batch_size=batch_size,
)

grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=5,
)

freeze_modules = [
    "encoder.embed",
    "encoder.audio_encoders",
    "encoder.vfea_extractor",
    "encoder.video_encoders",
    # "encoder.vfea_module",
]
freeze_model_callback = dict(
    type="FreezeModule",
    modules=[freeze_modules for _ in range(num_epochs)],
    step_or_epoch=list(range(num_epochs)),
    update_by="epoch",
)


def update_metric(metrics, batch, model_outs):
    loss, loss_att, loss_ctc, att_acc, ctc_out = model_outs
    if att_acc is not None:
        att_acc = torch.Tensor([att_acc]).to(loss.device)
    for metric in metrics:
        if isinstance(metric, LossShow):
            metric.update(
                {
                    "loss": loss,
                    "loss_att": loss_att,
                    "loss_ctc": loss_ctc,
                    "att_acc": att_acc,
                }
            )
        if isinstance(metric, WER):
            tokens = [" ".join(_tokens) for _tokens in batch["tokens"]]
            metric.update(ctc_out, tokens)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=step_log_freq,
    reset_metrics_by="log",
    epoch_log_freq=epoch_log_freq,
    log_prefix=model_prefix,
)


def update_tb_train(writer, model_outs, global_step_id, **kwargs):
    mode = "train"
    metrics = kwargs["train_metrics"]
    for metric in metrics:
        if isinstance(metric, LossShow):
            head = f"{mode}/loss"
        elif isinstance(metric, WER):
            head = f"{mode}/wer"
        else:
            continue
        names, values = metric.get()
        if not isinstance(names, (list, tuple)):
            names = (names,)
            values = (values,)
        state = {k: v for k, v in zip(names, values)}
        writer.add_scalars(head, state, global_step_id)


train_tb_callback = dict(
    type=TensorBoard,
    save_dir=tb_log_path,
    overwrite=True,
    update_freq=step_log_freq,
    update_by="step",
    tb_update_funcs=[update_tb_train],
)


def update_tb_val(writer, epoch_id, **kwargs):
    mode = "val"
    metrics = kwargs["train_metrics"]
    for metric in metrics:
        if isinstance(metric, LossShow):
            head = f"{mode}/loss"
        elif isinstance(metric, WER):
            head = f"{mode}/wer"
        else:
            continue
        names, values = metric.get()
        if not isinstance(names, (list, tuple)):
            names = (names,)
            values = (values,)
        state = {k: v for k, v in zip(names, values)}
        writer.add_scalars(head, state, epoch_id)


val_tb_callback = dict(
    type=TensorBoard,
    save_dir=tb_log_path,
    overwrite=True,
    update_freq=epoch_log_freq,
    update_by="epoch",
    tb_update_funcs=[update_tb_val],
)

val_callbacks = []

for db_level in val_db_levels:
    val_metric_updater = dict(
        type="MetricUpdater",
        metric_update_func=update_metric,
        step_log_freq=step_log_freq,
        reset_metrics_by="epoch",
        epoch_log_freq=epoch_log_freq,
        log_prefix=model_prefix + f" {db_level}",
    )
    each_callback = [val_metric_updater]
    val_callbacks.append(each_callback)


val_callback = dict(
    type="Validation",
    data_loader=data_loaders,
    batch_processor=val_batch_processor,
    callbacks=val_callbacks,
    log_interval=0,
    interval_by="epoch",
    share_callbacks=False,
    val_on_train_begin=False,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    interval_by="epoch",
    mode=None,
    monitor_metric_key="att_acc",
)

fckpt_callback = dict(
    type=FlexibleStateDict,
    loaded_state_dict=None,
    load_strict=True,
    save_root=ckpt_dir,
    name_prefix=f"{training_step}-{model_prefix}",
    save_interval=epoch_log_freq,
    interval_by="epoch",
    save_on_train_end=True,
)


def audio_state_dict_update_func(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder"):
            if "encoder.encoders" in key:
                key = key.replace("encoders", "audio_encoders")
            new_state_dict[key] = value
        # else:
        #     new_state_dict[key] = value

    return new_state_dict


def video_state_dict_update_func(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder"):
            new_state_dict[key] = value

    return new_state_dict


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=ao_checkpoint_path,
                allow_miss=True,
                ignore_extra=True,
                state_dict_update_func=audio_state_dict_update_func,
                verbose=True,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=vo_checkpoint_path,
                allow_miss=True,
                ignore_extra=True,
                state_dict_update_func=video_state_dict_update_func,
                verbose=True,
            ),
        ],
    ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={},
        lr=0.001,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type=NoamLrUpdater,
            d_model=25000,
            warmup_step=25000,
            step_log_interval=100,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
        fckpt_callback,
        # train_tb_callback,
        grad_scale_callback,
        freeze_model_callback,
    ],
    train_metrics=[
        dict(type=LossShow),
        dict(type=WER, name="wer"),
    ],
    val_metrics=[
        dict(type=LossShow),
        dict(type=WER, name="wer"),
    ],
)

# float_solver = dict(
#     trainer=float_trainer,
#     quantize=False,
#     allow_not_init=True,
#     strict_match=True,
#     pre_step="float",
#     pre_step_checkpoint="job_data/models/float-checkpoint-last-d805135d.pth.tar",
#     # os.path.join(
#     #     ckpt_dir, f"{qat_pre_step}-checkpoint-best.pth.tar"
#     # ),
# )

# test
# test_db_levels = ["QR clean", "QR 10db", "QR 5db", "QR 0db", "QR m5db"]
QR_db_levels = [
    "QR clean",
    "QR 10db",
    "QR 5db",
    "QR 0db",
    "QR m5db",
    "QR m10db",
]
Qianren_db_levels = [
    "Qianren clean",
    "Qianren 10db",
    "Qianren 5db",
    "Qianren 0db",
    "Qianren m5db",
    "Qianren m10db",
]
test_db_levels = [*QR_db_levels, *Qianren_db_levels]
test_dataloaders = []
test_callbacks = []
for each in test_db_levels:
    data_type = each.split()[0]  # FYZ QR
    db_level = each.split()[1]  # "25db"
    db_test_info_txt = test_info_txt[data_type]
    db_test_hdf5_audio_path = os.path.join(
        test_hdf5_audio_path[data_type],
        f"{data_type}_mmasr_test_mic1_{db_level}.hdf5",
    )
    audio_reader = dict(
        type=HDF5WaveformReader,
        hdf5_file=db_test_hdf5_audio_path,
        ret_dtype="float32",
        channel_first=True,
    )
    image_reader = dict(
        type=TendisVideoImageReader,
        fps=25,
        origin_fps=30,
        zfill_num=6,
        missing_image_complete_mode="nearest",
        filter_rules={"missing_ratio": {"threshold": 0.3}},
        tendis_kwargs=tendis_kwargs,
    )
    test_dataset = dict(
        type=CocktailDatasetV0,
        audio_reader=audio_reader,
        image_reader=image_reader,
        info_list=dict(
            type=MMASRInfoList, path=db_test_info_txt, right_limit=10.0
        ),
        basic_info_reader=dict(
            type=TendisBasicInfoReader,
            tendis_kwargs=tendis_kwargs,
        ),
        mode="test",
        vad_expand=0.2,
        transforms=dict(
            type=torchvision.transforms.Compose,
            transforms=[
                # 音频处理
                dict(type=ReSample, new_freq=16000),
                # dict(
                #     type=OnLineAugmentPipeline,
                #     config=aug_config,
                #     noise_hdf5_file=noise_hdf5_file,
                #     ret_dtype="float32",
                #     gpu_rir_path=gpu_rir_path,
                # ),
                # dict(
                #     type=HisfInferencePipeline_v200_online,
                #     pre_pad=True,
                #     channle_select=True,
                #     p=1.0,
                # ),
                dict(type=ChannelSelect, channel=0),
                dict(
                    type=FBank,
                    num_mel_bins=fbank_size,
                    frame_shift=10,
                    frame_length=25,
                    dither=0.0,
                ),
                # 强制对齐
                dict(type=ForceAlignV2, audio_left_extend=7),
                # 图片处理
                dict(type=ImageListStack, hwc2chw=True),
                dict(type=ImageListToYUV444),
                dict(
                    type=ImageListNormalize,
                    mean=[128.0, 128.0, 128.0],
                    std=[128.0, 128.0, 128.0],
                ),
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=symbol_table,
                    bpe_model_path=None,
                    non_lang_syms=None,
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )
    test_dataloader = dict(
        type=DataLoader,
        dataset=test_dataset,
        batch_size=8,
        pin_memory=False,
        collate_fn=dict(type=CocktailCollate, ignore_id=-1),
        num_workers=2,
        sampler=dict(
            type=DistributedSampler,
            dataset=test_dataset,
            shuffle=False,
        ),
    )

    test_dataloaders.append(test_dataloader)

    test_metric_updater = dict(
        type="MetricUpdater",
        metric_update_func=update_metric,
        step_log_freq=step_log_freq,
        reset_metrics_by="epoch",
        epoch_log_freq=epoch_log_freq,
        log_prefix=model_prefix + f" {db_level}",
    )
    dump_audio_reader = deepcopy(audio_reader)
    dump_audio_reader["channel_first"] = False
    dump_badcase = dict(
        type=DumpBadCase,
        image_reader=image_reader,
        audio_reader=dump_audio_reader,
        out_root=f"dump/{task_name}",
        wer_threshold=0.5,
        prefix=f"{data_type}_{db_level}-",
    )
    each_callback = [test_metric_updater, dump_badcase]

    each_callback = [test_metric_updater]
    test_callbacks.append(each_callback)


# predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path="/mnt/mnt-data-4/shujie.luo/Projects/HAT/job_data/models/exp0305_strcture_asr_model_ao12_vo6_resnet18_pretrain_both_float-checkpoint-average-min_loss-30-fed55c81.pth.tar",
                verbose=True,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=test_dataloaders,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type=LossShow),
        dict(type=WER, name="wer"),
    ],
    callbacks=test_callbacks,
    share_callbacks=False,
    log_interval=100,
)

# ! SUBMIT_CONFIG
k8s_config = dict(
    # REQUIRED
    job_name=task_name,
    job_password="yaoyaoqiekenao",
    num_machines=1,
    num_gpus_per_machine=max(cluster_device_ids) + 1,
    # OPTIONAL
    framework="pytorch",
    task_label="CARP",
    # project_id="PDT2021005",  # default ProjectID, replace with your ProjectID
    project_id="LTCS2020101",
    input_bucket="J2MM",  # default use HDLTAlgorithm Bucket
    priority=5,
    docker_image=(
        "docker.hobot.cc/imagesys/hat:runtime-py3.8-torch1.10.2-cu111-1.1.1-mxnet"
    ),
    max_jobtime=10000,  # default 7200 = 5days
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        os.path.join(RootHelper.HAT_ROOT, "hat"),
        os.path.join(RootHelper.HAT_ROOT, "tools"),
        os.path.join(RootHelper.HAT_ROOT, "projects"),
    ],
    job_list=[
        f"python3 tools/trainv2.py \
        --config {os.path.relpath(__file__, RootHelper.HAT_ROOT)} \
            --stage float",
    ],
    custom_cmds=[
        "export PYTHONUNBUFFERED=0",
        "export LANG=en_US.UTF-8",
        "export NCCL_P2P_LEVEL=NVL",
        "export NCCL_DEBUG=INFO",
        "ln -s /job_data/models ${WORKING_PATH}/tmp_models",
        "export PYTHONPATH=${WORKING_PATH}:${WORKING_PATH}/projects/carp/plugins/wenet:$PYTHONPATH",  # noqa: E501
        "export LD_LIBRARY_PATH=${WORKING_PATH}/projects/halo/avspeech/data/noise/carmhisf/hisf_loader_v200:$LD_LIBRARY_PATH",  # noqa: E501
        "pip3 install torchmetrics==0.5.1 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",
    ],
)
