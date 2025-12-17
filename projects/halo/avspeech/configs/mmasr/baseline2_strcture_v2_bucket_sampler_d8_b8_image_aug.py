# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.


# flake8: noqa

import copy
import os
import pathlib

import torch
import torchvision
from aidisdk.utils import running_in_cluster
from horizon_plugin_pytorch.march import March
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hat.callbacks.checkpoint import FlexibleStateDict
from hat.callbacks.lr_updater import NoamLrUpdater
from hat.callbacks.tensorboard import TensorBoard
from hat.data.collates.collates import CocktailCollate
from hat.data.datasets.avspeech.audio import HDF5WaveformReader
from hat.data.datasets.avspeech.basic_data_info import TendisBasicInfoReader
from hat.data.datasets.avspeech.image import TendisVideoImageReader
from hat.data.datasets.avspeech.info import MMASRInfoList
from hat.data.datasets.avspeech.symbol_table import SymbolTable
from hat.data.datasets.avspeech_dataset import CocktailDatasetV0
from hat.data.samplers.dist_bucket_sampler import DistOrderBucketSampler
from hat.data.transforms.avspeech.align import ForceAlign
from hat.data.transforms.avspeech.audio import FBank, ReSample, SpecAug
from hat.data.transforms.avspeech.images import (
    CoarseDropout,
    FancyPCA,
    HueSaturationValue,
    ImageListNormalize,
    ImageListStack,
    ImageListToYUV444,
    RandomBrightnessContrast,
)
from hat.data.transforms.avspeech.text import LabelToTensor, Tokenize
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.metrics.avspeech_metric import WER
from hat.metrics.loss_show import LossShow
from hat.models.backbones.vargnetv2 import CocktailVargNetV2
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.losses.smoothing_loss import LabelSmoothingLoss
from hat.models.structures.carp_structure_sA import CocktailE2EStructureA
from hat.models.task_modules.avspeech.carp_encoder import (
    CocktailE2EStructureAEncoder,
)
from hat.models.task_modules.avspeech.carp_feature import (
    CausalConv2DFeature,
    SimpleMixBlock,
)
from hat.models.task_modules.avspeech.cmvn import GlobalCMVN, load_cmvn
from hat.models.task_modules.avspeech.ctc_greedy import CTCGreedyDecoder
from hat.models.task_modules.avspeech.embed import Embedding
from hat.models.task_modules.avspeech.mask import ChunkMask, TargetMask
from hat.models.task_modules.avspeech.speech_decoder import (
    CTCDecoder,
    TransformerDecoder,
)
from hat.utils.root_helper import RootHelper

join = os.path.join

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

# ! HAT ENGINE 配置
# 使用配置文件的名字作为 task_name
# 可以手动设置名字
task_name = pathlib.Path(__file__).stem
# 保存模型的前缀名称
model_prefix = "MMASR-SA"

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES
qat_mode = "fuse_bn"


# ! 训练相关的配置
CARP_ROOT = RootHelper.get_project_root("halo/avspeech")
data_root = os.path.join(CARP_ROOT, "data")
symbol_table_path = os.path.join(data_root, "aishell/dict/lang_char.txt")
bpe_model_path = None

is_json_cmvn = True
cmvn_file = os.path.join(data_root, "aishell/train/global_cmvn")
if cmvn_file is not None:
    mean, istd = load_cmvn(cmvn_file, is_json_cmvn)
    global_cmvn = GlobalCMVN(
        torch.from_numpy(mean).float(), torch.from_numpy(istd).float()
    )
else:
    global_cmvn = None

cluster_device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
local_device_ids = [0]
if running_in_cluster():
    # 集群训练的一些配置
    device_ids = cluster_device_ids
    batch_size = 12
    hdf5_audio_path = "/bucket/input/J2MM/speech/hdf5/audio.hdf5"  # cluster
    test_hdf5_audio_path = (
        "/bucket/input/J2MM/speech/hdf5/mmasr_test/QR_test_mixed_no_hisf"
    )
    ckpt_dir = f"/job_data/models/{task_name}/"
    epoch_log_freq = 1
    step_log_freq = 100
    train_info_txt = os.path.join(data_root, "fyz_mmasr_chg/train.info")
    val_info_txt = os.path.join(data_root, "fyz_mmasr_chg/val.info")
    test_info_txt = os.path.join(data_root, "mmasr_test/test_noalpha.info")
    sampler_bucket_size = 10000
    checkpoint_path = "/cluster_home/plat_gpu/exp0056_strcture_v2_bucket_sampler_d8_b8-20220727-191619.983136/output/models/exp0056_strcture_v2_bucket_sampler_d8_b8/float-checkpoint-epoch-0062-4593876c.pth.tar"
else:
    # 本地训练的一些配置
    device_ids = local_device_ids
    batch_size = 4
    hdf5_audio_path = "/horizon-bucket/J2MM/speech/hdf5/audio.hdf5"  # local
    test_hdf5_audio_path = (
        "/horizon-bucket/J2MM/speech/hdf5/mmasr_test/QR_test_mixed_no_hisf"
    )
    ckpt_dir = f"./job_data/models/{model_prefix}/"
    epoch_log_freq = 1
    step_log_freq = 10
    train_info_txt = os.path.join(data_root, "fyz_mmasr_chg_dev/train.info")
    val_info_txt = os.path.join(data_root, "fyz_mmasr_chg/val.info")
    test_info_txt = os.path.join(data_root, "mmasr_test/test_noalpha.info")
    sampler_bucket_size = 200
    checkpoint_path = "/mnt/mnt-data-4/shujie.luo/Projects/HAT/job_data/models/float-checkpoint-last-d805135d.pth.tar"


test_db_level = ["clean", "10db", "0db", "m5db"]  # clean, 10db, 0db, m5db
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
num_epochs = 10000

# 网络结构相关配置
encoder_output_size = 256
bn_kwargs = dict(eps=1e-5, momentum=0.1)
# model 构造
model = dict(
    type=CocktailE2EStructureA,
    vocab_size=vocab_size,
    vfea_extractor=dict(
        type=CocktailVargNetV2,
        bn_kwargs=bn_kwargs,
        model_type="tinyvargnetv2",
        alpha=1.0,
        group_base=8,
        factor=1,
        bias=True,
        flat_output=True,
        input_channels=3,
        head_factor=1,
        input_resize_scale=None,
        top_layer=dict(
            type=ConvModule2d,
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            groups=1,
            bias=False,
            norm_layer=dict(type=BatchNorm2d, num_features=256, **bn_kwargs),
            act_layer=None,
        ),
    ),
    afea_extractor=global_cmvn,
    encoder_mask=dict(
        type=ChunkMask,
        use_dynamic_chunk=True,
        use_dynamic_left_chunk=False,
        decoding_chunk_size=0,
        static_chunk_size=24,
        num_decoding_left_chunks=-1,
    ),
    encoder=dict(
        type=CocktailE2EStructureAEncoder,
        vfea_module=dict(
            type=CausalConv2DFeature,
            in_channels=256,
            out_channels=256,
            L=2,
            downsample=False,
            dropout_rate=0.1,
            activation=nn.ReLU(),
            bn_kwargs=bn_kwargs,
        ),
        afea_module=dict(
            type=CausalConv2DFeature,
            in_channels=80,
            out_channels=256,
            L=2,
            downsample=True,
            dropout_rate=0.1,
            activation=nn.ReLU(),
            bn_kwargs=bn_kwargs,
        ),
        mix_module=dict(
            type=SimpleMixBlock,
            mode="cat",
            dim=1,
        ),
        downsample_module=dict(
            type=CausalConv2DFeature,
            in_channels=512,
            out_channels=256,
            L=0,
            dropout_rate=0.1,
        ),
        n_feat=encoder_output_size,
        dropout_rate=0.1,
        conformer_units=6,
        feed_forward_hidden_dim=2048,
        feed_forward_dropout_rate=0.1,
        attn_head=4,
        attn_dropout_rate=0.1,
        conv_module_inner_channels=encoder_output_size * 2,
        conv_module_kernel_size=7,
        conv_module_bias=True,
        activation=nn.SiLU(),
    ),
    ctc=dict(
        type=CTCDecoder,
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        dropout_rate=0.1,
    ),
    ctc_decoder=dict(
        type=CTCGreedyDecoder,
        vocab_path=symbol_table_path,
        blank_id=0,
        keep_blank=True,
    ),
    ctc_weight=0.3,
    ctc_loss=dict(
        type=nn.CTCLoss,
        reduction="sum",
        zero_infinity=True,
    ),
    ignore_id=-1,
    att_loss=dict(
        type=LabelSmoothingLoss,
        size=vocab_size,
        padding_idx=-1,
        smoothing=0.1,
        normalize_length=False,
    ),
    reverse_weight=0.3,
    att_decoder_embed=dict(
        type=Embedding,
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        positional_dropout_rate=0.1,
    ),
    att_decoder_mask=dict(
        type=TargetMask,
    ),
    att_decoder=dict(
        type=TransformerDecoder,
        n_feat=encoder_output_size,
        vocab_size=vocab_size,
        dropout_rate=0.1,
        decoder_units=3,
        self_attn_n_head=4,
        self_attn_dropout_rate=0.1,
        src_attn_n_head=4,
        src_attn_dropout_rate=0.1,
        feed_forward_hidden_dim=encoder_output_size * 8,
        feed_forward_dropout_rate=0.1,
        activation=nn.ReLU(),
    ),
    reverse_att_decoder=dict(
        type=TransformerDecoder,
        n_feat=encoder_output_size,
        vocab_size=vocab_size,
        dropout_rate=0.1,
        decoder_units=3,
        self_attn_n_head=4,
        self_attn_dropout_rate=0.1,
        src_attn_n_head=4,
        src_attn_dropout_rate=0.1,
        feed_forward_hidden_dim=encoder_output_size * 4,
        feed_forward_dropout_rate=0.1,
        activation=nn.ReLU(),
    ),
)

train_dataset = dict(
    type=CocktailDatasetV0,
    image_reader=dict(
        type=TendisVideoImageReader,
        fps=25,
        origin_fps=30,
        zfill_num=6,
        missing_image_complete_mode="zero",
        filter_rules=None,
        tendis_kwargs=tendis_kwargs,
    ),
    audio_reader=dict(
        type=HDF5WaveformReader,
        hdf5_file=hdf5_audio_path,
        ret_dtype="float32",
    ),
    info_list=dict(type=MMASRInfoList, path=train_info_txt),
    basic_info_reader=dict(
        type=TendisBasicInfoReader, tendis_kwargs=tendis_kwargs
    ),
    mode="train",
    transforms=dict(
        type=torchvision.transforms.Compose,
        transforms=[
            # 音频处理
            dict(type=ReSample, new_freq=16000),
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
            ),
            # 强制对齐
            dict(type=ForceAlign),
            # 图片处理
            dict(
                type=RandomBrightnessContrast,
                brightness_limit=0.8,
                contrast_limit=0.4,
                brightness_by_max=True,
                p=0.5,
            ),
            dict(
                type=CoarseDropout,
                max_holes=1,
                max_height=96,
                max_width=40,
                fill_value=[183, 197, 248],
                p=0.5,
            ),
            dict(
                type=HueSaturationValue,
                hue_shift_limit=54,
                sat_shift_limit=40,
                val_shift_limit=0,
                p=0.5,
            ),
            dict(
                type=FancyPCA,
                alpha=0.8,
                p=0.5,
            ),
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

train_dataloader = dict(
    type=DataLoader,
    dataset=train_dataset,
    batch_size=batch_size,
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=-1),
    num_workers=6,
    sampler=dict(
        type=DistOrderBucketSampler,
        dataset=train_dataset,
        shuffle=True,
        bucket_size=sampler_bucket_size,
        reverse=False,
    ),
)

val_dataset = dict(
    type=CocktailDatasetV0,
    image_reader=dict(
        type=TendisVideoImageReader,
        fps=25,
        origin_fps=30,
        zfill_num=6,
        missing_image_complete_mode="zero",
        filter_rules=None,
        tendis_kwargs=tendis_kwargs,
    ),
    audio_reader=dict(
        type=HDF5WaveformReader,
        hdf5_file=hdf5_audio_path,
        ret_dtype="float32",
    ),
    info_list=dict(type=MMASRInfoList, path=val_info_txt),
    basic_info_reader=dict(
        type=TendisBasicInfoReader,
        tendis_kwargs=tendis_kwargs,
    ),
    mode="val",
    transforms=dict(
        type=torchvision.transforms.Compose,
        transforms=[
            # 音频处理
            dict(type=ReSample, new_freq=16000),
            dict(
                type=FBank,
                num_mel_bins=fbank_size,
                frame_shift=10,
                frame_length=25,
                dither=0.0,
            ),
            # 强制对齐
            dict(type=ForceAlign),
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
    sampler=dict(type=DistributedSampler, dataset=val_dataset, shuffle=False),
)

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


def update_metric(metrics, batch, model_outs):
    loss, loss_att, loss_ctc, att_acc, ctc_out = model_outs
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

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=step_log_freq,
    reset_metrics_by="epoch",
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

val_callback = dict(
    type="Validation",
    data_loader=val_dataloader,
    batch_processor=val_batch_processor,
    callbacks=[
        val_metric_updater,
        val_tb_callback,
    ],
    log_interval=0,
    interval_by="epoch",
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


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=checkpoint_path,
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
        train_tb_callback,
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

float_solver = dict(
    trainer=float_trainer,
    quantize=False,
    allow_not_init=True,
    strict_match=True,
)

step2solver = dict(float=float_solver)

# predictor

test_dataloaders = []
test_callbacks = []
for db in test_db_level:
    db_val_hdf5_audio_path = os.path.join(
        test_hdf5_audio_path, f"QR_mmasr_test_{db}.hdf5"
    )
    test_dataset = dict(
        type=CocktailDatasetV0,
        image_reader=dict(
            type=TendisVideoImageReader,
            fps=25,
            origin_fps=30,
            zfill_num=6,
            missing_image_complete_mode="zero",
            filter_rules=None,
            tendis_kwargs=tendis_kwargs,
        ),
        audio_reader=dict(
            type=HDF5WaveformReader,
            hdf5_file=db_val_hdf5_audio_path,
            ret_dtype="float32",
        ),
        info_list=dict(type=MMASRInfoList, path=test_info_txt),
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
                dict(
                    type=FBank,
                    num_mel_bins=fbank_size,
                    frame_shift=10,
                    frame_length=25,
                    dither=0.0,
                ),
                # 强制对齐
                dict(type=ForceAlign),
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
        batch_size=batch_size,
        pin_memory=False,
        collate_fn=dict(type=CocktailCollate, ignore_id=-1),
        num_workers=2,
        sampler=dict(
            type=DistributedSampler, dataset=test_dataset, shuffle=False
        ),
    )

    test_metric_updater = dict(
        type="MetricUpdater",
        metric_update_func=update_metric,
        step_log_freq=step_log_freq,
        reset_metrics_by="epoch",
        epoch_log_freq=epoch_log_freq,
        log_prefix=model_prefix + f"_{db}",
    )
    test_dataloaders.append(test_dataloader)
    test_callbacks.append(test_metric_updater)


float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path="/mnt/mnt-data-4/shujie.luo/Projects/HAT/job_data/models/mmasr_image_aug_float-checkpoint-epoch-0080-2c7f65ae.pth.tar",
            ),
        ],
    ),
    data_loader=test_dataloaders,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type=WER, name="wer"),
    ],
    callbacks=test_callbacks,
    share_callbacks=False,
    log_interval=50,
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
            --stage float"
    ],
    custom_cmds=[
        "export PYTHONUNBUFFERED=0",
        "export LANG=en_US.UTF-8",
        "export NCCL_P2P_LEVEL=NVL",
        "export NCCL_DEBUG=INFO",
        "ln -s /job_data/models ${WORKING_PATH}/tmp_models",
        "export PYTHONPATH=${WORKING_PATH}:${WORKING_PATH}/projects/carp/plugins/wenet:$PYTHONPATH",  # noqa: E501
        "pip3 install torchmetrics==0.5.1 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",
    ],
)
