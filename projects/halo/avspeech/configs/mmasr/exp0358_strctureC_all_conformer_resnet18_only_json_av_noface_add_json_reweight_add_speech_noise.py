# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.
# flake8: noqa

import math
import os
import pathlib
from copy import deepcopy

import torch
from aidisdk.utils import running_in_cluster
from common.default import bucket, global_keys
from datasets.audio_datasets import get_wenet_train_dataset
from datasets.av_datasets import (
    get_json_eval_dataset,
    get_json_train_dataset,
    get_train_dataset,
)
from easydict import EasyDict as edict
from horizon_plugin_pytorch.march import March
from models.models import get_avsr_conformer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hat.data.collates.collates import CocktailCollate
from hat.data.dataloaders.chain_dataloader import ChainDataloader
from hat.data.datasets.avspeech.symbol_table import SymbolTable
from hat.data.samplers.dist_bucket_sampler import (
    DistRandomBucketBatchSampler,
    DistRandomBucketSampler,
)
from hat.models.task_modules.avspeech.cmvn import GlobalCMVN, load_cmvn
from hat.models.task_modules.avspeech.utils import IGNORE_ID
from hat.utils.root_helper import RootHelper
from projects.halo.avspeech.utils.global_config import get_config, set_config

join = os.path.join

# -------------------------------------------------------------------------
#
# ! HAT ENGINE 配置
#
# ------------------------------------------------------------------------
seed = None
march = March.BAYES
qat_mode = "fuse_bn"
log_rank_zero_only = True
cudnn_benchmark = False
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

# -------------------------------------------------------------------------
#
# 通用配置
#
# ------------------------------------------------------------------------
task_name = pathlib.Path(__file__).stem
current_config = os.path.relpath(__file__, RootHelper.HAT_ROOT)
model_prefix = "MMASR-" + task_name.split("_")[0].upper()

_in_cluster = running_in_cluster()

cluster_device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
device_ids = cluster_device_ids if _in_cluster else [1]

num_epochs = 1000
batch_size = 32 if _in_cluster else 4
num_workers = 4 if _in_cluster else 2
epoch_log_freq = 1
step_log_freq = 100 if _in_cluster else 100

carp_root = RootHelper.get_project_root("halo/avspeech")
data_root = os.path.join(carp_root, "data")

fbank_size = 80
bpe_model_path = join(data_root, "xingchen/train/bpe_model.model")
symbol_table_path = join(data_root, "xingchen/dict/lang_char.txt")
symbol_table = SymbolTable(symbol_table_path)
vocab_size = len(symbol_table)

# -------------------------------------------------------------------------
#
# 初始化全局配置，主要用途是 callbacks 需要的一些参数
#
# ------------------------------------------------------------------------
for k in global_keys:
    if k in globals():
        if get_config(k) is None:
            set_config(k, eval(k))
        # 将本地的相应字段删除，避免使用本地变量
        globals().pop(k)

# -------------------------------------------------------------------------
#
# 模型结构相关配置
#
# ------------------------------------------------------------------------
import baseline10_strcture_asr_model_ao12_vo6_3dconv_resnet18_adamw_add_datatang as baseline

model_config = edict()  # model_config_config

model_config.cmvn_file = join(data_root, "xingchen/train/global_cmvn")
model_config.mean, model_config.istd = load_cmvn(model_config.cmvn_file, True)
model_config.global_cmvn = GlobalCMVN(
    torch.from_numpy(model_config.mean).float(),
    torch.from_numpy(model_config.istd).float(),
)

model_config.encoder_output_size = 384

model_config.bn_kwargs = dict(eps=1e-5, momentum=0.1)
model = get_avsr_conformer(model_config)

# -------------------------------------------------------------------------
#
# 训练集配置
#
# ------------------------------------------------------------------------

# 多模数据依赖路径
tendis_train_data_config = edict()
tendis_train_data_config.use_hisf = False
tendis_train_data_config.rir = (
    f"{bucket}/J2MM/speech/hdf5/gpu_rirs/gpu_rirs.hdf5"
)
tendis_train_data_config.noise_info = (
    f"{bucket}/J2MM/speech/hdf5/noise_info_add_asr_noise.txt"
)
tendis_train_data_config.noise_hdf5_file = (
    f"{bucket}/J2MM/speech/hdf5/noise_add_asr_noise.hdf5"
)
tendis_train_data_config.augment_config_path = "projects/halo/avspeech/data/noise/exp_mmasr_augment_linearly_no_norm_low_snr_add_cmd.yaml"
tendis_train_data_config.info = f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/train_fyz_j2mm_1000id_batch01-03_datatang_no-j2mm-cmd.info"
tendis_train_data_config.noise_prob = 0.5


tendis_train_dataset = get_train_dataset(config=tendis_train_data_config)

tendis_train_dataloader = dict(
    type=DataLoader,
    dataset=tendis_train_dataset,
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=IGNORE_ID),
    num_workers=1,
    batch_sampler=dict(
        type=DistRandomBucketBatchSampler,
        dataset=tendis_train_dataset,
        max_times_in_batch=120,
        shuffle=True,
        bucket_size=1024,
    ),
    prefetch_factor=8,
)

# 多模命令词数据集
cmd_train_data_config = deepcopy(tendis_train_data_config)
cmd_train_data_config.info = (
    f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/train_mmcmd_filter.info"
)
cmd_train_dataset = get_train_dataset(config=cmd_train_data_config)
cmd_train_dataloader = dict(
    type=DataLoader,
    dataset=cmd_train_dataset,
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=IGNORE_ID),
    num_workers=1,
    batch_sampler=dict(
        type=DistRandomBucketBatchSampler,
        dataset=cmd_train_dataset,
        max_times_in_batch=120,
        shuffle=True,
        bucket_size=1024,
    ),
    prefetch_factor=8,
)

# 短视频数据集
json_train_data_config = deepcopy(tendis_train_data_config)
json_train_data_config.use_hisf = False
json_train_data_config.info = (
    f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/face_batch16-batch38_add_merge_filter_tn.json"
    if _in_cluster
    else f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/dev.json"
)


json_train_dataset = get_json_train_dataset(json_train_data_config)

json_train_dataloader = dict(
    type=DataLoader,
    dataset=json_train_dataset,
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=IGNORE_ID),
    num_workers=get_config("num_workers"),
    # num_workers=4,
    batch_sampler=dict(
        type=DistRandomBucketBatchSampler,
        dataset=json_train_dataset,
        max_times_in_batch=120,
        shuffle=True,
        bucket_size=1024,
    ),
    prefetch_factor=8,
)


# 单模bigdata数据集
bigdata_train_config = edict(
    info=f"{bucket}/J2MM/speech/hdf5/mmasr_data/bigdata/train/data.list"
)
bigdata_train_dataset = get_wenet_train_dataset(bigdata_train_config)
bigdata_train_dataloader = dict(
    type=DataLoader,
    dataset=bigdata_train_dataset,
    batch_size=None,
    pin_memory=True,
    num_workers=4,
    # num_workers=get_config("num_workers"),
    prefetch_factor=8,
)

train_dataloader = dict(
    type=ChainDataloader,
    dataloaders=[
        tendis_train_dataloader,
        cmd_train_dataloader,
        json_train_dataloader,
        bigdata_train_dataloader,
    ],
    proportions=[0.18, 0.02, 0.3, 0.5],
    # batches=[1, 1, 2, 2],
)

# -------------------------------------------------------------------------
#
# 验证集和跑验证时的配置
#
# ------------------------------------------------------------------------
from common.callbacks import val_batch_processor, val_metric_updater

val_json_info = (
    f"{bucket}/J2MM/speech/hdf5/mmasr_info/val/json_batch23-26_val.json"
)
val_config = edict(
    use_hisf=False,
    info=val_json_info,
    mode="val",
)
json_val_dataset = get_json_eval_dataset(val_config)
json_val_dataloader = dict(
    type=DataLoader,
    dataset=json_val_dataset,
    batch_size=8,
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=IGNORE_ID),
    num_workers=1,
    sampler=dict(
        type=DistributedSampler, dataset=json_val_dataset, shuffle=False
    ),
)
json_callback = deepcopy(val_metric_updater)
json_callback["log_prefix"] = get_config("model_prefix") + " json_val"

val_callback = deepcopy(baseline.val_callback)
val_callback["data_loader"].append(json_val_dataloader)
val_callback["callbacks"].append(json_callback)

# -------------------------------------------------------------------------
#
# 训练配置
#
# ------------------------------------------------------------------------
from common.callbacks import (
    metrics,
    multibatch_train_batch_processor,
    train_batch_processor,
    train_callbacks,
)

train_batch_processor["batch_transforms"][0][
    "image_feat_size"
] = model_config.encoder_output_size
trainer_config = edict()
trainer_config.ao_pt = f"/{bucket}/J2MM/speech/hdf5/mmasr_pretrain_model/conformer_bigmodel_ziyu.pt"
trainer_config.vo_pt = f"{bucket}/J2MM/speech/hdf5/mmasr_pretrain_model/exp0224_strctureA_vo6_resnet18_adamw_json_relu_refactor_floor_float-checkpoint-average-last-7-9ca66a88.pth.tar"
freeze_modules = [
    # "audio_encoder",
    # "audio_cmvn",
    # "shared_encoder",
    # "decoder",
    # "ctc",
    # "vfea_extractor",
]

freeze_callback = dict(
    type="FreezeModule",
    modules=[freeze_modules for _ in range(num_epochs)],
    step_or_epoch=list(range(num_epochs)),
    update_by="epoch",
)

train_callbacks = deepcopy(train_callbacks)
train_callbacks.extend([val_callback]),

multibatch_train_batch_processor["enable_amp"] = True


def state_dict_update_func(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("audio_encoder.embed.linear"):
            value = value * math.sqrt(384.0)
        new_state_dict[key] = value

    return new_state_dict


# ckpt = f"{bucket}/J2MM/speech/hdf5/mmasr_pretrain_model/exp0322_strctureC_ao12_fusionformer_vo6_conformer_resnet18_add_merge_float-checkpoint-average-min_wer-5-594900e8.pth.tar"
# ckpt = f"{bucket}/J2MM/speech/hdf5/mmasr_pretrain_model/exp0350_strctureC_all_conformer_resnet18_only_json_av_noface_add_json_reweight_float-checkpoint-average-min_wer-15-4d3a6987.pth.tar"
ckpt = "http://fm-shujie-luo.train.hogpu.cc/plat_gpu/exp0355_strctureC_all_conformer_resnet18_only_json_av_noface_add_json_reweight_negative_samples_005-20230327_094113/output/models/exp0358_strctureC_all_conformer_resnet18_only_json_av_noface_add_json_reweight_add_speech_noise/float-checkpoint-epoch-0016-0c29494e.pth.tar"
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    # start_step=200000,
    # start_epoch=0,
    # resume_epoch_or_step=True,
    # resume_optimizer=True,
    # model_convert_pipeline=dict(
    #     type="ModelConvertPipeline",
    #     converters=[
    #         dict(
    #             type="LoadCheckpoint",
    #             checkpoint_path=ckpt,
    #             verbose=True,
    #             allow_miss=True,
    #             # state_dict_update_func=state_dict_update_func,
    #         ),
    #     ],
    # ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={},
        # weight_decay=1e-4,
        lr=0.001,
    ),
    batch_processor=train_batch_processor,
    num_epochs=num_epochs,
    device=None,
    callbacks=train_callbacks,
    train_metrics=metrics,
    val_metrics=metrics,
)


# -------------------------------------------------------------------------
#
# 测试配置
#
# ------------------------------------------------------------------------
from typing import Dict


def predict_state_update_func(state_dict: Dict[str, torch.Tensor]):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("audio_encoder.embed.linear"):
            value = value * math.sqrt(384.0)
        elif key.startswith("decoder") and "norm" in key:
            value = value.unsqueeze(1).unsqueeze(1)
        new_state_dict[key] = value

    return new_state_dict


if not _in_cluster:
    from common.predictor import load_av_predictor

    ckpt = "/mnt/mnt-data-4/shujie.luo/Projects/HAT_mmasr/job_data/models/exp0358_strctureC_all_conformer_resnet18_only_json_av_noface_add_json_reweight_add_speech_noise_float-checkpoint-average-min_wer-5-8eaff59c.pth.tar"
    float_predictor = load_av_predictor(
        model=model,
        ckpt=ckpt,
        use_hisf=False,
    )

    float_predictor.update(
        dict(
            model_convert_pipeline=dict(
                type="ModelConvertPipeline",
                converters=[
                    dict(
                        type="LoadCheckpoint",
                        checkpoint_path=ckpt,
                        allow_miss=True,
                        ignore_extra=True,
                        state_dict_update_func=predict_state_update_func,
                        verbose=True,
                    ),
                ],
            ),
        )
    )

# -------------------------------------------------------------------------
#
# 提交任务
#
# ------------------------------------------------------------------------
from common.submit_config import k8s_config
