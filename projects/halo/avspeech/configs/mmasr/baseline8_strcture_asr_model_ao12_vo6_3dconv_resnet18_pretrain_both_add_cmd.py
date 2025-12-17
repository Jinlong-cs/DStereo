# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.

# flake8: noqa

import os
import pathlib
from copy import deepcopy

import torch
from aidisdk.utils import running_in_cluster
from common.default import bucket, global_keys
from datasets.av_datasets import get_eval_dataset, get_train_dataset
from easydict import EasyDict as edict
from horizon_plugin_pytorch.march import March
from models.models import get_structureC
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hat.data.collates.collates import CocktailCollate
from hat.data.dataloaders.chain_dataloader import ChainDataloader
from hat.data.datasets.avspeech.symbol_table import SymbolTable
from hat.data.samplers.dist_bucket_sampler import DistRandomBucketSampler
from hat.models.task_modules.avspeech.cmvn import GlobalCMVN, load_cmvn
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
batch_size = 32 if _in_cluster else 8
num_workers = 8 if _in_cluster else 2
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
model_config = edict()  # model_config_config

model_config.cmvn_file = join(data_root, "xingchen/train/global_cmvn")
model_config.mean, model_config.istd = load_cmvn(model_config.cmvn_file, True)
model_config.global_cmvn = GlobalCMVN(
    torch.from_numpy(model_config.mean).float(),
    torch.from_numpy(model_config.istd).float(),
)

model_config.encoder_output_size = 256

model_config.bn_kwargs = dict(eps=1e-5, momentum=0.1)
model_config.mode = "AV"
model = get_structureC(model_config)
model["encoder"]["video_frontend"] = "conv3d+resnet18_bak"
# -------------------------------------------------------------------------
#
# 训练集配置
#
# ------------------------------------------------------------------------

mmasr_train_data_config = edict()

# 多模数据依赖路径
mmasr_train_data_config.use_hisf = False
mmasr_train_data_config.rir = (
    f"{bucket}/J2MM/speech/hdf5/gpu_rirs/gpu_rirs.hdf5"
)
mmasr_train_data_config.noise_info = (
    f"{bucket}/J2MM/speech/hdf5/noise_info.txt"
)
mmasr_train_data_config.noise_hdf5_file = (
    f"{bucket}/J2MM/speech/hdf5/noise.hdf5"
)
mmasr_train_data_config.augment_config_path = "projects/halo/avspeech/data/noise/exp_mmasr_augment_linearly_no_norm_low_snr_test.yaml"
mmasr_train_data_config.info = f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/train_fyz_j2mm_1000id_batch01-03_datatang_no-j2mm-cmd.info"
# mmasr_train_data_config.info = join(data_root, "fyz_mmasr_chg_dev/train.info")

mmcmd_train_data_config = mmasr_train_data_config
mmcmd_train_data_config.info = (
    f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/train_mmcmd.info"
)

mmasr_train_dataset = get_train_dataset(config=mmasr_train_data_config)
mmcmd_train_dataset = get_train_dataset(config=mmcmd_train_data_config)

mmasr_train_dataloader = dict(
    type=DataLoader,
    dataset=mmasr_train_dataset,
    batch_size=get_config("batch_size"),
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=-1),
    num_workers=get_config("num_workers"),
    sampler=dict(
        type=DistRandomBucketSampler,
        dataset=mmasr_train_dataset,
        batch_size=get_config("batch_size"),
        shuffle=True,
        bucket_size=10000,
    ),
    prefetch_factor=8,
)
mmcmd_train_dataloader = dict(
    type=DataLoader,
    dataset=mmcmd_train_dataset,
    batch_size=get_config("batch_size"),
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=-1),
    num_workers=get_config("num_workers"),
    sampler=dict(
        type=DistRandomBucketSampler,
        dataset=mmcmd_train_dataset,
        batch_size=get_config("batch_size"),
        shuffle=True,
        bucket_size=10000,
    ),
    prefetch_factor=8,
)
train_dataloader = dict(
    type=ChainDataloader,
    dataloaders=[mmasr_train_dataloader, mmcmd_train_dataloader],
    proportions=[0.95, 0.05],
)

# -------------------------------------------------------------------------
#
# 验证集和跑验证时的配置
#
# ------------------------------------------------------------------------
from common.callbacks import val_batch_processor, val_metric_updater  # noqa

val_hdf5 = {
    "FYZ": f"{bucket}/J2MM/speech/hdf5/mmasr_val/FYZ_val_mixed_no_hisf",
    "QR": f"{bucket}/J2MM/speech/hdf5/mmasr_val/QR_val_mixed_no_hisf",
}
val_info = {
    "FYZ": join(data_root, "fyz_mmasr_chg/val.info"),
    "QR": join(data_root, "fyz_mmasr_chg/qr_choose_500_for_val.info"),
}

dbs = ["clean", "0db", "m5db"]
sets = list(val_hdf5.keys())
subsets = [f"{_set} {_db}" for _set in sets for _db in dbs]

val_dataloaders = []
val_callbacks = []
for subset in subsets:
    _set, _db = subset.split()  # FYZ QR

    val_config = edict(
        use_hisf=False,
        hdf5=join(val_hdf5[_set], f"{_set}_mmasr_val_{_db}.hdf5"),
        info=val_info[_set],
        mode="val",
    )
    _dataset = get_eval_dataset(val_config)

    _dataloader = dict(
        type=DataLoader,
        dataset=_dataset,
        batch_size=8,
        pin_memory=False,
        collate_fn=dict(type=CocktailCollate, ignore_id=-1),
        num_workers=2,
        sampler=dict(type=DistributedSampler, dataset=_dataset, shuffle=False),
    )
    _callback = deepcopy(val_metric_updater)
    _callback["log_prefix"] = get_config("model_prefix") + f" {subset}"
    val_dataloaders.append(_dataloader)
    val_callbacks.append([_callback])

val_callback = dict(
    type="Validation",
    data_loader=val_dataloaders,
    batch_processor=deepcopy(val_batch_processor),
    callbacks=val_callbacks,
    log_interval=0,
    interval_by="epoch",
    share_callbacks=False,
)

# -------------------------------------------------------------------------
#
# 训练配置
#
# ------------------------------------------------------------------------
from common.callbacks import (  # noqa
    metrics,
    train_batch_processor,
    train_callbacks,
)

_trainer = edict()
_trainer.ao_pt = (
    "/horizon-bucket/J2MM/speech/hdf5/mmasr_pretrain_model/avg_30_xingchen.pt"
)
_trainer.vo_pt = "/horizon-bucket/J2MM/speech/hdf5/mmasr_pretrain_model/exp0107_strcture_asr_model_vo12_3dconv_resnet18_vision_6layers_float-checkpoint-average-20-7a3adb80.pth.tar"


def audio_state_dict_update_func(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder"):
            if "encoder.encoders" in key:
                key = key.replace("encoders", "audio_encoders")
            new_state_dict[key] = value

    return new_state_dict


def video_state_dict_update_func(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder"):
            new_state_dict[key] = value

    return new_state_dict


freeze_modules = [
    "encoder.embed",
    "encoder.audio_encoders",
    "encoder.vfea_extractor",
    "encoder.video_encoders",
    # "encoder.vfea_module",
]

freeze_callback = dict(
    type="FreezeModule",
    modules=[freeze_modules for _ in range(num_epochs)],
    step_or_epoch=list(range(num_epochs)),
    update_by="epoch",
)

train_callbacks.extend([freeze_callback, val_callback]),
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=_trainer.ao_pt,
                allow_miss=True,
                state_dict_update_func=audio_state_dict_update_func,
                # verbose=True,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=_trainer.vo_pt,
                allow_miss=True,
                state_dict_update_func=video_state_dict_update_func,
                # verbose=True,
            ),
        ],
    ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={},
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
from common.predictor import load_av_predictor  # noqa

ckpt = "job_data/models/exp0305_strcture_asr_model_ao12_vo6_resnet18_pretrain_both_float-checkpoint-average-min_loss-10-0acbc5b6.pth.tar"
float_predictor = load_av_predictor(
    model=model,
    ckpt=ckpt,
    use_hisf=False,
)


# -------------------------------------------------------------------------
#
# 提交任务
#
# ------------------------------------------------------------------------
from common.submit_config import k8s_config  # noqa
