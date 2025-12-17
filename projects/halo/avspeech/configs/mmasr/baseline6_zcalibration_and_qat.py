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
device_ids = cluster_device_ids if _in_cluster else [3]

num_epochs = 1000
batch_size = 32 if _in_cluster else 8
num_workers = 8 if _in_cluster else 2
epoch_log_freq = 1
step_log_freq = 100


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

# ! IMPORT FROM BASIC CONFIG
from common.callbacks import (  # noqa: E402
    calib_batch_processor,
    metrics,
    train_batch_processor,
    train_callbacks,
)
from config.tasks.mmasr.baseline6_strcture_asr_model_ao12_vo6_resnet18_pretrain_both_arange import (  # noqa: E408, E402
    model,
    train_dataloader,
)

# ?
# calibration 集合配置
calib_data_config = edict()

# 多模数据依赖路径
calib_data_config.use_hisf = False
calib_data_config.rir = f"{bucket}/J2MM/speech/hdf5/gpu_rirs/gpu_rirs.hdf5"
calib_data_config.noise_info = f"{bucket}/J2MM/speech/hdf5/noise_info.txt"
calib_data_config.noise_hdf5_file = f"{bucket}/J2MM/speech/hdf5/noise.hdf5"
calib_data_config.augment_config_path = "projects/halo/avspeech/data/noise/exp_mmasr_augment_linearly_no_norm_low_snr.yaml"
calib_data_config.info = (
    f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/train_fyz_j2mm_1000id.info"
)

calib_dataset = get_train_dataset(config=calib_data_config)

calib_dataloaer = dict(
    type=DataLoader,
    dataset=calib_dataset,
    batch_size=get_config("batch_size"),
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=-1, mode="calibration"),
    num_workers=get_config("num_workers"),
)


# ! "NEED PUT CKPT HERE!"
float_checkpoint_path = "NEED PUT CKPT HERE!"
float_checkpoint_path = "/mnt/mnt-data-1/ziyang01.wang/HAT/HAT/job_data/models/exp0158/float-checkpoint-average-min_wer-30-8447307f.pth.tar"
calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_checkpoint_path,
                ignore_extra=True,
                allow_miss=True,
                # state_dict_update_func=update_func,
                verbose=True,
            ),
            dict(type="Float2Calibration"),
        ],
    ),
    data_loader=calib_dataloaer,
    batch_processor=calib_batch_processor,
    num_steps=1250,
    device=device_ids[0],
    callbacks=train_callbacks,
    val_metrics=metrics,
    log_interval=get_config("step_log_freq"),
)


# ! "NEED PUT CKPT HERE!"
calib_checkpoint_path = "NEED PUT CKPT HERE!"
calib_checkpoint_path = "/mnt/mnt-data-1/ziyang01.wang/HAT/HAT/job_data/models/exp0126_split_fusion_norm_data_update_qat_on_baozhong0158/MMASR-EXP0126checkpoint-last-8ce5977a.pth.tar"
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=calib_checkpoint_path,
                ignore_extra=True,
                allow_miss=True,
                # state_dict_update_func=update_func,
                verbose=True,
            ),
        ],
        qconfig_params=dict(
            activation_qkwargs=dict(
                averaging_constant=0.0,
            ),
            weight_qkwargs=dict(
                averaging_constant=0.0,
            ),
        ),
    ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=1e-5,
        momentum=0.9,
    ),
    batch_processor=train_batch_processor,
    num_epochs=num_epochs,
    device=None,
    callbacks=train_callbacks,
    train_metrics=metrics,
    val_metrics=metrics,
)
