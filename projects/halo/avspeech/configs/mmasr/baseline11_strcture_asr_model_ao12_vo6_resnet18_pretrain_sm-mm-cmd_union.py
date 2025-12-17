# Copyright (c) Horizon Robotics, All rights reserved.
# flake8: noqa

import os
import pathlib
from copy import deepcopy

import torch
from aidisdk.utils import running_in_cluster
from common.default import bucket, global_keys
from datasets.audio_datasets import (
    get_wenet_eval_dataset,
    get_wenet_train_dataset,
)
from datasets.av_datasets import get_eval_dataset, get_train_dataset
from easydict import EasyDict as edict
from horizon_plugin_pytorch.march import March
from models.models import get_avsr_fusionformer
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
device_ids = cluster_device_ids if _in_cluster else [0]

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
model = get_avsr_fusionformer(model_config)
# model["encoder"]["video_frontend"] = "resnet18"


# -------------------------------------------------------------------------
#
# 训练集配置
#
# ------------------------------------------------------------------------
mm_train_data_config = edict()

# 多模数据依赖路径
mm_train_data_config.use_hisf = False
mm_train_data_config.rir = f"{bucket}/J2MM/speech/hdf5/gpu_rirs/gpu_rirs.hdf5"
mm_train_data_config.noise_info = (
    f"{bucket}/J2MM/speech/hdf5/noise_info_add_asr_noise.txt"
)
mm_train_data_config.noise_hdf5_file = (
    f"{bucket}/J2MM/speech/hdf5/noise_add_asr_noise.hdf5"
)
mm_train_data_config.augment_config_path = join(
    data_root, "noise/exp_mmasr_augment_linearly_no_norm_low_snr_add_cmd.yaml"
)
mm_train_data_config.info = f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/train_fyz_j2mm_1000id_batch01-03_datatang_no-j2mm-cmd.info"

sm_train_data_config = edict()

# 单模数据依赖路径
sm_train_data_config.info = (
    f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/bigdata/data.list"
)

cmd_train_data_config = deepcopy(mm_train_data_config)

# 多模命令词数据依赖路径
cmd_train_data_config.info = (
    f"{bucket}/J2MM/speech/hdf5/mmasr_info/train/train_mmcmd.info"
)

mm_train_dataset = get_train_dataset(config=mm_train_data_config)
sm_train_dataset = get_wenet_train_dataset(config=sm_train_data_config)
cmd_train_dataset = get_train_dataset(config=cmd_train_data_config)

mm_train_dataloader = dict(
    type=DataLoader,
    dataset=mm_train_dataset,
    batch_size=get_config("batch_size"),
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=-1),
    num_workers=get_config("num_workers"),
    sampler=dict(
        type=DistRandomBucketSampler,
        dataset=mm_train_dataset,
        batch_size=get_config("batch_size"),
        shuffle=True,
        bucket_size=10000,
    ),
    prefetch_factor=4,
)

sm_train_dataloader = dict(
    type=DataLoader,
    dataset=sm_train_dataset,
    batch_size=None,
    pin_memory=False,
    num_workers=get_config("num_workers"),
    prefetch_factor=4,
)

cmd_train_dataloader = dict(
    type=DataLoader,
    dataset=cmd_train_dataset,
    batch_size=get_config("batch_size"),
    pin_memory=False,
    collate_fn=dict(type=CocktailCollate, ignore_id=-1),
    num_workers=get_config("num_workers"),
    sampler=dict(
        type=DistRandomBucketSampler,
        dataset=cmd_train_dataset,
        batch_size=get_config("batch_size"),
        shuffle=True,
        bucket_size=10000,
    ),
    prefetch_factor=4,
)

train_dataloader = dict(
    type=ChainDataloader,
    dataloaders=[
        mm_train_dataloader,
        sm_train_dataloader,
        cmd_train_dataloader,
    ],
    proportions=[0.64, 0.31, 0.05],
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
    "Qianren": f"{bucket}/J2MM/speech/hdf5/mmasr_val/Qianren_val_mixed_no_hisf",
}
val_info = {
    "FYZ": f"{bucket}/J2MM/speech/hdf5/mmasr_info/val/val.info",
    "QR": f"{bucket}/J2MM/speech/hdf5/mmasr_info/val/QR_batch01_mmasr_val_source_proc_check.info",
    "Qianren": f"{bucket}/J2MM/speech/hdf5/mmasr_info/val/Qianren_batch01_mmasr_hy_val_source_proc_check.info",
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
    # val_on_train_begin=True,
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
_trainer.av_pt = f"{bucket}/J2MM/speech/hdf5/mmasr_pretrain_model/exp0308_strcture_asr_model_ao12_vo6_resnet18_pretrain_both_norm_float-checkpoint-average-min_wer-30-6f1ff3cc.pth.tar"


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
    "audio_cmvn",
    "vfea_extractor",
    "encoder.audio_encoder",
    "encoder.video_encoder",
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
                checkpoint_path=_trainer.av_pt,
                ignore_extra=True,
                verbose=True,
                allow_miss=True,
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
