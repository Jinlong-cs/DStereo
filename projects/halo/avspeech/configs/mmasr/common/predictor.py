# Copyright (c) Horizon Robotics, All rights reserved.

# flake8: noqa

import os
from copy import deepcopy

from common.callbacks import (
    test_batch_processor,
    test_metric_updater,
    test_metrics,
    vo_test_batch_processor,
)
from common.default import bucket, tendis_kwargs
from datasets.audio_datasets import get_wenet_eval_dataset
from datasets.av_datasets import get_eval_dataset, get_json_eval_dataset
from datasets.video_datasets import (
    get_vo_eval_dataset,
    get_vo_json_eval_dataset,
)
from easydict import EasyDict as edict
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hat.data.collates.collates import CocktailCollate
from projects.halo.avspeech.utils.error_analysis import (
    DumpBadCase,
    DumpJsonBadCase,
    DumpResult,
)
from projects.halo.avspeech.utils.global_config import get_config

join = os.path.join


STAGES = ["float", "qat", "quantize"]


def get_converters(ckpt, stage):
    assert stage in STAGES, f"ERROR stage {stage} not in {STAGES}"

    converters = []
    if stage == "qat" or stage == "quantize":
        converters.append(dict(type="Float2QAT"))
    converters.append(
        dict(type="LoadCheckpoint", checkpoint_path=ckpt, verbose=True)
    )
    if stage == "quantize":
        converters.append(dict(type="QAT2Quantize"))
    return converters


def merge_predictor(predictors):
    if len(predictors) == 1:
        return predictors[0]

    merged_predictor = deepcopy(predictors[0])
    callbacks = merged_predictor["callbacks"]
    dataloaders = merged_predictor["data_loader"]
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]
        callbacks = [callbacks]

    for pred in predictors[1:]:
        _dataloaders = pred["data_loader"]
        _callbacks = pred["callbacks"]
        if not isinstance(_dataloaders, list):
            _dataloaders = [_dataloaders]
            _callbacks = [_callbacks]
        dataloaders.extend(_dataloaders)
        callbacks.extend(_callbacks)

    merged_predictor["callbacks"] = callbacks
    merged_predictor["data_loader"] = dataloaders
    merged_predictor["share_callbacks"] = False

    return merged_predictor


def load_av_predictor(
    model, ckpt, use_hisf=True, stage="float", dump_badcase=False
):
    info_dict = {
        "FYZ": f"{bucket}/J2MM/speech/hdf5/mmasr_info/val/val.info",
        "QR": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/QR_batch01_mmasr_test_alpha_proc_check.info",
        "Qianren": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/Qianren_batch01_mmasr_test_no_alpha_proc_check.info",
    }

    hdf5_dict = {
        "FYZ": f"{bucket}/J2MM/speech/hdf5/mmasr_val/FYZ_val_mixed_no_hisf",
        "QR": f"{bucket}/J2MM/speech/hdf5/mmasr_test/QR_test_mixed_no_hisf",
        "Qianren": f"{bucket}/J2MM/speech/hdf5/mmasr_test/Qianren_test_mixed_no_hisf",
    }

    sets = ["QR", "Qianren"]
    dbs = ["clean", "10db", "5db", "0db", "m5db", "m10db"]
    # subsets = [f"{_set} {_db}" for _set in sets for _db in dbs[-2:]]
    subsets = [f"{_set} {_db}" for _set in sets for _db in dbs]

    test_dataloaders = []
    test_callbacks = []
    for subset in subsets:
        _set, _db = subset.split()  # FYZ clean

        config = edict(
            use_hisf=use_hisf,
            info=info_dict[_set],
            hdf5=join(hdf5_dict[_set], f"{_set}_mmasr_test_mic1_{_db}.hdf5"),
            mode="test",
        )
        dataset = get_eval_dataset(config)

        dataloader = dict(
            type=DataLoader,
            dataset=dataset,
            batch_size=8,
            pin_memory=False,
            collate_fn=dict(type=CocktailCollate, ignore_id=-1),
            num_workers=2,
            sampler=dict(
                type=DistributedSampler,
                dataset=dataset,
                shuffle=False,
            ),
        )

        callback = deepcopy(test_metric_updater)
        callback["log_prefix"] = get_config("model_prefix") + f" {subset}"

        if dump_badcase:
            image_reader = deepcopy(dataset["image_reader"])
            audio_reader = deepcopy(dataset["audio_reader"])
            audio_reader["channel_first"] = False
            dumpbadcase = dict(
                type=DumpBadCase,
                image_reader=image_reader,
                audio_reader=audio_reader,
                out_root=f"dump/{subset.replace(' ', '_')}",
                wer_threshold=0.5,
                prefix=f"{subset.replace(' ', '_')}_",
            )
            test_callbacks.append([callback, dumpbadcase])
        else:
            test_callbacks.append([callback])

        test_dataloaders.append(dataloader)

    # predictor
    predictor = dict(
        type="Predictor",
        model=model,
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=get_converters(ckpt=ckpt, stage=stage),
        ),
        data_loader=test_dataloaders,
        batch_processor=deepcopy(test_batch_processor),
        device=None,
        metrics=test_metrics,
        callbacks=test_callbacks,
        share_callbacks=False,
        log_interval=100,
    )

    return predictor


def load_av_negative_predictor(
    model, ckpt, use_hisf=True, stage="float", dump_badcase=False
):
    info_dict = {
        "FYZ": f"{bucket}/J2MM/speech/hdf5/mmasr_info/val/val.info",
        "QR": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/QR_batch01_mmasr_test_alpha_proc_check.info",
        "Qianren": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/Qianren_batch01_mmasr_test_no_alpha_proc_check.info",
    }

    hdf5_dict = {
        "FYZ": f"{bucket}/J2MM/speech/hdf5/mmasr_val/FYZ_val_mixed_no_hisf",
        "QR": f"{bucket}/J2MM/speech/hdf5/mmasr_test/QR_test_mixed_no_hisf",
        "Qianren": f"{bucket}/J2MM/speech/hdf5/mmasr_test/Qianren_test_mixed_no_hisf",
    }

    sets = ["QR", "Qianren"]
    dbs = ["clean"]
    # subsets = [f"{_set} {_db}" for _set in sets for _db in dbs[-2:]]
    subsets = [f"{_set} {_db}" for _set in sets for _db in dbs]

    test_dataloaders = []
    test_callbacks = []
    for subset in subsets:
        _set, _db = subset.split()  # FYZ clean

        config = edict(
            use_hisf=use_hisf,
            info=info_dict[_set],
            hdf5=join(hdf5_dict[_set], f"{_set}_mmasr_test_mic1_{_db}.hdf5"),
            mode="test",
            negative_info=f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/QR_test_negative.info",
        )
        dataset = get_eval_dataset(config)

        dataloader = dict(
            type=DataLoader,
            dataset=dataset,
            batch_size=32,
            pin_memory=False,
            collate_fn=dict(type=CocktailCollate, ignore_id=-1),
            num_workers=2,
            sampler=dict(
                type=DistributedSampler,
                dataset=dataset,
                shuffle=False,
            ),
        )

        callback = deepcopy(test_metric_updater)
        callback["log_prefix"] = get_config("model_prefix") + f" {subset}_neg"

        if dump_badcase:
            image_reader = deepcopy(dataset["image_reader"])
            audio_reader = deepcopy(dataset["audio_reader"])
            audio_reader["channel_first"] = False
            dumpbadcase = dict(
                type=DumpBadCase,
                image_reader=image_reader,
                audio_reader=audio_reader,
                out_root=f"dump/{subset.replace(' ', '_')}",
                wer_threshold=0.5,
                prefix=f"{subset.replace(' ', '_')}_",
            )
            test_callbacks.append([callback, dumpbadcase])
        else:
            test_callbacks.append([callback])

        test_dataloaders.append(dataloader)

    # predictor
    predictor = dict(
        type="Predictor",
        model=model,
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=get_converters(ckpt=ckpt, stage=stage),
        ),
        data_loader=test_dataloaders,
        batch_processor=deepcopy(test_batch_processor),
        device=None,
        metrics=test_metrics,
        callbacks=test_callbacks,
        share_callbacks=False,
        log_interval=100,
    )

    return predictor


def load_json_av_predictor(
    model, ckpt, use_hisf=True, dump_badcase=False, stage="float"
):
    sets = ["batch23", "batch24", "batch25", "batch26"]
    # sets = ["batch26"]
    sets = []
    # dbs = ["clean", "10db", "5db", "0db", "m5db", "m10db"]
    dbs = [
        "clean",
        # "5db_cabin_denoised",
        "0db_cabin_noised_1",
        "0db_cabin_denoised",
        # "m5db_cabin_noised_1",
        "m5db_cabin_denoised",
    ]
    subsets = [f"{_set} {_db}" for _set in sets for _db in dbs]

    test_dataloaders = []
    test_callbacks = []
    for subset in subsets:
        _set, _db = subset.split()
        info = f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/{_set}_test_fix_1216_{_db}.json"

        config = edict(
            use_hisf=use_hisf,
            info=info,
            mode="test",
        )
        dataset = get_json_eval_dataset(config)

        dataloader = dict(
            type=DataLoader,
            dataset=dataset,
            batch_size=32,
            pin_memory=False,
            collate_fn=dict(type=CocktailCollate, ignore_id=-1),
            num_workers=2,
            sampler=dict(
                type=DistributedSampler,
                dataset=dataset,
                shuffle=False,
            ),
        )

        if dump_badcase:
            dumpbadcase = dict(
                type=DumpJsonBadCase,
                dataset=dataset,
                out_root=f"dump/{subset}",
                wer_threshold=0.5,
                prefix=f"{subset}",
            )
            dump_result = dict(
                type=DumpResult,
                out_root="dump/",
                prefix=get_config("model_prefix") + f"_{subset}",
            )
            # callback = [deepcopy(test_metric_updater), dumpbadcase, dump_result]
            callback = [deepcopy(test_metric_updater), dump_result]
            callback[0]["log_prefix"] = (
                get_config("model_prefix") + f" {subset}"
            )
            test_callbacks.append(callback)
        else:
            callback = deepcopy(test_metric_updater)
            callback["log_prefix"] = get_config("model_prefix") + f" {subset}"
            test_callbacks.append([callback])

        test_dataloaders.append(dataloader)

    infos = {
        # "10174_clean": "/jfs-hdfs/user/baozhong.ma/DATA/WAT_10174_train_noise_test1/dump_json_mp4/clean/data.json",
        # "10174_5db": "/jfs-hdfs/user/baozhong.ma/DATA/WAT_10174_train_noise_test1/dump_json_mp4/5db/data.json",
        # "10174_0db": "/jfs-hdfs/user/baozhong.ma/DATA/WAT_10174_train_noise_test1/dump_json_mp4/0db/data.json",
        # "10174_m5db": "/jfs-hdfs/user/baozhong.ma/DATA/WAT_10174_train_noise_test1/dump_json_mp4/m5db/data.json",
        # "10175_clean": "/jfs-hdfs/user/baozhong.ma/DATA/WAT_10175_train_noise_test1/dump_json_mp4/clean/data.json",
        # "10175_5db": "/jfs-hdfs/user/baozhong.ma/DATA/WAT_10175_train_noise_test1/dump_json_mp4/5db/data.json",
        # "10175_0db": "/jfs-hdfs/user/baozhong.ma/DATA/WAT_10175_train_noise_test1/dump_json_mp4/0db/data.json",
        # "10175_m5db": "/jfs-hdfs/user/baozhong.ma/DATA/WAT_10175_train_noise_test1/dump_json_mp4/m5db/data.json",
        "tiyan": "/jfs-hdfs/user/baozhong.ma/DATA/offline_test/taijia/0404_19/data.json",
        "noa": "/jfs-hdfs/user/lin03.wang/dms_raw/offline-test-data/badcase/car_en/dump_json/json/20230409_204545_511.json",
    }

    for subset, info in infos.items():
        config = edict(
            use_hisf=use_hisf,
            info=info,
            mode="test",
        )
        dataset = get_json_eval_dataset(config)

        dataloader = dict(
            type=DataLoader,
            dataset=dataset,
            batch_size=32,
            pin_memory=False,
            collate_fn=dict(type=CocktailCollate, ignore_id=-1),
            num_workers=2,
            sampler=dict(
                type=DistributedSampler,
                dataset=dataset,
                shuffle=False,
            ),
        )
        dumpbadcase = dict(
            type=DumpJsonBadCase,
            dataset=dataset,
            out_root=f"dump/{subset}",
            wer_threshold=0.5,
            prefix=f"{subset}",
        )
        if dump_badcase:
            # dump_result = dict(type=DumpResult, prefix=get_config("model_prefix") + f"_{subset}")
            dump_result = dict(
                type=DumpResult,
                out_root="dump/",
                prefix=get_config("model_prefix") + f"_{subset}",
            )
            # callback = [deepcopy(test_metric_updater), dumpbadcase]
            callback = [
                deepcopy(test_metric_updater),
                dumpbadcase,
                dump_result,
            ]
            callback[0]["log_prefix"] = (
                get_config("model_prefix") + f" {subset}"
            )
            test_callbacks.append(callback)
        else:
            callback = deepcopy(test_metric_updater)
            callback["log_prefix"] = get_config("model_prefix") + f" {subset}"
            test_callbacks.append([callback])

        test_dataloaders.append(dataloader)

    # predictor
    predictor = dict(
        type="Predictor",
        model=model,
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=get_converters(ckpt=ckpt, stage=stage),
        ),
        data_loader=test_dataloaders,
        batch_processor=deepcopy(test_batch_processor),
        device=None,
        metrics=test_metrics,
        callbacks=test_callbacks,
        share_callbacks=False,
        log_interval=100,
    )

    return predictor


def load_json_vo_predictor(model, ckpt, stage="float", dump_badcase=False):
    info_dict = {
        "batch23": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/batch23_test.json",
        "batch24": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/batch24_test.json",
        "batch25": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/batch25_test.json",
        "batch26": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/batch26_test.json",
    }

    test_dataloaders = []
    test_callbacks = []
    for subset, info in info_dict.items():
        config = edict(
            info=info,
            mode="test",
        )
        dataset = get_vo_json_eval_dataset(config)

        dataloader = dict(
            type=DataLoader,
            dataset=dataset,
            batch_size=8,
            pin_memory=False,
            collate_fn=dict(type=CocktailCollate, ignore_id=-1),
            num_workers=2,
            sampler=dict(
                type=DistributedSampler,
                dataset=dataset,
                shuffle=False,
            ),
        )
        dumpbadcase = dict(
            type=DumpJsonBadCase,
            dataset=dataset,
            out_root=f"dump/{subset}",
            wer_threshold=0.5,
            prefix=f"{subset}",
        )
        if dump_badcase:
            callback = [deepcopy(test_metric_updater), dumpbadcase]
            callback[0]["log_prefix"] = (
                get_config("model_prefix") + f" {subset}"
            )
            test_callbacks.append(callback)
        else:
            callback = deepcopy(test_metric_updater)
            callback["log_prefix"] = get_config("model_prefix") + f" {subset}"
            test_callbacks.append([callback])

        test_dataloaders.append(dataloader)

    # predictor
    predictor = dict(
        type="Predictor",
        model=model,
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=[
                dict(
                    type="LoadCheckpoint",
                    checkpoint_path=ckpt,
                    verbose=True,
                    allow_miss=True,
                    ignore_extra=True,
                ),
            ],
        ),
        data_loader=test_dataloaders,
        batch_processor=deepcopy(vo_test_batch_processor),
        device=None,
        metrics=test_metrics,
        callbacks=test_callbacks,
        share_callbacks=False,
        log_interval=100,
    )

    return predictor


def load_ao_predictor(model, ckpt, stage="float"):
    data_lists = {
        "common3500": "/mnt/mnt-data-4/shujie.luo/Data/MMASR/test/common_3500/data.list",
        "control_5500": "/mnt/mnt-data-4/shujie.luo/Data/MMASR/test/control_5500/data.list",
        "music_6000": "/mnt/mnt-data-4/shujie.luo/Data/MMASR/test/music_6000/data.list",
        "navigation_5000": "/mnt/mnt-data-4/shujie.luo/Data/MMASR/test/navigation_5000/data.list",
    }

    dataloaders = []
    callbacks = []
    for subset, data_list in data_lists.items():
        config = edict(info=data_list)
        dataset = get_wenet_eval_dataset(config)

        dataloader = dict(
            type=DataLoader,
            dataset=dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        )

        callback = deepcopy(test_metric_updater)
        callback["log_prefix"] = get_config("model_prefix") + f" {subset}"
        dataloaders.append(dataloader)
        callbacks.append(callback)
    # predictor
    predictor = dict(
        type="Predictor",
        model=model,
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=get_converters(ckpt=ckpt, stage=stage),
        ),
        data_loader=dataloaders,
        batch_processor=deepcopy(test_batch_processor),
        device=None,
        metrics=test_metrics,
        callbacks=callbacks,
        share_callbacks=False,
        log_interval=100,
    )

    return predictor


def load_vo_predictor(model, ckpt, stage="float"):
    info_dict = {
        # "FYZ": f"{bucket}/J2MM/speech/hdf5/mmasr_info/val/val.info",
        "QR": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/QR_batch01_mmasr_test_alpha_proc_check.info",
        "Qianren": f"{bucket}/J2MM/speech/hdf5/mmasr_info/test/Qianren_batch01_mmasr_test_no_alpha_proc_check.info",
    }

    test_dataloaders = []
    test_callbacks = []
    for subset, info in info_dict.items():
        config = edict(mode="test", info=info)
        dataset = get_vo_eval_dataset(config)
        dataloader = dict(
            type=DataLoader,
            dataset=dataset,
            batch_size=8,
            pin_memory=False,
            collate_fn=dict(type=CocktailCollate, ignore_id=-1),
            num_workers=2,
            sampler=dict(
                type=DistributedSampler,
                dataset=dataset,
                shuffle=False,
            ),
        )

        callback = deepcopy(test_metric_updater)
        callback["log_prefix"] = get_config("model_prefix") + f" {subset}"

        test_dataloaders.append(dataloader)
        test_callbacks.append([callback])

    # predictor
    predictor = dict(
        type="Predictor",
        model=model,
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=get_converters(ckpt=ckpt, stage=stage),
        ),
        data_loader=test_dataloaders,
        batch_processor=deepcopy(test_batch_processor),
        device=None,
        metrics=test_metrics,
        callbacks=test_callbacks,
        share_callbacks=False,
        log_interval=100,
    )

    return predictor
