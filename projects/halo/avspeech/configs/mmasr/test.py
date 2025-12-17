# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.

from copy import deepcopy


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


from exp0358_strctureC_all_conformer_resnet18_only_json_av_noface_add_json_reweight_add_speech_noise import (  # noqa
    model,
)

ckpt = "/mnt/mnt-data-4/shujie.luo/Projects/HAT_mmasr/job_data/models/exp0358_strctureC_all_conformer_resnet18_only_json_av_noface_add_json_reweight_add_speech_noise_float-checkpoint-average-min_wer-5-8eaff59c.pth.tar"  # noqa

device_ids = [2]
assert len(device_ids) == 1
from common.predictor import (  # noqa
    load_ao_predictor,
    load_av_negative_predictor,
    load_av_predictor,
    load_json_av_predictor,
)

predictors = [load_av_predictor(model, ckpt, False, dump_badcase=False)]
# predictors = [load_ao_predictor(model, ckpt)]


float_predictor = merge_predictor(predictors)
