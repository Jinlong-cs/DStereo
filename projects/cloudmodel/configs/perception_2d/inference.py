import os
from importlib import import_module

import torch

from hat.data.collates.collates import collate_2d
from projects.cloudmodel.configs.perception_2d.build_model import build_model
from projects.cloudmodel.configs.perception_2d.build_visualization import (
    build_visualization,
)
from projects.cloudmodel.configs.perception_2d.common import (
    batch_size_per_gpu,
    checkpoint_path_inference,
    common_inputs,
    data_path_inference,
    num_workers,
    save_root_inference,
    transforms,
)
from projects.cloudmodel.configs.perception_2d.multitasks import TASKS

# ------------------------------- HAT settings ------------------------------ #
log_rank_zero_only = True
cudnn_benchmark = True
seed = None
march = "bayes"

output_dir = os.path.join(save_root_inference, "inference")

# -------------------------- build multitask models ------------------------- #
TASK_CONFIGS = [import_module(t) for t in TASKS]
infer_model = build_model(TASK_CONFIGS, common_inputs, mode="infer")

# ----------------------------- build dataloader ---------------------------- #
infer_dataloader = {
    "batch_size": batch_size_per_gpu["test"],
    "collate_fn": collate_2d,
    "dataset": {
        "type": "Auto2dFromImage",
        "data_path": data_path_inference,
        "to_rgb": True,
        "transforms": transforms["infer"],
        "return_orig_img": True,
    },
    "num_workers": num_workers["test"],
    "sampler": {"type": "DistSamplerHook"},
    "type": torch.utils.data.dataloader.DataLoader,
}

# --------------------------- build batch processors ------------------------ #
infer_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    enable_amp=False,
)

# ------------------------------ build callbacks ---------------------------- #
callbacks = []
vis_callback = build_visualization(
    TASK_CONFIGS,
    save_root_inference,
    det_threshold=0.3,
    mode="infer",
    max_save_num=-1,
    compose_detection=True,
)
callbacks.extend(vis_callback)

# --------------------------------------------------------------------------- #
model_convert_pipeline = dict(
    type="LoadCheckpoint",
    checkpoint_path=checkpoint_path_inference,
    state_dict_update_func=None,
    check_hash=False,
    allow_miss=True,
    ignore_extra=True,
    verbose=True,
)

float_predictor = dict(
    type="Predictor",
    model=infer_model,
    model_convert_pipeline=model_convert_pipeline,
    data_loader=infer_dataloader,
    batch_processor=infer_batch_processor,
    device=None,
    log_interval=10,
    callbacks=callbacks,
)
