import os
import time
from functools import partial

import torch
from aidisdk.model import DeviceMeta
from aidisdk.utils.env import token_from_env

from hat.data.collates.collates import collate_2d
from hat.engine.AIDIPredictor import AIDIPredictor
from projects.cloudmodel.configs.perception_2d.evaluation import (
    dict_to_camera_frame,
    reformat_mc_prediction_fn,
)
from projects.cloudmodel.model_service.model_configs.segmentation.singletask__base_segmentation_config import (
    postprocess_reverse_for_mask,
)

# --------------------------------- Cluster --------------------------------- #
is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
cur_time = time.strftime("%Y%m%d%H%M%S")
save_root = f"./work_dirs/exp_{cur_time}" if is_local_train else "/job_log"
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"


# --------------------------------- DataSet --------------------------------- #
local_datapath = os.path.join(
    bucket_root, "auto_eval/adas_eval/eval_platform/fs/"
)
num_workers = 2
batch_size = 1
leaderboard_id = [6028550]

# task_name should end with _segmentation/_detection/_classfication
task_name = "cloudmodel_lane_instance_segmentation"

# ---------------------------------- Model ---------------------------------- #
infer_mode = AIDIPredictor.InferenceMode.InferModelLocal
cfg = dict(
    aidi_config="projects/cloudmodel/model_service/aidi_configs/segmentation/aidi_config_singletask_lane_instanceseg_prelabel_config.py",
    run_device=DeviceMeta("gpu", 0),
)

test_model = {
    "type": "AIDIPredictor",
    "infer_mode": infer_mode,
    "init_config": cfg,
}


collate_2d = partial(collate_2d, verbose=pipeline_test)

dataloaders = []
for li in leaderboard_id:
    data_path = os.path.join(local_datapath, str(li), "datasets")
    dataloader = {
        "type": torch.utils.data.dataloader.DataLoader,
        "batch_size": batch_size,
        "collate_fn": collate_2d,
        "dataset": {
            "type": "Auto2dFromImage",
            "data_path": data_path,  # 根据leader_board id直接构造
            "to_rgb": True,
            "transforms": None,
            "return_orig_img": True,
        },
        "num_workers": num_workers,
        "persistent_workers": (num_workers > 0),
        "sampler": dict(
            type=torch.utils.data.DistributedSampler,
            shuffle=False,
            drop_last=False,
        ),
    }
    dataloaders.append(dataloader)


test_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    batch_transforms=dict_to_camera_frame,
    inverse_transforms=postprocess_reverse_for_mask,
    enable_amp=False,
)

tags = ("resize", "merge")
loaders_list = []
callbacks_list = []
for loader in dataloaders:
    dataset_id = loader["dataset"]["data_path"].split("/")[-2]
    aidi_eval_callback = dict(
        type="AIDIEval",
        output_root=os.path.join(save_root, "prediction"),
        prediction_name=task_name,
        prediction_tags=list(tags),
        project_id=os.getenv("PROJECT_ID", None),
        aidi_eval_token=token_from_env(),
        aidi_eval_dataset_name=None,
        aidi_eval_dataset_id=[dataset_id],
        reformat_input_fn=None,
        reformat_output_fn=reformat_mc_prediction_fn,
        reformat_out_fn_kwargs=dict(
            task_name=task_name,
        ),
        overwrite=True,
        cpu=4,
        cpu_mem_ratio=6,
    )
    callbacks_list.append(aidi_eval_callback)

    task_loader = dict(
        type="MultitaskLoader",
        loaders={task_name: dataloader},
        return_task=True,
        custom_length=50 if pipeline_test else None,
        wrap_batch=True,
    )
    loaders_list.append(task_loader)


float_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=None,
    batch_processor=test_batch_processor,
    data_loader=loaders_list,
    callbacks=callbacks_list,
    device=None,
    metrics=None,
    log_interval=10 if pipeline_test else 50,
    share_callbacks=False,  # share_callbacks should be False
)
