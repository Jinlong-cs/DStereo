import os
import sys

import torch
from common import (
    docker_image,
    folder_list,
    framework,
    get_json_dataset_list,
    get_val_sparse_transforms,
    input_bucket,
    job_list,
    job_name,
    job_password,
    launcher,
    max_jobtime,
    num_gpus_per_machine,
    num_machines,
    priority,
    project_id,
    sub_dirs,
    task_label,
    upload_folder_name,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval_utils import reformat_bev3d_to_det3d_aidi_eval
from horizon_plugin_pytorch.march import March
from sparse4d_dynamic_3d_detection import get_sparse_model

from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

device_ids = list(range(num_gpus_per_machine))

if os.environ.get("CLUSTER"):
    redirect_config_logging_path = f"/job_data/config_log_{training_step}.log"
else:
    redirect_config_logging_path = os.path.join(
        f"./tmp_models/config_log_{training_step}.log"
    )

# hdflow model zoo data_transform
val_transforms = get_val_sparse_transforms("x3c")

seed = 1280
cudnn_benchmark = False
log_rank_zero_only = True
march = March.BAYES


model = get_sparse_model()


k8s_config = dict(
    job_name=job_name,
    job_password=job_password,
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework=framework,
    task_label=task_label,
    project_id=project_id,
    input_bucket=input_bucket,
    priority=priority,
    docker_image=docker_image,
    max_jobtime=max_jobtime,
    launcher=launcher,
    upload_folder_name=upload_folder_name,
    folder_list=folder_list,
    job_list=job_list,
)

eval_dataset_map = {
    6040986: dict(  # BYD 4D mini test set, Just for MVT4Dv2 quick experiment.  # noqa
        json_file="/jfs-public/adas/zixiang.pei/workspace/data_worksapce/eval/BYD72/BYD72_20221118/extract_annos/BYD72_20221118_vehicle_sampled.json",  # noqa
        img_dir="/jfs-public/adas/zixiang.pei/workspace/data_worksapce/eval/BYD72/BYD72_20221118/extract_annos/vehicle/imgs",  # noqa
    ),
    6042946: dict(  # Galaxy 4D badcase mini test set, Just for MVT4Dv2 quick experiment.  # noqa
        json_file="/horizon-bucket/matrix2/users/zixiang.pei/data/auto_eval/dataset/6042946/galaxy_test_vehicle_sampled.json",  # noqa
        img_dir="/horizon-bucket/matrix2/users/zixiang.pei/data/auto_eval/dataset/6042946/imgs",  # noqa
    ),
    6039539: dict(  # Galaxy LX513 single frame test set, For MVT4Dv1.  # noqa
        json_file="/horizon-bucket/matrix2/users/zixiang.pei/data/auto_eval/dataset/6039539/LX513_20220806_vehicle_day_sampled.json",  # noqa
        img_dir="/horizon-bucket/matrix2/users/zixiang.pei/data/auto_eval/dataset/6039539/imgs",  # noqa
    ),
    6040536: dict(  # X3C big single frame test set,  For MVT4Dv1.  # noqa
        json_file="/horizon-bucket/matrix2/users/zixiang.pei/data/auto_eval/dataset/6040536/base.json",  # noqa
        img_dir="/horizon-bucket/matrix2/users/zixiang.pei/data/auto_eval/dataset/6040536/imgs",  # noqa
    ),
    6042966: dict(  # Galaxy 4D badcase test set, For MVT4Dv2.  # noqa
        json_file="/horizon-bucket/matrix2/users/zixiang.pei/data/auto_eval/dataset/6042966/galaxy_4badcase_vehicle_sampled.json",  # noqa
        img_dir="/horizon-bucket/matrix2/users/zixiang.pei/data/auto_eval/dataset/6042966/imgs",  # noqa
    ),
}


# eval_ckpt_path="dmpv2://adas/big_model/model_zoo/small_galaxy_c385_float-checkpoint-last-2520071f-formatted.pth.tar"
# eval_ckpt_path = "http://fm-zixiang-pei.train.hogpu.cc/plat_gpu/hobot-dag-3359148_hat-job-mvt4d-r50-256-10fps-6v-galaxy-badcase-20230710-161156/output/models/mvt4d_r50_256_10fps_6v_galaxy_badcase/float-checkpoint-last-00741c7f.pth.tar"
eval_ckpt_path = "http://fm-zixiang-pei.train.hogpu.cc/plat_gpu/hobot-dag-3502658_hat-job-sparse4d-galaxy-baseline-2e-4-20230724-163930/output/models/sparse4d_galaxy_baseline_2e-4/float-checkpoint-last-47dd18df.pth.tar"


val_samples_per_gpu = 1

eval_dataset_id = 6042946

val_dataset_list = get_json_dataset_list(
    [eval_dataset_map[eval_dataset_id]], model_setting="x3c", mode="val"
)

from functools import partial

from mmcv.parallel import collate

val_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        datasets=val_dataset_list,
        with_flag=True,
        accumulate_flag=True,
    ),
    batch_size=1,
    num_workers=4,
    pin_memory=False,
    # shuffle=False,
    # sampler=dict(type=torch.utils.data.DistributedSampler),
    sampler=None,
    batch_sampler=dict(
        type="DistributedGroupInBatchSampler",
        dataset=dict(
            type="ConcatDataset",
            datasets=val_dataset_list,
            with_flag=True,
            accumulate_flag=True,
        ),
        batch_size=val_samples_per_gpu,
        stop_by_epoch=True,
    ),
    collate_fn=partial(collate, samples_per_gpu=val_samples_per_gpu),
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    batch_transforms=[],
    loss_collector=None,
    enable_amp=False,
)

reformat_out_fn_kwargs = dict(
    obj_key="vehicle",
    dump_obj_key="vehicle",
    idx2cam={
        0: "front_right",
        1: "rear_right",
        2: "front_left",
        3: "rear_left",
        4: "rear",
        5: "front",
    },
    camera_view_names=sub_dirs,
    class_key_id_map=dict(
        person=0,
        vehicle=1,
        cyclist=2,
    ),
    score_thresh=0.1,
    center_type="cube_center",
)
reformat_output_fn = reformat_bev3d_to_det3d_aidi_eval


eval_callback = dict(
    type="AIDIEval",
    aidi_eval_dataset_id=eval_dataset_id,
    output_root=f"./eval_res/{eval_dataset_id}",
    prediction_name="sparse4dv2_merge_master_1x1_single_frame",
    prediction_tags=None,
    project_id=project_id,
    reformat_output_fn=reformat_output_fn,
    reformat_out_fn_kwargs=reformat_out_fn_kwargs,
    cpu=8,
    cpu_mem_ratio=4,
)

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.getenv(
                    "HAT_PILOT_MODEL_CHECKPOINT", eval_ckpt_path
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    batch_processor=val_batch_processor,
    data_loader=[val_dataloader],
    device=None,
    callbacks=[eval_callback],
    log_interval=2,
)
