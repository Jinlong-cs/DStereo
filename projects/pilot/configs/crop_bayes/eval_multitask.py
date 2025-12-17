import os
from copy import deepcopy
from importlib import import_module

os.environ["NO_HDFLOW"] = "1"

import torch
from common import (
    enable_model_tracking,
    eval_data_setting,
    model_checkpoint,
    model_name,
    model_setting,
    model_version,
    pred_batch_size,
    tasks,
    training_step,
    val_transforms,
    vis_tasks,
)
from models import march, val_model  # noqa
from schedule import get_fuse_patterns_by_stage, train_stages

from hat.core.data_struct.app_struct import (
    reformat_det_to_aidi_eval,
    reformat_seg_to_aidi_eval,
)
from hat.core.eval_platform_adaptor import BBoxScoreFilterAdapter
from hat.core.proj_spec.parsing import colormap
from hat.utils import Config
from hat.utils.bucket import url_to_local_path

suffix = os.getenv("HAT_PILOT_PREDICTION_NAME_SUFFIX", "")
prediction_name = f"{model_name}_{suffix}" if suffix else model_name

device_ids = [0]
log_rank_zero_only = True
# If eval_part_graph_model_only == True, forward only part graph model
# for diff task datasets, it usually speeds eval up
eval_part_graph_model_only = False

eval_tags = ["crop", "crop_region"]

if "rear" in model_setting.lower() or "test" in model_setting.lower():
    roi_region = (704, 568, 1216, 760)
    if eval_data_setting and "galaxy_x03_rear" in eval_data_setting:
        roi_region = (704, 568 - 180, 1216, 760 - 180)
elif "side_right" in model_setting.lower():
    if "niofy" in model_setting.lower():
        roi_region = (1536, 528, 1920, 720)  # right rear
    else:
        roi_region = (1536, 431, 1920, 623)  # right rear
elif "side_left" in model_setting.lower():
    if "niofy" in model_setting.lower():
        roi_region = (0, 528, 384, 720)  # left rear
    else:
        roi_region = (0, 431, 384, 623)  # left rear
val_transforms[1].update(static_roi=roi_region)
roi_from_img_meta = True
if roi_from_img_meta:
    val_transforms[1]["type"] = "DynamicCropImgPatch"
    val_transforms[1]["roi_source"] = "img_meta"
    val_transforms[1]["bpu_align"] = False

# task args
task_args = dict(
    vehicle_wheel_detection=dict(
        dump_obj_key="common_box",
    ),
    rear_detection=dict(
        dump_obj_key="vehicle",
    ),
    rear_plate_detection=dict(
        dump_obj_key="plate",
    ),
    person_face_detection=dict(
        dump_obj_key="face",
    ),
    person_pose_classification=dict(
        dump_extra_obj_key="person_pose_classification",
    ),
    person_occlusion_classification=dict(
        dump_extra_obj_key="person_occlusion_classification"
    ),
    person_orientation_classification=dict(
        dump_extra_obj_key="person_orientation_classification"
    ),
    cyclist_classification=dict(
        dump_extra_obj_key="cyclist_classification",
    ),
)

task_adapter_func = dict(
    vehicle_wheel_detection=BBoxScoreFilterAdapter(
        take_keys=("object_key", "detection_task_key", "decoder_results"),
        box_score_threshold=0.3,
    ),
)

all_callbacks = []
all_data_loaders = []

# data
if not eval_data_setting:
    eval_data_setting = model_setting.lower().replace("_lmdb", "")
ds_path = os.path.join(
    os.path.dirname(__file__),
    f"../datasets/{eval_data_setting.lower()}_crop_eval_datasets.py",
)
dataset_ids = Config.fromfile(ds_path).dataset_ids

local_datapath = url_to_local_path(
    "dmpv2://auto_eval/adas_eval/eval_platform/fs/"
)

task_names = [task["name"] for task in tasks]

for eval_type, all_datasets in dataset_ids.items():
    for task_cfg, datasets in all_datasets.items():
        if isinstance(task_cfg, tuple):
            extra_task_key, task_key = task_cfg
        else:
            extra_task_key = None
            task_key = task_cfg

        # skip datasets of tasks not included in current model
        if extra_task_key is not None:
            if extra_task_key not in task_names:
                continue
        elif task_key not in task_names:
            continue

        if eval_type == "detection":
            _kwargs = {}
            obj_key = import_module(task_key).object_type
            reformat_output_fn = reformat_det_to_aidi_eval

            _kwargs = dict(
                obj_key=obj_key,
                det_task_key=task_key,
                extra_task_key=extra_task_key,
            )
            if extra_task_key in task_adapter_func:
                _kwargs.update(
                    dict(custom_func=task_adapter_func[extra_task_key])
                )
            elif task_key in task_adapter_func:
                _kwargs.update(dict(custom_func=task_adapter_func[task_key]))

            if extra_task_key in task_args:
                _kwargs.update(task_args[extra_task_key])
            elif task_key in task_args:
                _kwargs.update(task_args[task_key])
            reformat_out_fn_kwargs = dict(
                **_kwargs,
            )

        elif eval_type == "semantic_segmentation":
            obj_key = import_module(task_key).task_name
            reformat_output_fn = reformat_seg_to_aidi_eval
            reformat_out_fn_kwargs = dict(
                obj_key=obj_key,
            )

        for ds in datasets:
            data_path = os.path.join(local_datapath, str(ds), "datasets")

            eval_callback = dict(
                type="AIDIEval",
                aidi_eval_dataset_id=ds,
                output_root=f"./eval_res/{ds}",
                prediction_name=prediction_name,
                prediction_tags=eval_tags,
                project_id=os.getenv("PROJECT_ID"),
                reformat_output_fn=reformat_output_fn,
                reformat_out_fn_kwargs=reformat_out_fn_kwargs,
            )

            visualize_callback = dict(
                type="ComposeVisualize",
                callbacks=[
                    dict(
                        type="DetMultitaskVisualize",
                        out_keys=[obj_key]
                        if eval_part_graph_model_only
                        else vis_tasks,
                        output_dir=f"./tmp_viz_imgs/{model_name}/{ds}",
                        vis_configs=dict(
                            vehicle=dict(
                                color=(0, 255, 0),
                                thickness=2,
                                points2=dict(),
                            ),
                            rear=dict(
                                color=(0, 255, 255),
                                thickness=2,
                            ),
                            person=dict(
                                color=(255, 0, 0),
                                thickness=2,
                            ),
                            cyclist=dict(
                                color=(255, 255, 0),
                                thickness=2,
                            ),
                            default_segmentation=dict(
                                colormap=colormap,
                                alpha=0.7,
                            ),
                            lane_segmentation=dict(
                                colormap=colormap,
                                alpha=0.7,
                            ),
                        ),
                        save_viz_imgs=False,
                    ),
                ],
            )
            callbacks = [
                visualize_callback,
                eval_callback,
                dict(
                    type="StatsMonitor",
                    log_freq=5,
                ),
            ]

            all_callbacks.append(callbacks)

            data_loader = dict(
                type=torch.utils.data.DataLoader,
                sampler=dict(
                    type=torch.utils.data.DistributedSampler,
                    shuffle=False,
                ),
                dataset=dict(
                    type="ModelEvalRawDataset",
                    data_path=data_path,
                    to_rgb=True,
                    buf_only=True,
                    return_orig_img=True,
                    transforms=val_transforms,
                ),
                batch_size=pred_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
                drop_last=False,
            )
            if eval_part_graph_model_only:
                all_data_loaders.append(
                    dict(
                        type="MultitaskLoader",
                        return_task=True,
                        loaders={obj_key: data_loader},
                    )
                )
            else:
                all_data_loaders.append(data_loader)

base_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="MultiBatchProcessor"
        if eval_part_graph_model_only
        else "BasicBatchProcessor",
        need_grad_update=False,
        inverse_transforms=dict(
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)


for stage in train_stages:
    pre, cur = get_fuse_patterns_by_stage(stage)

    predictor = deepcopy(base_predictor)
    predictor.update(
        model_convert_pipeline=dict(
            type="QATFuseBNConvertPipeline",
            qat_mode="with_bn_reverse_fold",
            pre_stage_fuse_patterns=pre,
            cur_stage_fuse_patterns=cur,
            fuse_part_configs=dict(
                fuse_method="fuse_norm",
                regex=True,
                strict=False,
            ),
            checkpoint_mode="resume",
            checkpoint_configs=dict(
                checkpoint_path=model_checkpoint
                if model_checkpoint
                else f"aidi_artifact://{model_name}/{training_step}/{model_version}/{training_step}-checkpoint-last.pth.tar",
                enable_tracking=enable_model_tracking,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            qconfig_params=None,
        )
    )
    globals()[f"{stage}_predictor"] = predictor
