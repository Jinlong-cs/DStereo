import os
from functools import partial
from importlib import import_module

import torch
from bev_crosspoint import save_pack_infer as save_crosspoint_pack_infer
from multitask import get_om_cpts_update_state_dict
from online_mapping import save_pack_infer as save_om_pack_infer

from hat.data.collates.collates import collate_3d, expand_ori_data
from hat.data.samplers.dist_clip_group_sampler import (
    ANCDistributedClipValSampler,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_base import (
    save_pack_infer as save_bev_3d_pack_infer,
)
from projects.pilot.configs.bev_7v_temporal.bev_discobj_base import (
    save_pack_infer as save_bev_discobj_pack_infer,
)
from projects.pilot.configs.bev_7v_temporal.bev_elevation_base import (
    save_freespace_pack_infer,
    save_vismask_pack_infer,
)
from projects.pilot.configs.bev_7v_temporal.common import (
    H_persp_view_scale,
    block_warp_padding,
    camera_view_names,
    common_transforms,
    grid_quant_scale,
    img_ori_size,
    ipm_output_size,
    model_checkpoint,
    model_setting,
    narrow_block_warp_padding,
    narrow_warp_sizes,
    num_views,
    pack_infer_consist,
    pack_infer_vis,
    pipeline_test,
    save_prefix,
    spatial_resolution,
    tasks,
    temporal_fusion_scale,
    vcs_plane_heights,
    vcs_range,
    views_dist,
    warp_sizes,
    world_size,
)
from projects.pilot.configs.bev_7v_temporal.models import val_model
from projects.pilot.configs.project_utils.enum import BEVModelSetting
from projects.pilot.configs.project_utils.pack_infer_utils import (
    build_pack_datasets,
    get_homo_transforms,
    update_pack_transform,
)
from projects.pilot.configs.project_utils.sensor_module import (
    BEV_SENSOR_MODULES,
)

device_ids = [0]


# -------------------------- Data Configs --------------------------
multiview_pack_path = os.environ.get("HAT_PILOT_MULTIVIEW_PACK")
bev_pack_path = os.environ.get("HAT_PILOT_BEV_PACK")
# software provided homo offset
homo_offset_path = os.environ.get("HAT_PILOT_HOMO_OFFSET")
temporal_homo_offset_path = os.environ.get("HAT_PILOT_TEMPORAL_HOMO_OFFSET")

# Master
# multiview_pack_path = "/horizon-bucket/matrix2/users/wentao.xie/new_temporal_pack/pack_time/ADAS_20230107-092200_855_$Index.pack"  # noqa
# bev_pack_path = "/horizon-bucket/matrix2/users/wentao.xie/new_temporal_pack/BEVBR_20240129-172341_659_0.pack"  # noqa
# homo_offset_path = "/horizon-bucket/matrix2/users/wentao.xie/new_temporal_pack/homo_offset_large"  # noqa
# temporal_homo_offset_path = "/horizon-bucket/matrix2/users/wentao.xie//new_temporal_pack/homo_offset_large/homooffset_temporal_$timestamp.bin"  # noqa

# EK
# multiview_pack_path = "/horizon-bucket/matrix2/users/wentao.xie/pack_EK/packs/ADAS_20231207-142222_318_$Index.pack"
# bev_pack_path = "/horizon-bucket/matrix2/users/wentao.xie/pack_EK/BEVBR_20240129-155928_466_0.pack"  # noqa
# homo_offset_path = "/horizon-bucket/matrix2/users/wentao.xie/pack_EK/homo_offset_large"  # noqa
# temporal_homo_offset_path = "/horizon-bucket/matrix2/users/wentao.xie/pack_EK/homo_offset_large/homooffset_temporal_$timestamp.bin"  # noqa


if pack_infer_consist:
    assert (
        homo_offset_path and temporal_homo_offset_path
    ), "homo_offset and temporal_homo_offset must provide for consist!"


# -------------------------- task --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

batch_size_per_gpu = 1
input_odo_length = 12
return_odo = False
interpolated_odometry = False
# pyramid resize layer index， 0:1/2，1:1/4 ···
pyramid_layer_index_list = [1, 0, 0, 0, 0, 0, 1]
view_idxs = [0, 1, 2, 3, 4, 5]


# -------------------------- Dataset --------------------------
# Pack-Transforms
homo_transforms = get_homo_transforms(
    list(common_transforms.values()), camera_view_names=camera_view_names
)
homo_gen = dict(
    camera_view_names=camera_view_names,
    task_camera_view_names=camera_view_names,
    spatial_resolution=spatial_resolution,
    vcs_range=vcs_range,
    homo_transforms=homo_transforms,
    per_view_shape=img_ori_size,
    use_distorted_offset=True,
    H_persp_view_scale=H_persp_view_scale,
    vcs_plane_heights=vcs_plane_heights,
    return_offset_in_meta_info=True,
)

convert_software_offset = None
if homo_offset_path:
    convert_software_offset = dict(
        offset_path=homo_offset_path,
        ipm_output_size=ipm_output_size,
        warp_sizes=warp_sizes + narrow_warp_sizes,
        vcs_plane_heights=vcs_plane_heights,
        block_warp_padding=block_warp_padding + narrow_block_warp_padding,
        grid_quant_scale=grid_quant_scale,
    )

transforms = [
    dict(
        type="ANCConvertPackDataTo3DV",
        calib=True,
        ground_level=False,
        nv12_format=True,
        num_frames_per_iter=1,
        temporal_bev=True,
        homo_gen=homo_gen,
        convert_software_offset=convert_software_offset,
    ),
    common_transforms[
        "ANCResize3DV"
    ],  # only for visualize with nv12 format input
    dict(
        type="ANCNV12Transform3DV",
        ori_size=list(img_ori_size.values()),
        pyramid_layer_index=pyramid_layer_index_list,
        save_nv12=True,
    ),
    common_transforms["ANCCrop3DV"],
    dict(
        type="ANCVisualizeIpm",
        bev_size=ipm_output_size,
        view_idxs=view_idxs,
        img_scale=1.0,
        block_warp_padding=block_warp_padding,
        enable_vis=False,
        meta_name="meta_info",
        vcs_plane_heights=vcs_plane_heights,
        return_name="ipm",
    ),
    dict(type="DeleteKeys", keys=["pil_imgs"]),
    common_transforms["ANCPrepareDataBEV"],
    common_transforms["ANCNormalize3DV"],
    dict(type="AddKeys", kv={"task_name": "multi_task"}),
]

if not temporal_homo_offset_path:
    transforms += [
        dict(
            type="ANCHisOdometryCollector",
            max_his_odo_len=input_odo_length,
        ),
        dict(
            type="ANCObtainHomographyTemporal",
            bev_size=ipm_output_size,
            vcs_range=vcs_range,
            return_relative=True,
        ),
    ]
else:
    transforms.append(
        dict(
            type="ObtainHomoOffsetTemporal",
            homooffset_temporal_url=temporal_homo_offset_path,
            bev_size=ipm_output_size,
            grid_quant_scale=temporal_fusion_scale,
        )
    )

transforms = update_pack_transform(
    transforms,
    sensor_module=BEV_SENSOR_MODULES[model_setting],
    camera_view_names=camera_view_names,
)

# Pack-Dataset
datasets = build_pack_datasets(
    pack_path=multiview_pack_path,
    transforms=transforms,
    camera_view_names=camera_view_names,
    bev_pack_path=bev_pack_path,
    max_num_frame=2 if pack_infer_consist or pipeline_test else None,
    return_odometry=False if pack_infer_consist else True,
    interpolated_odometry=interpolated_odometry,
    using_odo_diagnostic_code=True,
)
if (homo_offset_path or temporal_homo_offset_path) and len(datasets) > 1:
    raise NotImplementedError(
        "Only support load one clip homo_offset and temporal_homo_offset"
    )

# Pack-Dataloader
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ResampleDataset",
        resample_interval=1,
        with_pack_flag=True,
        dataset=dict(
            type="ConcatDataset",
            datasets=datasets,
            with_pack_flag=True,
        ),
    ),
    sampler=dict(type=ANCDistributedClipValSampler)
    if world_size > 1
    else dict(type=torch.utils.data.DistributedSampler, shuffle=False),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=0,
    collate_fn=partial(
        collate_3d,
        ignore_keys=["homo_transforms", "motr_targets"],
        stack_keys=["odo_info"],
        custom_handle_keys=["origin_imgs"],
        custom_func=expand_ori_data,
    ),
)

# -------------------------- Callbacks --------------------------
callbacks = []
if pack_infer_consist:
    callbacks += [
        dict(
            type="ANCSaveHomoOffset",
            output_dir=os.path.join(save_prefix, "inference", "homo_offset"),
            vcs_plane_nums=len(vcs_plane_heights),
            block_warp_padding=block_warp_padding + narrow_block_warp_padding,
            grid_quant_scale=grid_quant_scale,
            num_views=num_views,
        ),
        dict(
            type="ANCSaveEachViewFeature",
            output_dir=os.path.join(save_prefix, "inference", "bev_stage1"),
            views=views_dist,
            task_key="bev_each_view_feat",
        ),
        dict(
            type="ANCSaveFusionFeature",
            output_dir=os.path.join(
                save_prefix, "inference", "bev_temporal_fusion"
            ),
            task_key="bev_temporal_feat",
            result_key="bev_stage2_temporal_feat_head_predict",
        ),
        save_bev_3d_pack_infer,
        save_freespace_pack_infer,
        save_vismask_pack_infer,
        save_om_pack_infer,
        save_crosspoint_pack_infer,
    ]
    if model_setting in [BEVModelSetting.pilot51_master]:
        callbacks.append(save_bev_discobj_pack_infer)

if pack_infer_vis:
    for task_c in TASK_CONFIGS:
        if hasattr(task_c, "visualize_callbacks"):
            callbacks += task_c.visualize_callbacks


# -------------------------- Predictor --------------------------
pack_infer_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_checkpoint,
                state_dict_update_func=get_om_cpts_update_state_dict(),
                allow_miss=True,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=data_loader,
    batch_processor=dict(
        type="MultiBatchProcessor",
        batch_transforms=None,
        need_grad_update=False,
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
    log_interval=10,
)
