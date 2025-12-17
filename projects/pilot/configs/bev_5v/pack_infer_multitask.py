import copy
import os
from functools import partial
from importlib import import_module

import torch

from hat.data.collates.collates import collate_3d
from projects.pilot.configs.bev_5v.bev_3d_base import (
    save_pack_infer as save_bev_3d_pack_infer,
)
from projects.pilot.configs.bev_5v.bev_discobj_base import (
    high_sp_resolution,
    high_sp_vcs_range,
    img_scale_before_ipm,
)
from projects.pilot.configs.bev_5v.bev_discobj_base import (
    save_pack_infer as save_bev_discobj_pack_infer,
)
from projects.pilot.configs.bev_5v.bev_discobj_base import (
    vis_common_transforms,
)
from projects.pilot.configs.bev_5v.bev_elevation_base import (
    save_freespace_pack_infer,
    save_vismask_pack_infer,
)
from projects.pilot.configs.bev_5v.bev_parkingrod import (
    save_pack_infer as save_parkingrod_pack_infer,
)
from projects.pilot.configs.bev_5v.bev_psd import (
    save_pack_infer as save_psd_pack_infer,
)
from projects.pilot.configs.bev_5v.common import (
    H_persp_view_scale,
    block_warp_padding,
    camera_view_names,
    common_transforms,
    grid_quant_scale,
    img_ori_size,
    ipm_output_size,
    model_checkpoint,
    model_setting,
    num_views,
    pack_infer_consist,
    pack_infer_vis,
    pipeline_test,
    save_prefix,
    spatial_resolution,
    tasks,
    vcs_plane_heights,
    vcs_range,
    views_dist,
    warp_sizes,
)
from projects.pilot.configs.bev_5v.models import val_model
from projects.pilot.configs.bev_5v.online_mapping import (
    save_pack_infer as save_om_pack_infer,
)
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

# Master
# multiview_pack_path="/horizon-bucket/pilot_tmp/users/wentao.xie/parking_packs/ADAS_20230711-095335_728_$Index.pack"
# bev_pack_path = "/home/users/wentao.xie/dataset_verify/parking_consist/BEVSR_20240130-195228_824_1.pack"
# homo_offset_path = "/home/users/wentao.xie/dataset_verify/parking_consist/homo_offset_small"

# EK
# multiview_pack_path = "/horizon-bucket/matrix2/users/wentao.xie/pack_EK/packs/ADAS_20231207-142222_318_$Index.pack"  # noqa
# bev_pack_path = "/horizon-bucket/matrix2/users/wentao.xie//pack_EK/BEVSR_20240131-110157_832_0.pack"
# homo_offset_path = "/horizon-bucket/matrix2/users/wentao.xie/pack_EK/homo_offset_small"

if pack_infer_consist:
    assert homo_offset_path, "homo_offset must provide for consist!"


# -------------------------- task --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

batch_size_per_gpu = 1
# pyramid resize layer index， 0:1/2，1:1/4 ···
pyramid_layer_index_list = [1, 0, 0, 0, 0]
view_idxs = [0, 1, 2, 3]


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
    # here set True, for `VisualizeIpm`
    # the `VisualizeIpm` which
    # generate the ipm in cpu transform, so we should
    # calculate homo_offset on cpu, this pipeline only
    # in task validation, will not cost cpu oom.
)
homo_gen_high_sp = copy.deepcopy(homo_gen)
homo_transforms = copy.deepcopy(homo_gen_high_sp["homo_transforms"])
homo_gen_high_sp.update(
    vcs_range=high_sp_vcs_range,
    spatial_resolution=(high_sp_resolution, high_sp_resolution),
    H_persp_view_scale=img_scale_before_ipm,
    homo_transforms=homo_transforms,
)

convert_software_offset = None
if homo_offset_path:
    convert_software_offset = dict(
        offset_path=homo_offset_path,
        ipm_output_size=ipm_output_size,
        warp_sizes=warp_sizes,
        vcs_plane_heights=vcs_plane_heights,
        block_warp_padding=block_warp_padding,
        grid_quant_scale=grid_quant_scale,
    )

transforms = [
    dict(
        type="ANCConvertPackDataTo3DV",
        calib=True,
        ground_level=False,
        nv12_format=True,
        num_frames_per_iter=1,
        homo_gen=homo_gen,
        homo_gen_high_sp=homo_gen_high_sp,
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
    vis_common_transforms,
    dict(type="DeleteKeys", keys=["pil_imgs"]),
    common_transforms["ANCPrepareDataBEV"],
    common_transforms["ANCNormalize3DV"],
]

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
    return_odometry=False,
    interpolated_odometry=False,
)
if homo_offset_path and len(datasets) > 1:
    raise NotImplementedError("Only support load one clip homo_offset")

# Pack-Dataloader
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        datasets=datasets,
        with_pack_flag=True,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler, shuffle=False),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=4,
    collate_fn=partial(
        collate_3d,
        ignore_keys=["homo_transforms"],
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
            block_warp_padding=block_warp_padding,
            grid_quant_scale=grid_quant_scale,
            num_views=num_views,
        ),
        dict(
            type="ANCSaveEachViewFeature",
            output_dir=os.path.join(save_prefix, "inference", "bev_stage1"),
            views=views_dist,
            task_key="bev_each_view_feat",
        ),
        save_bev_3d_pack_infer,
        save_bev_discobj_pack_infer,
        save_freespace_pack_infer,
        save_vismask_pack_infer,
        save_om_pack_infer,
        save_psd_pack_infer,
        save_parkingrod_pack_infer,
    ]

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
