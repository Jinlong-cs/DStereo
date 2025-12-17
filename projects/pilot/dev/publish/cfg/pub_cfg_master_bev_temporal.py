import os

from projects.pilot.configs.project_utils.enum import (  # noqa
    BEVModelName,
    BEVModelPackName,
    BEVModelReleaseName,
    BEVModelSetting,
)

cfg_dir = os.path.join(os.path.dirname(__file__), "../../../configs")

march = "bayes"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_{march}"  # 编译后模型存放目录
if os.path.exists("/running_package"):
    output_dir = f"/job_data/compile_{march}"
compiled_hbm_name = None  # hbm文件名, 在pack_graoups中配置
jobs_num = 128  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "2.0.0"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.43.3"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_pilot51_bev"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = BEVModelReleaseName.master_bev_temporal

desc = publish_name  # 模型的描述

# 拆分编译模型
models = dict()


# ======== 时序7V 需要编译的模型 ========
bev_7v_temporal_ckpt = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/wentao.xie/plat_gpu/hobot-dag-8299650_pilot-multitask-bev-7v-temporal-pilot5-1-master-v3-2-0-release-20240124-205145/output/models/pilot_multitask_bev_7v_temporal/qat-checkpoint-last-84819eb8.pth.tar"  # noqa

models[BEVModelName.bev_multitask_stage1_fov120] = dict(
    cfg_path=os.path.join(cfg_dir, "bev_7v_temporal/multitask.py"),
    model_setting=BEVModelSetting.pilot51_master,
    input_shape="1x3x512x960",
    input_type="dict",
    input_source="pyramid",
    input_key="camera_front",
    torch_native="False",
    jobs_num=jobs_num,
    extra_args="--max-time-per-fc 1000 --balance 0",
    output_layout="NCHW",
    input_layout="NCHW",
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(),
    split_flag=BEVModelName.bev_multitask_stage1_fov120,
    override=True,
    debug=True,
    dump_cmp_info=True,
    model_address=bev_7v_temporal_ckpt,
    output_packs=BEVModelPackName.bev_large_range,
)

models[BEVModelName.bev_multitask_stage1_side] = dict(
    cfg_path=os.path.join(cfg_dir, "bev_7v_temporal/multitask.py"),
    model_setting=BEVModelSetting.pilot51_master,
    input_shape="1x3x640x960",
    input_type="dict",
    input_source="pyramid",
    input_key="camera_front_left",
    torch_native="False",
    jobs_num=jobs_num,
    extra_args="--max-time-per-fc 1000 --balance 2",
    output_layout="NCHW",
    input_layout="NCHW",
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(),
    split_flag=BEVModelName.bev_multitask_stage1_side,
    override=True,
    debug=True,
    dump_cmp_info=True,
    model_address=bev_7v_temporal_ckpt,
    output_packs=BEVModelPackName.bev_large_range,
)

models[BEVModelName.bev_multitask_stage1_narrow] = dict(
    cfg_path=os.path.join(cfg_dir, "bev_7v_temporal/multitask.py"),
    model_setting=BEVModelSetting.pilot51_master,
    input_shape="1x3x512x960",
    input_type="dict",
    input_source="pyramid",
    input_key="camera_front_30fov",
    torch_native="False",
    jobs_num=jobs_num,
    extra_args="--max-time-per-fc 1000 --balance 0",
    output_layout="NCHW",
    input_layout="NCHW",
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(),
    split_flag=BEVModelName.bev_multitask_stage1_narrow,
    override=True,
    debug=True,
    dump_cmp_info=True,
    model_address=bev_7v_temporal_ckpt,
    output_packs=BEVModelPackName.bev_large_range,
)


# homo_offset
bev_7v_num_view = 7
bev_7v_num_side_view = 5
bev_7v_num_vcs_plane_heights = 4
bev_7v_input_key = [
    f"homo_offset_{i}"
    for i in range(bev_7v_num_view * bev_7v_num_vcs_plane_heights)
]
input_shape = (
    ["1x264x256x2"] * bev_7v_num_vcs_plane_heights
    + ["1x352x128x2"]
    * (bev_7v_num_side_view - 1)
    * bev_7v_num_vcs_plane_heights
    + ["1x88x256x2"] * bev_7v_num_vcs_plane_heights
    + ["1x264x256x2"] * bev_7v_num_vcs_plane_heights
)
torch_native = ["True"] * bev_7v_num_vcs_plane_heights * bev_7v_num_view
# stage1 feat
bev_7v_input_key += [f"pred_segs_frame0_{i}" for i in range(7)]
input_shape += (
    ["1x4x128x240"] + ["1x4x160x240"] * bev_7v_num_side_view + ["1x4x128x240"]
)
torch_native += ["False"] * bev_7v_num_view
# history feat
bev_7v_input_key += ["pre_fusion_feat"]
input_shape += ["1x16x352x256"]
torch_native += ["False"]
# tmporal homo offset
bev_7v_input_key += ["homooffset_temporal"]
input_shape += ["1x352x256x2"]
torch_native += ["True"]

models[BEVModelName.bev_multitask_stage2_temporal] = dict(
    cfg_path=os.path.join(cfg_dir, "bev_7v_temporal/multitask.py"),
    model_setting=BEVModelSetting.pilot51_master,
    input_shape="^".join(input_shape),
    input_type="dict",
    input_source="ddr",
    input_key="^".join(bev_7v_input_key),  # noqa
    torch_native="^".join(torch_native),
    jobs_num=jobs_num,
    extra_args="--max-time-per-fc 1000 --balance 0",
    output_layout="NCHW",
    input_layout="NCHW",
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        bev_3d_vehicle=dict(
            score_threshold=0.155,
            roi_score_threshold=[0.155] * 7,
        ),
        bev_3d_vrumerge=dict(
            score_threshold=[0.19, 0.185],
            roi_score_threshold=[0.19, 0.185],
        ),
        online_mapping=dict(
            lane=0.57,
            roadedge=0.45,
        ),
        bev_arrow=dict(
            score_threshold=0.2,
            iou_threshold=0.2,
        ),
        bev_junction=dict(
            score_threshold=0.2,
            iou_threshold=0.2,
        ),
        bev_roadmarking=dict(
            score_threshold=0.2,
            iou_threshold=0.2,
        ),
    ),
    split_flag=BEVModelName.bev_multitask_stage2_temporal,
    override=True,
    debug=True,
    dump_cmp_info=True,
    model_address=bev_7v_temporal_ckpt,
    output_packs=BEVModelPackName.bev_large_range,
)


# datamasking, iqa, lane_parsing
models[BEVModelName.datamasking_fov100] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_20231103-205913/hbms/datamasking_fov100.hbm",  # noqa
    output_packs=BEVModelPackName.bev_large_range,
)
models[BEVModelName.iqa_parsing_fov100] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_20231103-205913/hbms/iqa_parsing_fov100_opt_O3_20230704_md5_4ac66212281750a3aab58e51c8bae754.hbm",  # noqa
    output_packs=BEVModelPackName.bev_large_range,
)
models[BEVModelName.lane_parsing_fov100] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_20231103-205913/hbms/lane_parsing_fov100_opt_O3_20230702_md5_23608fb0fe624c561ceb1229ee08fff5.hbm",  # noqa
    output_packs=BEVModelPackName.bev_large_range,
)


# ======== 5V 需要编译的模型 ========
bev_5v_ckpt = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/zixiang.pei/plat_gpu/hobot-dag-8073602_pilot-multitask-bev-5v-pilot5-1-master-v3-2-0-release-20240122-195311/output/models/pilot_multitask_bev_5v/qat-checkpoint-last-5ce18c2b.pth.tar"  # noqa

models[BEVModelName.bev_multitask_small_stage1_fov120] = dict(
    cfg_path=os.path.join(cfg_dir, "bev_5v/multitask.py"),
    model_setting=BEVModelSetting.pilot51_master,
    input_shape="1x3x512x960",
    input_type="dict",
    input_source="pyramid",
    input_key="camera_front",
    torch_native="False",
    jobs_num=jobs_num,
    extra_args="--max-time-per-fc 1000 --balance 100",
    output_layout="NCHW",
    input_layout="NCHW",
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(),
    split_flag=BEVModelName.bev_multitask_small_stage1_fov120,
    override=True,
    debug=True,
    dump_cmp_info=True,
    model_address=bev_5v_ckpt,
    output_packs=BEVModelPackName.bev_small_range,
)


models[BEVModelName.bev_multitask_small_stage1_fisheye] = dict(
    cfg_path=os.path.join(cfg_dir, "bev_5v/multitask.py"),
    model_setting=BEVModelSetting.pilot51_master,
    input_shape="1x3x768x960",
    input_type="dict",
    input_source="pyramid",
    input_key="fisheye_front",
    torch_native="False",
    jobs_num=jobs_num,
    extra_args="--max-time-per-fc 1000 --balance 1",
    output_layout="NCHW",
    input_layout="NCHW",
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(),
    split_flag=BEVModelName.bev_multitask_small_stage1_fisheye,
    override=True,
    debug=True,
    dump_cmp_info=True,
    model_address=bev_5v_ckpt,
    output_packs=BEVModelPackName.bev_small_range,
)


# homo_offset
bev_5v_num_view = 5
bev_5v_num_fisheye_view = 4
bev_5v_num_vcs_plane_heights = 4
bev_5v_input_key = [
    f"homo_offset_{i}"
    for i in range(bev_5v_num_view * bev_5v_num_vcs_plane_heights)
]
bev_5v_input_shape = (
    ["1x256x256x2"]
    * bev_5v_num_vcs_plane_heights
    * (bev_5v_num_fisheye_view - 2)
    + ["1x128x256x2"] * bev_5v_num_vcs_plane_heights
    + ["1x384x128x2"]
    * bev_5v_num_vcs_plane_heights
    * (bev_5v_num_fisheye_view - 2)
)
bev_5v_torch_native = ["True"] * bev_5v_num_vcs_plane_heights * bev_5v_num_view
# stage1 feat
bev_5v_input_key += [f"pred_segs_frame0_{i}" for i in range(5)]
bev_5v_input_shape += ["1x4x128x240"] + [
    "1x4x192x240"
] * bev_5v_num_fisheye_view
bev_5v_torch_native += ["False"] * bev_5v_num_view


models[BEVModelName.bev_multitask_small_stage2] = dict(
    cfg_path=os.path.join(cfg_dir, "bev_5v/multitask.py"),
    model_setting=BEVModelSetting.pilot51_master,
    input_shape="^".join(bev_5v_input_shape),
    input_type="dict",
    input_source="ddr",
    input_key="^".join(bev_5v_input_key),  # noqa
    torch_native="^".join(bev_5v_torch_native),
    jobs_num=jobs_num,
    extra_args="--max-time-per-fc 1000 --balance 2",
    output_layout="NCHW",
    input_layout="NCHW",
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        bev_3d_vehicle=dict(
            score_threshold=0.35,
            roi_score_threshold=[0.35] * 7,
        ),
        bev_3d_vrumerge=dict(
            score_threshold=[0.27, 0.33],
            roi_score_threshold=[0.27, 0.33],
        ),
        online_mapping=dict(
            lane=0.6,
            roadedge=0.45,
        ),
        bev_arrow=dict(
            score_threshold=0.2,
            iou_threshold=0.1,
        ),
        bev_roadmarking=dict(
            score_threshold=0.2,
            iou_threshold=0.2,
        ),
        bev_static_obstacle=dict(
            score_threshold=0.3,
            iou_threshold=0.2,
        ),
        bev_parkingrod=dict(
            threshold_parkingrod_feature=0.2,
        ),
        bev_sod3d=dict(
            score_threshold=0.308,
        ),
    ),
    split_flag=BEVModelName.bev_multitask_small_stage2,
    override=True,
    debug=True,
    dump_cmp_info=True,
    model_address=bev_5v_ckpt,
    output_packs=BEVModelPackName.bev_small_range,
)


# datamasking, iqa, lane_parsing
models[BEVModelName.datamasking_fisheye] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_20231103-205913/hbms/datamasking_fisheye.hbm",  # noqa
    output_packs=BEVModelPackName.bev_small_range,
)
models[BEVModelName.iqa_parsing_fisheye] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_20231103-205913/hbms/iqa_parsing_fisheye_opt_O3_20230704_md5_d288a9c8be6ae1ddcc9f8b0df0654082.hbm",  # noqa
    output_packs=BEVModelPackName.bev_small_range,
)
models[BEVModelName.lane_parsing_fisheye] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_20231103-205913/hbms/lane_parsing_fisheye_opt_O3_20230702_md5_3b9b21c13bb2ebbf666bce9b706f5faf.hbm",  # noqa
    output_packs=BEVModelPackName.bev_small_range,
)

# if local upload to gallery
gallery_config = dict(
    project="master",
    group="auto.pilot_algo.pilot5_1_model_release",
)
