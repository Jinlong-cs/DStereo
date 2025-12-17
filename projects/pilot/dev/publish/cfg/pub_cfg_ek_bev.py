import os
from copy import deepcopy

from hat.utils.config import Config
from projects.pilot.configs.project_utils.enum import (
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
publish_name = BEVModelReleaseName.ek_bev

desc = publish_name  # 模型的描述

# 拆分编译模型
models = dict()

_base_models = Config.fromfile(
    os.path.join(os.path.dirname(__file__), "pub_cfg_master_bev.py")
)["models"]


# ======== 7V 需要编译的模型 ========
for _name in [
    BEVModelName.bev_multitask_stage1_fov120,
    BEVModelName.bev_multitask_stage1_side,
    BEVModelName.bev_multitask_stage1_narrow,
    BEVModelName.bev_multitask_stage2,
]:
    models[_name] = deepcopy(_base_models[_name])
    models[_name].update(dict(model_setting=BEVModelSetting.ek_bev))

# datamasking, iqa, lane_parsing
models[BEVModelName.datamasking_fov100] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_ek_20231106-140609/hbms/datamasking_fov100.hbm",  # noqa
    output_packs=BEVModelPackName.bev_large_range,
)
models[BEVModelName.iqa_parsing_fov100] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_ek_20231106-140609/hbms/iqa_parsing_fov100_v8.1.2.hbm",  # noqa
    output_packs=BEVModelPackName.bev_large_range,
)
models[BEVModelName.lane_parsing_fov100] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_ek_20231106-140609/hbms/lane_parsing_fov100_opt_O3_20230702_md5_23608fb0fe624c561ceb1229ee08fff5.hbm",  # noqa
    output_packs=BEVModelPackName.bev_large_range,
)


# ======== 5V 需要编译的模型 ========
for _name in [
    BEVModelName.bev_multitask_small_stage1_fov120,
    BEVModelName.bev_multitask_small_stage1_fisheye,
    BEVModelName.bev_multitask_small_stage2,
]:
    models[_name] = deepcopy(_base_models[_name])
    models[_name].update(dict(model_setting=BEVModelSetting.ek_bev))

models[BEVModelName.bev_multitask_small_stage1_fisheye].update(
    dict(input_shape="1x3x640x960")
)
# update stage2 input shape
bev_5v_num_fisheye_view = 4
bev_5v_num_vcs_plane_heights = 4
bev_5v_input_shape = (
    ["1x128x128x2"]
    * bev_5v_num_vcs_plane_heights
    * (bev_5v_num_fisheye_view - 2)
    + ["1x64x128x2"] * bev_5v_num_vcs_plane_heights
    + ["1x192x64x2"]
    * bev_5v_num_vcs_plane_heights
    * (bev_5v_num_fisheye_view - 2)
)
bev_5v_input_shape += ["1x4x128x240"] + [
    "1x4x160x240"
] * bev_5v_num_fisheye_view

models[BEVModelName.bev_multitask_small_stage2].update(
    dict(input_shape="^".join(bev_5v_input_shape))
)

# datamasking, iqa, lane_parsing
models[BEVModelName.datamasking_fisheye] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_ek_20231106-140609/hbms/datamasking_fov100_fisheye_640960_v8_0_0.hbm",  # noqa
    output_packs=BEVModelPackName.bev_small_range,
)
models[BEVModelName.iqa_parsing_fisheye] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_ek_20231106-140609/hbms/iqa_parsing_fisheye_v8.1.2.hbm",  # noqa
    output_packs=BEVModelPackName.bev_small_range,
)
models[BEVModelName.lane_parsing_fisheye] = dict(
    compiled_file="dmpv2://matrix2/users/feng02.li/model_package/pilot_perception_v1.0_ek_20231106-140609/hbms/lane_parsing_fisheye_v9.0.5.hbm",  # noqa
    output_packs=BEVModelPackName.bev_small_range,
)


# if local upload to gallery
gallery_config = dict(
    project="byd_ek",
    group="auto.pilot_algo.pilot5_1_model_release",
)
