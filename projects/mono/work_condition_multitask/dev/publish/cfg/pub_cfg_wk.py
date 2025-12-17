import os

cfg_dir = os.path.join(
    os.path.dirname(__file__), "../../../config/work_condition"
)

march = "bayes"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_wk_{march}"  # 编译后模型存放目录
compiled_hbm_name = "scene_multitask.hbm"  # "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "1.3.3"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.45.1"  # 编译用hbdk版本号

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "sd_wk_multitask"

desc = "sd, wk, mtl, hat"  # 模型的描述

models = dict(
    work_condition_multitask=dict(
        cfg_path=os.path.join(cfg_dir, "multitask.py"),
        input_shape="1x3x234x456",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--dev-remove-extra-output-cpu-op --debug",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="multitask",
    ),
)
