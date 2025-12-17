import copy
import os

# from base_compile_cfg import cfg_dir
from base_compile_cfg import models as _models

_models = copy.deepcopy(_models)

cfg_dir = os.path.join(
    os.path.dirname(__file__), "../../../config/traffic_light"
)

march = "bayes"
input_layout = "NCHW"
output_layout = "NCHW"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_tl_{march}"  # 编译后模型存放目录
compiled_hbm_name = "traffic_light.hbm"  # "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "1.3.3"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.45.1"  # 编译用hbdk版本号

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "sd_tl_multitask"

desc = "sd, tl, day, hat"  # 模型的描述

models = dict(
    traffic_light_multitask=dict(
        cfg_path=os.path.join(cfg_dir, "multitask.py"),
        input_shape="8x3x96x96^8x1x1x96^8x1x1x96",
        input_type="dict",
        input_source="resizer,ddr,ddr",
        input_key="img^mask_height^mask_width",
        jobs_num=jobs_num,
        extra_args="--dev-remove-extra-output-cpu-op --debug",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="multitask",
        # model_address="http://fm-fan-lv.tcloud.hogpu.cc/plat_gpu/hobot-dag-2853250_traffic-light-mono-cn-2pe-day-multitask-v0-1-1-fan-lv-20230601-194401/output/models/traffic_light_multitask/freeze_bn_2-checkpoint-last.pth.tar",  # noqa
    ),
)
