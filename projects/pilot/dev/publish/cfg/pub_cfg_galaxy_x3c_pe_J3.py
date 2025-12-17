from pub_cfg_galaxy_x3c_pe import models  # noqa

march = "bernoulli2"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_{march}_x3c_pe"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 8  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "0.16.2"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.35.2"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_galaxy_x3c_pe_J3"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP5.0_galaxy_x3c_pe_J3"

desc = "MCP5.0-cc02-x3c-pe"  # 模型的描述
