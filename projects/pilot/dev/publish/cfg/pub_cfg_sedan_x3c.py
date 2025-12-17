import copy
import os

from base_compile_cfg import cfg_dir
from base_compile_cfg import models as _models

_models = copy.deepcopy(_models)

march = "bernoulli2"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_sedan_x3c_{march}"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "0.16.2.1"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.35.4"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_sedan_x3c"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
# notice sedan-x3c include c385-x3c、c673-x3c
publish_name = "MCP3.0-AlgoModel5V-OV-SEDAN"

desc = "MCP3.0-sedan-x3c"  # 模型的描述

models = dict(
    pilot_legorcnn_multitask_resize_2=_models[
        "pilot_legorcnn_multitask_resize_2"
    ],
    pilot_legorcnn_multitask_resize_2_night=_models[
        "pilot_legorcnn_multitask_resize_2_night"
    ],
    pilot_legorcnn_multitask_resize_4=_models[
        "pilot_legorcnn_multitask_resize_4"
    ],
    pilot_legorcnn_multitask_resize_4_night=_models[
        "pilot_legorcnn_multitask_resize_4_night"
    ],
    pilot_legorcnn_multitask_crop=_models["pilot_legorcnn_multitask_crop"],
    pilot_legorcnn_multitask_crop_night=_models[
        "pilot_legorcnn_multitask_crop_night"
    ],
    pilot_legorcnn_iqa_parsing_resize_4=_models[
        "pilot_legorcnn_iqa_parsing_resize_4"
    ],
    pilot_legorcnn_iqa_parsing_resize_4_night=_models[
        "pilot_legorcnn_iqa_parsing_resize_4_night"
    ],
)

models["pilot_legorcnn_multitask_resize_2"].update(
    dict(
        model_setting="sedan_x3c_day",
        input_shape="1x3x160x240^1x3x640x960",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride ,960 -g --max-time-per-fc 1000",
        input_source="ddr,pyramid",
        input_key="coordinate_map^img",
        input_layout=input_layout,
        output_layout=output_layout,
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.45,
                vehicle=0.3,
                rear=0.3,
            ),
            det_thresh=dict(
                person=0.467,
                cyclist=0.562,
                vehicle=0.732,
                rear=0.446,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.439,
                person=0.764,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.05,
                cyclist=0.1,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-feng01-wang.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_c385_x3c_day_v6.0-20230201_202340/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-9076afd6.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_2_night"].update(
    dict(
        model_setting="sedan_x3c_night",
        input_shape="1x3x160x240^1x3x640x960",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride ,960 -g --max-time-per-fc 1000",
        input_source="ddr,pyramid",
        input_key="coordinate_map^img",
        input_layout=input_layout,
        output_layout=output_layout,
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.45,
                vehicle=0.3,
                rear=0.3,
            ),
            det_thresh=dict(
                person=0.441,
                cyclist=0.664,
                vehicle=0.621,
                rear=0.576,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.55,
                person=0.77,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.05,
                cyclist=0.1,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-feng01-wang.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_c385_x3c_night_v6.0-20230201_202425/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-1fd866a1.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4"].update(
    dict(
        model_setting="sedan_x3c_day",
        input_shape="1x3x80x128^1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride ,480 -g --max-time-per-fc 1000",
        input_source="ddr,pyramid",
        input_key="coordinate_map^img",
        input_layout=input_layout,
        output_layout=output_layout,
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.45,
                vehicle=0.3,
                rear=0.3,
            ),
            det_thresh=dict(
                person=0.582,
                cyclist=0.583,
                vehicle=0.645,
                rear=0.374,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.426,
                person=0.58,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.05,
                cyclist=0.2,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-jianhang-he.train.hogpu.cc/plat_gpu/pilot_multitask_resize4_c385_x3c_day_v6.0-20230202_132803/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-a4240db1.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4_night"].update(
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
        model_setting="sedan_x3c_night",
        input_shape="1x3x80x128^1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride ,480 -g --max-time-per-fc 1000",
        input_source="ddr,pyramid",
        input_key="coordinate_map^img",
        input_layout=input_layout,
        output_layout=output_layout,
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.45,
                vehicle=0.3,
                rear=0.3,
            ),
            det_thresh=dict(
                person=0.43,
                cyclist=0.743,
                vehicle=0.599,
                rear=0.52,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.59,
                person=0.56,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.05,
                cyclist=0.1,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-feng01-wang.train.hogpu.cc/plat_gpu/pilot_multitask_resize4_c385_x3c_night_v6.0-20230201_202517/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-997dfc78.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_crop"].update(
    dict(
        model_setting="sedan_x3c_day",
        input_shape="1x3x192x512",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 1920 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.3,
                cyclist=0.45,
                vehicle=0.3,
                rear=0.3,
            ),
            det_thresh=dict(
                person=0.487,
                cyclist=0.69,
                vehicle=0.51,
                rear=0.47,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.3,
                person=0.3,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-liang-xu.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_c385_x3c_day_4.0-20221019-195352/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-b660fdb1.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_crop_night"].update(
    dict(
        model_setting="sedan_x3c_night",
        input_shape="1x3x192x512",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 1920 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.3,
                cyclist=0.45,
                vehicle=0.3,
                rear=0.3,
            ),
            det_thresh=dict(
                person=0.32,
                cyclist=0.48,
                vehicle=0.50,
                rear=0.603,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.3,
                person=0.3,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-qingzhou-shen.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_3.0_cc02_x3c_night-20220409-143816/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-8d3bf8a3.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_resize_4"].update(
    dict(
        model_setting="sedan_x3c_day",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/image_fail_parsing_c385_v5_step5w_id1-20221122-214425/output/models/image_fail_parsing/qat-checkpoint-last-0d609317.pth.tar",  # noqa
    )
)
models["pilot_legorcnn_iqa_parsing_resize_4_night"].update(
    dict(
        model_setting="sedan_x3c_night",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/image_fail_parsing_c385_v5_step5w_id1-20221122-214425/output/models/image_fail_parsing/qat-checkpoint-last-0d609317.pth.tar",  # noqa
    )
)
