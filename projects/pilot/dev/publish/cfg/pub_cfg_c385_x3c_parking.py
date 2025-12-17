import copy

from base_compile_cfg import models as _models

_models = copy.deepcopy(_models)

march = "bernoulli2"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_c385_parking_x3c_{march}"  # 编译后模型存放目录
compiled_hbm_name = "model_parking.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "0.16.2.1"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.35.4"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_c385_x3c_parking"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP3.0-AlgoModel5V-OV-C385-Parking"

desc = "MCP3.0-c385-parking-x3c"  # 模型的描述

models = dict(
    pilot_legorcnn_multitask_resize_2_parking=_models[
        "pilot_legorcnn_multitask_resize_2"
    ],
    pilot_legorcnn_multitask_resize_4_parking=_models[
        "pilot_legorcnn_multitask_resize_4"
    ],
    pilot_legorcnn_iqa_parsing_resize_4_parking=_models[
        "pilot_legorcnn_iqa_parsing_resize_4"
    ],
)

models["pilot_legorcnn_multitask_resize_2_parking"].update(
    dict(
        model_setting="c385_x3c_parking",
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
                person=0.561,
                cyclist=0.674,
                vehicle=0.740,
                rear=0.663,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.48,
                person=0.71,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.02,
                cyclist=0.1,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-jianhang-he.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_c385_x3c_parking_v6.0-20230201_202811/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-e32c2a83.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4_parking"].update(
    dict(
        model_setting="c385_x3c_parking",
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
                person=0.617,
                cyclist=0.754,
                vehicle=0.646,
                rear=0.547,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.483,
                person=0.55,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.02,
                cyclist=0.1,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-jianhang-he.train.hogpu.cc/plat_gpu/pilot_multitask_resize4_c385_x3c_parking_v6.0-resume-freezebn-20230207_222741/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-bff248f4.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_resize_4_parking"].update(
    dict(
        model_setting="c385_x3c_parking",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-bo-chen.alitrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-7-input-320x512-hat-cu111-20220602_old-docker-20220621-010745/output/models/image_fail_parsing/qat-checkpoint-last-3b4136e9.pth.tar",  # noqa
    )
)
