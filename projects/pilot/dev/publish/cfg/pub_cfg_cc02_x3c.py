import copy

from base_compile_cfg import models as _models

models = copy.deepcopy(_models)

march = "bernoulli2"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_cc02_x3c_{march}"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "0.16.2"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.35.2"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_cc02_x3c"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP3.0-AlgoModel5V-OV-CC02"

desc = "MCP3.0-cc02-x3c"  # 模型的描述

# 需要编译的模型
models["pilot_legorcnn_multitask_resize_2"].update(
    dict(
        model_setting="cc02_x3c_day",
        input_shape="1x3x640x960",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 960 -g --max-time-per-fc 1000",
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
                person=0.6,
                cyclist=0.75,
                vehicle=0.65,
                rear=0.31,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.3,
                person=0.3,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.03,
                cyclist=0.21,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-xinjie-wang.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize2_cc02_x3c_day_v4.0-20220513-214951/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-b3874d59.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_2_night"].update(
    dict(
        model_setting="cc02_x3c_night",
        input_shape="1x3x640x960",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 960 -g --max-time-per-fc 1000",
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
                cyclist=0.56,
                vehicle=0.67,
                rear=0.50,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.3,
                person=0.3,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.02,
                cyclist=0.226,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="aidi://pilot_multitask_resize2_cc02_x3c_night_5.0/v0.0.6/sparse_3d_freeze_bn_2",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4"].update(
    dict(
        model_setting="cc02_x3c_day",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g --max-time-per-fc 1000",
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
                person=0.58,
                cyclist=0.73,
                vehicle=0.67,
                rear=0.375,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.3,
                person=0.3,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.02,
                cyclist=0.17,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="aidi://pilot_multitask_resize4_cc02_x3c_day_5.0/v0.0.5/sparse_3d_freeze_bn_2",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4_night"].update(
    dict(
        model_setting="cc02_x3c_night",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g --max-time-per-fc 1000",
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
                person=0.65,
                cyclist=0.5,
                vehicle=0.63,
                rear=0.377,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.3,
                person=0.3,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.02,
                cyclist=0.08,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-qingzhou-shen.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize4_cc02_x3c_night_4.0-20220513-213936/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-41a0a2a8.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_crop"].update(
    dict(
        model_setting="cc02_x3c_day",
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
                person=0.45,
                cyclist=0.69,
                vehicle=0.50,
                rear=0.485,
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
        model_address="http://fm-qingzhou-shen.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_3.0_cc02_x3c_day-20220409-143311/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-870e279a.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_crop_night"].update(
    dict(
        model_setting="cc02_x3c_night",
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
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-bo-chen.ucloudtrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-6-input-320x512-hat-cu111-20220602_old-docker-20220613-230811/output/models/image_fail_parsing/qat-checkpoint-last-405de1b6.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_resize_4_night"].update(
    dict(
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-bo-chen.ucloudtrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-6-input-320x512-hat-cu111-20220602_old-docker-20220613-230811/output/models/image_fail_parsing/qat-checkpoint-last-405de1b6.pth.tar",  # noqa
    )
)

models["pilot_veh_reid"].update(
    dict(
        input_shape="1x3x128x128",
        jobs_num=jobs_num,
        extra_args="-g ",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="aidi://pilot3_algomodel_reid_meng01.wang_20220224/v0.0.1/WithoutBN",  # noqa
    )
)

models["pilot_veh_reid_night"].update(
    dict(
        input_shape="1x3x128x128",
        jobs_num=jobs_num,
        extra_args="-g ",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="aidi://pilot3_algomodel_reid_meng01.wang_20220224/v0.0.1/WithoutBN",  # noqa
    )
)

models["pilot_depth_resize_4"].update(
    dict(
        model_setting="cc02_x3c_day",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="aidi://pilot_depth_resize_4_20220224/v0.0.1/WithoutBN",  # noqa
    )
)

models["pilot_depth_resize_4_night"].update(
    dict(
        model_setting="cc02_x3c_night",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="aidi://pilot_depth_resize_4_20220224/v0.0.1/WithoutBN",  # noqa
    )
)
