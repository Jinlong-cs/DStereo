import copy

from base_compile_cfg import models as _models

models = copy.deepcopy(_models)

march = "bernoulli2"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_as33_{march}"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "0.16.2"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.35.2"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_as33"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP3.0-AlgoModel5V-ON"

desc = "MCP3.0-as33"  # 模型的描述

# 需要编译的模型
models["pilot_legorcnn_multitask_resize_2"].update(
    dict(
        model_setting="as33_day",
        input_shape="1x3x640x1024",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 1024 -g --max-time-per-fc 1000",
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
                person=0.53,
                cyclist=0.48,
                vehicle=0.68,
                rear=0.54,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.55,
                person=0.7,
            ),
            thresh_3d=dict(
                person=0.02,
                cyclist=0.1,
                vehicle=0.2,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize2_as33_day_v13.1-20220930-143332/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-fb2c8402.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_2_night"].update(
    dict(
        model_setting="as33_night",
        input_shape="1x3x640x1024",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 1024 -g --max-time-per-fc 1000",
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
                person=0.42,
                cyclist=0.50,
                vehicle=0.55,
                rear=0.60,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.52,
                person=0.67,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.02,
                cyclist=0.07,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize2_as33_night_v13.1-20220930-143555/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-9a8270f2.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4"].update(
    dict(
        model_setting="as33_day",
        input_shape="1x3x320x512",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g --max-time-per-fc 1000",
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
                person=0.60,
                cyclist=0.56,
                vehicle=0.71,
                rear=0.60,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.5,
                person=0.5,
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
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize4_as33_day_v13.1-20220930-143855/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-3e97ab56.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4_night"].update(
    dict(
        model_setting="as33_night",
        input_shape="1x3x320x512",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g --max-time-per-fc 1000",
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
                person=0.45,
                cyclist=0.60,
                vehicle=0.52,
                rear=0.62,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.49,
                person=0.48,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.02,
                cyclist=0.05,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/pilot_multitask_resize4_as33_night_v13.1-withbn-resume-20221003-092538/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-2980c534.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_crop"].update(
    dict(
        model_setting="as33_day",
        input_shape="1x3x192x512",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 2048 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.4,
                cyclist=0.25,
                vehicle=0.3,
                rear=0.3,
            ),
            det_thresh=dict(
                person=0.6,
                cyclist=0.42,
                vehicle=0.50,
                rear=0.495,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.3,
                person=0.3,
            ),
            thresh_3d=dict(
                person=0.03,
                cyclist=0.2,
                vehicle=0.2,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://fm-tianzhongpeng.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_as33_day_v12-20220630-204931/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-81c58c46.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_crop_night"].update(
    dict(
        model_setting="as33_night",
        input_shape="1x3x192x512",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 2048 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.35,
                cyclist=0.4,
                vehicle=0.3,
                rear=0.3,
            ),
            det_thresh=dict(
                person=0.52,
                cyclist=0.65,
                vehicle=0.50,
                rear=0.726,
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
        model_address="aidi://pilot_multitask_crop_night_ON_v11.1/v0.0.4/freeze_bn_3",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_resize_4"].update(
    dict(
        model_setting="as33_day",
        input_shape="1x3x320x512",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-bo-chen.alitrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-7-input-320x512-hat-cu111-20220602_old-docker-20220621-010745/output/models/image_fail_parsing/qat-checkpoint-last-3b4136e9.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_resize_4_night"].update(
    dict(
        model_setting="as33_night",
        input_shape="1x3x320x512",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-bo-chen.alitrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-7-input-320x512-hat-cu111-20220602_old-docker-20220621-010745/output/models/image_fail_parsing/qat-checkpoint-last-3b4136e9.pth.tar",  # noqa
    )
)
