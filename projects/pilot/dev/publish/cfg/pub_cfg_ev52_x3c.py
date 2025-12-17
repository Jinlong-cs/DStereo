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
output_dir = f"tmp_compile_ev52_x3c_{march}"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "0.16.2.1"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.35.5"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_ev52_x3c"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP3.0-AlgoModel5V-OV-EV52"

desc = "MCP3.0-ev52-x3c"  # 模型的描述

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
        model_setting="ev52_x3c_day_lmdb",
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
                person=0.544,
                cyclist=0.478,
                vehicle=0.795,
                rear=0.501,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.495,
                person=0.699,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.272,
                cyclist=0.03,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/jiaxi.wu/plat_gpu/hobot-dag-3047636_resize-2-training-ev52-x3c-day-lmdb/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-e4d4f3ef.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_2_night"].update(
    dict(
        model_setting="ev52_x3c_night_lmdb",
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
                person=0.492,
                cyclist=0.585,
                vehicle=0.653,
                rear=0.55,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.468,
                person=0.75,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.19,
                cyclist=0.04,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/jiaxi.wu/plat_gpu/hobot-dag-3047636_resize-2-training-ev52-x3c-night-lmdb/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-332e99be.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4"].update(
    dict(
        model_setting="ev52_x3c_day_lmdb",
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
                person=0.62,
                cyclist=0.619,
                vehicle=0.675,
                rear=0.48,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.634,
                person=0.577,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.25,
                cyclist=0.07,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/xinwen.hu/plat_gpu/hobot-dag-3073361_pilot-multitask-resize4-ev52-x3c-day-lmdb-ev52-v6-release-resume-fbn1-lr0-00002-2w-20230619-112650/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-4b8b23b3.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_resize_4_night"].update(
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
        model_setting="ev52_x3c_night_lmdb",
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
                person=0.48,
                cyclist=0.618,
                vehicle=0.614,
                rear=0.627,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                rear=0.485,
                person=0.581,
            ),
            thresh_3d=dict(
                vehicle=0.2,
                person=0.17,
                cyclist=0.55,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                rear=False,
            ),
        ),
        model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/jiaxi.wu/plat_gpu/hobot-dag-3047636_resize-4-training-ev52-x3c-night-lmdb/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-bf8c0a9d.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_multitask_crop"].update(
    dict(
        model_setting="ev52_x3c_day_lmdb",
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
        model_setting="ev52_x3c_night_lmdb",
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
        model_setting="ev52_x3c_day",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/hobot-dag-2002759_pilot-image-fail-segmentation-all-data-input-x3c-lmdb-id5-20230315-213122/output/models/image_fail_parsing/qat-checkpoint-last-3e3e6cf3.pth.tar",  # noqa
    )
)
models["pilot_legorcnn_iqa_parsing_resize_4_night"].update(
    dict(
        model_setting="ev52_x3c_day",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/hobot-dag-2002759_pilot-image-fail-segmentation-all-data-input-x3c-lmdb-id5-20230315-213122/output/models/image_fail_parsing/qat-checkpoint-last-3e3e6cf3.pth.tar",  # noqa
    )
)
