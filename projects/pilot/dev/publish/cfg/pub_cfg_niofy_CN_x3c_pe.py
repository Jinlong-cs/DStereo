import copy
import os

from base_compile_cfg import models as _models

cfg_dir = os.path.join(os.path.dirname(__file__), "../../../configs")

march = "bayes"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_{march}_x3c_pe"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "1.8.1"  # 编译用horizon_plugin_pytorch版本号
niofy_intinfer_plugin_version = "1.8.1"  # int_infer用horizon_plugin_pytorch版本号
hbcc_version = "v3.44.6"  # 编译用hbdk版本号

model_type = "s5v"

# 模型推理相关参数
infer_result_prefix = "infer_niofy_CN_x3c_pe"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP5.0_niofy_CN_x3c_pe"

desc = "MCP5.0-niofy-CN-x3c-pe"  # 模型的描述
# 需要编译的模型
models = dict(
    pilot_legorcnn_iqa_parsing_rear_resize_4=_models[
        "pilot_legorcnn_iqa_parsing_resize_4"
    ],
    pilot_legorcnn_iqa_parsing_rear_resize_4_night=_models[
        "pilot_legorcnn_iqa_parsing_resize_4_night"
    ],
    pilot_legorcnn_iqa_parsing_side_resize_4=copy.deepcopy(
        _models["pilot_legorcnn_iqa_parsing_resize_4"]
    ),
    pilot_legorcnn_iqa_parsing_side_resize_4_night=copy.deepcopy(
        _models["pilot_legorcnn_iqa_parsing_resize_4_night"]
    ),
)

with_cam_standiardization = True
cam_standiardization_cfg_rear = dict(
    input_shape="1x3x160x240^1x3x640x960^1x640x960x2",
    input_source="ddr,pyramid,ddr",
    input_key="coordinate_map^img^uv_map",
    extra_args="-g --max-time-per-fc 1000 --skip-check",
    torch_native="False^False^True",
)
cam_standiardization_cfg_side = copy.deepcopy(cam_standiardization_cfg_rear)
cam_standiardization_cfg_side[
    "input_shape"
] = "1x3x80x120^1x3x640x960^1x640x960x2"

models["pilot_legorcnn_multitask_rear_resize_2_day"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_rear_bayes/multitask.py"),
    model_setting="niofy_CN_x3c_rear_lmdb",
    input_shape="1x3x160x240^1x3x640x960",
    input_type="dict",
    input_source="ddr,pyramid",
    input_key="coordinate_map^img",
    jobs_num=jobs_num,
    extra_args="-g --max-time-per-fc 1000",
    output_layout=output_layout,
    input_layout=input_layout,
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.4,
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.49,
            cyclist=0.60,
            vehicle=0.63,
            rear=0.40,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.1,
            person=0.1,
        ),
        thresh_3d=dict(
            person=0.8257,
            cyclist=0.3033,
            vehicle=0.7297,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://fm-jun01-guan.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3303365_resize-2-rear-bayes-training-niofy-CN-x3c-rear-lmdb/output/models/pilot5_multitask_resize2_rear_bayes/freeze_bn_3-checkpoint-last-d27c0cbe.pth.tar",  # noqa
)
if with_cam_standiardization:
    models["pilot_legorcnn_multitask_rear_resize_2_day"].update(
        cam_standiardization_cfg_rear
    )

models["pilot_legorcnn_multitask_rear_resize_2_night"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_rear_bayes/multitask.py"),
    model_setting="niofy_CN_x3c_rear_lmdb",
    input_shape="1x3x160x240^1x3x640x960",
    input_type="dict",
    input_source="ddr,pyramid",
    input_key="coordinate_map^img",
    jobs_num=jobs_num,
    extra_args="-g --max-time-per-fc 1000",
    output_layout=output_layout,
    input_layout=input_layout,
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.4,
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.46,
            cyclist=0.55,
            vehicle=0.72,
            rear=0.56,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.1,
            person=0.1,
        ),
        thresh_3d=dict(
            person=0.8049,
            cyclist=0.2693,
            vehicle=0.6514,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://fm-jun01-guan.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3303365_resize-2-rear-bayes-training-niofy-CN-x3c-rear-lmdb/output/models/pilot5_multitask_resize2_rear_bayes/freeze_bn_3-checkpoint-last-d27c0cbe.pth.tar",  # noqa
)
if with_cam_standiardization:
    models["pilot_legorcnn_multitask_rear_resize_2_night"].update(
        cam_standiardization_cfg_rear
    )

models["pilot_legorcnn_multitask_side_resize_2_day"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_side_bayes/multitask.py"),
    model_setting="niofy_CN_x3c_side_lmdb",
    input_shape="1x3x80x120^1x3x640x960",
    input_type="dict",
    input_source="ddr,pyramid",
    input_key="coordinate_map^img",
    jobs_num=jobs_num,
    extra_args="-g --max-time-per-fc 1000",
    output_layout=output_layout,
    input_layout=input_layout,
    task_type="detection",
    framework="PyTorch",
    # 更新模型阈值中的相关参数
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.4,
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.5043,
            cyclist=0.5476,
            vehicle=0.67,
            rear=0.54,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.1,
            person=0.1,
        ),
        thresh_3d=dict(
            person=0.6697,
            cyclist=0.3277,
            vehicle=0.6834,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/jun01.guan/plat_gpu/hobot-dag-5619990_resize-2-side-bayes-training-niofy-CN-x3c-side-lmdb/output/models/pilot5_multitask_resize2_side_bayes/freeze_bn_3-checkpoint-last-f7dcfb82.pth.tar",  # noqa
)
# if with_cam_standiardization:
#     models["pilot_legorcnn_multitask_side_resize_2_day"].update(
#         cam_standiardization_cfg_side
#     )

models["pilot_legorcnn_multitask_side_resize_2_night"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_side_bayes/multitask.py"),
    model_setting="niofy_CN_x3c_side_lmdb",
    input_shape="1x3x80x120^1x3x640x960",
    input_type="dict",
    input_source="ddr,pyramid",
    input_key="coordinate_map^img",
    jobs_num=jobs_num,
    extra_args="-g --max-time-per-fc 1000",
    output_layout=output_layout,
    input_layout=input_layout,
    task_type="detection",
    framework="PyTorch",
    # 更新模型阈值中的相关参数
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.4,
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.58,
            cyclist=0.62,
            vehicle=0.81,
            rear=0.62,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.1,
            person=0.1,
        ),
        thresh_3d=dict(
            person=0.6677,
            cyclist=0.3012,
            vehicle=0.6849,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/jun01.guan/plat_gpu/hobot-dag-5619990_resize-2-side-bayes-training-niofy-CN-x3c-side-lmdb/output/models/pilot5_multitask_resize2_side_bayes/freeze_bn_3-checkpoint-last-f7dcfb82.pth.tar",  # noqa
)
# if with_cam_standiardization:
#     models["pilot_legorcnn_multitask_side_resize_2_night"].update(
#         cam_standiardization_cfg_side
#     )

models["pilot_legorcnn_multitask_rear_crop_day"] = dict(
    cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
    model_setting="niofy_CN_x3c_rear_lmdb",
    input_shape="1x3x192x512",
    input_type="dict",
    input_source="pyramid",
    input_key="img",
    jobs_num=jobs_num,
    extra_args="--pyramid-stride 1920 -g --max-time-per-fc 1000",
    output_layout=output_layout,
    input_layout=input_layout,
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.4,
            cyclist=0.25,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.38,
            cyclist=0.4,
            vehicle=0.54,
            rear=0.35,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.1,
            person=0.1,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://fm-jianhang-he.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3663259_pilot5-multitask-crop-niofy-cn-x3c-rear-lmdb-fix-freeze-bug-20230809-233706/output/models/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last-cabaab49.pth.tar",  # noqa
)
models["pilot_legorcnn_multitask_rear_crop_night"] = dict(
    cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
    model_setting="niofy_CN_x3c_rear_lmdb",
    input_shape="1x3x192x512",
    input_type="dict",
    input_source="pyramid",
    input_key="img",
    jobs_num=jobs_num,
    extra_args="--pyramid-stride 1920 -g --max-time-per-fc 1000",
    input_layout=input_layout,
    output_layout=output_layout,
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.4,
            cyclist=0.25,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.49,
            cyclist=0.39,
            vehicle=0.57,
            rear=0.38,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.1,
            person=0.1,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://fm-jianhang-he.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3663259_pilot5-multitask-crop-niofy-cn-x3c-rear-lmdb-fix-freeze-bug-20230809-233706/output/models/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last-cabaab49.pth.tar",  # noqa
)


# models["pilot_legorcnn_multitask_side_crop_day"] = dict(
#     cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
#     model_setting="niofy_CN_x3c_side_lmdb",
#     input_shape="1x3x192x384",
#     input_type="dict",
#     input_source="pyramid",
#     input_key="img",
#     jobs_num=jobs_num,
#     extra_args="--pyramid-stride 1920 -g --max-time-per-fc 1000",
#     input_layout=input_layout,
#     output_layout=output_layout,
#     task_type="detection",
#     framework="PyTorch",
#     update_cfg=dict(
#         rpn_thresh=dict(
#             person=0.4,
#             cyclist=0.45,
#             vehicle=0.3,
#             rear=0.3,
#         ),
#         det_thresh=dict(
#             person=0.37,
#             cyclist=0.55,
#             vehicle=0.52,
#             rear=0.37,
#         ),
#         roi_det_thresh=dict(
#             vehicle=0.3,
#             rear=0.1,
#             person=0.1,
#         ),
#         bbox_clipping=dict(
#             person=False,
#             cyclist=False,
#             vehicle=False,
#             rear=False,
#         ),
#     ),
#     model_address="http://fm-jiahui-chen.train.hogpu.cc/plat_gpu/hobot-dag-1761074_pilot5-multitask-crop-galaxy-0233-rear-lmdb-v12-crop-test-20230222-162403/output/models/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last-88d5987a.pth.tar",  # noqa
# )

# models["pilot_legorcnn_multitask_side_crop_night"] = dict(
#     cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
#     model_setting="niofy_CN_x3c_side_lmdb",
#     input_shape="1x3x192x384",
#     input_type="dict",
#     input_source="pyramid",
#     input_key="img",
#     jobs_num=jobs_num,
#     extra_args="--pyramid-stride 1920 -g --max-time-per-fc 1000",
#     input_layout=input_layout,
#     output_layout=output_layout,
#     task_type="detection",
#     framework="PyTorch",
#     update_cfg=dict(
#         rpn_thresh=dict(
#             person=0.4,
#             cyclist=0.45,
#             vehicle=0.3,
#             rear=0.3,
#         ),
#         det_thresh=dict(
#             person=0.55,
#             cyclist=0.66,
#             vehicle=0.63,
#             rear=0.46,
#         ),
#         roi_det_thresh=dict(
#             vehicle=0.3,
#             rear=0.1,
#             person=0.1,
#         ),
#         bbox_clipping=dict(
#             person=False,
#             cyclist=False,
#             vehicle=False,
#             rear=False,
#         ),
#     ),
#     model_address="http://fm-jiahui-chen.train.hogpu.cc/plat_gpu/hobot-dag-1761074_pilot5-multitask-crop-galaxy-0233-rear-lmdb-v12-crop-test-20230222-162403/output/models/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last-88d5987a.pth.tar",  # noqa
# )

models["pilot_legorcnn_iqa_parsing_rear_resize_4"].update(
    dict(
        model_setting="niofy_CN_x3c_rear",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/hobot-dag-2002759_pilot-image-fail-segmentation-all-data-input-x3c-lmdb-id5-20230315-213122/output/models/image_fail_parsing/qat-checkpoint-last-3e3e6cf3.pth.tar",  # noqa
    )
)


models["pilot_legorcnn_iqa_parsing_rear_resize_4_night"].update(
    dict(
        model_setting="niofy_CN_x3c_rear",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/hobot-dag-2002759_pilot-image-fail-segmentation-all-data-input-x3c-lmdb-id5-20230315-213122/output/models/image_fail_parsing/qat-checkpoint-last-3e3e6cf3.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_side_resize_4"].update(
    dict(
        model_setting="niofy_CN_x3c_side",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/hobot-dag-2002759_pilot-image-fail-segmentation-all-data-input-x3c-lmdb-id5-20230315-213122/output/models/image_fail_parsing/qat-checkpoint-last-3e3e6cf3.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_side_resize_4_night"].update(
    dict(
        model_setting="niofy_CN_x3c_side",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.train.hogpu.cc/plat_gpu/hobot-dag-2002759_pilot-image-fail-segmentation-all-data-input-x3c-lmdb-id5-20230315-213122/output/models/image_fail_parsing/qat-checkpoint-last-3e3e6cf3.pth.tar",  # noqa
    )
)
