import copy
import os

from base_compile_cfg import models

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
plugin_version = "1.1.3"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.39.2"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_galaxy_x3c_pe"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP5.0_galaxy_x3c_pe"

desc = "MCP5.0-cc02-x3c-pe"  # 模型的描述
# 需要编译的模型
models = dict(
    pilot_legorcnn_iqa_parsing_rear_resize_4=models[
        "pilot_legorcnn_iqa_parsing_resize_4"
    ],
    pilot_legorcnn_iqa_parsing_rear_resize_4_night=models[
        "pilot_legorcnn_iqa_parsing_resize_4_night"
    ],
    pilot_legorcnn_iqa_parsing_side_resize_4=copy.deepcopy(
        models["pilot_legorcnn_iqa_parsing_resize_4"]
    ),
    pilot_legorcnn_iqa_parsing_side_resize_4_night=copy.deepcopy(
        models["pilot_legorcnn_iqa_parsing_resize_4_night"]
    ),
)

with_cam_standiardization = False
cam_standiardization_cfg_rear = dict(
    input_shape="1x3x128x240^1x3x512x960^1x512x960x2",
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
    model_setting="galaxy_0233_rear_lmdb",
    input_shape="1x3x128x240^1x3x512x960",
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
            person=0.515,
            cyclist=0.62,
            vehicle=0.66,
            rear=0.35,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.4248,
            person=0.7443,
        ),
        thresh_3d=dict(
            vehicle=0.2,
            person=0.2,
            cyclist=0.2,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/wenyuan.zeng/plat_gpu/hobot-dag-3649187_resize-2-rear-bayes-training-galaxy-0233-rear-lmdb/output/models/pilot5_multitask_resize2_rear_bayes/freeze_bn_3-checkpoint-last-db1490f7.pth.tar",  # noqa
)
if with_cam_standiardization:
    models["pilot_legorcnn_multitask_rear_resize_2_day"].update(
        cam_standiardization_cfg_rear
    )

models["pilot_legorcnn_multitask_rear_resize_2_night"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_rear_bayes/multitask.py"),
    model_setting="galaxy_0233_rear_lmdb",
    input_shape="1x3x128x240^1x3x512x960",
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
            person=0.55,
            cyclist=0.63,
            vehicle=0.7,
            rear=0.52,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.1,
            person=0.1,
        ),
        thresh_3d=dict(
            vehicle=0.2,
            person=0.2,
            cyclist=0.2,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/wenyuan.zeng/plat_gpu/hobot-dag-3649187_resize-2-rear-bayes-training-galaxy-0233-rear-lmdb/output/models/pilot5_multitask_resize2_rear_bayes/freeze_bn_3-checkpoint-last-db1490f7.pth.tar",  # noqa
)
if with_cam_standiardization:
    models["pilot_legorcnn_multitask_rear_resize_2_night"].update(
        cam_standiardization_cfg_rear
    )

models["pilot_legorcnn_multitask_side_resize_2_day"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_side_bayes/multitask.py"),
    model_setting="galaxy_x3c_side_lmdb",
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
            person=0.40,
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.54,
            cyclist=0.50,
            vehicle=0.68,
            rear=0.6,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.3664,
            person=0.4579,
        ),
        thresh_3d=dict(
            vehicle=0.2,
            person=0.2,
            cyclist=0.2,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/jianhang.he/plat_gpu/hobot-dag-4445957_resize-2-side-bayes-training-galaxy-x3c-side-lmdb/output/models/pilot5_multitask_resize2_side_bayes/freeze_bn_3-checkpoint-last-721050b6.pth.tar",  # noqa
)
if with_cam_standiardization:
    models["pilot_legorcnn_multitask_side_resize_2_day"].update(
        cam_standiardization_cfg_side
    )

models["pilot_legorcnn_multitask_side_resize_2_night"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_side_bayes/multitask.py"),
    model_setting="galaxy_x3c_side_lmdb",
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
            person=0.50,
            cyclist=0.50,
            vehicle=0.71,
            rear=0.49,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            rear=0.1,
            person=0.1,
        ),
        thresh_3d=dict(
            vehicle=0.2,
            person=0.2,
            cyclist=0.2,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            rear=False,
        ),
    ),
    model_address="http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/jianhang.he/plat_gpu/hobot-dag-4445957_resize-2-side-bayes-training-galaxy-x3c-side-lmdb/output/models/pilot5_multitask_resize2_side_bayes/freeze_bn_3-checkpoint-last-721050b6.pth.tar",  # noqa
)
if with_cam_standiardization:
    models["pilot_legorcnn_multitask_side_resize_2_night"].update(
        cam_standiardization_cfg_side
    )

models["pilot_legorcnn_multitask_rear_crop_day"] = dict(
    cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
    model_setting="galaxy_0233_rear_lmdb",
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
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.51,
            cyclist=0.53,
            vehicle=0.45,
            rear=0.4,
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
    model_address="http://fm-yiding-liu.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3335533_crop-bayes-training-galaxy-0233-rear-lmdb/output/models/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last-b71ff423.pth.tar",  # noqa
)
models["pilot_legorcnn_multitask_rear_crop_night"] = dict(
    cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
    model_setting="galaxy_0233_rear_lmdb",
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
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.61,
            cyclist=0.6,
            vehicle=0.57,
            rear=0.32,
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
    model_address="http://fm-yiding-liu.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3335533_crop-bayes-training-galaxy-0233-rear-lmdb/output/models/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last-b71ff423.pth.tar",  # noqa
)


models["pilot_legorcnn_multitask_side_crop_day"] = dict(
    cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
    model_setting="galaxy_x3c_side_lmdb",
    input_shape="1x3x192x384",
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
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.51,
            cyclist=0.53,
            vehicle=0.59,
            rear=0.28,
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
    model_address="http://fm-yiding-liu.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3335533_crop-bayes-training-galaxy-0233-rear-lmdb/output/models/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last-b71ff423.pth.tar",  # noqa
)

models["pilot_legorcnn_multitask_side_crop_night"] = dict(
    cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
    model_setting="galaxy_x3c_side_lmdb",
    input_shape="1x3x192x384",
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
            cyclist=0.45,
            vehicle=0.3,
            rear=0.3,
        ),
        det_thresh=dict(
            person=0.61,
            cyclist=0.60,
            vehicle=0.66,
            rear=0.33,
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
    model_address="http://fm-yiding-liu.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3335533_crop-bayes-training-galaxy-0233-rear-lmdb/output/models/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last-b71ff423.pth.tar",  # noqa
)

models["pilot_legorcnn_iqa_parsing_rear_resize_4"].update(
    dict(
        model_setting="galaxy_0233_rear",
        input_shape="1x3x256x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/image_fail_parsing_galaxy_v9_rear_step5w_id1-20221221_123207/output/models/image_fail_parsing/qat-checkpoint-last-1da8cd54.pth.tar",  # noqa
    )
)


models["pilot_legorcnn_iqa_parsing_rear_resize_4_night"].update(
    dict(
        model_setting="galaxy_0233_rear",
        input_shape="1x3x256x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/image_fail_parsing_galaxy_v9_rear_step5w_id1-20221221_123207/output/models/image_fail_parsing/qat-checkpoint-last-1da8cd54.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_side_resize_4"].update(
    dict(
        model_setting="galaxy_x3c_side",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/image_fail_parsing_galaxy_v9_side_step5w_id3-20221221_164246-COPY/output/models/image_fail_parsing/qat-checkpoint-last-f5d708ff.pth.tar",  # noqa
    )
)

models["pilot_legorcnn_iqa_parsing_side_resize_4_night"].update(
    dict(
        model_setting="galaxy_x3c_side",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        model_address="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/image_fail_parsing_galaxy_v9_side_step5w_id3-20221221_164246-COPY/output/models/image_fail_parsing/qat-checkpoint-last-f5d708ff.pth.tar",  # noqa
    )
)
