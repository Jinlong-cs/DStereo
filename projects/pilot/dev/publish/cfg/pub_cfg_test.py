import copy
import os
import time

cfg_dir = os.path.join(os.path.dirname(__file__), "../../../configs")

test_level = os.getenv("HAT_PILOT_TEST_LEVEL")
assert test_level is not None

march = "bernoulli2"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O0"
timestamp = time.strftime("%y%m%d%H%M", time.localtime())

# 本地编译模式相关参数
output_dir = f"tmp_compile_test_{march}"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

plugin_version = "1.3.0"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.42.1"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = f"infer_test_{timestamp}"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP-AlgoModel5V-test"

desc = "MCP3.0-build-ci-test"  # 模型的描述

model_resize_2 = "tmp_output/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last.pth.tar"
model_resize_4 = "tmp_output/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last.pth.tar"
model_crop = (
    "tmp_output/pilot_multitask_crop/freeze_bn_3-checkpoint-last.pth.tar"
)
model_resize_2_side_bayes = "tmp_output/pilot5_multitask_resize2_side_bayes/sparse_3d_freeze_bn_2-checkpoint-last.pth.tar"
model_resize_2_rear_bayes = "tmp_output/pilot5_multitask_resize2_rear_bayes/sparse_3d_freeze_bn_2-checkpoint-last.pth.tar"
model_crop_bayes = "tmp_output/pilot5_multitask_crop_bayes/freeze_bn_3-checkpoint-last.pth.tar"
model_image_fail_segmentation = (
    "tmp_output/image_fail_parsing/qat-checkpoint-last.pth.tar"
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

models = dict(
    pilot_legorcnn_multitask_rear_resize_2=dict(
        cfg_path=os.path.join(cfg_dir, "resize_2_rear_bayes/multitask.py"),
        model_setting="test_lmdb",
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
                person=0.533,
                cyclist=0.657,
                vehicle=0.65,
                rear=0.56,
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
        model_address=model_resize_2_rear_bayes,
    ),
    pilot_legorcnn_multitask_side_resize_2=dict(
        cfg_path=os.path.join(cfg_dir, "resize_2_side_bayes/multitask.py"),
        model_setting="test_lmdb",
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
                person=0.517,
                cyclist=0.497,
                vehicle=0.67,
                rear=0.54,
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
        model_address=model_resize_2_side_bayes,  # noqa
    ),
    pilot_legorcnn_multitask_rear_crop=dict(
        cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
        model_setting="test_lmdb",
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
                person=0.62,
                cyclist=0.505,
                vehicle=0.45,
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
        model_address=model_crop_bayes,  # noqa
    ),
    pilot_legorcnn_iqa_parsing_resize_4=dict(
        cfg_path=os.path.join(
            cfg_dir, "single_task/image_fail_segmentation.py"
        ),
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="segmentation",
        model_setting="test_lmdb",
        model_address=model_image_fail_segmentation,
    ),
)

if test_level == "daily":
    models.update(
        dict(
            pilot_legorcnn_multitask_resize_2=dict(
                cfg_path=os.path.join(cfg_dir, "resize_2/multitask.py"),
                input_shape="1x3x640x1024",
                input_type="dict",
                input_source="pyramid",
                input_key="img",
                task_type="detection",
                model_setting="test_lmdb",
                jobs_num=jobs_num,
                extra_args="--pyramid-stride 1024 -g --max-time-per-fc 1000",
                input_layout=input_layout,
                output_layout=output_layout,
                # 更新模型阈值中的相关参数
                update_cfg=dict(
                    rpn_thresh=dict(
                        person=0.40,
                        cyclist=0.45,
                        vehicle=0.3,
                        rear=0.3,
                    ),
                    det_thresh=dict(
                        person=0.52,
                        cyclist=0.58,
                        vehicle=0.55,
                        rear=0.489,
                    ),
                    roi_det_thresh=dict(
                        vehicle=0.3,
                        rear=0.58,
                        person=0.6,
                    ),
                    thresh_3d=dict(
                        person=0.02,
                        cyclist=0.21,
                        vehicle=0.2,
                    ),
                    bbox_clipping=dict(
                        person=False,
                        cyclist=False,
                        vehicle=False,
                        rear=False,
                    ),
                ),
                model_address=model_resize_2,  # noqa
            ),
            pilot_legorcnn_multitask_resize_4=dict(
                cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
                input_shape="1x3x320x512",
                input_type="dict",
                input_source="pyramid",
                input_key="img",
                task_type="detection",
                model_setting="test_lmdb",
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
                        person=0.58,
                        cyclist=0.48,
                        vehicle=0.690,
                        rear=0.540,
                    ),
                    roi_det_thresh=dict(
                        vehicle=0.3,
                        rear=0.5,
                        person=0.49,
                    ),
                    thresh_3d=dict(
                        vehicle=0.2,
                        person=0.02,
                        cyclist=0.09,
                    ),
                    bbox_clipping=dict(
                        person=False,
                        cyclist=False,
                        vehicle=False,
                        rear=False,
                    ),
                ),
                model_address=model_resize_4,  # noqa
            ),
            pilot_legorcnn_multitask_crop=dict(
                cfg_path=os.path.join(cfg_dir, "crop/multitask.py"),
                input_shape="1x3x192x512",
                input_type="dict",
                input_source="pyramid",
                input_key="img",
                task_type="detection",
                model_setting="test_lmdb",
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
                model_address=model_crop,  # noqa
            ),
        )
    )
