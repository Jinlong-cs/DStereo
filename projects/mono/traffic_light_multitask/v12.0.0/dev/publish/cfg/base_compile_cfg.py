import os

cfg_dir = os.path.join(os.path.dirname(__file__), "../../../configs")

march = "bernoulli2"
input_layout = "NHWC"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_{march}"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
plugin_version = "0.16.2"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.35.2"  # 编译用hbdk版本号

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = f"MCP_{march}"
desc = "MCP3.0-torch-test"  # 模型的描述


# 需要编译的模型
models = dict(
    pilot_legorcnn_multitask_resize_2=dict(
        cfg_path=os.path.join(cfg_dir, "resize_2/multitask.py"),
        input_shape="1x3x640x1024",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 1024 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="detection",
        # 更新模型阈值中的相关参数
        update_cfg=dict(
            rpn_thresh=dict(),
            det_thresh=dict(),
            roi_det_thresh=dict(),
            thresh_3d=dict(),
            bbox_clipping=dict(),
        ),
        model_address="",  # noqa
    ),
    pilot_legorcnn_multitask_resize_2_night=dict(
        cfg_path=os.path.join(cfg_dir, "resize_2/multitask.py"),
        model_setting="as33_night",
        input_shape="1x3x640x1024",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 1024 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="detection",
        update_cfg=dict(
            rpn_thresh=dict(),
            det_thresh=dict(),
            roi_det_thresh=dict(),
            thresh_3d=dict(),
            bbox_clipping=dict(),
        ),
        model_address="",  # noqa
    ),
    pilot_legorcnn_multitask_resize_4=dict(
        cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="detection",
        update_cfg=dict(
            rpn_thresh=dict(),
            det_thresh=dict(),
            roi_det_thresh=dict(),
            thresh_3d=dict(),
            bbox_clipping=dict(),
        ),
        model_address="",  # noqa
    ),
    pilot_legorcnn_multitask_resize_4_night=dict(
        cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="detection",
        update_cfg=dict(
            rpn_thresh=dict(),
            det_thresh=dict(),
            roi_det_thresh=dict(),
            thresh_3d=dict(),
            bbox_clipping=dict(),
        ),
        model_address="",
    ),
    pilot_legorcnn_multitask_crop=dict(
        cfg_path=os.path.join(cfg_dir, "crop/multitask.py"),
        input_shape="1x3x192x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 2048 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="detection",
        update_cfg=dict(
            rpn_thresh=dict(),
            det_thresh=dict(),
            roi_det_thresh=dict(),
            bbox_clipping=dict(),
        ),
        model_address="",
    ),
    pilot_legorcnn_multitask_crop_night=dict(
        cfg_path=os.path.join(cfg_dir, "crop/multitask.py"),
        input_shape="1x3x192x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 2048 -g --max-time-per-fc 1000",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="detection",
        update_cfg=dict(
            rpn_thresh=dict(),
            det_thresh=dict(),
            roi_det_thresh=dict(),
            bbox_clipping=dict(),
        ),
        model_address="",
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
        model_address="",
    ),
    pilot_legorcnn_iqa_parsing_resize_4_night=dict(
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
        model_address="",
    ),
    pilot_veh_reid=dict(
        cfg_path=os.path.join(cfg_dir, "single_task/vehicle_reid.py"),
        input_shape="1x3x128x128",
        input_type="dict",
        input_source="resizer",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="-g ",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="segmentation",
        model_address="",
    ),
    pilot_veh_reid_night=dict(
        cfg_path=os.path.join(cfg_dir, "single_task/vehicle_reid.py"),
        input_shape="1x3x128x128",
        input_type="dict",
        input_source="resizer",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="-g ",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="segmentation",
        model_address="",
    ),
    pilot_depth_resize_4=dict(
        cfg_path=os.path.join(cfg_dir, "single_task/depth_estimation.py"),
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="segmentation",
        model_address="",
    ),
    pilot_depth_resize_4_night=dict(
        cfg_path=os.path.join(cfg_dir, "single_task/depth_estimation.py"),
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 512 -g",
        input_layout=input_layout,
        output_layout=output_layout,
        task_type="segmentation",
        model_address="",
    ),
)
