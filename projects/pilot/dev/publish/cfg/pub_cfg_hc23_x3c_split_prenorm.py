import os

cfg_dir = os.path.join(os.path.dirname(__file__), "../../../configs")

march = "bayes"
input_layout = {}
input_layout["BEV"] = "NCHW"
input_layout["2D"] = "NHWC"
output_layout = {}
output_layout["BEV"] = "NCHW"
output_layout["2D"] = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_{march}_x3c"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 48  # 编译worker数量

# aidi编译模式相关参数
# plugin_version = "0.16.3"  # 编译用horizon_plugin_pytorch版本号
# hbcc_version = "v3.33.1"  # 编译用hbdk版本号
plugin_version = "1.1.3"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.41.3"  # 编译用hbdk版本号
# hbcc_version = "v3.38.8"  # 编译用hbdk版本号

# 模型推理相关参数
infer_result_prefix = "infer_pilot5.0_bev"


# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
publish_name = "MCP5.0_hc23_x3c_split"

desc = "MCP5.0-hc23-x3c-split"  # 模型的描述
# 需要编译的模型
models = dict()
models["pilot_bev_5v_multitask_day_stage1"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_bev/multitask.py"),
    model_setting="hc23_x3c_day",
    input_shape="1x3x640x960^1x640x960x2",  # noqa
    input_type="dict",
    input_source="pyramid,ddr",
    input_key="img^uv_map",  # noqa
    torch_native="False^True",
    jobs_num=jobs_num,
    extra_args="-g --debug=True --skip-check",
    output_layout=output_layout["BEV"],
    input_layout=input_layout["BEV"],
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.40,
            cyclist=0.45,
            vehicle=0.3,
            vehicle_rear=0.3,
            face=1.5,
            plate=1.5,
            vehicle_plate=1.5,
        ),
        det_thresh=dict(
            face_detection=0.5,
            vehicle_plate_detection=0.5,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            vehicle_rear=0.3,
            person=0.3,
        ),
        thresh_3d=dict(
            vehicle=0.2,
            person=0.03,
            cyclist=0.21,
        ),
        bev_det_thresh=dict(
            bev_3d_vehicle_cls=0.24,
            bev_3d_pedestrian=0.33,
            bev_3d_cyclist_cls=0.35,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            vehicle_rear=False,
        ),
    ),
    split_index=0,
    override=True,
    model_address="http://fm-yiding-liu.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_bev_hc23_x3c_day_3.5.0_qat-20221217-105458-COPY/output/models/pilot_multitask_resize2_bev/qat-checkpoint-last-e3f4bf8c.pth.tar",  # noqa
)

models["pilot_bev_5v_multitask_day_stage2"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_bev/multitask.py"),
    model_setting="hc23_x3c_day",
    input_shape="1x256x128x2^1x256x128x2^1x256x128x2^1x256x128x2^1x180x256x2^1x16x160x240^1x16x160x240^1x16x160x240^1x16x160x240^1x16x160x240",  # noqa
    input_type="dict",
    input_source="ddr,ddr,ddr,ddr,ddr,ddr,ddr,ddr,ddr,ddr",
    input_key="homo_offset_0^homo_offset_1^homo_offset_2^homo_offset_3^homo_offset_4^pred_segs_frame0_0^pred_segs_frame0_1^pred_segs_frame0_2^pred_segs_frame0_3^pred_segs_frame0_4",  # noqa
    torch_native="True^True^True^True^True^False^False^False^False^False",
    jobs_num=jobs_num,
    extra_args="-g --debug=True",
    output_layout=output_layout["BEV"],
    input_layout=input_layout["BEV"],
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.40,
            cyclist=0.45,
            vehicle=0.3,
            vehicle_rear=0.3,
            face=1.5,
            plate=1.5,
            vehicle_plate=1.5,
        ),
        det_thresh=dict(
            face_detection=0.5,
            vehicle_plate_detection=0.5,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            vehicle_rear=0.3,
            person=0.3,
        ),
        thresh_3d=dict(
            vehicle=0.2,
            person=0.03,
            cyclist=0.21,
        ),
        bev_det_thresh=dict(
            bev_3d_vehicle_cls=0.24,
            bev_3d_pedestrian=0.33,
            bev_3d_cyclist_cls=0.35,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            vehicle_rear=False,
        ),
    ),
    split_index=1,
    override=True,
    model_address="http://fm-yiding-liu.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_bev_hc23_x3c_day_3.5.0_qat-20221217-105458-COPY/output/models/pilot_multitask_resize2_bev/qat-checkpoint-last-e3f4bf8c.pth.tar",  # noqa
)
models["pilot_bev_5v_multitask_night_stage1"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_bev/multitask.py"),
    model_setting="hc23_x3c_day",
    input_shape="1x3x640x960^1x640x960x2",  # noqa
    input_type="dict",
    input_source="pyramid,ddr",
    input_key="img^uv_map",  # noqa
    torch_native="False^True",
    jobs_num=jobs_num,
    extra_args="-g --debug=True --skip-check",
    output_layout=output_layout["BEV"],
    input_layout=input_layout["BEV"],
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.40,
            cyclist=0.45,
            vehicle=0.3,
            vehicle_rear=0.3,
            face=1.5,
            plate=1.5,
            vehicle_plate=1.5,
        ),
        det_thresh=dict(
            face_detection=0.5,
            vehicle_plate_detection=0.5,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            vehicle_rear=0.3,
            person=0.3,
        ),
        thresh_3d=dict(
            vehicle=0.2,
            person=0.03,
            cyclist=0.21,
        ),
        bev_det_thresh=dict(
            bev_3d_vehicle_cls=0.15,
            bev_3d_pedestrian=0.32,
            bev_3d_cyclist_cls=0.35,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            vehicle_rear=False,
        ),
    ),
    split_index=0,
    override=True,
    model_address="http://fm-yiding-liu.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_bev_hc23_x3c_day_3.5.0_qat-20221217-105458-COPY/output/models/pilot_multitask_resize2_bev/qat-checkpoint-last-e3f4bf8c.pth.tar",  # noqa
)

models["pilot_bev_5v_multitask_night_stage2"] = dict(
    cfg_path=os.path.join(cfg_dir, "resize_2_bev/multitask.py"),
    model_setting="hc23_x3c_day",
    input_shape="1x256x128x2^1x256x128x2^1x256x128x2^1x256x128x2^1x180x256x2^1x16x160x240^1x16x160x240^1x16x160x240^1x16x160x240^1x16x160x240",  # noqa
    input_type="dict",
    input_source="ddr,ddr,ddr,ddr,ddr,ddr,ddr,ddr,ddr,ddr",
    input_key="homo_offset_0^homo_offset_1^homo_offset_2^homo_offset_3^homo_offset_4^pred_segs_frame0_0^pred_segs_frame0_1^pred_segs_frame0_2^pred_segs_frame0_3^pred_segs_frame0_4",  # noqa
    torch_native="True^True^True^True^True^False^False^False^False^False",
    jobs_num=jobs_num,
    extra_args="-g --debug=True",
    output_layout=output_layout["BEV"],
    input_layout=input_layout["BEV"],
    task_type="detection",
    framework="PyTorch",
    update_cfg=dict(
        rpn_thresh=dict(
            person=0.40,
            cyclist=0.45,
            vehicle=0.3,
            vehicle_rear=0.3,
            face=1.5,
            plate=1.5,
            vehicle_plate=1.5,
        ),
        det_thresh=dict(
            face_detection=0.5,
            vehicle_plate_detection=0.5,
        ),
        roi_det_thresh=dict(
            vehicle=0.3,
            vehicle_rear=0.3,
            person=0.3,
        ),
        thresh_3d=dict(
            vehicle=0.2,
            person=0.03,
            cyclist=0.21,
        ),
        bev_det_thresh=dict(
            bev_3d_vehicle_cls=0.15,
            bev_3d_pedestrian=0.32,
            bev_3d_cyclist_cls=0.35,
        ),
        bbox_clipping=dict(
            person=False,
            cyclist=False,
            vehicle=False,
            vehicle_rear=False,
        ),
    ),
    split_index=1,
    override=True,
    model_address="http://fm-yiding-liu.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_bev_hc23_x3c_day_3.5.0_qat-20221217-105458-COPY/output/models/pilot_multitask_resize2_bev/qat-checkpoint-last-e3f4bf8c.pth.tar",  # noqa
)
models["pilot_legorcnn_multitask_rear_crop_day"] = dict(
    cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
    model_setting="hc23_x3c_rear",
    input_shape="1x3x192x512",
    input_type="dict",
    input_source="pyramid",
    input_key="img",
    jobs_num=jobs_num,
    extra_args="--pyramid-stride 1920 -g --max-time-per-fc 1000",
    output_layout=output_layout["2D"],
    input_layout=input_layout["2D"],
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
            person=0.408,
            cyclist=0.45,
            vehicle=0.528,
            rear=0.179,
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
    # model_address="aidi://pilot5_multitask_crop_bayes_galaxy_0233_rear_train/v0.0.22/freeze_bn_3",  # noqa
    model_address="http://fm-kai-liu.train.hogpu.cc/plat_gpu/pilot5_multitask_rear_crop_galaxy_galaxy_0233_rear_train-20221007-175407/output/models/pilot5_multitask_crop_bayes_with_cyc_cls/freeze_bn_3-checkpoint-last-327e38da.pth.tar",  # noqa
)
models["pilot_legorcnn_multitask_rear_crop_night"] = dict(
    cfg_path=os.path.join(cfg_dir, "crop_bayes/multitask.py"),
    model_setting="hc23_x3c_rear",
    input_shape="1x3x192x512",
    input_type="dict",
    input_source="pyramid",
    input_key="img",
    jobs_num=jobs_num,
    extra_args="--pyramid-stride 1920 -g --max-time-per-fc 1000",
    output_layout=output_layout["2D"],
    input_layout=input_layout["2D"],
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
            person=0.408,
            cyclist=0.45,
            vehicle=0.528,
            rear=0.179,
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
    # model_address="aidi://pilot5_multitask_crop_bayes_galaxy_0233_rear_train/v0.0.22/freeze_bn_3",  # noqa
    model_address="http://fm-kai-liu.train.hogpu.cc/plat_gpu/pilot5_multitask_rear_crop_galaxy_galaxy_0233_rear_train-20221007-175407/output/models/pilot5_multitask_crop_bayes_with_cyc_cls/freeze_bn_3-checkpoint-last-327e38da.pth.tar",  # noqa
)

models["pilot_legorcnn_iqa_parsing_resize_4"] = dict(
    dict(
        cfg_path=os.path.join(
            cfg_dir, "single_task/image_fail_segmentation.py"
        ),
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        task_type="segmentation",
        framework="PyTorch",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g --max-time-per-fc 1000",
        input_layout=input_layout["2D"],
        output_layout=output_layout["2D"],
        model_address="http://fm-rui01-wang.alitrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-13-input-320x512-hat-cu111-20220602_old-docker-20220829-020227/output/models/image_fail_parsing/qat-checkpoint-last-7cb117fd.pth.tar",  # noqa
    )
)
models["pilot_legorcnn_iqa_parsing_resize_4_night"] = dict(
    dict(
        cfg_path=os.path.join(
            cfg_dir, "single_task/image_fail_segmentation.py"
        ),
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        task_type="segmentation",
        framework="PyTorch",
        input_shape="1x3x320x480",
        jobs_num=jobs_num,
        extra_args="--pyramid-stride 480 -g --max-time-per-fc 1000",
        input_layout=input_layout["2D"],
        output_layout=output_layout["2D"],
        model_address="http://fm-rui01-wang.alitrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-13-input-320x512-hat-cu111-20220602_old-docker-20220829-020227/output/models/image_fail_parsing/qat-checkpoint-last-7cb117fd.pth.tar",  # noqa
    )
)
