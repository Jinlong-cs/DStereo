import os

# 实验配置
task_name = "cloudsparse4d_v1_1"
project_id = "PD20230003"
tags = ["all", "quick_exp", "wide"]
num_gpus = 8
num_gpus_debug = 1
num_gpus_eval = 4
local_or_remote_debug = True

# classes
# SD评测数据的name id
category_name2id = {
    "pedestrian": 1,
    "vehicle": 2,
    "cyclist": 3,
}
sub_category_name2id = {
    "Car": 0,
    "Bus": 0,
    "Truck": 0,
    "Tricycle": 0,
    "Construction": 0,
    "Special_vehicle": 0,
    "Tiny_car": 0,
    "Lorry": 0,
    "MiniVan": 0,
    "Sedan_Car": 0,
    "SUV": 0,
    "BigTruck": 0,
    "Motor-Tricycle": 0,
    "Flatbed_Trucks": 0,
    "Car_transporter": 0,
    "Tank_truck": 0,
    "Garbage_truck": 0,
    "Digger": 0,
    "Loader": 0,
    "Blur": 0,
    "Pedestrian": -99,
    "Cyclist": -99,
    "Other": -99,
    "Ignore": -99,
    "Unknown": -99,
}
# pilot数据的name id映射
category_id_dict = {
    1: 0,  # Pedestrian
    2: 1,  # Car        -> Car
    3: 2,  # Cyclist
    4: 1,  # Bus        -> Car
    5: 1,  # Truck      -> Car
    6: 1,  # SpecialCar -> Car
    7: 1,  # Blur       -> Car
    8: -99,  # Other    -> ignore
}  # 'Dontcare' -> Ignore
num_classes = 3


# network input
img_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[128.0, 128.0, 128.0], to_rgb=False
)
point_cloud_range = [-108.8, -51.2, -3.0, 108.8, 51.2, 5.0]
camera_view_names = [
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
    "camera_front",
    "camera_front_30fov",
]
sub_dirs = [view.replace("camera_", "") for view in camera_view_names]

# 虚拟crop相机
virtual_crop_views = ["front", "rear"]
virtual_crop_rois = [
    [960, 560, 2879, 1639],
    [480, 320, 1439, 959],
]
num_cameras = (
    len(camera_view_names) + len(virtual_crop_views) - 1
    if "front" in virtual_crop_views
    else len(camera_view_names) + len(virtual_crop_views)
)

# backbone拆分
view2category = {
    "front": ["front", "front_30fov"],
    "side": [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
        "rear",
        "virtual_rear",
    ],
}
# resolutions
view2target_dim = {
    "front": (1280, 1920),
    "side": (640, 960),
}
view2pad_dim = {
    "front": (1280, 1920),
    "side": (640, 960),
}

# temporal
max_interval = 600
max_len_in_clip = 20

# trainer
trainer_type = "distributed_data_parallel_trainer"
train_batch_size = 1
val_batch_size = int(os.getenv("val_batch_size", 4))
train_num_workers = 0
val_num_workers = 0
num_epochs = 3
num_steps = 200000
warmup_steps = 500
step_log_interval = 500
stat_log_freq = 1000
save_interval = 20000
interval_by = "step"
lr = 2e-5
weight_decay = 0.01

pretrain = "http://fm-fan-lv.bcloud-2nd.hogpu.cc/plat_gpu/hobot-dag-3953959_hat-job-cloudsparse4d-v0-vovnet-cpfpn-rear-crop-virtual-cam-480-320-1439-959-20230826-125900/output/models/cloudsparse4d_v0_vovnet_cpfpn_rear_crop_virtual_cam_480_320_1439_959/float-checkpoint-last.pth.tar"  # noqa


# eval
temporal_eval = os.getenv("temporal_eval", "False") == "True"
eval_ckpt = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/fan.lv/plat_gpu/hobot-dag-4467663_hat-job-cloudsparse4d-v4-20231010-095727/output/models/cloudsparse4d_v4/float-checkpoint-last.pth.tar"  # noqa

# hdflow model-zoo release
release_score_threshold = 0.1
release_class_id_map = {0: "person", 1: "vehicle", 2: "cyclist"}
