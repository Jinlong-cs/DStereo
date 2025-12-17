import copy
from collections import OrderedDict

import numpy as np
import torch
from bev_common import (
    bevfusion_output_size,
    camera_view_names,
    get_common_transforms,
    get_dataloader,
    get_homo_transforms,
    organize_data_type,
    per_view_shape,
    resize_hw_list,
    roi_region_list,
    spatial_resolution,
    standardized_cam_calibs,
    use_distorted_offset,
    vcs_range,
    view_num,
)
from common import (
    batch_size_bev,
    bev_task_names,
    datapaths,
    input_hw,
    log_freq,
    training_step,
    with_cam_standiardization,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.metrics.bev_3d import BEVDetEval
from hat.models.losses.real3d_losses import sigmoid_and_clip

train_batch_size_per_gpu = batch_size_bev
val_batch_size_per_gpu = 8


train_num_workers = 4
val_num_workers = 4
test_num_workers = 0

test_image_dir = ""
test_image_calibration = ""
test_image_dist_coeffs = ""
test_attribte_json_path = ""
test_batch_size_per_gpu = 1
test_homo_path = ""

# --------------------------BEV BASE --------------------------
use_multi_head = False
use_stage1_loss = False
# assert use_multi_head, "current bev only support use_multi_head"

# HAT metric setting
save_eval_results = True
save_real3d_results = True
save_vis_dir = None
vis_setting = None
get_pack_dir = None
use_nms = False


task_num_classes = {
    "bev_3d_vehicle_cls": 8,
    "bev_3d_vehicle": 1,
    "bev_3d_pedestrian": 1,
    "bev_3d_cyclist_cls": 2,
    "bev_3d_cyclist": 1,
}

task_use_ignore_mask = {
    "bev_3d_vehicle_cls": True,
    "bev_3d_vehicle": True,
    "bev_3d_pedestrian": False,
    "bev_3d_cyclist": False,
    "bev_3d_cyclist_cls": False,
}

eval_setting = {
    "bev_3d_vehicle_cls": {
        "eval_category_ids": (0,),
        "score_threshold": 0.1,
        "iou_threshold": 0.2,
        "gt_max_depth": 100,
    },
    "bev_3d_vehicle": {
        "eval_category_ids": (0,),
        "score_threshold": 0.1,
        "iou_threshold": 0.2,
        "gt_max_depth": 100,
    },
    "bev_3d_pedestrian": {
        "eval_category_ids": (0,),
        "score_threshold": 0.2,
        "iou_threshold": 0.2,
        "gt_max_depth": 100,
    },
    "bev_3d_cyclist": {
        "eval_category_ids": (0,),
        "score_threshold": 0.2,
        "iou_threshold": 0.2,
        "gt_max_depth": 100,
    },
    "bev_3d_cyclist_cls": {
        "eval_category_ids": (0,),
        "score_threshold": 0.2,
        "iou_threshold": 0.2,
        "gt_max_depth": 100,
    },
}

# ------------------ BEV3D TRANSFORM SETTING -------------------
bev_3d_out_size = (
    bevfusion_output_size[0] // 2,
    bevfusion_output_size[1] // 2,
)
max_objs = 300
sub_dirs = [view.replace("camera_", "") for view in camera_view_names]

# ------------------- BEV3D DATASET SETTING --------------------
bev_common_transforms = get_common_transforms(
    _size=resize_hw_list,
    crop_roi_list=roi_region_list,
    _organize_data_type=organize_data_type,
    _H_persp_view_scale=1 / 4,
    to_yuv=False,  # ? maybe True but lazy_yuv=True?
    lazy_yuv=True,
)

load_data_types = ["gt_bev_3d", "img_name", "timestamp", "pack_dir"]
if use_stage1_loss:
    load_data_types.append("gt_seg")

collect_3dv = bev_common_transforms["Collect3DV"]
collect_3dv["gt_bev_3d_idx"] = 0
collect_3dv["load_data_types"] = load_data_types

# ------------------- REAL3D DATASET TRANSFORM SETTING ------------------
repeat_image = dict(
    type="RepeatImage", times=1, input_key="img", output_key="pil_imgs"
)


def get_dataset_list(path_dict, transforms_list, homo_transforms=None):
    datasets_list = [
        dict(
            type="MultiViewRecDataset",
            rec_path=rec_path,
            rec_idx_file=rec_path + ".idx",
            camera_view_names=sub_dirs,
            view_shapes=per_view_shape,
            to_rgb=True,
            decode_img=True,
            transforms=transforms_list,
            standardized_cam_calibs=standardized_cam_calibs,
            homo_cfg=dict(
                homo_path=None,
                calib_path=None,
                spatial_resolution=spatial_resolution,
                vcs_range=vcs_range,
                camera_view_names=camera_view_names,
                per_view_shape=per_view_shape,
                norm_homo=True,
                use_distorted_offset=use_distorted_offset,
                homo_transforms=homo_transforms,
            ),
        )
        for rec_path in path_dict["rec_path"]
    ]
    return datasets_list


def get_val_dataset_list(path_list, transforms_list, homo_transforms=None):
    datasets_list = [
        dict(
            type="MultiViewImgDataset",
            img_dir=data_path["img_dir"],
            anno_json_file=data_path["json_file"],
            camera_list=sub_dirs,
            view_shapes=per_view_shape,
            to_rgb=True,
            transforms=transforms_list,
            standardized_cam_calibs=standardized_cam_calibs,
            homo_cfg=dict(
                homo_path=None,
                calib_path=None,
                spatial_resolution=spatial_resolution,
                vcs_range=vcs_range,
                camera_view_names=camera_view_names,
                per_view_shape=per_view_shape,
                norm_homo=True,
                use_distorted_offset=use_distorted_offset,
                homo_transforms=homo_transforms,
            ),
        )
        for data_path in path_list
    ]
    return datasets_list


def get_val_raw_img_dataset_list(
    path_list, transforms_list, homo_transforms=None
):
    datasets_list = [
        dict(
            type="MultiViewImgDataset",
            img_dir=data_path["img_dir"],
            calib_path=data_path["calib_path"],
            camera_list=sub_dirs,
            view_shapes=per_view_shape,
            to_rgb=True,
            transforms=transforms_list,
            homo_cfg=dict(
                homo_path=None,
                calib_path=None,
                spatial_resolution=spatial_resolution,
                vcs_range=vcs_range,
                camera_view_names=camera_view_names,
                per_view_shape=per_view_shape,
                norm_homo=True,
                use_distorted_offset=use_distorted_offset,
                homo_transforms=homo_transforms,
            ),
        )
        for data_path in path_list
    ]
    return datasets_list


def get_bev3d_sub_category_id_map(task_name):
    if task_name == "bev_3d_vehicle_cls":
        sub_category_id_dict = {
            "Sedan_Car": 1,  # Small_Medium_Car
            "SUV": 1,  # Small_Medium_Car
            "Bus": 0,  # Bus
            "BigTruck": 2,  # Trucks
            "Lorry": 6,  # Lorry
            "Bike": -99,
            "MiniVan": 7,  # MiniVan
            "Special_vehicle": 4,  # Special_vehicle
            "Motorcycle_electrombile": 3,  # Motors
            "Tricycle": 3,  # Motors
            "Motor-Tricycle": 3,  # Motors
            "Vehicle_others": -99,
            "Non-Motor Vehicle_others": -99,
            "Tiny_car": 5,  # Tiny_car
            "unknown": -99,
            "Flatbed_Trucks": 2,  # Trucks
            "Car_transporter": 2,  # Trucks
            "Tank_truck": 4,  # Special_vehicle
            "Garbage_truck": 4,  # Special_vehicle
            "Digger": 4,  # Special_vehicle
            "Loader": 4,  # Special_vehicle
            "Vehicle_light": 1,  # Small_Medium_Car
        }
    elif task_name == "bev_3d_cyclist_cls":
        sub_category_id_dict = {
            "PersonRideBicycle": 0,  # PersonRideBicycle
            "PersonRideMotorcycle": 1,  # PersonRideMotorcycle
            "AEB_PersonRideBicycle": 0,  # PersonRideBicycle
            "AEB_PersonRideMotorcycle": 1,  # PersonRideMotorcycle
        }  # Others -> ignore
    else:
        raise KeyError(
            f"task_name {task_name} not support attrs_type as sub category id,"
            "Please Check the task_name"
        )
    return sub_category_id_dict


def get_real3d_category_id_map(task_name):
    if task_name == "bev_3d_vehicle":
        category_id_dict = {
            1: -99,  # Pedestrian -> ignore
            2: 0,  # Car        -> Car
            3: -99,  # Cyclist    -> ignore
            4: 0,  # Bus        -> Car
            5: 0,  # Truck      -> Car
            6: 0,  # SpecialCar -> Car
            7: 0,  # Blur       -> Car
            8: -99,  # Other    -> ignore
        }  # 'Dontcare' -> Ignore
    elif task_name == "bev_3d_vehicle_cls":
        category_id_dict = {
            i: i for i in range(task_num_classes["bev_3d_vehicle_cls"])
        }
    elif task_name == "bev_3d_cyclist":
        category_id_dict = {
            1: -99,  # Pedestrian -> ignore
            2: -99,  # Car        -> ignore
            3: 0,  # Cyclist    -> Cyclist
            4: -99,  # Bus        -> ignore
            5: -99,  # Truck      -> ignore
            6: -99,  # SpecialCar -> ignore
            7: -99,  # Blur       -> ignore
            8: -99,  # Other    -> ignore
        }  # 'Dontcare' -> Ignore
    elif task_name == "bev_3d_cyclist_cls":
        category_id_dict = {
            i: i for i in range(task_num_classes["bev_3d_cyclist_cls"])
        }
    elif task_name == "bev_3d_pedestrian":
        category_id_dict = {
            1: 0,  # Pedestrian -> Pedestrian
            2: -99,  # Car        -> ignore
            3: -99,  # Cyclist    -> ignore
            4: -99,  # Bus        -> ignore
            5: -99,  # Truck      -> ignore
            6: -99,  # SpecialCar -> ignore
            7: -99,  # Blur       -> ignore
            8: -99,  # Other    -> ignore
        }  # 'Dontcare' -> Ignore
    elif task_name == "bev_3d_full":
        category_id_dict = {
            1: 0,  # Pedestrian -> Pedestrian
            2: 0,  # Car        -> Car
            3: 0,  # Cyclist    -> Cyclist
            4: 0,  # Bus        -> Car
            5: 0,  # Truck      -> Car
            6: 0,  # SpecialCar -> Car
            7: 0,  # Blur       -> Car
            8: -99,  # Other    -> ignore
        }  # 'Dontcare' -> Ignore
    else:
        raise KeyError(
            f"task_name {task_name} not support, Please Check the task_name"
        )
    return category_id_dict


# ------------------- BEV3D SHARED DATALOADER --------------------
bev_transform_cfgs = {
    "bev_3d_vehicle_cls": [
        "bev_3d_vehicle",
        [1.7761209, 1.8237954, 4.79461],
        9,
    ],
    "bev_3d_vehicle": [
        "bev_3d_vehicle",
        [1.7761209, 1.8237954, 4.79461],
        9,
    ],
    "bev_3d_pedestrian": [
        "bev_3d_pedestrian",
        [1.6383535, 0.60629284, 0.5058226],
        3,
    ],
    "bev_3d_cyclist_cls": [
        "bev_3d_cyclist",
        [1.5518998, 0.73560804, 1.7083853],
        3,
    ],
    "bev_3d_cyclist": [
        "bev_3d_cyclist",
        [1.5518998, 0.73560804, 1.7083853],
        3,
    ],
}
sub_transforms = {}
for task_name in bev_task_names:
    bev_cfg = bev_transform_cfgs[task_name]
    num_classes = task_num_classes[task_name]
    use_ignore_mask = task_use_ignore_mask[task_name]
    cls_dimension = np.array([bev_cfg[1]] * num_classes)
    cls_hm_kernel = {i: bev_cfg[2] for i in range(num_classes)}

    sub_transforms[task_name] = [
        dict(
            type="ANCBev3dTargetGenerator",
            num_classes=num_classes,
            max_objs=max_objs,
            bev_size=bev_3d_out_size,
            vcs_range=vcs_range,
            cls_dimension=cls_dimension,
            cls_hm_kernel=cls_hm_kernel,
            category2id_map=get_real3d_category_id_map(
                bev_cfg[0]
            ),  # big category
            subcategory2id_map=None
            if "cls" not in task_name
            else get_real3d_category_id_map(task_name),  # sub category
            enable_ignore=use_ignore_mask,
        ),
        bev_common_transforms["PrepareDataBEV"],
    ]

bev_3d_transforms = [
    dict(
        type="ConvertMultiViewAnnoTo3DV",
        category_id_dict=get_real3d_category_id_map(
            "bev_3d_full"
        ),  # big class all
        camera_list=sub_dirs,
        per_view_shape=per_view_shape,
        pad_value=128,
        using_sub_category=False,
    ),
    repeat_image,
    bev_common_transforms["Resize3DV"],
    bev_common_transforms["Crop3DV"],
    bev_common_transforms["ResizeHomo"],
    bev_common_transforms["ToTensor3DV"],
    # bev_common_transforms["Normalize3DV"],
    dict(
        type="MultiTaskAnnoWrapper",
        sub_transforms=sub_transforms,
        unikeys=("gt_bev_3d", "annos_bev_3d"),
        repkeys=(
            "homography",
            "homo_offset",
            "timestamp",
            "view",
            "pack_dir",
        ),
    ),
]
homo_transforms = get_homo_transforms(
    transforms_list=bev_3d_transforms, camera_view_names=camera_view_names
)

train_data_paths = datapaths.multiview_vehicle_3d_detection.train_data_paths

# bev dataloader should be the same as that in each bev task
train_datasets_list = get_dataset_list(
    train_data_paths[0], bev_3d_transforms, homo_transforms
)
bev_3d_full_data_loader = get_dataloader(
    train_datasets_list, train_num_workers
)
bev_3d_full_data_loader["sampler"] = dict(
    type=torch.utils.data.DistributedSampler
)
bev_3d_full_data_loader["shuffle"] = True
# To fix reload imgrec problem, use persistent_workers.
# persistent_workers option needs num_workers > 0
bev_3d_full_data_loader["persistent_workers"] = train_num_workers > 0


# -------------------------- MODEL --------------------------
def get_inputs(num_classes_=8):
    inputs = dict(
        homography=torch.randn((1, view_num, 3, 3)),
        homo_offset=torch.randn((view_num, 256, 256, 2)),
        timestamp=torch.randn((1,)),
        gt_bev_3d={
            "bev3d_hm": torch.zeros(
                1, num_classes_, 256, 256, dtype=torch.float32
            ),
            "bev3d_dim": torch.zeros(1, 3, 256, 256, dtype=torch.float32),
            "bev3d_rot": torch.zeros(1, 2, 256, 256, dtype=torch.float32),
            "bev3d_ct_offset": torch.zeros(
                1, 2, 256, 256, dtype=torch.float32
            ),
            "bev3d_loc_z": torch.zeros(1, 1, 256, 256, dtype=torch.float32),
            "bev3d_weight_hm": torch.zeros(
                1, 1, 256, 256, dtype=torch.float32
            ),
            "bev3d_point_pos_mask": torch.zeros(
                1, 1, 256, 256, dtype=torch.float32
            ),
        },
        annos_bev_3d={
            "vcs_loc_": torch.zeros(1, 300, 3, dtype=torch.float32),
            "vcs_cls": torch.zeros(1, 300, dtype=torch.float32),
            "vcs_rot_z_": torch.zeros(1, 300, dtype=torch.float32),
            "vcs_dim_": torch.zeros(1, 300, 3, dtype=torch.float32),
            "vcs_ignore_": torch.zeros(1, 300, dtype=torch.bool),
        },
        view="",
        pack_dir="",
    )
    if with_cam_standiardization:
        inputs["uv_map"] = torch.randn((view_num, *input_hw, 2))
    if use_multi_head:
        inputs["view"] = ["front_side"]
        inputs["side_img"] = [
            torch.rand((5, 3, 640, 1024)),
            torch.rand((5, 3, 640, 1024)),
        ]

    val_inputs = copy.deepcopy(inputs)

    if training_step == "int_infer":
        test_inputs = dict()
    else:
        test_inputs = dict(
            homography=torch.randn((2, view_num, 3, 3)),
            img_name=[
                "xxx.jpg",
            ],
        )
        if use_multi_head:
            test_inputs["side_img"] = None
            test_inputs["view"] = None
        if get_pack_dir:
            test_inputs["pack_dir"] = None
    return inputs, val_inputs, test_inputs


# -------------------- BEV3D COMMON SOLVER ----------------
def get_metrics_patterns(task_name):
    metrics = [
        dict(type="LossShow", name=f"{task_name}_hm_loss"),
        dict(type="LossShow", name=f"{task_name}_dim_loss"),
        dict(type="LossShow", name=f"{task_name}_rot_loss"),
        dict(type="LossShow", name=f"{task_name}_ct_offset_loss"),
        dict(type="LossShow", name=f"{task_name}_loc_z_loss"),
    ]

    per_metric_patterns = [  # corresponding to metrics
        dict(
            label_pattern=None, pred_pattern=f"^.*{task_name}.*bev3d_hm_loss$"
        ),
        dict(
            label_pattern=None, pred_pattern=f"^.*{task_name}.*bev3d_dim_loss$"
        ),
        dict(
            label_pattern=None, pred_pattern=f"^.*{task_name}.*bev3d_rot_loss$"
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}.*bev3d_ct_offset_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}.*bev3d_loc_z_loss$",
        ),
    ]

    return metrics, per_metric_patterns


def get_metric_updater(metrics, per_metric_patterns, task_name):

    metric_updater = dict(
        type="MetricUpdater",
        metrics=metrics,
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=per_metric_patterns
        ),
        step_log_freq=log_freq,
        epoch_log_freq=1,
        log_prefix=task_name,
        reset_metrics_by="log",
    )
    return metric_updater


def get_update_metric_func(mode, task_name):
    assert mode in ["val"]

    def _val(metrics, batch, model_outs):
        pred_dict = {}
        pred_keys = [
            "bev3d_ct",
            "bev3d_cls_id",
            "bev3d_score",
            "bev3d_rot",
            "bev3d_dim",
            "bev3d_loc_z",
        ]
        pred_dict = OrderedDict()

        index_bev3d = -1
        for i in range(len(metrics)):
            if isinstance(metrics[i], BEVDetEval):
                index_bev3d = i
        assert index_bev3d != -1, "BEVDetEval is not in metrics list!"

        if task_name in model_outs:
            for k, v in model_outs[task_name][0].items():
                for key in pred_keys:
                    if "predict_" + key in k:
                        pred_dict[key] = v
            metrics[index_bev3d].update(batch[0], pred_dict)

    if mode == "val":
        return _val
    else:
        raise NotImplementedError


visibility_intervals = None


def get_val_metric_updater(
    task_name,
    eval_category_ids,
    score_threshold,
    iou_threshold,
    gt_max_depth,
    save_path,
    use_ignore_mask,
    save_vis_dir,
    vis_setting,
    prcurv_save_path,
):
    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=[
            dict(
                type="BEVDetEval",
                eval_category_ids=eval_category_ids,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                gt_max_depth=gt_max_depth,
                save_path=save_path,
                save_real3d_res=save_real3d_results,
                save_vis_dir=save_vis_dir,
                vis_setting=vis_setting,
                enable_ignore=use_ignore_mask,
                prcurv_save_path=prcurv_save_path,
            )
        ],
        metric_update_func=get_update_metric_func("val", task_name),
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix="Validation " + task_name,
    )

    return val_metric_updater


# -------------------------- TENSORBOADD --------------------------


def get_gt_hm_vis(data):
    data = data.detach().permute(1, 2, 0).cpu().numpy()
    data *= 255.0
    num_classes = data.shape[2]
    data = data[:, :, tuple(range(num_classes)[::-1])]
    data = data.astype("uint8")
    return data


def get_pred_hm_vis(data):
    data_vis = sigmoid_and_clip(data.detach())
    data_vis = data_vis / data_vis.max()
    data_vis = data_vis * 255.0
    data_vis[data_vis != data_vis.max(dim=1, keepdim=True)[0]] = 0
    data_vis = data_vis[0].permute([1, 2, 0]).cpu().numpy()
    num_classes = data_vis.shape[2]
    data = data[:, :, tuple(range(num_classes)[::-1])]
    data_vis = data_vis.astype("uint8")
    return data_vis


def get_bev3d_tb_update_func(writer, model_outs, global_step_id, **kwargs):
    # model_outs = to_flat_ordered_dict(model_outs)
    if "bev_3d" in list(model_outs.keys())[0]:
        for k, v in model_outs.items():
            if "target_gt_bev_3d_bev3d_hm" in k:
                data_vis = get_gt_hm_vis(v[0])
                writer.add_images(
                    k, data_vis, global_step_id, dataformats="HWC"
                )
            elif "predict_bev3d_hm" in k:
                data_vis = get_pred_hm_vis(v[:1])
                writer.add_images(
                    k, data_vis, global_step_id, dataformats="HWC"
                )
            else:
                continue
