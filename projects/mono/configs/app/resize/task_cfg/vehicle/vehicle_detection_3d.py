import os
from functools import partial

from projects.superparking.app.fisheye.lib.utils import get_data_path
from ...common import (
    bucket_root,
    ctimestr,
    local_train,
    log_freq,
    save_prefix,
    tasks_batch_size,
)
from ..real3d_common import (  # noqa
    base_data_version,
    camera_names,
    data_version,
    dataset_root,
    fisheye,
    get_data_info_fn,
    get_inputs,
    get_real3d_aidi_eval_callback,
    get_real3d_aidi_eval_loaders,
    get_real3d_dataloader,
    get_real3d_transforms,
    get_train_metric_updater,
    get_val_metric_updater,
    input_sequence_length,
    keep_org_img,
    num_dist,
    rescale,
    save_val_results,
    use_dynamic_loss_weight,
    visual_save_base,
)
from ..vdvru_common import (  # get_model_mtf3d,
    category_id_to_name,
    head_channels,
    multibin_margin,
    use_multibin,
)
from .common import (  # noqa
    disentangled_corner3D,
    get_model,
    num_classes,
    object_type,
)

task_type = "detection_3d"
dataset_name = "vehicle3d"

max_dep = 80
score_threshold = 0.4
flip = True

task_name = f"{object_type}_{task_type}"
batch_size = tasks_batch_size[task_name]


# -------------------- data ----------------------------------
fov = object_type

visual_save_path = f"{visual_save_base}/Vehicle"

if save_val_results:
    save_path = os.path.join(save_prefix, f"real3d/fov_{fov}", ctimestr)
else:
    save_path = None


# -------------------------------- dataset -----------------------------------
CLASS_PED = -1
CLASS_CYC = -1
CLASS_CAR = 0

# remap dataset cls ids
need_eval_categories = {
    # name_to_val, id_in_gt, id_id_pred
    "MergedCar": [[2, 4, 5, 6], CLASS_CAR],
    "Car": [[2], CLASS_CAR],
}
category_id_dict = {
    1: CLASS_PED,  # Pedestrian -> Pedestrian
    2: CLASS_CAR,  # Car        -> Car
    3: CLASS_CYC,  # Cyclist    -> Cyclist
    4: CLASS_CAR,  # Bus        -> Car
    5: CLASS_CAR,  # Truck      -> Car
    6: CLASS_CAR,  # SpecialCar -> Car
    7: CLASS_CAR,  # Tricycle   -> Car
    8: -1,  # Other    -> ignore
}  # 'Dontcare' -> Ignore

# data version
train_data_info = base_data_version[fov][data_version]["train"]
train_data_version = train_data_info["data_version"]
val_data_info = base_data_version[fov][data_version]["val"]
val_data_version = val_data_info["data_version"]

# data path
data_paths = get_data_path(bucket_root, f"{dataset_root}/vehicle.yaml")
train_rec_paths = []
val_rec_paths = []
for i in train_data_version:
    if data_paths[fov][i]["train"]["rec"]:
        train_rec_paths.extend(data_paths[fov][i]["train"]["rec"])
for i in val_data_version:
    if data_paths[fov][i]["val"]["rec"]:
        val_rec_paths.extend(data_paths[fov][i]["val"]["rec"])
val_rec_paths = sorted(set(val_rec_paths))
get_data_info = get_data_info_fn(
    data_version,
    train_rec_paths,
    val_rec_paths,
)
if local_train:
    train_rec_paths = train_rec_paths[:1]
    val_rec_paths = val_rec_paths  # [:1]

# transforms
get_transforms = partial(
    get_real3d_transforms,
    max_depth=max_dep,
    num_classes=num_classes,
    category_id_dict=category_id_dict,
    repeat_times=input_sequence_length,
    rescale=rescale,
    keep_org_img=keep_org_img,
    flip=flip,
)
train_transforms = get_transforms(
    mode="train",
    flip=flip,
)
val_transforms = get_transforms(
    mode="val",
    flip=False,
)
test_transforms = get_real3d_transforms(
    mode="test",
    max_depth=max_dep,
    rescale=rescale,
)

# dataloader
get_dataloader = partial(
    get_real3d_dataloader,
    num_classes=num_classes,
    transforms=train_transforms,
    rec_paths=train_rec_paths,
    num_dist=num_dist,
    batch_size=batch_size,
)
data_loader = get_dataloader(
    mode="train",
)
val_data_loader = get_dataloader(
    mode="val",
)
test_data_loader = get_dataloader(
    mode="test",
)

# --------------------metric-----------------------
metric_updater = get_train_metric_updater(
    task_name,
    log_freq=log_freq,
    head_channels=head_channels,
    use_dynamic_weight=use_dynamic_loss_weight,
    disentangled_corner3D=disentangled_corner3D,
    multibin_margin=multibin_margin,
    use_multibin=use_multibin,
)
val_metric_updater = get_val_metric_updater(
    task_name=task_name,
    need_eval_categories=need_eval_categories,
    eval_camera_names=camera_names,
    depth_intervals=(
        5,
        10,
        15,
        20,
    ),  # (5, 10, 15, 20, 30, 40, ),
    score_threshold=score_threshold,
    gt_max_depth=max_dep,
    save_path=save_path,
    num_dist=num_dist,
    fisheye=fisheye,
    match_basis="det2d",  # proj2d
    save_name="vehicle.xlsx",
)


# -----------------------aidi eval-----------------------------


(
    aidi_eval_callback,
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_real3d_aidi_eval_callback(
    dataset_name, task_name, category_id_to_name, category_id_dict
)

aidi_eval_loader = get_real3d_aidi_eval_loaders(
    batch_size["val"], dataset_name, val_transforms
)
