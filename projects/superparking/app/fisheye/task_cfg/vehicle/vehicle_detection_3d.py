import functools
import os
from collections import OrderedDict

from projects.superparking.app.fisheye.lib.utils import get_data_path
from ...common import (
    bucket_root,
    log_freq,
    pipeline_test,
    save_prefix,
    tasks_batch_size,
)
from ..real3d_common import (  # noqa
    base_data_version,
    camera_names,
    camera_type,
    current_time,
    data_version,
    dataset_root,
    depth_type,
    fisheye,
    get_data_info_fn,
    get_inputs,
    get_real3d_aidi_eval_info,
    get_real3d_aidi_eval_loaders,
    get_real3d_transforms,
    get_train_metric_updater,
    get_val_metric_updater,
    input_sequence_length,
    is_virtual,
    keep_org_img,
    lmdb_version,
    num_dist,
    pred_json_name,
    rescale,
    resize,
    save_pred_json,
    save_val_results,
    use_bev_lmdb,
    use_dynamic_loss_weight,
    view,
    visual_save_base,
)
from ..vdvru_common import (  # get_model_mtf3d,
    category_id_to_name,
    get_real3d_dataloader,
    head_channels,
    multibin_margin,
    use_multibin,
)
from .common import get_model  # noqa
from .common import disentangled_corner3D, num_classes, object_type

task_type = "detection_3d"
# vis_tasks.append("vehicle")

task_name = f"{object_type}_{task_type}"
batch_size = tasks_batch_size[task_name]

val_decoders = {}

# inputs
# -------------------- data ----------------------------------
fov = object_type  # connect with dataset default:superparking\fisheye

visual_save_path = os.path.join(visual_save_base, object_type)

if save_val_results:
    save_path = os.path.join(save_prefix, f"real3d/fov_{fov}", current_time)
else:
    save_path = None

if (
    os.path.exists(
        os.path.join(visual_save_path, "output_json", pred_json_name)
    )
    and save_pred_json
):
    raise OSError("pred.json is append write, please del old json")

if use_dynamic_loss_weight:
    loss_weights = OrderedDict(
        hm=1.0,
        rot=10.0,
        dep=3.0,
        dim=1.0,
        loc_offset=3.0,
        wh=0.01,
    )

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

max_dep = 30
score_threshold = 0.4
flip = False
translate = False
cpu_warp = False

# data version
train_data_info = base_data_version[fov][data_version]["train"]
train_data_version = train_data_info["data_version"]
val_data_info = base_data_version[fov][data_version]["val"]
val_data_version = val_data_info["data_version"]

# data path
data_paths = get_data_path(
    bucket_root, f"{dataset_root}/{camera_type}/vehicle_filtered.yaml"
)
lmdb_data_path = os.path.join(dataset_root, "4DGT", "vehicle_dataset.yaml")
lmdb_train_version_path = os.path.join(
    dataset_root, "4DGT", "vehicle_train_version.yaml"
)
lmdb_val_version_path = os.path.join(
    dataset_root, "4DGT", "vehicle_val_version.yaml"
)
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
if pipeline_test:
    train_rec_paths = train_rec_paths[:1]
    val_rec_paths = val_rec_paths  # [:1]

# transforms
get_real3d_transforms_fn = functools.partial(
    get_real3d_transforms,
    max_depth=max_dep,
    num_classes=num_classes,
    depth_type=depth_type,
    category_id_dict=category_id_dict,
    repeat_times=input_sequence_length,
    rescale=rescale,
    keep_org_img=keep_org_img,
    is_virtual=is_virtual,
    resize=resize,
)
train_transforms = get_real3d_transforms_fn(
    mode="train",
    flip=flip,
    translate=translate,
    warp=cpu_warp,
)
bev_lmdb_train_transforms = get_real3d_transforms_fn(
    mode="train",
    flip=flip,
    camera_type="Fisheye",
    is_virtual=True,
    translate=translate,
    warp=cpu_warp,
)
bev_lmdb_val_transforms = get_real3d_transforms_fn(
    mode="val",
    flip=False,
    camera_type="Fisheye",
    is_virtual=True,
)
val_transforms = get_real3d_transforms_fn(
    mode="val",
    flip=False,
)
test_transforms = get_real3d_transforms(
    mode="test",
    max_depth=max_dep,
    rescale=rescale,
)

# dataloader
data_loader = get_real3d_dataloader(
    mode="train",
    batch_size=batch_size["train"],
    num_classes=num_classes,
    transforms=train_transforms,
    rec_paths=train_rec_paths,
    num_dist=num_dist,
    view=view,
    use_bev_lmdb=use_bev_lmdb,
    version_yaml=lmdb_train_version_path,
    spec_yaml=lmdb_data_path,
    version=lmdb_version,
    lmdb_transforms=bev_lmdb_train_transforms,
    lmdb_weight=0.2,
)
val_data_loader = get_real3d_dataloader(
    mode="val",
    batch_size=batch_size["val"],
    num_classes=num_classes,
    transforms=val_transforms,
    rec_paths=val_rec_paths,
    num_dist=num_dist,
    view=view,
    use_bev_lmdb=use_bev_lmdb,
    version_yaml=lmdb_val_version_path,
    spec_yaml=lmdb_data_path,
    version=lmdb_version,
    lmdb_transforms=bev_lmdb_val_transforms,
    lmdb_weight=0.1,
)
test_data_loader = get_real3d_dataloader(
    mode="test",
    batch_size=batch_size["test"],
    num_classes=num_classes,
    transforms=test_transforms,
    rec_paths=val_rec_paths,
    view=view,
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
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_callback,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_real3d_aidi_eval_info(task_name, category_id_to_name, category_id_dict)

aidi_eval_loader = get_real3d_aidi_eval_loaders(
    batch_size["val"], task_name, val_transforms
)
