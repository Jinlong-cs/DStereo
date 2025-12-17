from __future__ import absolute_import, print_function
import os

from hat.data.packer.utils import (
    get_default_pack_args_from_environment,
    ls_img_folder_and_anno_file,
)

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose


parent_class_name = "vehicle"
children_class_name = "vehicle_kps_8"
task_name = "vehicle_ground_line"
input_task_name = "vehicle_keypoints"
task_attribute = "train"

input_root_dir = f"data/train/{input_task_name}"
output_dir = f"data/lmdb/{task_name}/"
excluded_annos_pattern = None

folder_anno_pairs = ls_img_folder_and_anno_file(
    root_path=input_root_dir,
    anno_ext=".json",
    recursive=True,
    excluded_annos_pattern=excluded_annos_pattern,
)

idx_path = os.path.join(output_dir, "idx")
img_path = os.path.join(output_dir, "img")
anno_path = os.path.join(output_dir, "anno")


max_num_imgs = -1  # maximum number of samples to be packed

anno_transformer = dict(
    type="VehicleFlankAnnoTs",
    parent_classname="vehicle",
    child_classname="vehicle_flank",
    anno_adapter=dict(
        # the adapter for input dataset
        type="AnnoAdapterKps8",
        parent_classname="vehicle",
        child_classname="vehicle_kps_8",
        target_classname="vehicle_flank",
        empty_value=-10000,
    ),
    min_flank_width=4,
    empty_value=-10000,
)

# pack image data
data_packer = dict(
    type="DetSeg2DPackerLocal",
    folder_anno_pairs=folder_anno_pairs,
    output_dir=output_dir,
    num_workers=num_workers,
    anno_transform=anno_transformer,
)
