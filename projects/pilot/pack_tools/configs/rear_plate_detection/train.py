from __future__ import absolute_import, print_function
import os

import yaml

from hat.data.packer.utils import (
    get_default_pack_args_from_environment,
    ls_img_folder_and_anno_file,
)

CLASS_NAMES = [
    "CLASS_01",
    "CLASS_02",
    "CLASS_03",
    "CLASS_04",
    "CLASS_05",
    "CLASS_06",
    "CLASS_07",
    "CLASS_08",
    "CLASS_09",
    "CLASS_10",
    "CLASS_11",
    "CLASS_12",
    "CLASS_13",
    "CLASS_14",
    "CLASS_15",
    "CLASS_16",
    "CLASS_17",
    "CLASS_18",
    "CLASS_19",
    "CLASS_20",
    "CLASS_21",
    "CLASS_22",
    "CLASS_23",
    "CLASS_24",
    "CLASS_25",
    "CLASS_26",
    "CLASS_27",
    "CLASS_28",
    "CLASS_29",
    "CLASS_30",
    "CLASS_31",
    "CLASS_32",
    "CLASS_33",
    "CLASS_34",
    "CLASS_35",
    "CLASS_36",
    "CLASS_37",
    "CLASS_38",
    "CLASS_39",
    "CLASS_40",
    "CLASS_41",
    "CLASS_42",
    "CLASS_43",
    "CLASS_44",
    "CLASS_45",
    "CLASS_46",
    "CLASS_47",
    "CLASS_48",
    "CLASS_49",
    "CLASS_50",
]

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose


task_name = "rear_plate_detection"

anno_ts_fn_config_path = f"configs/{task_name}/anno_ts_config.yaml"
with open(anno_ts_fn_config_path) as fin:
    anno_ts_fn_config = yaml.load(fin, Loader=yaml.FullLoader)

input_root_dir = f"data/train/{task_name}"
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

viz_num_show = 20
viz_save_flag = True
viz_save_path = os.path.join(output_dir, "viz")

sub_classname = anno_ts_fn_config["children_box_classname"]
anno_transformer = [
    dict(
        type="RearPlateKps4ToBBox",
        classname=sub_classname,
    ),
    dict(
        type="DenseBoxSubboxDetAnnoTs",
        verbose=False,
        anno_config=anno_ts_fn_config,
        root_dir=input_root_dir,
    ),
]

data_packer = dict(
    type="DetSeg2DPackerLocal",
    folder_anno_pairs=folder_anno_pairs,
    output_dir=output_dir,
    num_workers=num_workers,
    anno_transform=anno_transformer,
)


# dataset for visualization
viz_dataset = dict(
    type="DetSeg2DAnnoDataset",
    idx_path=idx_path,
    img_path=img_path,
    anno_path=anno_path,
)

viz_fn = dict(
    type="VizRoiDenseBoxDetAnno",
    save_flag=True,
    save_path=viz_save_path,
    viz_class_id=list(range(1, 51)),
    class_name=CLASS_NAMES,
    lt_point_id=0,
    rb_point_id=2,
    roi_lt_point_id=10,
    roi_rb_point_id=12,
)
