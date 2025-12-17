from __future__ import absolute_import, print_function
import os

from hat.data.packer.transformer.anno_ts_utils import (
    anno_to_contours_fn_with_label_mapping,
)
from hat.data.packer.utils import (
    get_colors_and_class_names_for_lane_parsing,
    get_default_pack_args_from_environment,
    ls_img_folder_and_anno_file,
)
from hat.utils import Config

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose

# logger

task_name = "lane_parsing"

cls = "5_2"  # lane parsing
label_map_config = Config.fromfile(
    os.path.join("configs", task_name, "lane_parsing_labelmap_5_2cls.py")
)  # noqa
src_label = label_map_config.src_label
dst_label = label_map_config.dst_label
if "dst_label_map" in label_map_config.keys():
    dst_label_map = label_map_config.dst_label_map
else:
    dst_label_map = None
color_map = label_map_config.color_map
if "class_config" in label_map_config.keys():
    class_config = label_map_config.class_config
else:
    class_config = None


input_root_dir = f"data/train/{task_name}"
output_dir = f"data/lmdb/{task_name}/"
label_map_output_dir = f"{output_dir}/gt/anno_{cls}"
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


reuse_prelabel = False  # whether to combine prelabel results to annotation


anno_ts_fn_config = dict(shuffle=True)
colors, clsnames = get_colors_and_class_names_for_lane_parsing(color_map)

anno_transformer = [
    dict(
        type="DefaultGenerateLabelMapAnnoTs",
        __build_recursive=False,
        output_dir=label_map_output_dir,
        src_label=src_label,
        dst_label=dst_label,
        dst_label_map=dst_label_map,
        reuse_prelabel=reuse_prelabel,
        colors=colors,
        clsnames=clsnames,
        # anno_to_contours_fn=default_anno_to_contours_fn,
        anno_to_contours_fn=anno_to_contours_fn_with_label_mapping,
        check_parsing_ignore=True,
    ),
    dict(
        type="DenseBoxSegAnnoTs",
        __build_recursive=False,
        class_ids=list(range(1, 20)),
        verify_image=True,
        verify_label=True,
    ),
]

data_packer = dict(
    type="DetSeg2DPackerLocal",
    folder_anno_pairs=folder_anno_pairs,
    output_dir=output_dir,
    num_workers=num_workers,
    anno_transform=anno_transformer,
)
