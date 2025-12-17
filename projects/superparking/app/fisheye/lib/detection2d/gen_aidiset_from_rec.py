import json
import os
import subprocess

import cv2

from hat.data.datasets.dataset_wrappers import ConcatDataset
from hat.data.datasets.legacy_densebox import LegacyDenseBoxImageRecordDataset
from ...datasets.auto_2d.dataset import datapaths
from ..constants import AIDI_EVAL_SCRIPT

BUCKET_ROOT = "/horizon-bucket/"
DMP_ROOT = "dmpv2://"
TARGET_SIZE = (576, 704)
CLASS_MAP = {
    12: "parking_column",
    13: "traffic_bollard",
    14: "aframe_sign",
    9: "traffic_cone",
    15: "parking_lock_open",
    16: "parking_lock_close",
}
NAME2DATASETID = {
    "parking_column": (6036670, "parking_column", []),
    "traffic_bollard": (
        6037038,
        "traffic_bollard",
        [
            "traffic_cone",
            "aframe_sign",
            "TrafficCone",
            "OtherBollard",
            "AFrameSign",
            "CrashBarrel",
            "Tripod",
        ],
    ),
    "traffic_cone": (
        6037039,
        "traffic_cone",
        [
            "traffic_bollard",
            "aframe_sign",
            "TrafficBollard",
            "IsolationBollard",
            "OtherBollard",
            "AFrameSign",
            "CrashBarrel",
            "Tripod",
        ],
    ),
    "aframe_sign": (
        6037040,
        "aframe_sign",
        [
            "traffic_cone",
            "traffic_bollard",
            "TrafficCone",
            "OtherBollard",
            "IsolationBollard",
            "OtherBollard",
            "CrashBarrel",
            "Tripod",
        ],
    ),
    "parking_lock_open": (
        6036677,
        "parking_lock_open",
        ["parking_lock_close"],
    ),
    "parking_lock_close": (
        6036678,
        "parking_lock_close",
        ["parking_lock_open"],
    ),
}


def vis_same_source_dataset(stat_dict):
    from pylab import mpl

    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    try:
        from matplotlib_venn import venn2, venn3
    except ImportError:
        return

    sets = {}
    for key, val in stat_dict.items():
        for cls_name in NAME2DATASETID:
            if cls_name in key:
                if cls_name not in sets:
                    sets[cls_name] = set()
                for ii in range(val):
                    sets[cls_name].add("{}_{}".format(key, ii))

    if len(sets) == 3:
        venn3(
            list(sets.values()),
            list(sets.keys()),
            set_colors=("r", "g", "#1181A2"),
            alpha=0.4,
        )
    elif len(sets) == 2:
        venn2(list(sets.values()), list(sets.keys()), alpha=0.4)
    else:
        raise ValueError
    plt.title("common source data overlaps", fontsize=18)
    plt.savefig("tmp.png")


def vis_histogram(data, range, num_bins):
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.array(data)
    hist, bin_edges = np.histogram(data, range=range, bins=num_bins)
    x = [
        (bin_edges[i] + bin_edges[i + 1]) / 2
        for i in range(len(bin_edges) - 1)
    ]
    plt.bar(x, hist, width=bin_edges[1] - bin_edges[0])
    plt.savefig("tmp_vis_histogram.png")


def load_json_annotation(json_path):
    with open(json_path, "rt") as fr:
        lines = fr.readlines()

    annotations = {}
    length = len(lines)
    for ii, line in enumerate(lines):
        print(f"\rloading {ii}/{length} raw annotations...", end="")
        anno = json.loads(line)
        img_key = os.path.basename(anno["image_url"]).split(".")[0]
        annotations[img_key] = anno
    print("")

    return annotations


def generate_eval_set(
    data_dicts,
    save_dir,
    dclass_name,
    statistic_only=False,
    ignore_classes=None,
):
    """Get image path and anno json and pack to tar."""
    denseboxAnno = ConcatDataset(
        datasets=[
            LegacyDenseBoxImageRecordDataset(
                rec_path=data_dict["rec_path"],
                anno_path=data_dict["anno_path"],
            )
            for data_dict in data_dicts
        ]
    )
    total_types = set()
    raw_annotations = {}
    for data_dict in data_dicts:
        anno_path = data_dict["anno_path"]
        filename = os.path.basename(anno_path)
        raw_json_path = anno_path.replace(filename, "raw_data.json").replace(
            "train", "val"
        )
        if not os.path.exists(raw_json_path):
            print(f"{raw_json_path} not exist!")
            continue
        annotation = load_json_annotation(raw_json_path)
        raw_annotations.update(annotation)

    data_name = "data"
    print("combining json file...")
    print("and copying image to data directory...")
    dst_anno = []
    stat_dict = {}
    n_bboxes = 0
    n_hard_bboxes = 0
    img_with_no_easy = 0
    n_images_written = 0
    width_list = []
    save_image_dir = os.path.join(save_dir, f"{data_name}")
    os.makedirs(save_image_dir, exist_ok=True)
    data_length = len(denseboxAnno)
    for index in range(data_length):
        print("\rprocessing {}/{}".format(index + 1, data_length), end="")
        _, anno = denseboxAnno[index]
        anno = anno.to_dict()
        img_path = os.path.join(
            BUCKET_ROOT, anno["img_url"].split(BUCKET_ROOT)[1]
        )
        aidi_rec = {
            "image_key": os.path.basename(img_path),
            "image_source": "",  # img_path.replace(BUCKET_ROOT, DMP_ROOT),
            "video_name": "1",
            "video_index": "1",
            "width": TARGET_SIZE[1],
            "height": TARGET_SIZE[0],
            dclass_name: [],
        }
        ori_size = int(anno["img_h"]), int(anno["img_w"])
        raw_anno = raw_annotations.get(
            os.path.basename(img_path).split(".")[0], None
        )
        hs = TARGET_SIZE[0] / ori_size[0]
        ws = TARGET_SIZE[1] / ori_size[1]
        n_hard_this = 0
        n_bboxes_this = 0
        current_cls_set = set()
        has_pos = False
        for ins in anno["instances"]:
            class_name = CLASS_MAP[ins["class_id"][0]]
            current_cls_set.add(class_name)
            if class_name == dclass_name:
                has_pos = True
            current_len = len(aidi_rec[dclass_name])
            is_ignore = False
            if ins["is_hard"][0] or class_name in ignore_classes:
                is_ignore = True
            dst_ins = {
                "id": current_len + 1,
                "track_id": -1,
                "struct_type": "rect",  # 2D detection bbox signature
                "label_type": "boxes",  # 2D detection bbox signature
                "attrs": {
                    "ignore": "yes" if is_ignore else "no",
                    "type": class_name,
                },  # for data analysisz;
            }
            bbox = ins["points_data"][0] + ins["points_data"][2]
            if not is_ignore:
                width_list.append(bbox[2] - bbox[0])
            dst_ins["data"] = [
                bbox[0] * ws,
                bbox[1] * hs,
                bbox[2] * ws,
                bbox[3] * hs,
            ]
            n_bboxes += 1
            if ins["is_hard"][0]:
                n_hard_bboxes += 1
                n_hard_this += 1
            aidi_rec[dclass_name].append(dst_ins)
            n_bboxes_this += 1
        if not has_pos:
            continue
        if raw_anno is not None:
            ori_size = int(raw_anno["height"]), int(raw_anno["width"])
            rhs = TARGET_SIZE[0] / ori_size[0]
            rws = TARGET_SIZE[1] / ori_size[1]
            for box in raw_anno.get("common_box", []):
                if "attr" not in box:
                    continue
                atype = box["attrs"]["type"]
                total_types.add(atype)
                if atype not in ignore_classes:
                    continue
                dst_ins = {
                    "id": len(aidi_rec[dclass_name]) + 1,
                    "track_id": -1,
                    "struct_type": "rect",  # 2D detection bbox signature
                    "label_type": "boxes",  # 2D detection bbox signature
                    "attrs": {
                        "ignore": "yes",
                        "type": atype,
                    },  # for data analysisz;
                    "data": [
                        box["data"][0] * rws,
                        box["data"][1] * rhs,
                        box["data"][2] * rws,
                        box["data"][3] * rhs,
                    ],
                }
                aidi_rec[dclass_name].append(dst_ins)

        for ins in anno["ignore_regions"]:
            class_id = ins["class_id"][0]
            if (class_id not in CLASS_MAP) or (
                CLASS_MAP[class_id] != dclass_name
            ):
                continue
            class_name = CLASS_MAP[class_id]
            dst_ins = {
                "id": len(aidi_rec[dclass_name]) + 1,
                "track_id": -1,
                "struct_type": "rect",  # 2D detection bbox signature
                "label_type": "boxes",  # 2D detection bbox signature
                "attrs": {
                    "ignore": "yes",
                },  # for data analysisz;
                "data": [
                    ins["contour"][0][0] * ws,
                    ins["contour"][0][1] * hs,
                    ins["contour"][1][0] * ws,
                    ins["contour"][1][1] * hs,
                ],
            }
            aidi_rec[dclass_name].append(dst_ins)
            n_bboxes_this += 1
        if n_hard_this == len(anno["instances"]):
            img_with_no_easy += 1

        current_cls_list = list(current_cls_set)
        current_cls_str = "|".join(sorted(current_cls_list))
        if current_cls_str not in stat_dict:
            stat_dict[current_cls_str] = 0
        stat_dict[current_cls_str] += 1
        if n_bboxes_this == 0:
            continue

        dst_anno.append(json.dumps(aidi_rec) + "\n")
        img_name = os.path.basename(img_path)
        target_path = os.path.join(save_image_dir, img_name)

        n_images_written += 1
        if statistic_only:
            continue

        if not ori_size == TARGET_SIZE:
            src_img = cv2.imread(img_path)
            dst_img = cv2.resize(src_img, TARGET_SIZE[::-1])
            cv2.imwrite(target_path, dst_img)
        else:
            os.system(f"cp {img_path} {target_path}")
    print("")
    print(
        f"{n_bboxes} bbox labels has been written, "
        f"including {n_hard_bboxes} hard ones."
    )
    print(stat_dict)
    assert n_images_written == len(dst_anno)
    if statistic_only:
        print(f"{n_images_written}/{data_length} images for {dclass_name}")
        # vis_same_source_dataset(stat_dict)
        # vis_histogram(width_list, (0, 100), 10)
        return
    os.makedirs(save_dir, exist_ok=True)
    json_save_path = os.path.join(save_dir, f"{data_name}.json")
    with open(json_save_path, "wt", encoding="utf-8") as fw:
        fw.writelines(dst_anno)
    os.system(
        "cd {}/.. && tar cvf {}.tar {} > generate.log".format(
            save_dir,
            os.path.basename(save_dir),
            os.path.basename(save_dir),
        )
    )
    tar_path = f"{save_dir}.tar"
    print(
        f"dataset [{tar_path}] has been created."
        f"including {n_images_written} images."
    )
    return tar_path


def upload_eval_set(class_name, version, dataset_dir, setting_root):
    name = f"sp_sod_{class_name}_{version}"
    setting_path = os.path.join(setting_root, f"{class_name}.yaml")
    assert os.path.exists(dataset_dir), f"{dataset_dir} does not exists!"
    assert os.path.exists(setting_path), f"{setting_path} does not exists!"
    command = [
        f"python {AIDI_EVAL_SCRIPT}",
        f"--dataset-name {name}",
        f"--dataset-path {dataset_dir}",
        f"--setting-name {name}",
        f"--setting-path {setting_path}",
    ]
    print("=" * 100)
    print(f"start uploading dataset for {class_name} ...")
    result = subprocess.run(
        " ".join(command),
        shell=True,
    )
    print(result)
    print("=" * 100)


def update_eval_setting(class_name, dataset_id, setting_root):
    setting_path = os.path.join(setting_root, f"{class_name}.yaml")
    assert os.path.exists(setting_path), f"{setting_path} does not exists!"
    command = [
        f"python {AIDI_EVAL_SCRIPT}",
        f"--dataset-id {dataset_id}",
        f"--setting-path {setting_path}",
    ]
    print("=" * 100)
    print(
        f"start updating setting for dataset {dataset_id} for {class_name} ..."
    )
    result = subprocess.run(
        " ".join(command),
        shell=True,
    )
    print(result)
    print("=" * 100)


if __name__ == "__main__":
    split = "val"  # train or val
    version = "V1.8.0"
    statistic_only = False
    do_generate = True
    do_upload = True
    do_update_setting = False
    save_root = (
        f"/horizon-bucket/SuperParking/fangquan.hu/SOD/AIDI_EVAL/{version}"
    )
    setting_root = os.path.join(save_root, "settings")

    for obstacle, (
        dataset_id,
        dataset_name,
        ignore_classes,
    ) in NAME2DATASETID.items():
        save_dir = os.path.join(save_root, obstacle)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if do_generate:
            dataset_path = generate_eval_set(
                datapaths[dataset_name][f"{split}_data_paths"],
                save_dir,
                obstacle,
                statistic_only,
                ignore_classes,
            )
        else:
            dataset_path = f"{save_dir}.tar"
            assert os.path.exists(dataset_path)

        if do_upload and not statistic_only:
            upload_eval_set(obstacle, version, dataset_path, setting_root)

        if do_update_setting:
            update_eval_setting(obstacle, dataset_id, setting_root)
