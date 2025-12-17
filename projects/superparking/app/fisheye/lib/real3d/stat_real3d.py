import os
from multiprocessing import Pool

import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from hat.registry import build_from_registry
from projects.superparking.app.fisheye.lib.utils import (
    get_bev_real3d_lmdb,
    get_data_path,
)

config_file_root = "projects/superparking/app/fisheye"
config_file_root = os.path.abspath(config_file_root)
dataset_root = os.path.join(config_file_root, "datasets/real3d")
bucket_root = "/horizon-bucket"
camera_type = "Cylindrical"


def get_rec_dataset(object_type, data_version, split):
    base_data_version = yaml.load(
        open(os.path.join(dataset_root, "data_version.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    data_info = base_data_version[object_type][data_version][split]
    data_version_info = data_info["data_version"]

    data_paths = get_data_path(
        bucket_root,
        f"{dataset_root}/{camera_type}/{object_type}_filtered.yaml",
    )
    rec_paths = []
    for i in data_version_info:
        if data_paths[object_type][i][split]["rec"]:
            rec_paths.extend(data_paths[object_type][i][split]["rec"])
    rec_dataset = dict(
        type="Real3DDatasetRec",
        paths=rec_paths,
        num_classes=1,
        transforms=None,
        num_dist=8,
        to_rgb=True,
        select_sample=False,
    )
    return (
        build_from_registry(rec_dataset),
        f"{object_type}_{split}_depth_yaw_hist_rec_{data_version}",
    )


def get_lmdb_dataset(object_type, data_version, split):
    version_yaml = os.path.join(
        dataset_root, "4DGT", f"{object_type}_{split}_version.yaml"
    )
    spec_yaml = os.path.join(
        dataset_root, "4DGT", f"{object_type}_dataset.yaml"
    )
    rec_dataset = get_bev_real3d_lmdb(
        version_yaml=version_yaml,
        spec_yaml=spec_yaml,
        version=data_version,
        num_dist=8,
        bucket_root="/horizon-bucket",
    )
    return (
        build_from_registry(rec_dataset),
        f"{object_type}_{split}_depth_yaw_hist_lmdb_{data_version}",
    )


def _stat_real3d_rec_depth_roty(data):
    objects = data["annotations"]
    yaws, depths = [], []
    for obj in objects:
        if obj["ignore"]:
            continue
        if "in_camera" in obj:
            obj.update(obj.pop("in_camera"))
        if "depth" not in obj:
            obj["depth"] = np.linalg.norm(obj["location"][[0, 2]])
        yaws.append(np.rad2deg(obj["rotation_y"]))
        depths.append(obj["depth"])
    return yaws, depths


def _plot_stat_rec(depth_list, yaw_list, save_name):
    plt.figure()
    axes = plt.subplot(2, 1, 1)
    num_bins = 10
    hrange = (0, 30)
    axes.hist(depth_list, bins=num_bins, range=hrange)
    xticks = np.arange(hrange[0], hrange[1], 2)
    axes.set_xlim(hrange)
    axes.set_xticks(xticks)

    axes = plt.subplot(2, 1, 2)
    num_bins = 50
    hrange = (-180, 180)
    axes.hist(yaw_list, bins=num_bins, range=hrange)
    xticks = np.arange(hrange[0], hrange[1], 30)
    axes.set_xlim(hrange)
    axes.set_xticks(xticks)
    plt.savefig(f"{save_name}.png")


def stat_real3d_rec_depth_roty():
    num_workers = 0

    yaw_list = []
    depth_list = []

    # dataset, save_name = get_rec_dataset(object_type="vehicle", data_version="v1.1.6", split="val")
    dataset, save_name = get_lmdb_dataset(
        object_type="vehicle", data_version="4DGT_SD", split="val"
    )

    if num_workers > 0:
        results = []
        with Pool(processes=num_workers) as pool:
            for i in tqdm(range(len(dataset))):
                r = pool.apply_async(
                    _stat_real3d_rec_depth_roty, args=(dataset, i)
                )
                results.append(r)
        pool.join()
        for r in results:
            yaws, depths = r.get()
            depth_list.extend(depths)
            yaw_list.extend(yaws)
    else:
        for i_data in tqdm(range(len(dataset))):
            data = dataset[i_data]
            yaws, depths = _stat_real3d_rec_depth_roty(data)
            depth_list.extend(depths)
            yaw_list.extend(yaws)

    # use plt.subplot to draw multi-plot histgram in one figure and save it
    _plot_stat_rec(depth_list, yaw_list, save_name)


if __name__ == "__main__":
    stat_real3d_rec_depth_roty()
