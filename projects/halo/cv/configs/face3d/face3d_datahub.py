import os
from typing import Dict, List

__all__ = ["get_dataset"]


if os.path.isdir("/horizon-bucket"):
    PREFIX = "/horizon-bucket"
elif os.path.isdir("/bucket/output"):
    PREFIX = "/bucket/output"
elif os.path.isdir("/bucket/input"):
    PREFIX = "/bucekt/input"


Dataset = {
    # Data captured by azure kinect. 471ID.
    "kinect471": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/lmdb/azure_kinect_depth_fitting_471ID_20230214",  # noqa
    },
    # Data captured by azure kinect. 120ID.
    "kinect120": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/lmdb/azure_kinect_depth_fitting_120ID_20230214",  # noqa
    },
    "workshop_10270_kinect_train": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/workshop_train_pts_c",  # noqa
    },
    "workshop_10270_kinect_val_cam09": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/workshop_val_pts_c_cam09",  # noqa
    },
    "workshop_10270_kinect_2dw3d_train": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/workshop_10270_2dw3d_train",  # noqa
    },
    "workshop_10293_kinect_2dw3d_train": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/workshop_10293_2dw3d_train",  # noqa
    },
    "workshop_10270_kinect_2dw3d_val": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/workshop_10270_2dw3d_val",  # noqa
    },
    "workshop_10293_kinect_2dw3d_val": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/workshop_10293_2dw3d_val",  # noqa
    },
    "workshop_10270_dms_train_crop640": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/dms_10270_train_crop640",  # noqa
    },
    "workshop_10293_dms_train_crop640": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/dms_10293_train_crop640",  # noqa
    },
    "workshop_10307_dms_train_crop640": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/dms_10307_train_crop640",  # noqa
    },
    "cd569_10308_ov2311_val_crop640": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/dms_incar_10308_val_crop640",  # noqa
    },
    "workshop_10270_dms_val_crop640": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/dms_10270_val_crop640",  # noqa
    },
    "workshop_10293_dms_val_crop640": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/dms_10293_val_crop640",  # noqa
    },
    "workshop_10307_dms_val_crop640": {
        "prelabel": "",
        "json_path": "",
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/workshop/lmdb/dms_10307_val_crop640",  # noqa
    },
    "cd569_10308_ov2311_train_toy": {
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/realcar/lmdb/cd569_10308_ov2311_train_toy"  # noqa
    },
    "cd569_10308_ov2311_val_toy": {
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/realcar/lmdb/cd569_10308_ov2311_val_toy"  # noqa
    },
    "cd569_10308_ov2311_val_stereo_cam0": {
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/realcar/lmdb/cd569_10308_ov2311_val_stereo_cam0"  # noqa
    },
    "cd569_10308_ov2311_val_fitting_cam0": {
        "lmdb_path": f"{PREFIX}/interaction/active/face3d/data/realcar/lmdb/cd569_10308_ov2311_val_fitting_cam0"  # noqa
    },
}


def get_dataset(name: List[str]) -> Dict:
    datasets = [Dataset[key] for key in name]
    results = {}
    results["anno_list"] = [
        os.path.join(d["lmdb_path"], "anno_lmdb") for d in datasets
    ]
    results["image_list"] = [
        os.path.join(d["lmdb_path"], "image_lmdb") for d in datasets
    ]
    results["mask_list"] = [
        os.path.join(d["lmdb_path"], "mask_lmdb") for d in datasets
    ]
    return results
