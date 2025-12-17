import os
from typing import Dict, List

__all__ = ["get_dataset", "get_eval_id_dataset"]


if os.path.isdir("/horizon-bucket"):
    BUCKET_PREFIX = "/horizon-bucket"
elif os.path.isdir("/bucket/output"):
    BUCKET_PREFIX = "/bucket/output"
elif os.path.isdir("/bucket/input"):
    BUCKET_PREFIX = "/bucekt/input"


TRAIN_DATA_PREFIX = f"{BUCKET_PREFIX}/mono"
VAL_DATASET_PREFIX = f"{BUCKET_PREFIX}/auto_eval/adas_eval/eval_platform/fs/"


TrainDataset = {
    "cyc_kps_20190802_new": {
        "train_rec": f"{TRAIN_DATA_PREFIX}/data/lele.liu/kps/cyc_kps/20190802_new/data.train.pb_rec",  # noqa
        "train_json": f"{TRAIN_DATA_PREFIX}/data/lele.liu/kps/cyc_kps/20190802_new/label.train.pb_rec",  # noqa
    },
    "cyc_kps_mini": {
        "train_rec": f"{TRAIN_DATA_PREFIX}/data/lele.liu/kps/cyc_kps/mini/data.train.pb_rec",  # noqa
        "train_json": f"{TRAIN_DATA_PREFIX}/data/lele.liu/kps/cyc_kps/mini/label.train.pb_rec",  # noqa
    },
    "cyclist_2_kps_390_20201207": {
        "train_rec": f"{TRAIN_DATA_PREFIX}/data/lele.liu/kps/cyc_kps/cyclist_2_kps_390_20201207/data.train.pb_rec",  # noqa
        "train_json": f"{TRAIN_DATA_PREFIX}/data/lele.liu/kps/cyc_kps/cyclist_2_kps_390_20201207/label.train.pb_rec",  # noqa
    },
    "cyclist_keypoints_0820": {
        "train_rec": f"{TRAIN_DATA_PREFIX}/data/ca-qi.zhang/auto_data/01_cyclist_keypoints/train/0820/2022-04-13/train.pb_rec",  # noqa
        "train_json": f"{TRAIN_DATA_PREFIX}/data/ca-qi.zhang/auto_data/01_cyclist_keypoints/train/0820/2022-04-13/train.anno.pb_rec",  # noqa
    },
    "cyclist_keypoints_10652": {
        "train_rec": f"{TRAIN_DATA_PREFIX}/data/ca-qi.zhang/auto_data/01_cyclist_keypoints/train/10652/2022-04-13/train.pb_rec",  # noqa
        "train_json": f"{TRAIN_DATA_PREFIX}/data/ca-qi.zhang/auto_data/01_cyclist_keypoints/train/10652/2022-04-13/train.anno.pb_rec",  # noqa
    },
    "cyclist_keypoints_X8B": {
        "train_rec": f"{TRAIN_DATA_PREFIX}/data/ca-qi.zhang/auto_data/01_cyclist_keypoints/train/X8B/2022-05-09/2022-05-09/train.pb_rec",  # noqa
        "train_json": f"{TRAIN_DATA_PREFIX}/data/ca-qi.zhang/auto_data/01_cyclist_keypoints/train/X8B/2022-05-09/2022-05-09/train.anno.pb_rec",  # noqa
    },
    "cyc_kps_220_323": {
        "train_rec": f"{TRAIN_DATA_PREFIX}/data/lele.liu/kps/cyc_kps/220_323/train.pb_rec",  # noqa
        "train_json": f"{TRAIN_DATA_PREFIX}/data/lele.liu/kps/cyc_kps/220_323/train.anno.pb_rec",  # noqa
    },
    "mini": {
        "train_rec": "/jfs-public/smartauto/users/yisu.zhou/cyckps/horizon_mini_rec/data.train.pb.rec",  # noqa
        "train_json": "/jfs-public/smartauto/users/yisu.zhou/cyckps/horizon_mini_rec/label.train.pb_rec",  # noqa
    },
}


ValDataset = {
    "6027226": f"{VAL_DATASET_PREFIX}/6027226/datasets/10652_normal/dataset",
    "6029054": f"{VAL_DATASET_PREFIX}/6029054/datasets/X8B/2022-04-24",
    "mini": "/jfs-public/smartauto/users/yisu.zhou/cyckps/mini_val_data/tmp_mini_val_img",  # noqa
}


def get_dataset(name: List[str]) -> Dict:
    datasets = [TrainDataset[key] for key in name]
    results = {}
    results["img_rec"] = [d["train_rec"] for d in datasets]
    results["label_rec"] = [d["train_json"] for d in datasets]
    return results


def get_eval_id_dataset(name: List[str]) -> Dict:
    valimg = [ValDataset[eval_id] for eval_id in name]
    return valimg
