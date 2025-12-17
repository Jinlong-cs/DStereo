import os

import pytest
import torch
import yaml

from hat.data.datasets.densebox2detectron2 import (
    build_dataset,
    build_detectron2_data_loader,
)
from tests import AIDI_PUBLIC_DATA_EXISTS, AIDI_PUBLIC_DATA_PATH

try:
    from detectron2.config.defaults import _C
    from detectron2.data.common import AspectRatioGroupedDataset

    _DETECTRON2_IMPORTED = True
except ImportError:
    _DETECTRON2_IMPORTED = False

ADASMINI_ROOT = os.path.join(AIDI_PUBLIC_DATA_PATH, "adasmini")

data_path = {
    "test_rec_file": "vehicle_rear/detection/720_v1/rec/val.rec",  # noqa
    "test_json_file": "vehicle_rear/detection/720_v1/rec/val.json",  # noqa
    "train_rec_file": "vehicle_rear/detection/720_v1/rec/train.rec",  # noqa
    "train_json_file": "vehicle_rear/detection/720_v1/rec/train.json",  # noqa
}
classes = "vehicle_rear"

# create densebox dataset
dataset_cfg = dict(
    vehicle_rear=dict(
        common=dict(
            task_type="detection",
            class_id=5,
            category=0,
            to_rgb=False,
        ),
        train=dict(
            data_path=os.path.join(ADASMINI_ROOT, data_path["train_rec_file"]),
            anno_path=os.path.join(
                ADASMINI_ROOT, data_path["train_json_file"]
            ),
        ),
        test=dict(
            data_path=os.path.join(ADASMINI_ROOT, data_path["test_rec_file"]),
            anno_path=os.path.join(ADASMINI_ROOT, data_path["test_json_file"]),
        ),
    )
)


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("test"),
    ],
)
@pytest.mark.skipif(
    not AIDI_PUBLIC_DATA_EXISTS, reason="adas-mini is required"
)
@pytest.mark.skipif(not _DETECTRON2_IMPORTED, reason="detectron2 is required")
def test_build_dataset(tmpdir, mode):

    # create densebox dataset yaml
    dataset_yaml = os.path.join(tmpdir, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        yaml.dump(dataset_cfg, f)
    data_cfg = {
        "rec_dataset_path_file": dataset_yaml,
        "task_name": "vehicle_rear",
    }
    dataset = build_dataset(data_cfg, mode=mode)
    if mode == "train":
        assert len(dataset) == 50000
    else:
        assert len(dataset) == 10000


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("test"),
    ],
)
@pytest.mark.skipif(
    not AIDI_PUBLIC_DATA_EXISTS, reason="adas-mini is required"
)
@pytest.mark.skipif(not _DETECTRON2_IMPORTED, reason="detectron2 is required")
def test_build_detectron2_data_loader(tmpdir, mode):

    # create densebox dataset yaml
    dataset_yaml = os.path.join(tmpdir, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        yaml.dump(dataset_cfg, f)

    #  detectron2 cfg for test
    detectron2_cfg = {
        "DATASETS": {
            "TRAIN": (
                {
                    "rec_dataset_path_file": dataset_yaml,
                    "task_name": "vehicle_rear",
                },
            ),
            "TEST": (
                {
                    "rec_dataset_path_file": dataset_yaml,
                    "task_name": "vehicle_rear",
                },
            ),
        }
    }
    detectron2_yaml = os.path.join(tmpdir, "test.yaml")
    with open(detectron2_yaml, "w") as f:
        yaml.dump(detectron2_cfg, f)

    # detectron CfgNode config
    cfg = _C.clone()
    cfg.merge_from_file(detectron2_yaml)

    dataloader = build_detectron2_data_loader(cfg, mode=mode)
    if mode == "train":
        assert isinstance(dataloader, AspectRatioGroupedDataset)
    else:
        assert isinstance(dataloader, torch.utils.data.dataloader.DataLoader)
