import os

import pytest

from tests import AIDI_PUBLIC_DATA_PATH

try:
    import mmcv
    import mmdet
    from mmcv.utils import ConfigDict
    from mmdet.datasets import build_dataset

    _MMDET_IMPORTED = True
except ImportError:
    _MMDET_IMPORTED = False

ADASMINI_ROOT = os.path.join(AIDI_PUBLIC_DATA_PATH, "adasmini")
ADASMINI_AVAILABLE = os.path.exists(ADASMINI_ROOT)

data_path = {
    "val_rec_file": "vehicle_rear/detection/720_v1/rec/val.rec",  # noqa
    "val_json_file": "vehicle_rear/detection/720_v1/rec/val.json",  # noqa
    "train_rec_file": "vehicle_rear/detection/720_v1/rec/train.rec",  # noqa
    "train_json_file": "vehicle_rear/detection/720_v1/rec/train.json",  # noqa
}
classes = [
    "vehicle_rear",
]

train_data = dict(
    type="DenseboxDataset",
    data_path=os.path.join(ADASMINI_ROOT, data_path["train_rec_file"]),
    anno_path=os.path.join(ADASMINI_ROOT, data_path["train_json_file"]),
    task_type="detection",
    class_id=5,
    category=0,
    to_rgb=True,
)
val_data = dict(
    type="DenseboxDataset",
    data_path=os.path.join(ADASMINI_ROOT, data_path["val_rec_file"]),
    anno_path=os.path.join(ADASMINI_ROOT, data_path["val_json_file"]),
    task_type="detection",
    class_id=5,
    category=0,
    to_rgb=True,
)

cfg = dict(
    custom_imports=dict(imports=["hat.data.datasets.densebox2mmdet"]),
    data=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        val=dict(
            type="DenseboxDataset2MMDet",
            dataset=val_data,
            classes=classes,
            pipeline=[
                dict(type="LoadAnnotations", with_bbox=True),
                dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
                dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
            ],
        ),
    ),
)

cfg_multi_data = dict(
    custom_imports=dict(imports=["hat.data.datasets.densebox2mmdet"]),
    data=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # for test
        val=dict(
            type="DenseboxDataset2MMDet",
            dataset=[train_data, val_data],
            classes=classes,
            pipeline=[
                dict(type="LoadAnnotations", with_bbox=True),
                dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
                dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
            ],
        ),
    ),
)


@pytest.mark.parametrize(
    ["cfg", "dataset_num"],
    [
        pytest.param(cfg, 1),
        pytest.param(cfg_multi_data, 2),
    ],
)
@pytest.mark.skipif(not ADASMINI_AVAILABLE, reason="adas-mini is required")
@pytest.mark.skipif(not _MMDET_IMPORTED, reason="mmdet and mmcv is required")
def test_densebox2mmdet(cfg, dataset_num):

    assert mmcv.__version__
    assert mmdet.__version__

    cfg_mmdet = ConfigDict(cfg)
    # import
    mmcv.utils.import_modules_from_strings(cfg_mmdet.custom_imports.imports)

    densebox_data = build_dataset(cfg_mmdet.data.val)
    assert densebox_data.dataset
    if dataset_num == 1:
        assert len(densebox_data.dataset) == 10000
    elif dataset_num == 2:
        assert len(densebox_data.dataset) == 60000
