# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torchvision

from hat.data.transforms.detection import ToTensor
from hat.registry import build_from_registry

transforms = torchvision.transforms.Compose([ToTensor()])


@pytest.mark.parametrize("transforms", [(transforms)])
def test_eye_status_dataset(transforms):

    data_cfg = dict(
        type="EyeStatusDataset",
        rec_paths=[
            "./tmp_orig_data/face/eye_status/train_02.rec",
            "./tmp_orig_data/face/eye_status/train_03.rec",
        ],
        transforms=transforms,
    )
    dataset = build_from_registry(data_cfg)

    for idx, batch in enumerate(dataset):
        assert batch["eye_status"].shape == (8,)
        assert batch["gt_ldmk"].shape == (16, 3)
        assert batch["gt_ldmk_attr"].shape == (16,)
        assert batch["gt_eye_cls_labels"].shape == (10,)

        if idx > 2:
            break


@pytest.mark.parametrize("transforms", [(transforms)])
def test_eye_vis_dataset(transforms):

    data_cfg = dict(
        type="EyeStatusDataset",
        rec_paths=[
            "./tmp_orig_data/face/eye-vis/train_04.rec",
            "./tmp_orig_data/face/eye-vis/train_07.rec",
        ],
        dataset_type="eye_vis",
        transforms=transforms,
    )
    dataset = build_from_registry(data_cfg)

    for idx, batch in enumerate(dataset):
        assert batch["eye_vis_labels"].shape == (5,)
        assert batch["gt_eye_cls_labels"].shape == (4,)

        if idx > 2:
            break
