# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest

from hat.data.datasets.eval_2pe_dataset import MtlEvalRaw2PEDataset

AIDI_EVAL_BUCKET_PATH = "/horizon-bucket/auto_eval"
local_datapath = os.path.join(
    AIDI_EVAL_BUCKET_PATH, "adas_eval/eval_platform/fs/"
)
PATH_EXIST = os.path.exists(local_datapath)

val_transforms = [
    dict(
        type="RoiTransformer",
        roi_crop_parm=dict(
            norm_len=86,
            norm_method="max_width_height",
            output_wh=[96, 96],
            input_wh=None,
            min_crop_scale=1.0,
            max_crop_scale=1.0,
            max_coord_jitter_ratio=0.0,
            img_min_scale=0.01,
            img_max_scale=100,
            padd_val=0.0,
            random_roi_ratio=0.0,
            restrict_roi_in_center=False,
            flip_ratio=0,
        ),
        img_crop_parm=dict(
            target_wh=[96, 96],
            inter_method=2,
            use_pyramid=True,
            pyramid_min_step=0.8,
            pyramid_max_step=0.8,
            pixel_center_aligned=False,
        ),
        bbox_ts_parm=dict(
            clip=False,
            min_valid_area=10,
            min_valid_clip_area_ratio=0,
            min_edge_size=1,
            label_type="classification",
        ),
        convert_roi=True,
    ),
]


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
@pytest.mark.parametrize(
    ["task_type", "dataset_id", "transforms"],
    [
        pytest.param("classification", "6042272", None),
        pytest.param("detection", "6042410", None),
        pytest.param("detection", "6042410", None),
    ],
)
def test_tl_eval_from_raw_dataset(task_type, dataset_id, transforms):
    data_path = os.path.join(local_datapath, str(dataset_id), "datasets")
    dataset = MtlEvalRaw2PEDataset(
        data_path=data_path,
        task_type=task_type,
        to_rgb=False,
        buf_only=True,
        return_orig_img=False,
        transforms=transforms,
    )
    for index in range(len(dataset)):
        data = dataset[index]
        if len(data) > 0:
            assert "img" in data[0].keys()
            if task_type == "detection" and transforms is not None:
                assert "inverse_affine_aug_param" in data[0].keys()
        if index > 10:
            break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
