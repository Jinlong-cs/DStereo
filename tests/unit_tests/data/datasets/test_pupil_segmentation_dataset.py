import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    "data_type, only_one_channel",
    [("train", False), ("predict", True)],
)
def test_facequality_dataset(data_type, only_one_channel):
    train_transforms = [
        dict(
            type="CropRecROI",
            crop_type="center",
            target_shape=(64, 64, 3),
            base_roi=[32, 32, 96, 96],
            crop_jitter_range=0,
        ),
        dict(type="RandomFlip", px=1, py=0),
        dict(
            type="GaussianBlur",
            p=1,
            kernel_size_min=2,
            kernel_size_max=5,
            sigma_min=2,
            sigma_max=7,
        ),
        dict(
            type="GaussianNoise",
            prob=1,
            mean=0,
            sigma=2,
        ),
        dict(
            type="RandomShiftRotateScale",
            rotate_prob=1,
            max_rotate_angle=30,
            resize=True,
            border_value=4,
        ),
        dict(type="GenerateEllipseMask"),
        dict(type="GenerateEdgeWeightMap"),
        dict(type="GenerateDistMap"),
        dict(type="NormEllipseParam"),
        dict(
            type="RandomGray",
            p=1,
            rgb_data=True,
            only_one_channel=only_one_channel,
        ),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=128.0, std=128.0),
    ]
    test_transforms = [
        dict(
            type="RandomGray",
            p=1,
            rgb_data=True,
            only_one_channel=only_one_channel,
        ),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=128.0, std=128.0),
    ]

    if data_type == "predict":
        dataset = build_from_registry(
            dict(
                type="PupilSegDataset",
                image_path="./tmp_orig_data/face/gaze/pupil_segmentation/lmdb/image_lmdb",  # noqa
                data_type=data_type,
                transforms=test_transforms,
            )
        )
    else:
        dataset = build_from_registry(
            dict(
                type="PupilSegDataset",
                image_path="./tmp_orig_data/face/gaze/pupil_segmentation/lmdb/image_lmdb",  # noqa
                anno_path="./tmp_orig_data/face/gaze/pupil_segmentation/lmdb/anno_lmdb",  # noqa
                data_type=data_type,
                transforms=train_transforms,
            )
        )
    for idx, batch in enumerate(dataset):
        assert isinstance(batch["img"], torch.Tensor)
        assert (~(batch["img"] > -1) & (batch["img"] < 1)).to(
            torch.int
        ).sum() == 0
        if only_one_channel:
            assert batch["img"].shape[0] == 1
        else:
            assert batch["img"].shape[0] == 3
        _, height, width = batch["img"].shape
        if data_type == "train":
            assert batch["gt_pupil_mask"].shape == torch.Size([height, width])
            assert batch["spat_weights"].shape == torch.Size([height, width])
            assert batch["dist_map"].shape == torch.Size([height, width])
            assert batch["gt_pupil_center"].shape == torch.Size([2])
            assert batch["gt_norm_pupil_ellipse_param"].shape == torch.Size(
                [5]
            )

        if idx > 2:
            break
