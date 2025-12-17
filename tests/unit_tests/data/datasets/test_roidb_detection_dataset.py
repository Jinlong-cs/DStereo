import pytest
import torchvision

from hat.data.datasets.roidb_detection_dataset import (
    RoidbDataset,
    RoidbDetectionDataset,
)
from hat.data.transforms.detection import Resize

height, width = 352, 640
img_scale = (height, width)
transforms_1 = torchvision.transforms.Compose(
    [
        Resize(
            img_scale=img_scale,
            keep_ratio=False,
        ),
    ]
)

transforms_2 = torchvision.transforms.Compose(
    [
        Resize(
            img_scale=img_scale,
            keep_ratio=True,
        ),
    ]
)


def test_roidb_dataset():
    dataset = RoidbDataset(
        roidb_path="./tmp_data/halo_detection/CD569_IR_hand_background_wi_dolls_badcases_baseline/CD569_IR_hand_background_wi_dolls_badcases_roidb_baseline.pkl",  # noqa: E501
        rec_path="./tmp_data/halo_detection/CD569_IR_hand_background_wi_dolls_badcases_baseline/CD569_IR_hand_background_wi_dolls_badcases.rec",  # noqa: E501
        data_desc="CD569_IR_hand_background_wi_dolls_badcases_baseline",  # noqa: E501
    )

    for ind, data in enumerate(dataset):
        img, anno = data
        assert "image_index" in anno.keys()
        assert (
            "data_desc" in anno.keys()
            and anno["data_desc"]
            == "CD569_IR_hand_background_wi_dolls_badcases_baseline"
        )
        H, W = img.shape[0:2]
        if (W / H) > 1:
            assert dataset.flag[ind] == 1
        else:
            assert dataset.flag[ind] == 0
        if ind > 10:
            break


@pytest.mark.parametrize("transforms", [transforms_1, transforms_2])
@pytest.mark.parametrize("ignore_spec_cls_ids", [None, [7, 8]])
def test_roidb_detection_dataset(transforms, ignore_spec_cls_ids):
    num_point_per_person = 15
    dataset = RoidbDetectionDataset(
        data_path="./tmp_data/halo_detection/CD569_IR_hand_background_wi_dolls_badcases_baseline/CD569_IR_hand_background_wi_dolls_badcases.rec",  # noqa: E501
        anno_path="./tmp_data/halo_detection/CD569_IR_hand_background_wi_dolls_badcases_baseline/CD569_IR_hand_background_wi_dolls_badcases_roidb_baseline.pkl",  # noqa: E501
        data_desc="CD569_IR_hand_background_wi_dolls_badcases_baseline",  # noqa: E501
        selected_class_ids=[1],
        num_point_per_person=num_point_per_person,
        ldmk_pairs=[
            [0, 3],
            [1, 4],
            [2, 5],
            [6, 10],
            [7, 11],
            [8, 12],
            [9, 13],
        ],
        ignore_spec_cls_ids=ignore_spec_cls_ids,
        transforms=transforms,
        to_rgb=True,
    )

    for ind, data in enumerate(dataset):
        assert (
            "data_desc" in data.keys()
            and data["data_desc"]
            == "CD569_IR_hand_background_wi_dolls_badcases_baseline"
        )
        gt_bboxes, gt_classes = (
            data["gt_bboxes"],
            data["gt_classes"],
        )
        assert gt_bboxes.shape[0] == gt_classes.shape[0]

        img = data["img"]
        H, W = data["img_shape"][0:2]
        assert H == img.shape[0] and W == img.shape[1]
        assert data["gt_ldmk"].shape[1] == num_point_per_person
        assert data["gt_ldmk"].shape[2] == 3

        if ind > 10:
            break
