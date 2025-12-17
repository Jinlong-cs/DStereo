import torch
import torchvision
from torch.utils.data import DataLoader

from hat.data.datasets.roidb_detection_dataset import RoidbDetectionDataset
from hat.data.transforms.detection import (
    RandomFlip,
    Resize,
    ToLdmkRCNNData,
    ToTensor,
)
from hat.data.transforms.landmark import (
    ClipBoxes,
    GenerateGRMITarget,
    RandomShiftRotateScale,
)

REC_PATH = "./tmp_orig_data/landmark/hat_test/human_keypoints_test/changan_202003_RGB.rec"  # noqa
ROIDB_PATH = "./tmp_orig_data/landmark/hat_test/human_keypoints_test/changan_202003_RGB_kps_person_head_face_hand.pkl"  # noqa


def kps_collate(batch):
    images = []

    gt_ldmks = []
    gt_bboxes = []
    gt_classes = []
    gt_heatmap = []
    gt_heatmap_weight = []
    gt_offset = []
    gt_offset_weight = []

    for b in batch:
        images.append(b["img"])
        gt_ldmks.append(b["gt_ldmk"])
        gt_bboxes.append(b["gt_bboxes"])
        gt_classes.append(b["gt_classes"])
    images = torch.stack(images, 0)
    new_batch = {
        "img": images,
        "img_name": batch[0]["img_name"],
        "img_height": batch[0]["img_height"],
        "img_width": batch[0]["img_width"],
        "color_space": batch[0]["color_space"],
        "layout": batch[0]["layout"],
        "gt_ldmk": gt_ldmks,
        "gt_bboxes": gt_bboxes,
        "gt_classes": gt_classes,
        "img_shape": batch[0]["img_shape"],
    }

    if "gt_heatmap" in batch[0]:
        for b in batch:
            gt_heatmap.append(b["gt_heatmap"])
            gt_heatmap_weight.append(b["gt_heatmap_weight"])
        new_batch["gt_heatmap"] = gt_heatmap
        new_batch["gt_heatmap_weight"] = gt_heatmap_weight
    if "gt_offset" in batch[0]:
        for b in batch:
            gt_offset.append(b["gt_offset"])
            gt_offset_weight.append(b["gt_offset_weight"])
        new_batch["gt_offset"] = gt_offset
        new_batch["gt_offset_weight"] = gt_offset_weight

    return new_batch


def test_rotate():
    dataset = RoidbDetectionDataset(
        selected_class_ids=[1],
        data_path=REC_PATH,
        anno_path=ROIDB_PATH,
        transforms=torchvision.transforms.Compose(
            [
                RandomShiftRotateScale(
                    1.0,
                    0,
                    img_scale=True,
                    scale_range=(1.1, 1.0),
                    out_shape=(640, 380),
                    shift_prob=1.0,
                    max_shift_range=(0.2, 0.3),
                ),
                ToTensor(),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=kps_collate,
    )

    for idx, batch in enumerate(dataloader):
        assert "img" in batch
        assert "gt_ldmk" in batch
        assert "gt_bboxes" in batch
        assert "gt_classes" in batch
        img = batch["img"]
        assert img.size()[0] == 1
        if idx > 3:
            break


def test_resize():
    dataset = RoidbDetectionDataset(
        selected_class_ids=[1],
        data_path=REC_PATH,
        anno_path=ROIDB_PATH,
        transforms=torchvision.transforms.Compose(
            [
                Resize(
                    img_scale=tuple(
                        [(352 + i * 32, 640 + i * 32) for i in range(-3, 4)]
                    ),
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                ToTensor(),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=kps_collate,
    )

    for idx, batch in enumerate(dataloader):
        assert "img" in batch
        assert "gt_ldmk" in batch
        assert "gt_bboxes" in batch
        assert "gt_classes" in batch
        img = batch["img"]
        assert img.size()[0] == 1
        if idx > 3:
            break


def test_flip():
    dataset = RoidbDetectionDataset(
        selected_class_ids=[1],
        data_path=REC_PATH,
        anno_path=ROIDB_PATH,
        transforms=torchvision.transforms.Compose(
            [
                RandomFlip(px=1.0),
                ToTensor(),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=kps_collate,
    )

    for idx, batch in enumerate(dataloader):
        assert "img" in batch
        assert "gt_ldmk" in batch
        assert "gt_bboxes" in batch
        assert "gt_classes" in batch
        img = batch["img"]
        assert img.size()[0] == 1
        if idx > 2:
            break


def test_heatmap():
    dataset = RoidbDetectionDataset(
        selected_class_ids=[1],
        data_path=REC_PATH,
        anno_path=ROIDB_PATH,
        transforms=torchvision.transforms.Compose(
            [
                GenerateGRMITarget(
                    15, (16, 16), ldmk_loss_type=["pixel", "cross_entropy"]
                ),
                ToTensor(),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=kps_collate,
    )

    for idx, batch in enumerate(dataloader):
        assert "img" in batch
        assert "gt_ldmk" in batch
        assert "gt_bboxes" in batch
        assert "gt_classes" in batch
        img = batch["img"]
        assert img.size()[0] == 1
        if idx > 3:
            break


def test_cvt():
    dataset = RoidbDetectionDataset(
        selected_class_ids=[1],
        data_path=REC_PATH,
        anno_path=ROIDB_PATH,
        transforms=torchvision.transforms.Compose(
            [
                ToLdmkRCNNData(),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    for idx, batch in enumerate(dataloader):
        img = batch["img"]
        batch["gt_boxes"]
        assert img.size()[0] == 1
        if idx > 3:
            break


def test_clip():
    dataset = RoidbDetectionDataset(
        selected_class_ids=[1],
        data_path=REC_PATH,
        anno_path=ROIDB_PATH,
        transforms=torchvision.transforms.Compose(
            [
                ClipBoxes(),
                ToTensor(),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=kps_collate,
    )

    for idx, batch in enumerate(dataloader):
        assert "img" in batch
        assert "gt_ldmk" in batch
        assert "gt_bboxes" in batch
        assert "gt_classes" in batch
        img = batch["img"]
        assert img.size()[0] == 1
        if idx > 3:
            break
