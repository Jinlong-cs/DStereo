"""dataset mappings, include all the preprocess of images and metrics"""
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import InterpolationMode

from hat.core.adapter import TorchVisionAdapter
from hat.data.datasets.cityscapes import Cityscapes
from hat.data.datasets.imagenet import ImageNetFromImage
from hat.data.datasets.kitti3d import Kitti3D
from hat.data.datasets.mscoco import CocoFromImage
from hat.data.datasets.voc import VOCFromImage
from hat.data.transforms.classification import BgrToYuv444
from hat.data.transforms.common import PILToTensor, TensorToNumpy
from hat.data.transforms.detection import Normalize, Pad, Resize, ToTensor
from hat.data.transforms.lidar_utils.lidar_transform_3d import LidarReformat
from hat.data.transforms.segmentation import LabelRemap, SegOneHot, SegResize
from hat.metrics.acc import Accuracy
from hat.metrics.coco_detection import COCODetectionMetric
from hat.metrics.kitti3d_detection import Kitti3DMetricDet
from hat.metrics.mean_iou import MeanIOU
from hat.metrics.voc_detection import VOC07MApMetric
from hat.visualize.cls import ClsViz
from hat.visualize.det import DetViz
from hat.visualize.lidar_det import lidar_det_visualize
from hat.visualize.seg import SegViz


def nas_imagenet_mappings(cfg):
    dataset = ImageNetFromImage(
        root="./tmp_orig_data/imagenet/val",
        split="val",
        transforms=torchvision.transforms.Compose(
            [
                TorchVisionAdapter(
                    interface="Resize",
                    size=cfg.resize_shape,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                TorchVisionAdapter(
                    interface="CenterCrop", size=cfg.data_shape
                ),
                TorchVisionAdapter(interface="ToTensor"),
                TorchVisionAdapter(
                    interface="Normalize", mean=0.0, std=1 / 225.0
                ),
                BgrToYuv444(rgb_input=True),
                TorchVisionAdapter(
                    interface="Normalize",
                    mean=128.0,
                    std=128.0,
                ),
            ]
        ),
    )
    metrics = [Accuracy()]

    def metric_updater(metrics, batch, model_outs):
        target = batch["labels"]
        preds, losses = model_outs
        for metric in metrics:
            metric.update(target, preds)

    def viz_updater(img, model_outs):
        preds = model_outs
        preds = ClsViz(is_plot=cfg.is_plot)(img, preds)
        category = dataset.classes[int(preds)]
        return category

    return dataset, metrics, metric_updater, viz_updater


def vargconvnet_imagenet_mappings(cfg):
    dataset = ImageNetFromImage(
        root="./tmp_orig_data/imagenet/val",
        split="val",
        transforms=torchvision.transforms.Compose(
            [
                TorchVisionAdapter(interface="Resize", size=256),
                TorchVisionAdapter(interface="CenterCrop", size=224),
                TorchVisionAdapter(interface="PILToTensor"),
                BgrToYuv444(rgb_input=False),
                TorchVisionAdapter(
                    interface="Normalize",
                    mean=128.0,
                    std=128.0,
                ),
            ]
        ),
    )
    metrics = [Accuracy()]

    def metric_updater(metrics, batch, model_outs):
        target = batch["labels"]
        preds, losses = model_outs
        for metric in metrics:
            metric.update(target, preds)

    def viz_updater(img, model_outs):
        preds = model_outs
        preds = ClsViz(is_plot=cfg.is_plot)(img, preds)
        category = dataset.classes[int(preds)]
        return category

    return dataset, metrics, metric_updater, viz_updater


def imagenet_mappings(cfg):
    dataset = ImageNetFromImage(
        root="./tmp_orig_data/imagenet/val",
        split="val",
        transforms=torchvision.transforms.Compose(
            [
                TorchVisionAdapter(interface="Resize", size=256),
                TorchVisionAdapter(interface="CenterCrop", size=224),
                TorchVisionAdapter(interface="PILToTensor"),
                BgrToYuv444(rgb_input=True),
                TorchVisionAdapter(
                    interface="Normalize",
                    mean=128.0,
                    std=128.0,
                ),
            ]
        ),
    )
    metrics = [Accuracy()]

    def metric_updater(metrics, batch, model_outs):
        target = batch["labels"]
        preds, losses = model_outs
        for metric in metrics:
            metric.update(target, preds)

    def viz_updater(img, model_outs):
        preds = model_outs
        preds = ClsViz(is_plot=cfg.is_plot)(img, preds)
        category = dataset.classes[int(preds)]
        return category

    return dataset, metrics, metric_updater, viz_updater


def voc_mappings(cfg):
    dataset = VOCFromImage(
        root="./tmp_orig_data/voc/",
        year="2007",
        image_set="test",
        transforms=torchvision.transforms.Compose(
            [
                TorchVisionAdapter(interface="Resize", size=(416, 416)),
                TorchVisionAdapter(interface="PILToTensor"),
                BgrToYuv444(rgb_input=True),
                TorchVisionAdapter(
                    interface="Normalize",
                    mean=128.0,
                    std=128.0,
                ),
            ]
        ),
    )
    metrics = [VOC07MApMetric(cfg.num_classes)]

    def metric_updater(metrics, batch, model_outs):
        for metric in metrics:
            metric.update(model_outs)

    def viz_updater(img, model_outs):
        preds = model_outs["pred_bboxes"]
        bboxes = preds[0][:, :4]
        labels = preds[0][:, 4]
        scores = preds[0][:, 5]
        res = torch.cat(
            (
                bboxes,
                labels.unsqueeze(-1),
                scores.unsqueeze(-1),
            ),
            -1,
        )
        DetViz(is_plot=cfg.is_plot)(img, res)

    return dataset, metrics, metric_updater, viz_updater


def coco_mappings(cfg):
    ann_file = "./tmp_orig_data/mscoco/annotations/instances_val2017.json"
    dataset = CocoFromImage(
        root="./tmp_orig_data/mscoco/val2017",
        annFile=ann_file,
        transforms=torchvision.transforms.Compose(
            [
                Resize(img_scale=(800, 1024), keep_ratio=True),
                Pad(size=(1024, 1024)),
                ToTensor(to_yuv=True, use_yuv_v2=False),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    )
    metrics = [COCODetectionMetric(ann_file=ann_file)]

    def metric_updater(metrics, batch, model_outs):
        for metric in metrics:
            metric.update(model_outs)

    def viz_updater(img, model_outs):
        dets = model_outs["pred_bboxes"][0]
        DetViz(is_plot=cfg.is_plot)(img, dets)

    return dataset, metrics, metric_updater, viz_updater


def fcos_coco_mappings(cfg):
    deploy_input = cfg.get("deploy_inputs")
    img_scale = deploy_input.get("img").size()[2:]
    ann_file = "./tmp_orig_data/mscoco/annotations/instances_val2017.json"
    dataset = CocoFromImage(
        root="./tmp_orig_data/mscoco/val2017",
        annFile=ann_file,
        transforms=torchvision.transforms.Compose(
            [
                Resize(img_scale=img_scale, keep_ratio=True),
                Pad(size=img_scale),
                ToTensor(to_yuv=True, use_yuv_v2=False),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    )
    metrics = [COCODetectionMetric(ann_file=ann_file)]

    def metric_updater(metrics, batch, model_outs):
        for metric in metrics:
            metric.update(model_outs)

    def viz_updater(img, model_outs):
        dets = model_outs["pred_bboxes"][0]
        DetViz(is_plot=cfg.is_plot)(img, dets)

    return dataset, metrics, metric_updater, viz_updater


def yolo_coco_mappings(cfg):
    ann_file = "./tmp_orig_data/mscoco/annotations/instances_val2017.json"
    dataset = CocoFromImage(
        root="./tmp_orig_data/mscoco/val2017",
        annFile=ann_file,
        transforms=torchvision.transforms.Compose(
            [
                Resize(img_scale=(416, 416), keep_ratio=True),
                Pad(size=(416, 416)),
                ToTensor(to_yuv=True, use_yuv_v2=False),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    )
    metrics = [COCODetectionMetric(ann_file=ann_file)]

    def metric_updater(metrics, batch, model_outs):
        for metric in metrics:
            metric.update(model_outs)

    def viz_updater(img, model_outs):
        dets = model_outs["pred_bboxes"][0]
        DetViz(is_plot=cfg.is_plot)(img, dets)

    return dataset, metrics, metric_updater, viz_updater


def deeplab_cityscapes_mappings(cfg):
    dataset = Cityscapes(
        data_path="tmp_data/cityscapes/val_lmdb",
        transforms=torchvision.transforms.Compose(
            [
                PILToTensor(),
                LabelRemap(mapping=cfg.CITYSCAPES_LABLE_MAPPINGS),
                TensorToNumpy(),
                Resize(img_scale=(1024, 2048), keep_ratio=True),
                ToTensor(to_yuv=True, use_yuv_v2=False),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
        color_space="rgb",
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    metrics = [MeanIOU(seg_class=[str(i) for i in range(cfg.num_classes)])]

    def metric_updater(metrics, batch, model_outs):
        target = batch["gt_seg"]
        preds = model_outs
        for metric in metrics:
            metric.update(target, preds)

    def viz_updater(img, model_outs):
        img = np.transpose(img, (1, 2, 0))
        preds = model_outs[0]
        SegViz(is_plot=cfg.is_plot)(img, preds)

    return dataset, metrics, metric_updater, viz_updater


def unet_cityscapes_mappings(cfg):
    dataset = Cityscapes(
        data_path="tmp_data/cityscapes/val_lmdb",
        transforms=torchvision.transforms.Compose(
            [
                PILToTensor(),
                LabelRemap(mapping=cfg.CITYSCAPES_LABLE_MAPPINGS),
                SegOneHot(num_classes=cfg.num_classes),
                SegResize(size=cfg.data_shape[1:]),
                BgrToYuv444(rgb_input=True),
                TorchVisionAdapter(
                    interface="Normalize",
                    mean=128.0,
                    std=128.0,
                ),
            ]
        ),
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    metrics = [MeanIOU(seg_class=[str(i) for i in range(cfg.num_classes)])]

    def metric_updater(metrics, batch, model_outs):
        # Convert one hot to index
        target = batch["gt_seg"]
        ignore_points = target.sum(dim=1) == 0
        target = target.argmax(dim=1)
        target[ignore_points] = 255
        preds = model_outs[0][0]
        preds = torch.argmax(preds, dim=1, keepdim=True)
        preds = (
            F.interpolate(preds.float(), size=target.shape[1:], mode="nearest")
            .to(dtype=torch.uint8)
            .squeeze(1)
        )
        for metric in metrics:
            metric.update(target, preds)

    def viz_updater(img, model_outs):
        img = np.transpose(img, (1, 2, 0))
        preds = model_outs[0][0]
        preds = torch.argmax(preds[0], dim=0)
        SegViz(is_plot=cfg.is_plot)(img, preds)

    return dataset, metrics, metric_updater, viz_updater


def kitti3d_mappings(cfg):
    class_names = cfg.class_names
    dataset = Kitti3D(
        data_path="./tmp_data/kitti3d/val_lmdb",
        transforms=torchvision.transforms.Compose(
            [
                LidarReformat(),
            ]
        ),
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    metrics = [
        Kitti3DMetricDet(
            compute_aos=True,
            current_classes=class_names,
            difficultys=[0, 1, 2],
        )
    ]

    def metric_updater(metrics, batch, model_outs):
        preds = model_outs
        for metric in metrics:
            metric.update(preds, batch)

    def viz_updater(points, model_outs):
        preds = model_outs[0]
        lidar_det_visualize(
            points=points,
            predictions=preds,
            score_thresh=0.4,
            is_plot=cfg.is_plot,
        )

    return dataset, metrics, metric_updater, viz_updater


def dataset_mappings(dataset, config):
    DatasetMappings = {
        "imagenet": imagenet_mappings,
        "nas_imagenet": nas_imagenet_mappings,
        "vargconvnet": vargconvnet_imagenet_mappings,
        "voc": voc_mappings,
        "coco": coco_mappings,
        "fcos_coco": fcos_coco_mappings,
        "yolo_coco": yolo_coco_mappings,
        "fastscnn_cityscapes": deeplab_cityscapes_mappings,
        "deeplab_cityscapes": deeplab_cityscapes_mappings,
        "unet_cityscapes": unet_cityscapes_mappings,
        "kitti3d": kitti3d_mappings,
    }
    return DatasetMappings[dataset](config)
