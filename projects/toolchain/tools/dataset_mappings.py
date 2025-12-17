"""dataset mappings, include all the preprocess of images and metrics"""
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import InterpolationMode

from hat.core.adapter import TorchVisionAdapter
from hat.data.datasets.carfusion_keypoints_dataset import CarfusionPackData
from hat.data.datasets.cityscapes import Cityscapes
from hat.data.datasets.culane_dataset import CuLaneDataset
from hat.data.datasets.flyingchairs_dataset import FlyingChairs
from hat.data.datasets.imagenet import ImageNetFromImage
from hat.data.datasets.kitti3d import Kitti3D
from hat.data.datasets.mot17_dataset import Mot17FromImage
from hat.data.datasets.mscoco import CocoFromImage
from hat.data.datasets.nuscenes_dataset import (
    NuscenesFromImage,
    NuscenesFromImageSequence,
    NuscenesLidarDataset,
    NuscenesLidarWithSegDataset,
    NuscenesMonoFromImage,
)
from hat.data.datasets.sceneflow_dataset import SceneFlowFromImage
from hat.data.datasets.voc import VOCFromImage
from hat.data.transforms.classification import BgrToYuv444
from hat.data.transforms.common import PILToTensor, TensorToNumpy
from hat.data.transforms.detection import (
    FixedCrop,
    Normalize,
    Pad,
    Resize,
    Resize3D,
    ToTensor,
)
from hat.data.transforms.lidar_utils import AssignSegLabel, LidarReformat
from hat.data.transforms.multi_views import (
    MultiViewsImgCrop,
    MultiViewsImgResize,
    MultiViewsImgTransformWrapper,
)
from hat.data.transforms.segmentation import LabelRemap, SegOneHot, SegResize
from hat.metrics.acc import Accuracy
from hat.metrics.coco_detection import COCODetectionMetric
from hat.metrics.kitti3d_detection import Kitti3DMetricDet
from hat.metrics.mean_iou import MeanIOU
from hat.metrics.metric_keypoints import MeanKeypointDist, PCKMetric
from hat.metrics.metric_lane_detection import CulaneF1Score
from hat.metrics.metric_optical_flow import EndPointError
from hat.metrics.mot_metrics import MotMetric
from hat.metrics.nuscenes_metric import NuscenesMetric, NuscenesMonoMetric
from hat.metrics.voc_detection import VOC07MApMetric
from hat.visualize.cam3d import Cam3dViz
from hat.visualize.cls import ClsViz
from hat.visualize.det import DetViz
from hat.visualize.disparity import DispViz
from hat.visualize.keypoints import KeypointsViz
from hat.visualize.lane_lines import LanelineViz
from hat.visualize.lidar_det import lidar_det_visualize
from hat.visualize.nuscenes import NuscenesViz
from hat.visualize.opticalflow import FlowViz
from hat.visualize.seg import SegViz
from hat.visualize.track import TrackViz
from .seq_transform import SeqBgrToYuv444, SeqNormalize, SeqResize, SeqToTensor


def detr3d_nuscenes_mapping(cfg):
    data_shape = cfg.data_shape[1:]
    scale = data_shape[1] / 1600
    resize_shape = (round(900 * scale), round(1600 * scale))

    dataset = NuscenesFromImage(
        src_data_dir="./tmp_orig_data/nuscenes",
        version="v1.0-trainval",
        split_name="val",
        transforms=torchvision.transforms.Compose(
            [
                MultiViewsImgResize(size=resize_shape),
                MultiViewsImgCrop(size=data_shape),
                MultiViewsImgTransformWrapper(
                    transforms=[
                        PILToTensor(),
                        BgrToYuv444(rgb_input=True),
                        Normalize(mean=128.0, std=128.0),
                    ],
                ),
            ]
        ),
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        with_bev_mask=False,
        bev_range=cfg.bev_range,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    metrics = [
        NuscenesMetric(data_root=cfg.meta_rootdir, version="v1.0-trainval"),
    ]

    def metric_updater(metrics, batch, model_outs):
        # Convert one hot to index
        for metric in metrics:
            metric.update(batch, model_outs)

    def viz_updater(img, model_outs, meta):
        preds = {"ego_det": model_outs}
        NuscenesViz(is_plot=cfg.is_plot)(img, preds, meta)

    return dataset, metrics, metric_updater, viz_updater


def bev_nuscenes_mappings(cfg):
    data_shape = cfg.data_shape[1:]
    scale = data_shape[1] / 1600
    resize_shape = (round(900 * scale), round(1600 * scale))

    dataset = NuscenesFromImage(
        src_data_dir="./tmp_orig_data/nuscenes",
        version="v1.0-trainval",
        split_name="val",
        transforms=torchvision.transforms.Compose(
            [
                MultiViewsImgResize(size=resize_shape),
                MultiViewsImgCrop(size=data_shape),
                MultiViewsImgTransformWrapper(
                    transforms=[
                        PILToTensor(),
                        BgrToYuv444(rgb_input=True),
                        Normalize(mean=128.0, std=128.0),
                    ],
                ),
            ]
        ),
        bev_size=cfg.bev_size,
        map_size=cfg.map_size,
        map_path=cfg.meta_rootdir,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    metrics = [
        NuscenesMetric(data_root=cfg.meta_rootdir, version="v1.0-trainval"),
        MeanIOU(seg_class=cfg.seg_classes_name, ignore_index=-1),
    ]

    def metric_updater(metrics, batch, model_outs):
        # Convert one hot to index
        preds = model_outs[1]["bev_det"]
        metrics[0].update(batch, preds)

        target = batch["bev_seg_indices"]
        preds = model_outs[1]["bev_seg"]
        if cfg.use_bce is True:
            preds += 1
        metrics[1].update(target, preds)

    def viz_updater(img, model_outs, meta):
        preds = model_outs[1]
        NuscenesViz(
            is_plot=cfg.is_plot, bev_size=cfg.bev_size, use_bce=cfg.use_bce
        )(img, preds, meta)

    return dataset, metrics, metric_updater, viz_updater


def bev_nuscenes_sequence_mappings(cfg):
    data_shape = cfg.data_shape[1:]
    scale = data_shape[1] / 1600
    resize_shape = (round(900 * scale), round(1600 * scale))
    dataset = NuscenesFromImageSequence(
        src_data_dir="./tmp_orig_data/nuscenes",
        version="v1.0-trainval",
        split_name="val",
        transforms=torchvision.transforms.Compose(
            [
                MultiViewsImgResize(size=resize_shape),
                MultiViewsImgCrop(size=data_shape),
                MultiViewsImgTransformWrapper(
                    transforms=[
                        PILToTensor(),
                        BgrToYuv444(rgb_input=True),
                        Normalize(mean=128.0, std=128.0),
                    ],
                ),
            ]
        ),
        num_seq=cfg.num_seq,
        bev_size=cfg.bev_size,
        map_size=cfg.map_size,
        map_path=cfg.meta_rootdir,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    metrics = [
        NuscenesMetric(data_root=cfg.meta_rootdir, version="v1.0-trainval"),
        MeanIOU(seg_class=cfg.seg_classes_name, ignore_index=-1),
    ]

    def metric_updater(metrics, batch, model_outs):
        # Convert one hot to index
        preds = model_outs[1]["bev_det"]
        metrics[0].update(batch, preds)
        target = batch["bev_seg_indices"]
        preds = model_outs[1]["bev_seg"]
        if cfg.use_bce is True:
            preds += 1
        metrics[1].update(target, preds)

    def viz_updater(img, model_outs, meta):
        preds = model_outs[1]
        NuscenesViz(
            is_plot=cfg.is_plot, bev_size=cfg.bev_size, use_bce=cfg.use_bce
        )(img, preds, meta)

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


def detr_coco_mappings(cfg):
    ann_file = "./tmp_orig_data/mscoco/annotations/instances_val2017.json"
    dataset = CocoFromImage(
        root="./tmp_orig_data/mscoco/val2017",
        annFile=ann_file,
        transforms=torchvision.transforms.Compose(
            [
                Resize(img_scale=(800, 1333), keep_ratio=False),
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


def flyingchairs_mappings(cfg):
    dataset = FlyingChairs(
        data_path="./tmp_data/FlyingChairs/val_lmdb/",
        transforms=torchvision.transforms.Compose(
            [
                ToTensor(to_yuv=False),
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
    metrics = [EndPointError()]

    def metric_updater(metrics, batch, model_outs):
        labels = batch["gt_flow"]
        preds = model_outs
        preds = (
            F.interpolate(preds.float(), scale_factor=4, mode="bilinear") * 4.0
        )
        for metric in metrics:
            metric.update(labels, preds)

    def viz_updater(img, model_outs):
        preds = model_outs
        preds = (
            F.interpolate(preds.float(), scale_factor=4, mode="bilinear") * 4.0
        )
        preds = preds.permute((0, 2, 3, 1))
        FlowViz(is_plot=cfg.is_plot)(img, preds)

    return dataset, metrics, metric_updater, viz_updater


def culane_mappings(cfg):
    dataset = CuLaneDataset(
        data_path="./tmp_data/CULane/test_lmdb/",
        transforms=torchvision.transforms.Compose(
            [
                FixedCrop(size=(0, 270, 1640, 320)),
                Resize(
                    img_scale=(320, 800),
                    multiscale_mode="value",
                    keep_ratio=False,
                ),
                ToTensor(to_yuv=False),
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
    metrics = [CulaneF1Score()]

    def metric_updater(metrics, batch, model_outs):
        target = batch["ori_gt_lines"]
        for metric in metrics:
            metric.update(target, model_outs)

    def viz_updater(img, model_outs):
        preds = model_outs[0]
        LanelineViz(is_plot=cfg.is_plot)(img, preds)

    return dataset, metrics, metric_updater, viz_updater


def fcos3d_nuscenes_mappings(cfg):
    version = "v1.0-trainval"
    dataset = NuscenesMonoFromImage(
        version=version,
        src_data_dir="./tmp_orig_data/nuscenes/",
        split_name="val",
        transforms=torchvision.transforms.Compose(
            [
                Resize3D(img_scale=(896, 512)),
                Pad(size=(512, 896)),
                ToTensor(to_yuv=True, use_yuv_v2=False),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    )
    dataset = pickle.loads(pickle.dumps(dataset))

    metrics = [
        NuscenesMonoMetric(
            data_root="./tmp_data/nuscenes/meta",
            version=version,
        )
    ]

    def metric_updater(metrics, batch, model_outs):
        for metric in metrics:
            metric.update(batch, model_outs)

    def viz_updater(data, model_outs, save_dir, score_thr):
        Cam3dViz()(data, model_outs, save_dir, score_thr)

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


def mot17_mappings(cfg):

    dataset = Mot17FromImage(
        data_path="./tmp_orig_data/mot17/split_data/test",
        sampler_lengths=[1],
        sample_mode="fixed_interval",
        transforms=torchvision.transforms.Compose(
            [
                SeqResize(
                    img_scale=(800, 1422),
                    keep_ratio=False,
                ),
                SeqToTensor(to_yuv=False),
                SeqBgrToYuv444(rgb_input=True),
                SeqNormalize(mean=128.0, std=128.0),
            ]
        ),
    )

    metrics = [
        MotMetric(
            gt_dir="./tmp_data/mot17/test_gt",
        )
    ]

    def metric_updater(metrics, batch, model_outs):
        for metric in metrics:
            metric.update(model_outs)

    def viz_updater(img, model_outs):
        preds = model_outs
        TrackViz(is_plot=cfg.is_plot)(img, preds)

    return dataset, metrics, metric_updater, viz_updater


def sceneflow_mappings(cfg):
    dataset = SceneFlowFromImage(
        data_path="./tmp_orig_data/SceneFlow/",
        data_list="./tmp_orig_data/SceneFlow/SceneFlow_finalpass_test.txt",
        transforms=torchvision.transforms.Compose(
            [
                ToTensor(to_yuv=False),
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
    metrics = [EndPointError(use_mask=True)]
    maxdisp = 192

    def metric_updater(metrics, batch, model_outs):
        labels = batch["gt_disp"]
        preds = model_outs
        masks = (labels > 0) & (labels < maxdisp)
        metrics[0].update(labels, preds, masks)

    def viz_updater(img, model_outs):
        preds = model_outs.squeeze(0).cpu().numpy()
        depth = (1050 * 540 / preds) / 1000
        DispViz(is_plot=cfg.is_plot)(img, preds, depth)

    return dataset, metrics, metric_updater, viz_updater


def nuscenes_lidar_mappings(cfg):
    version = "v1.0-trainval"
    class_names = cfg.class_names
    dataset = NuscenesLidarDataset(
        test_mode=True,
        num_sweeps=9,
        data_path="./tmp_data/nuscenes/lidar_seg/v1.0-trainval/val_lmdb",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        classes=class_names,
        transforms=torchvision.transforms.Compose(
            [
                LidarReformat(),
            ]
        ),
    )

    metrics = [
        NuscenesMetric(
            data_root="./tmp_data/nuscenes/meta",
            version=version,
            use_lidar=True,
            classes=class_names,
        )
    ]

    def metric_updater(metrics, batch, model_outs):
        for metric in metrics:
            metric.update(batch, model_outs)

    def viz_updater(points, model_outs):
        preds = model_outs[0]
        lidar_det_visualize(
            points=points,
            predictions=preds,
            score_thresh=0.4,
            is_plot=cfg.is_plot,
            reverse=True,
        )

    return dataset, metrics, metric_updater, viz_updater


def carfusion_mappings(cfg):
    image_size = (128, 128)
    dataset = CarfusionPackData(
        data_path="./tmp_data/carfusion/test_lmdb",
        transforms=torchvision.transforms.Compose(
            [
                Resize(img_scale=image_size, keep_ratio=True),
                Pad(size=image_size),
                ToTensor(to_yuv=True, use_yuv_v2=False),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    )

    metrics = [
        PCKMetric(alpha=0.1, feat_stride=4, img_shape=image_size),
        MeanKeypointDist(
            feat_stride=4,
        ),
    ]

    def update_metric(metrics, batch, model_outs):
        data = {
            "gt_ldmk": batch["gt_ldmk"],
            "pr_ldmk": model_outs[1],
            "gt_ldmk_attr": batch["gt_ldmk_attr"],
        }
        for metric in metrics:
            metric.update(data)

    def viz_updater(img, scale, model_outs):
        pred_ldmk = model_outs[1][0]
        KeypointsViz(threshold=0.5, is_plot=cfg.is_plot)(
            image=img,
            keypoints=pred_ldmk,
            scale=scale,
        )

    return dataset, metrics, update_metric, viz_updater


def nuscenes_lidar_multi_mappings(cfg):
    version = "v1.0-trainval"
    class_names = cfg.det_class_names
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]
    dataset = NuscenesLidarWithSegDataset(
        test_mode=True,
        num_sweeps=9,
        data_path="./tmp_data/nuscenes/lidar_seg/v1.0-trainval/val_lmdb",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        classes=class_names,
        transforms=torchvision.transforms.Compose(
            [
                AssignSegLabel(
                    bev_size=[512, 512],
                    num_classes=2,
                    class_names=[0, 1],
                    point_cloud_range=point_cloud_range,
                    voxel_size=voxel_size[:2],
                ),
                LidarReformat(with_gt=True),
            ]
        ),
    )

    metrics = [
        NuscenesMetric(
            data_root="./tmp_data/nuscenes/meta",
            version=version,
            use_lidar=True,
            classes=class_names,
        ),
        MeanIOU(seg_class=cfg.seg_classes_name, ignore_index=-1),
    ]

    def metric_updater(metrics, batch, model_outs):
        preds = model_outs[1]["det"]
        metrics[0].update(batch, preds)

        target = batch["gt_seg_labels"]
        preds = model_outs[1]["seg"]
        metrics[1].update(target, preds)

    def viz_updater(points, model_outs):
        preds_det = model_outs[1]["det"][0]
        lidar_det_visualize(
            points=points,
            predictions=preds_det,
            score_thresh=0.4,
            is_plot=cfg.is_plot,
            reverse=True,
        )

        preds_seg = model_outs[1]["seg"][0]
        SegViz(is_plot=cfg.is_plot)(None, preds_seg)

    return dataset, metrics, metric_updater, viz_updater


def dataset_mappings(dataset, config):
    DatasetMappings = {
        "nas_imagenet": nas_imagenet_mappings,
        "vargconvnet": vargconvnet_imagenet_mappings,
        "imagenet": imagenet_mappings,
        "voc": voc_mappings,
        "coco": coco_mappings,
        "fcos_coco": fcos_coco_mappings,
        "yolo_coco": yolo_coco_mappings,
        "detr_coco": detr_coco_mappings,
        "fastscnn_cityscapes": deeplab_cityscapes_mappings,
        "deeplab_cityscapes": deeplab_cityscapes_mappings,
        "unet_cityscapes": unet_cityscapes_mappings,
        "flyingchairs": flyingchairs_mappings,
        "culane": culane_mappings,
        "nuscenes_mono": fcos3d_nuscenes_mappings,
        "sceneflow": sceneflow_mappings,
        "kitti3d": kitti3d_mappings,
        "mot17": mot17_mappings,
        "nuscenes_lidar": nuscenes_lidar_mappings,
        "bev_nuscenes": bev_nuscenes_mappings,
        "bev_nuscenes_sequence": bev_nuscenes_sequence_mappings,
        "detr3d_nuscenes": detr3d_nuscenes_mapping,
        "nuscenes_lidar_multi": nuscenes_lidar_multi_mappings,
        "carfusion": carfusion_mappings,
    }
    return DatasetMappings[dataset](config)
