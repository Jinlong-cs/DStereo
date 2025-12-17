"""dataset mappings, include all the preprocess of images and metrics"""
import os
import pickle

import torchvision

from hat.data.datasets.nuscenes_dataset import (
    NuscenesBevDataset,
    NuscenesBevSequenceDataset,
)
from hat.data.transforms.classification import BgrToYuv444
from hat.data.transforms.common import PILToTensor
from hat.data.transforms.detection import Normalize
from hat.data.transforms.multi_views import (
    MultiViewsImgCrop,
    MultiViewsImgResize,
    MultiViewsImgTransformWrapper,
)
from hat.metrics.mean_iou import MeanIOU
from hat.metrics.nuscenes_metric import NuscenesMetric
from hat.visualize.nuscenes import NuscenesViz


def detr3d_nuscenes_mapping(cfg):
    data_shape = cfg.data_shape[1:]
    scale = data_shape[1] / 1600
    resize_shape = (round(900 * scale), round(1600 * scale))

    dataset = NuscenesBevDataset(
        data_path=os.path.join(cfg.data_rootdir, "val_lmdb"),
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


def nuscenes_mappings(cfg):
    data_shape = cfg.data_shape[1:]
    scale = data_shape[1] / 1600
    resize_shape = (round(900 * scale), round(1600 * scale))

    dataset = NuscenesBevDataset(
        data_path=os.path.join(cfg.data_rootdir, "val_lmdb"),
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


def nuscenes_mappings_sequence(cfg):
    data_shape = cfg.data_shape[1:]
    scale = data_shape[1] / 1600
    resize_shape = (round(900 * scale), round(1600 * scale))
    dataset = NuscenesBevSequenceDataset(
        data_path=os.path.join(cfg.data_rootdir, "val_lmdb"),
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


def dataset_mappings(dataset, config):
    DatasetMappings = {
        "nuscenes": nuscenes_mappings,
        "nuscenes_sequence": nuscenes_mappings_sequence,
        "detr3d_nuscenes": detr3d_nuscenes_mapping,
    }
    return DatasetMappings[dataset](config)
