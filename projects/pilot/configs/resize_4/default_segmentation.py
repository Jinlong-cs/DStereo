# flake8: noqa

import os

import numpy as np
import torch
from common import (
    actual_input_hw,
    backbone,
    batch_size,
    bn_kwargs,
    datapaths,
    fpn_neck,
    input_hw,
    input_padding,
    lmdb_data,
    log_freq,
    resize_hw,
    roi_region,
    ufpn_seg_neck,
    val_transforms,
    vanishing_point,
    vis_tasks,
    with_rle,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import get_default_parsing_desc

task_name = "default_segmentation"
vis_tasks.append(task_name)

# model
head_out_strides = [2, 8, 16, 32, 64]
num_classes = 16


def get_model(mode):
    out_strides = (
        head_out_strides if mode == "train" else [min(head_out_strides)]
    )

    return dict(
        type="SegmentorV2",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, ufpn_seg_neck]),
        head=dict(
            type="FRCNNSegHead",
            group_base=4,
            in_strides=[4, 8, 16, 32, 64],
            in_channels=[16, 32, 64, 128, 128],
            out_strides=out_strides,
            out_channels=[num_classes] * len(out_strides),
            bn_kwargs=bn_kwargs,
            argmax_output="train" not in mode,
            with_score="test" in mode and with_rle,
            rle_label="test" in mode and with_rle,
            dequant_output="train" in mode,
            with_extra_conv=False,
            node_name=f"{task_name}_head",
        ),
        loss=dict(
            type="MultiStrideLosses",
            num_classes=num_classes,
            out_strides=head_out_strides,
            loss=dict(
                type="WeightedSquaredHingeLoss",
                reduction="mean",
                weight_low_thr=0.6,
                weight_high_thr=1.0,
                hard_neg_mining_cfg=dict(
                    keep_pos=True,
                    neg_ratio=0.999,
                    hard_ratio=1.0,
                    min_keep_num=255,
                ),
            ),
            loss_weights=[4, 2, 2, 2, 2],
            node_name=f"{task_name}_loss",
        )
        if mode == "train"
        else None,
        desc=dict(
            type="AddDesc",
            per_tensor_desc=get_default_parsing_desc(
                desc_id=f"us_{num_classes}",
                roi_regions=roi_region,
                vanishing_point=vanishing_point,
                input_padding=input_padding,
                with_rle=with_rle,
            ),
            node_name=f"{task_name}_desc",
        )
        if "test" in mode
        else None,
        postprocess=dict(
            type="VargNetSegDecoder",
            out_strides=out_strides,
            input_padding=input_padding,
            node_name=f"{task_name}_decoder",
        )
        if "val" in mode
        else None,
    )


# inputs
inputs = dict(
    train=dict(
        labels=[
            torch.zeros(
                (1, 1, actual_input_hw[0] // s, actual_input_hw[1] // s)
            )
            for s in head_out_strides
        ],
    ),
    val=dict(),
    test=dict(),
)


loss_names = [f"stride_{s}_loss" for s in head_out_strides]

metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=name) for name in loss_names],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_{name}$",
            )
            for name in loss_names
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


if lmdb_data:
    ds = datapaths.semantic_parsing
    data_paths = [d["data_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        drop_last=True,
        num_workers=1,
        persistent_workers=True,
        batch_size=batch_size // 2,
        dataset=dict(
            type="DistributedComposeRandomDataset",
            sample_weights=sample_weights,
            datasets=[
                dict(
                    type="DetSeg2DAnnoDataset",
                    idx_path=os.path.join(path, "idx"),
                    img_path=os.path.join(path, "img"),
                    anno_path=os.path.join(path, "anno"),
                    transforms=[
                        dict(
                            type="SemanticSegAffineAugTransformerEx",
                            target_wh=input_hw[::-1],
                            inter_method=10,
                            label_scales=[
                                1.0 / stride_i for stride_i in head_out_strides
                            ],
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            flip_prob=0.5,
                            label_padding_value=-1,
                            rand_translation_ratio=0.0,
                            center_aligned=False,
                            rand_scale_range=(0.8, 1.3),
                            resize_wh=resize_hw[::-1],
                            adapt_diff_resolution=False,
                            padding_in_network_lrub=input_padding,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )
else:
    ds = datapaths.semantic_parsing
    rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
    anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    data_loader = dict(
        type="MultiCachedDataLoader",
        __build_recursive=False,
        last_batch="pad",
        batch_size=batch_size // 2,
        num_workers=1,
        shuffle=True,
        chunk_size=32,
        min_prefetch=1,
        max_prefetch=2,
        min_chunk_num=2,
        max_chunk_num=4,
        batched_transform=True,
        skip_batchify=False,
        prefetcher_using_thread=True,
        dataset=dict(
            type="MultiFusedIterableDataset",
            dataset=[
                dict(
                    type="SplitDataset",
                    dataset=dict(
                        type="LegacyDenseBoxImageRecordDataset",
                        rec_path=rec_path_i,
                        anno_path=anno_path_i,
                        read_only=True,
                        with_seg_label=True,
                        to_rgb=True,
                    ),
                    even_split=False,
                )
                for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
            ],
            prob=sample_weights,
            balance=True,
        ),
        transform=[
            dict(
                type="LegacyDenseBoxImageRecordDatasetDecoder",
                to_rgb=True,
                with_seg_label=True,
                seg_label_dtype=np.int8,
            ),
            dict(
                type="DecodeDenseBoxDatasetToSemanticSegFormat",
            ),
            dict(
                type="MapSemanticSegLabels",
                src_values=[255],
                dst_values=[-1],
            ),
            dict(
                type="SemanticSegAffineAugTransformerEx",
                target_wh=input_hw[::-1],
                inter_method=10,
                label_scales=[1.0 / stride_i for stride_i in head_out_strides],
                use_pyramid=True,
                pyramid_min_step=0.7,
                pyramid_max_step=0.8,
                flip_prob=0.5,
                label_padding_value=-1,
                rand_translation_ratio=0.0,
                center_aligned=False,
                rand_scale_range=(0.8, 1.3),
                resize_wh=resize_hw[::-1],
                adapt_diff_resolution=False,
                padding_in_network_lrub=input_padding,
            ),
            dict(
                type="ReshapeAndCastSemanticSegLabels",
                label_dtype=np.float32,
            ),
            dict(
                type="PackImgAndLabels",
            ),
        ],
    )
