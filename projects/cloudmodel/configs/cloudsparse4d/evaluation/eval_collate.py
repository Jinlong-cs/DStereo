# Copyright (c) Horizon Robotics. All rights reserved.
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from mmcv.parallel.data_container import DataContainer
from torch.utils.data.dataloader import default_collate


def cloudsparse4d_eval_collate(batch, samples_per_gpu=1):
    """Collate function for cloud-sparse4d evaluation"""

    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i : i + samples_per_gpu]]
                )
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True
            )
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i : i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(
                                max_shape[dim - 1], sample.size(-dim)
                            )
                    padded_samples = []
                    for sample in batch[i : i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[
                                dim - 1
                            ] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(sample.data, pad, value=sample.padding_value)
                        )
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate(
                            [
                                sample.data
                                for sample in batch[i : i + samples_per_gpu]
                            ]
                        )
                    )
                else:
                    raise ValueError(
                        "pad_dims should be either None or integers (1-3)"
                    )

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i : i + samples_per_gpu]]
                )
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [
            cloudsparse4d_eval_collate(samples, samples_per_gpu)
            for samples in transposed
        ]
    elif isinstance(batch[0], Mapping):
        collate_data = {}
        special_key = ["image_keys"]
        for spe_key in special_key:
            if spe_key in batch[0].keys():
                collate_data[spe_key] = [d[spe_key] for d in batch]
        for key in batch[0]:
            if key not in special_key:
                collate_data[key] = cloudsparse4d_eval_collate(
                    [d[key] for d in batch], samples_per_gpu
                )
        return collate_data
    else:
        return default_collate(batch)
