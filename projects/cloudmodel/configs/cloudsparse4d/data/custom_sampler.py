# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict

import numpy as np
import torch.utils.data as data

from hat.data.datasets.temporal_dataset import (
    TemporalJsonDataset,
    TemporalLabelDataset,
    TemporalLMDBDataset,
    TemporalLmdbDataset,
)
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class ConcatTemporalMixedDataset(data.ConcatDataset):
    """A wrapper of concatenated dataset with group flag.
    Combine temporal & single-frame datasets.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`,
    addititionally concatenat the group flag of all dataset.

    Args:
        datasets: A list of datasets.
        with_flag: Whether to concatenate datasets flags.
            If True, concatenate all datasets flag (
            all datasets must has flag attribute in this case).
            Default to False.
        record_index: Whether to record the index. If True,
            record the index. Default to False.
    """

    TEMPORAL_DATASET_TYPES = [
        TemporalLMDBDataset,
        TemporalLabelDataset,
        TemporalJsonDataset,
        TemporalLmdbDataset,
    ]

    def __init__(
        self,
        datasets,
        with_flag: bool = True,
        record_index: bool = False,
        accumulate_flag: bool = True,
    ):
        super(ConcatTemporalMixedDataset, self).__init__(datasets)
        self._record_index = record_index
        if with_flag:

            if accumulate_flag:
                accumulate_sum = 0

            flags = []
            for dataset in datasets:
                assert hasattr(dataset, "flag"), "dataset must has group flag"
                assert isinstance(
                    dataset.flag, np.ndarray
                ), "dataset flag must is numpy array instance"
                assert (
                    len(dataset) == dataset.flag.shape[0]
                ), "dataset flag length at axis 0 must equal to the dataset length"  # noqa: E501
                if accumulate_flag:
                    flag_tmp = dataset.flag + accumulate_sum
                    flags.append(flag_tmp)
                    accumulate_sum += len(np.unique(dataset.flag))
                else:
                    flags.append(dataset.flag)
            self.flag = np.concatenate(flags)

        temporal_flag = np.concatenate(
            [
                np.ones_like(dataset.flag)
                if type(dataset) in self.TEMPORAL_DATASET_TYPES
                else np.zeros_like(dataset.flag)
                for dataset in datasets
            ]
        ).astype("bool")
        self.flag = self.convert_flag(self.flag, temporal_flag)
        self.len_list = [len(dataset) for dataset in self.datasets]

    def convert_flag(self, flag, temporal_flag):
        new_flag = (flag + 1) * temporal_flag
        flag_sets, counts = np.unique(
            new_flag[new_flag != 0], return_counts=True
        )
        if counts.shape[0] == 0:
            max_clip = 100
            num_single = (~temporal_flag).sum()
            group_num = int(num_single / max_clip)
            new_flag = np.random.randint(0, group_num + 1, flag.shape[0])
        else:
            max_clip = counts.max()
            num_single = (~temporal_flag).sum()
            group_num = int(num_single / max_clip)
            single_flag = np.arange(group_num + 1).repeat(max_clip)[
                :num_single
            ]
            new_flag[~temporal_flag] = single_flag + 1 + flag_sets.max()
            new_flag = new_flag - new_flag.min()
        assert np.bincount(new_flag).all()
        return new_flag

    def __getitem__(self, idx) -> Dict:
        res = super().__getitem__(idx)
        if self._record_index:
            assert isinstance(res, dict), "__getitem__ must return a dict"
            res["index"] = idx
        return res
