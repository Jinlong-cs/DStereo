# Copyright (c) Horizon Robotics. All rights reserved.
# This file defines the datasets for planning.
# Basic TDT datasets is from traj prediction.

import logging
import pickle
from typing import Callable, Dict, List, Optional, Sequence

import torch

from hat.core.traj_pred_typing import PathLike, TrajGroupIndex
from hat.data.datasets.dataset_wrappers import ConcatDataset
from hat.data.datasets.traj_pred_dataset import (
    AutoMultiAgentDataset,
    BaseTrajDataset,
)
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "P3CMultiAgentDataset",
    "P3CPickledTdtDataset",
]


class P3CMultiAgentDataset(AutoMultiAgentDataset):
    """P3C Urban Multi Agent Dataset."""

    ROUTE_COLS = ["timestamp", "frame_id", "pos_x", "pos_y", "yaw"]

    def __init__(
        self,
        df_path: PathLike,
        traj_group_indices: List[TrajGroupIndex],
        seq_length: int,
        seq_period: int,
        sample_step: int,
        context_frames: int,
        veh_type_id: int,
        transforms: Optional[Callable] = None,
        allow_incomplete_traj: bool = False,
        ego_as_obs: bool = False,
        have_index_col: bool = False,
        load_df_func: Optional[Callable] = None,
        # Below are customized parameters
        max_stamp_thr: Optional[float] = 550,
        path_prefix: str = "",
        check_map_exist_func: Optional[Callable] = None,
    ):
        all_kwargs = {
            "df_path": df_path,
            "traj_group_indices": traj_group_indices,
            "seq_length": seq_length,
            "seq_period": seq_period,
            "sample_step": sample_step,
            "context_frames": context_frames,
            "veh_type_id": veh_type_id,
            "transforms": transforms,
            "allow_incomplete_traj": allow_incomplete_traj,
            "ego_as_obs": ego_as_obs,
            "have_index_col": have_index_col,
            "load_df_func": load_df_func,
            "max_stamp_thr": max_stamp_thr,
            "path_prefix": path_prefix,
            "check_map_exist_func": check_map_exist_func,
        }
        super(P3CMultiAgentDataset, self).__init__(**all_kwargs)

    def __getitem__(self, index: int) -> dict:
        """Return a sample according to a integer index.

        Args:
            index: An integer index.

        Returns:
            sample: a dict which contains complete information of a
                trajectory sample. It mainly contains a DataFrame and a
                SeqIndex. The accompanying SeqIndex contains complete
                information of how we can get the sample DataFrame from the
                original big DataFrame read during initialization. We can also
                utilize this SeqIndex to slice the output DataFrame to extract
                DataFrame corresponding to a specific frame_id.
        """
        sample = self._get_basic_sample_dict(index)
        traj_group_index = sample["seq_index"].traj_group_index
        sample["dataset_prefix"] = self.path_prefix
        sample["date_token"] = traj_group_index[1]

        sample = self.gen_long_term_navi(sample)

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def gen_long_term_navi(self, sample: Dict) -> Dict:
        """Return a sample with long term ego coodinates as fake navi.

        Args:
            sample: original tdt sample data.
        Returns:
            sample: the updated sample with new cols.
        """
        traj_group_index = sample["seq_index"].traj_group_index
        traj_navi = BaseTrajDataset.filter_df_by_traj_group_index(
            self.df, traj_group_index
        )

        traj_navi = traj_navi[self.ROUTE_COLS]
        traj_navi = traj_navi.drop_duplicates()
        sample["traj_navi"] = traj_navi

        return sample


@OBJECT_REGISTRY.register
class P3CPickledTdtDataset(torch.utils.data.Dataset):
    """A wrapper Dataset to enable loading a pickled tdt dataset.

    This utility class is useful when you find building the online dataset
    time-consuming. Since our tdt datsets only has a DataFrame and a dict which
    is relatively small, the pickled object's size should be close to the csv.
    This kind of behavior means that you should assign 'transform=None' when
    before you dump your tdt dataset.
    """

    def __init__(
        self,
        pkl_path: PathLike,
        seq_length: int,
        seq_period: int,
        sample_step: int,
        context_frames: int,
        transforms: Optional[Callable] = None,
        indices: Optional[Sequence] = None,
        wrap_func: Optional[List] = None,
    ):
        """Initialize method.

        Args:
            pkl_path: path to a dumped tdt dataset.
            seq_length: length of the sequence w.r.t the original
                un-sampled frame index.
            seq_period: sampling period of the sequence.
            sample_step: sample step_size for the generated list of
                sequences. This value usually changes between training and
                validation.
            context_frames: number of historical frames to use as
                temporal context for prediction.
            transforms: torch-style transform. Defaults to
                None.
            indices: indices in the whole set selected
                for subset. Default to None.
            wrap_func: functions that will be called after
                loading pickle dataset. Default to None.
        """
        with open(pkl_path, "rb") as fp:
            self.loaded_ds = pickle.load(fp)
        if type(self.loaded_ds) is ConcatDataset:
            datasets_to_check = self.loaded_ds.datasets
        else:
            datasets_to_check = [self.loaded_ds]
        for dataset in datasets_to_check:
            assert (
                (seq_length == dataset.seq_length)
                and (seq_period == dataset.seq_period)
                and (sample_step == dataset.sample_step)
                and (context_frames == dataset.context_frames)
            ), "The pickle parameters are not consistent with the input"
        self.transforms = transforms
        # In default, self.loader_order is set as None and the dataloader
        #   will traverse and output all the data.
        if indices is not None:
            self.loaded_ds = torch.utils.data.Subset(self.loaded_ds, indices)

        if wrap_func is not None:
            for func in wrap_func:
                func(self)

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset.
        """
        return len(self.loaded_ds)

    def __getitem__(self, index: int):
        """Return a sample according to a integer index.

        Args:
            index: an integer index.

        Returns:
            raw_sample: a dict which contains information of a
                trajectory sample.
        """
        raw_sample = self.loaded_ds[index]
        raw_sample["dataset_index"] = index
        if self.transforms:
            return self.transforms(raw_sample)
        else:
            return raw_sample

    def __repr__(self):
        return "P3CPickledTdtDataset"
