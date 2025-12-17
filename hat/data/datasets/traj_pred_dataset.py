# Copyright (c) Horizon Robotics. All rights reserved.
# This file defines the datasets for trajectory prediction.

import itertools
import json
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from glob import glob
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import msgpack
import numpy as np
import pandas as pd
import torch
import yaml
from pandas import DataFrame
from tqdm import tqdm

from hat.core.traj_pred_typing import PathLike, SeqIndex, TrajGroupIndex
from hat.data.datasets.data_packer import Packer
from hat.data.datasets.jaad import JAAD
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.lmdb import Lmdb

logger = logging.getLogger(__name__)

__all__ = [
    "BaseTrajDataset",
    "AutoMultiAgentDataset",
    "AutoMultiAgentNaviDataset",
    "PickledTdtDataset",
    "ConcatTdtDataset",
    "TrajPredLMDBDateset",
    "TrajPredFPVPedDataset",
    "TrajPredJaadDataset",
    "BaseTrajDatasetV2",
    "ConcatTdtDatasetV2",
    "PickledTdtDatasetV2",
    "TrajPredBehavDataset",
]


@OBJECT_REGISTRY.register
class BaseTrajDataset(torch.utils.data.Dataset):  # noqa: D205,D400
    """Torch based trajectory dataset for TDT (Trajectory Dataset Template)
    format datasets.

    This class is responsible for building a pytorch map-style dataset
    which returns a dictionary containing an index and a pandas DataFrame
    with the __getitem__ method. This class is designed to deal with
    one file at a time, and it serves as the building block for more
    complex datasets.

    This class load customized TDT csv file. Each row of the csv records
    the information of an obstacle in a certain time stamp. The csv file
    has columns for obstacle information (in `OBS_COLS`) and columns for
    odometry of the ego vehicle (in `EGO_COLS`).

    Note: TDT is a csv format to save trajectory datasets. It must have
    all the columns in `BASIC_COLS`, `EGO_COLS` and `OBS_COLS`.
    """

    # DataFrame columns for the ego vehicle and the obstacles.
    BASIC_COLS = [
        "map_id",
        "date",
        "data_num",
        "data_version",
        "timestamp",
        "frame_id",
    ]
    EGO_COLS = [
        "center_car_id",
        "width",
        "length",
        "pos_x",
        "pos_y",
        "pos_z",
        "yaw",
        "ego_x0",
        "ego_x1",
        "ego_x2",
        "ego_x3",
        "ego_y0",
        "ego_y1",
        "ego_y2",
        "ego_y3",
        "ego_z0",
        "ego_z1",
        "ego_z2",
        "ego_z3",
    ]
    OBS_COLS = [
        "track_id",
        "classification",
        "obs_width",
        "obs_length",
        "x",
        "y",
        "z",
        "obs_yaw",
        "x0",
        "x1",
        "x2",
        "x3",
        "y0",
        "y1",
        "y2",
        "y3",
        "z0",
        "z1",
        "z2",
        "z3",
    ]

    # If we add ego vehicle to the obstacle list to predict, we use
    # -ANSWER as its `track_id`.
    ANSWER = 42

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
    ):
        """Initialize method.

        Args:
            df_path: the path to a TDT-format DataFrame.
            traj_group_indices (List[TrajGroupIndex]): a list of TrajGroupIndex
                obtained from dataset config file.
            seq_length: length of the sequence w.r.t the original
                un-sampled frame index.
            seq_period: sampling period of the sequence.
            sample_step: sample step_size for the generated list of
                sequences. This value usually changes between training and
                validation.
            context_frames: number of historical frames to use as
                temporal context for prediction.
            veh_type_id: the type id of the vehicle class in the input
                dataframe.
            transforms: torch-style transform. Defaults to
                None.
            allow_incomplete_traj (bool, optional): whether or not to allow the
                trajectory to stretch outside of the minimum and maximum frame
                index range and become incomplete. This feature is added to
                adapt to NuScenens prediction since the NuScenes-devkit do not
                support the metric calculation of an incomplete trajectory.
                Defaults to False. If the user is using NuScenes for
                validation, this parameter must be False.
            ego_as_obs (bool, optional): whether to add an additional row in
                the DataFrame of each sample to treat ego vehicle as one of the
                dynamic objects. This is the simplist solution to match with
                the down-stream pipeline and seamlessly add the information of
                the ego vechile. Defaults to False. The new row is assigned
                with a track_id of -self.ANSWER.
            have_index_col (bool, optional): whether the csv file have the
                index column. Default to False.
            load_df_func: the function to load
                dataframes from the TDT csv in `df_path`. Default to None.
                If it is None, the class will call the internal loading
                function `_load_df_func`. The `load_df_func` takes `df_path`
                and `have_index_col` as input.
        """
        super().__init__()

        self.df_path = df_path
        assert os.path.exists(
            df_path
        ), f"The dataframe csv file in the path {df_path} is not exist."
        self.load_df_func = (
            self.default_load_df_func if load_df_func is None else load_df_func
        )  # noqa: E501
        self.df = self.load_df_func(df_path, have_index_col)
        df_cols = self.df.columns
        for col in self.BASIC_COLS + self.EGO_COLS + self.OBS_COLS:
            assert col in df_cols, f"The input dataframe misses column {col}."
        self.df["date"] = self.df["date"].astype("str")

        self.traj_group_indices = traj_group_indices
        self.transforms = transforms
        self.allow_incomplete_traj = allow_incomplete_traj
        self.ego_as_obs = ego_as_obs
        self.veh_type_id = veh_type_id

        self.seq_length = seq_length
        self.seq_period = seq_period
        self.sample_step = sample_step
        self.context_frames = context_frames
        self.future_frames = seq_length // seq_period - context_frames

        self.df_indices = self.build_df_indices(
            self.df,
            self.traj_group_indices,
            self.seq_period,
            self.sample_step,
            self.context_frames,
            self.future_frames,
            self.allow_incomplete_traj,
        )
        # If we concat different `BaseTrajDataset` instances by
        # `torch.utils.data.ConcatDataset`, we use this parameter to
        # express the index in the ConcatDataset.
        self.concat_dataset_idx = 0

    @staticmethod
    def default_load_df_func(
        df_path, have_index_col: Optional[bool]
    ) -> DataFrame:
        """Load the dataframe.

        Args:
            df_path: the path to a TDT-format DataFrame.
            have_index_col (bool, optional): whether the csv file have the
                index column.

        Returns:
            df: the dataframe.
        """
        index_col = 0 if have_index_col else None
        df = pd.read_csv(df_path, index_col=index_col)
        return df

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset.
        """
        return len(self.df_indices)

    def _get_basic_sample_dict(self, index: int) -> dict:
        """Return the basic sample dictionary according to a integer index.

        Args:
            index: an integer index.

        Returns:
            sample: a dict which contains complete information of a
                trajectory sample. It mainly contains a DataFrame and a
                SeqIndex. The accompanying SeqIndex contains complete
                information of how we can get the sample DataFrame from the
                original big DataFrame read during initialization. We can also
                utilize this SeqIndex to slice the output DataFrame to extract
                DataFrame corresponding to a specific frame_id.
        """
        # Get the sequence index of this sample and its last context frame.
        seq_index, last_context_frame = self.df_indices[index]
        # Filter out a DataFrame that correspond to the current sequence.
        seq_df = self.filter_df_by_seq_index(self.df, seq_index)
        # Add one additional row for ego vehicle as obs if requested.
        if self.ego_as_obs:
            seq_df = self.add_ego_as_obs(seq_df, self.veh_type_id)
        # Select first row in case of possible duplicate rows.
        seq_df = seq_df.groupby(["frame_id", "track_id"]).first().reset_index()
        # Sort seq_df by frame id and track_id.
        seq_df.sort_values(by=["frame_id", "track_id"], inplace=True)
        # Build the sample, record the dataset index for later data retrieval.
        sample = {
            "concat_dataset_index": self.concat_dataset_idx,
            "dataset_index": index,
            "seq_index": seq_index,
            "seq_df": seq_df,
            "last_context_frame_id": last_context_frame,
        }
        return sample

    def __getitem__(self, index: int) -> dict:
        """Return a sample according to a integer index.

        Args:
            index: an integer index.

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
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def set_concat_dataset_index(self, idx: int) -> None:
        """Set the id of the current instance in the concatenated dataset.

        When we use multiple sub-datasets (BaseTrajDataset) to build an
        overall data iterator, each BaseTrajDataset instance does not record
        its index in the concatenated dataset. Therefore, it is unable to
        locate the original dataframe through the output information.

        Here, we enable the users to manually assign this index if some
        `trace back` operations are needed.

        Args:
            idx: the index of the current BaseTrajDataset instance
                in the concatenated dataset.
        """
        self.concat_dataset_idx = idx

    @staticmethod
    def build_df_indices(
        df: DataFrame,
        traj_group_indices: list,
        seq_period: int,
        sample_step: int,
        context_frames: int,
        future_frames: int,
        allow_incomplete_traj: bool,
    ) -> Dict[int, SeqIndex]:
        """Build an index mapping from integer indices to SeqIndex.

        Args:
            df: a preprocessed DataFrame.
            traj_group_indices (List[TrajGroupIndex]): a list of TrajGroupIndex
                obtained from dataset config file.
            seq_period: sampling period of the sequence.
            sample_step: user-defined sample_step.
            context_frames: user-defined length of historical trajectory
                frames.
            future_frames: use-defined length of future trajectory
                frames.
            allow_incomplete_traj: whether or not to allow the
                trajectory to stretch outside of the minimum and maximum frame
                index range and become incomplete. This feature is added to
                adapt to NuScenens dataset since the NuScenes-devkit do not
                support the metric calculation of an incomplete trajectory.

        Returns:
            out_indices (Dict[int, SeqIndex]): a dict which contains the index
                mapping from integer index to SeqIndex.
        """
        out_indices = {}  # the output index dictionary
        cur_index = 0  # the current key for the output dictionary

        for traj_group_index in traj_group_indices:
            # Filter out the DataFrame corresponding to the current trajectory
            # group and get the set of unique time frame indices.
            traj_group_index = TrajGroupIndex(*traj_group_index)
            traj_group_df = BaseTrajDataset.filter_df_by_traj_group_index(
                df, traj_group_index
            )
            traj_group_df_frame_ids = traj_group_df["frame_id"]
            unique_tg_frame_ids = set(traj_group_df_frame_ids)

            # Calculate a legal range for the last context frame (lcf).
            frame_id_min = traj_group_df_frame_ids.min()
            frame_id_max = traj_group_df_frame_ids.max()
            # -- if the `traj_group_df` is an empty dataframe, `frame_id_min`
            #   and `frame_id_max` will be nan and cause an error. Therefore,
            #   we directly continue here.
            if np.isnan(frame_id_min) or np.isnan(frame_id_max):
                continue

            # If we allow for incomplete trajectories, the range is only
            # limited by the available frame ids. We need at least one
            # context frame (a.k.a the last one) and at least one future frame.
            if allow_incomplete_traj:
                # The lcf takes the minimum frame_id.
                lcf_min = frame_id_min
                # The only future frame takes the maximum frame_id.
                lcf_max = frame_id_max - seq_period
            # If we only allow for complete trajectories, the range is bounded
            # by the number of historical context and future frames.
            else:
                # The first context frame takes the minimum frame_id.
                lcf_min = int(frame_id_min + (context_frames - 1) * seq_period)
                # The last future frame takes the maximum frame_id.
                lcf_max = int(frame_id_max - future_frames * seq_period)

            # We check the minimum and maximum frame_id for lcf.
            if lcf_max < lcf_min:
                logging.warning(
                    f"Trajectory group specified by traj_group_index: "
                    f"{traj_group_index} doesn't have enough frames required "
                    f"by context frame length: {context_frames}, "
                    f"future frame length: {future_frames}, "
                    f"sequence period: {seq_period}. "
                    f"The frames are {unique_tg_frame_ids}. Skipped!"
                )
                continue

            for lcf in range(lcf_min, lcf_max + 1, sample_step):
                # Clip the lower bound and upper bound for frame indices for
                # the current sequence. This is necessary for cases in which
                # `allow_incomplete_traj == True` and will not take effect
                # otherwise.
                min_seq_frame_id = lcf - (context_frames - 1) * seq_period
                max_seq_frame_id = lcf + future_frames * seq_period
                # -- make sure we are using lower and upper bounds that are
                #    seq_period aligned with respect to the last context frame.
                frame_id_min_aligned = (
                    lcf - (lcf - frame_id_min) // seq_period * seq_period
                )
                frame_id_max_aligned = (
                    lcf + (frame_id_max - lcf) // seq_period * seq_period
                )
                min_seq_frame_id = max(frame_id_min_aligned, min_seq_frame_id)
                max_seq_frame_id = min(frame_id_max_aligned, max_seq_frame_id)

                # Since the unique frame_id in the current trajectory group
                # may not be consecutive, we will check whether the lcf and
                # at least one future frame is a valid unique frame_id.
                # If not, we will skip this sequence.
                if not (
                    lcf in unique_tg_frame_ids
                    and lcf + seq_period in unique_tg_frame_ids
                ):
                    logging.warning(
                        f"Sequence "
                        f"{min_seq_frame_id}:{max_seq_frame_id}:{seq_period} "
                        f"in trajectory group with indextraj_group_index: "
                        f"{traj_group_index} does not have a valid last "
                        f"context frame {lcf} and a valid "
                        f"first future frame {lcf + seq_period}. Skipped!"
                    )
                    continue

                # Finally, if we are 100% sure this is a valid sequence, we
                # will build a SeqIndex item for it and add it to the dataset
                # index dictionary.
                seq_frame_slice = slice(
                    min_seq_frame_id, max_seq_frame_id + 1, seq_period
                )
                seq_index = SeqIndex(
                    traj_group_index=traj_group_index,
                    frame_slice=seq_frame_slice,
                )
                out_indices[cur_index] = [seq_index, lcf]
                cur_index += 1

        return out_indices

    @staticmethod
    def filter_df_by_traj_group_index(
        df: DataFrame, traj_group_index: TrajGroupIndex
    ) -> DataFrame:
        """Slice out a smaller DataFrame according to a traj_group_index tuple.

        Args:
            df: a dataframe.
            traj_group_index (TrajGroupIndex): index of a trajectory group.

        Returns:
            DataFrame: a sliced DataFrame whose traj_group_index columns
                correspond to the input traj_group_index.
        """
        map_id, date, data_num, data_version, center_car_id = traj_group_index
        flags = (
            (df["map_id"] == map_id)
            & ((df["date"] == date) | (df["date"] == str(date)))
            & (df["data_num"] == data_num)
            & (df["data_version"] == data_version)
            & (df["center_car_id"] == center_car_id)
        )
        df = df.loc[flags, :]
        return df

    @staticmethod
    def filter_df_by_seq_index(
        df: DataFrame, seq_index: SeqIndex
    ) -> DataFrame:
        """Slice out a trajectory DataFrame according to a SeqIndex.

        Args:
            df: a pandas DataFrame which contains traj_group_index
                columns and frame_id column.
            seq_index: a SeqIndex, (traj_group_index, frame_slice).

        Returns:
            DataFrame: a sliced DataFrame whose traj_group_index columns
                correspond to the input SeqIndex.
        """
        # Slice out the DataFrame for the current trajectory group.
        traj_group_idx = seq_index.traj_group_index
        df = BaseTrajDataset.filter_df_by_traj_group_index(df, traj_group_idx)

        # We filter the DataFrame of current trajectory group with the desired
        # frame indices of the current trajectory. Note that out_df may not
        # have enough frames compared to seq_length this is an intentional
        # choice, non-existing frame_ids correspond to empty DataFrames.
        frame_slice = seq_index.frame_slice
        seq_period = frame_slice.step
        seq_length = frame_slice.stop - frame_slice.start
        frame_offset = df["frame_id"] - frame_slice.start
        out_df = df.loc[
            (frame_offset >= 0)
            & (frame_offset < seq_length)
            & (frame_offset % seq_period == 0)
        ]
        return out_df

    @staticmethod
    def add_ego_as_obs(seq_df: DataFrame, veh_type_id: int) -> DataFrame:
        """Pivot the information of the ego vehicle as a new object row.

        This method pivot the information about the ego vehicle to columns
        that correspond to obstacle information so that the ego vehicle can
        be viewed as a separate obstacle.

        Args:
            seq_df: the original DataFrame for the sequence.
            veh_type_id: the type id of the vehicle class.

        Returns:
            seq_df: the transformed DataFrame for the sequence.
        """
        # All ego information is already inside the df, for convenience, we
        # add 1 additional row for every timestamp and give the ego car a track
        # id -self.ANSWER and the class id `veh_type_id`.
        _df = seq_df.drop_duplicates(subset=["frame_id"]).copy()
        # -- substitute state cols
        _df[BaseTrajDataset.OBS_COLS[2:]] = _df[BaseTrajDataset.EGO_COLS[1:]]
        _df["track_id"] = -BaseTrajDataset.ANSWER
        _df["classification"] = veh_type_id

        # Merge the rows for ego vehicle into the original seq_df.
        # Drop rows with track_ids == -BaseTrajDataset.ANSWER to avoid
        # duplicate.
        if any(seq_df.track_id == -BaseTrajDataset.ANSWER):
            logging.warning(
                f"Found objects with track_id == {-BaseTrajDataset.ANSWER}, "
                f"dropping it."
            )
            seq_df = seq_df[seq_df.track_id != -BaseTrajDataset.ANSWER]
        seq_df = pd.concat([seq_df, _df])
        # Drop rows containing NaNs track_ids.
        #   NaNs were originally added in to maintain the DataFrame data
        #   structure in cases where there are no obstacles to fill in columns
        #   that should store obstacle information. Since now we have at least
        #   one obstacle, the ego vehicle, we no longer need those NaN
        #   rows. To filter out those rows, we need to find a column that
        #   cannot be NaN valued when this is any obstacles at all. `track_id`
        #   seems like a good choice.
        seq_df.dropna(inplace=True, subset=["track_id"])

        return seq_df


@OBJECT_REGISTRY.register
class AutoMultiAgentDataset(BaseTrajDataset):
    """AutoUrban Multi Agent Dataset."""

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
        """Initialize method.

        Args:
            df_path: the path to a TDT-format DataFrame.
            traj_group_indices (List[TrajGroupIndex]): a list of TrajGroupIndex
                obtained from dataset config file.
            seq_length: length of the sequence w.r.t the original
                un-sampled frame index.
            seq_period: sampling period of the sequence.
            sample_step: sample step_size for the generated list of
                sequences. This value usually changes between training and
                validation.
            context_frames: number of historical frames to use as
                temporal context for prediction.
            veh_type_id: the type id of the vehicle class in the input
                dataframe.
            transforms: torch-style transform. Defaults to
                None.
            allow_incomplete_traj (bool, optional): whether or not to allow the
                trajectory to stretch outside of the minimum and maximum frame
                index range and become incomplete. Defaults to False.
            ego_as_obs (bool, optional): whether to add an additional row in
                the DataFrame of each sample to treat ego vehicle as one of the
                dynamic objects. Defaults to False.
            have_index_col (bool, optional): whether the csv file have the
                index column. Default to False.
            load_df_func: the function to load
                dataframes from the tdt csv in `df_path`. Default to None.
            max_stamp_thr (float, optional): max timestamp diff between two
                adjacent frames [ms]. If the value None, the timestamp check
                will be skipped.
            path_prefix: path prefix using for loading image.
            check_map_exist_func: custom function to
                check whether the map of a sample exists. Defaults to None.
        """
        self.path_prefix = path_prefix
        check_map = check_map_exist_func is not None

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
        }
        super(AutoMultiAgentDataset, self).__init__(**all_kwargs)
        if max_stamp_thr is not None:
            self.filter_bad_sample_miss_frames(max_stamp_thr)
        if check_map:
            self.filter_bad_sample_miss_map(check_map_exist_func)

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

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def filter_bad_sample_miss_frames(self, max_stamp_thr: float):
        """Filter out the samples that misses some frames.

        These bad samples may cause the time stamps discontinuous, resulting in
        unreasonable long trajectories.

        Args:
            max_stamp_thr (float): max timestamp difference between two
                adjacent frames.

        Returns:
            filtered dataset indices.
        """
        new_index = []
        for i, item in self.df_indices.items():
            seq_index, _ = item
            traj_group_index = seq_index.traj_group_index
            traj_group_df = BaseTrajDataset.filter_df_by_traj_group_index(
                self.df, traj_group_index
            )
            seq_df = BaseTrajDataset.filter_df_by_seq_index(
                traj_group_df, seq_index
            )
            df_stamp = seq_df.drop_duplicates(
                subset=["timestamp"], keep="first"
            )
            df_stamp = df_stamp.timestamp.values
            df_stamp_diff = df_stamp[1:] - df_stamp[:-1]
            if df_stamp_diff.max() <= max_stamp_thr:
                new_index.append(i)
        new_indices = {
            i: self.df_indices[new_index[i]] for i in range(len(new_index))
        }
        self.df_indices = new_indices

    def filter_bad_sample_miss_map(self, check_map_exist_func: Callable):
        """Filter the bad samples that miss map information.

        Args:
            check_map_exist_func: the function to check whether
                the map of a certain sample exists.

        Returns:
            filtered dataset indices.
        """
        new_index = []
        for i, item in self.df_indices.items():
            seq_index, lcf = item
            traj_group_index = seq_index.traj_group_index
            traj_group_df = BaseTrajDataset.filter_df_by_traj_group_index(
                self.df, traj_group_index
            )
            seq_df = BaseTrajDataset.filter_df_by_seq_index(
                traj_group_df, seq_index
            )
            seq_lcf_mask = seq_df["frame_id"] == lcf
            lcf_timestamp = str(
                int(seq_df.loc[seq_lcf_mask, "timestamp"].tolist()[0])
            )
            kwargs = {
                "dataset_prefix": self.path_prefix,
                "traj_group_index": traj_group_index,
                "timestamp": lcf_timestamp,
            }
            if check_map_exist_func(**kwargs):
                new_index.append(i)
        new_indices = {
            i: self.df_indices[new_index[i]] for i in range(len(new_index))
        }
        self.df_indices = new_indices


@OBJECT_REGISTRY.register
class AutoMultiAgentNaviDataset(AutoMultiAgentDataset):
    """AutoUrban Multi Agent Dataset with navigation information."""

    STRUCTURAL_KEYS = [
        "roadedge",
        "stopline",
        "crosswalk",
        "solid_lane",
        "logical_lane",
        "virtuallanelines",
        "virtual_lane",
        "zone",
        "parking_slot",
    ]

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
        max_stamp_thr: float = 550,
        path_prefix: str = "",
        check_map_exist_func: Optional[Callable] = None,
        # Below are customized parameters
        use_navi_info: bool = True,
        navi_save_mode: str = "lmdb",
        navi_file_path_func: Optional[Callable] = None,
        navi_file_dir: Optional[str] = None,
        navi_info_path_mapping: Optional[Dict] = None,
        navi_lmdb_path: Optional[Dict] = None,
        # Below are parameters for structural road info
        use_struct_road_info: bool = False,
        struct_road_save_mode: str = "lmdb",
        struct_road_lmdb_path: Optional[Dict] = None,
    ):
        """Initialize method.

        Args:
            the other parameters refer to the doc string of the base class.
            use_navi_info: whether to return navigation information
                when calling "__getitem__".
            navi_save_mode: the mode to save navigation information.
                We recommend to use mode 'lmdb'.
            navi_file_path_func: the function to generate the
                path to the navigation information file.
            navi_file_dir: the root path of the navigation files.
            navi_info_path_mapping: the mapping between date token
                and navigation file path.
            navi_lmdb_path: the path to save or load the navigation lmdb
                database. It will only be used when navi_save_mode == "lmdb".
            use_navi_info: whether to return structural road information
                when calling "__getitem__".
            struct_road_save_mode: the mode to save structural road
                information. We recommend to use mode 'lmdb'.
            struct_road_lmdb_path: the path to save or load the
                structural road lmdb database.
        """
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
        # Parameters for navigation information.
        self.navi_save_mode = navi_save_mode
        self.navi_file_path_func = navi_file_path_func
        self.navi_file_dir = navi_file_dir
        self.navi_info_path_mapping = navi_info_path_mapping
        self.navi_lmdb_path = navi_lmdb_path

        # Parameters for structural road map information.
        self.struct_road_save_mode = struct_road_save_mode
        self.struct_road_lmdb_path = struct_road_lmdb_path

        # Navigation info and strucal road info  shares the same
        # save path.
        if (
            navi_file_path_func is not None
            and navi_file_dir is not None
            and navi_info_path_mapping is not None
        ):
            self.use_navi_info = use_navi_info
            if self.use_navi_info:
                if navi_save_mode == "local":
                    self.navi_info = None
                elif navi_save_mode == "lmdb":
                    self.navi_lmdb_reader = None
                else:
                    raise ValueError(
                        f"Undefined save mode {navi_save_mode} for navigation "
                        "information."
                    )
            self.use_struct_road_info = use_struct_road_info
            if self.use_struct_road_info:
                if struct_road_save_mode == "local":
                    self.struct_road_info = None
                elif struct_road_save_mode == "lmdb":
                    self.struct_road_lmdb_reader = None
                else:
                    raise ValueError(
                        f"Undefined save mode {struct_road_save_mode} for "
                        "structural road information."
                    )
        else:
            self.use_navi_info = False
            self.use_struct_road_info = False

        super(AutoMultiAgentNaviDataset, self).__init__(**all_kwargs)

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

        last_context_frame = sample["last_context_frame_id"]
        seq_lcf_mask = sample["seq_df"]["frame_id"] == last_context_frame
        lcf_timestamp = (
            sample["seq_df"].loc[seq_lcf_mask, "timestamp"].tolist()[0]
        )
        lcf_timestamp = str(int(lcf_timestamp))
        if self.use_navi_info:
            if self.navi_save_mode == "local" and self.navi_info is not None:
                sample["navi_info"] = self.navi_info[lcf_timestamp]
            elif self.navi_save_mode == "lmdb":
                if self.navi_lmdb_reader is None:
                    lmdb_path = os.path.join(
                        self.navi_file_dir, self.navi_lmdb_path
                    )
                    self.navi_lmdb_reader = Lmdb(
                        lmdb_path, False, readonly=True
                    )
                navi_info = self.navi_lmdb_reader.read(lcf_timestamp)
                if navi_info:
                    navi_info = json.loads(navi_info.decode())
                    for _, data in navi_info.items():
                        data["bounding_box"] = np.array(
                            data["bounding_box"]
                        ).reshape([-1, 3])
                        data["drive_line"] = np.array(
                            data["drive_line"]
                        ).reshape([-1, 3])
                    sample["navi_info"] = navi_info
                else:
                    sample["navi_info"] = {}

        if self.use_struct_road_info:
            if (
                self.struct_road_save_mode == "local"
                and self.struct_road_info is not None
            ):
                sample["struct_road"] = self.struct_road_info[lcf_timestamp]
            elif self.struct_road_save_mode == "lmdb":
                if self.struct_road_lmdb_reader is None:
                    lmdb_path = os.path.join(
                        self.navi_file_dir, self.struct_road_lmdb_path
                    )
                    if "/bucket/output" in lmdb_path:
                        lmdb_path = lmdb_path.replace(
                            "/bucket/output", "/horizon-bucket"
                        )
                    if "02_user/wenke.wang" in lmdb_path:
                        lmdb_path = lmdb_path.replace(
                            "02_user/wenke.wang", "03_pack_package/wenke.wang"
                        )
                    self.struct_road_lmdb_reader = Lmdb(
                        lmdb_path, False, readonly=True, map_size=1e9
                    )
                    struct_road = self.struct_road_lmdb_reader.read(
                        lcf_timestamp
                    )
                    self.struct_road_lmdb_reader.close()
                    self.struct_road_lmdb_reader = None
                else:
                    struct_road = self.struct_road_lmdb_reader.read(
                        lcf_timestamp
                    )
                if struct_road:
                    struct_road = json.loads(struct_road.decode())
                    sample["struct_road"] = struct_road
                    # If use pipeline v5, navi_info is saved in struct_road.
                    if struct_road.get("navi"):
                        navi_info = struct_road["navi"]
                        for _, data in navi_info.items():
                            if data.get("bounding_box"):
                                data["bounding_box"] = np.array(
                                    data["bounding_box"]
                                ).reshape([-1, 3])
                            if data.get("drive_line"):
                                data["drive_line"] = np.array(
                                    data["drive_line"]
                                ).reshape([-1, 3])
                        sample["navi_info"] = navi_info
                    else:
                        sample["navi_info"] = {}
                else:
                    sample["struct_road"] = {}
                    sample["navi_info"] = {}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def get_navi_information(self, item: SeqIndex, save_mode: str):
        """Get navigation information.

        Args:
            item: the key parameters to split dataframes
                from the dataset.

        Returns:
            lcf_timestamp: the time stamp of the navigation
                information.
            navi_info: the navigation information.
        """
        seq_index, last_context_frame = item
        traj_group_index = seq_index.traj_group_index
        traj_group_df = BaseTrajDataset.filter_df_by_traj_group_index(
            self.df, traj_group_index
        )
        seq_df = BaseTrajDataset.filter_df_by_seq_index(
            traj_group_df, seq_index
        )
        seq_lcf_mask = seq_df["frame_id"] == last_context_frame
        lcf_timestamp = seq_df.loc[seq_lcf_mask, "timestamp"].tolist()[0]
        lcf_timestamp = str(int(lcf_timestamp))
        date_token = traj_group_index[1]
        plate = self.path_prefix
        date_key = plate + date_token.split("-")[0]
        navi_file = self.navi_file_path_func(
            self.navi_file_dir,
            self.navi_info_path_mapping[date_key],
            plate,
            date_token,
            lcf_timestamp,
            "/lane2driveline.yaml",
        )
        navi_info = None
        with open(navi_file, "r") as f:
            navi_info = yaml.load(f, Loader=yaml.FullLoader)

        if save_mode == "local":
            for _, data in navi_info.items():
                data["bounding_box"] = np.array(data["bounding_box"]).reshape(
                    [-1, 3]
                )
                data["drive_line"] = np.array(data["drive_line"]).reshape(
                    [-1, 3]
                )
                if data["lane_mark"]:
                    for _, value in data["lane_mark"].items():
                        value["bounding_box"] = np.array(
                            value["bounding_box"]
                        ).reshape([-1, 3])
            return lcf_timestamp, navi_info
        elif save_mode == "lmdb":
            cur_navi = {}
            cur_navi["key"] = lcf_timestamp
            cur_navi["value"] = json.dumps(navi_info)
            return cur_navi
        else:
            raise ValueError(
                f"Undefined save mode {save_mode} for navigation information."
            )

    def get_offline_navi_info(self):  # noqa: D205,D400
        """Traverse the dataset and get the navigation information
        for each samples.
        """
        all_results = []
        with ProcessPoolExecutor(max_workers=16) as pool:
            with tqdm(
                desc="Getting navi info",
                total=len(self.df_indices),
                unit="samples",
            ) as progress_bar:
                for _, item in self.df_indices.items():
                    result = pool.submit(
                        self.get_navi_information, item, self.navi_save_mode
                    )
                    all_results.append(result)
                # Check progress.
                for _ in as_completed(all_results):
                    progress_bar.update(n=1)

        if self.navi_save_mode == "local":
            self.navi_info = {}
            for res in all_results:
                lctx_stamp, navi_info = res.result()
                self.navi_info[lctx_stamp] = navi_info
        elif self.navi_save_mode == "lmdb":
            navi_info = [res.result() for res in all_results]
            lmdb_path = os.path.join(self.navi_file_dir, self.navi_lmdb_path)
            assert lmdb_path is not None
            os.makedirs(lmdb_path, exist_ok=True)
            packer = Navi2LMDB(navi_info, lmdb_path)
            packer()

    def get_structural_road_information(self, item: SeqIndex, save_mode: str):
        """Get structural road information.

        Args:
            item: the key parameters to split dataframes
                from the dataset.

        Returns:
            lcf_timestamp: the time stamp.
            structural_data: the structural road information.
        """
        seq_index, last_context_frame = item
        traj_group_index = seq_index.traj_group_index
        traj_group_df = BaseTrajDataset.filter_df_by_traj_group_index(
            self.df, traj_group_index
        )
        seq_df = BaseTrajDataset.filter_df_by_seq_index(
            traj_group_df, seq_index
        )
        seq_lcf_mask = seq_df["frame_id"] == last_context_frame
        lcf_timestamp = seq_df.loc[seq_lcf_mask, "timestamp"].tolist()[0]
        lcf_timestamp = str(int(lcf_timestamp))
        date_token = traj_group_index[1]
        plate = self.path_prefix
        date_key = plate + date_token.split("-")[0]
        navi_file = self.navi_file_path_func(
            self.navi_file_dir,
            self.navi_info_path_mapping[date_key],
            plate,
            date_token,
            lcf_timestamp,
            "/",
        )
        structural_data = {}
        for key in self.STRUCTURAL_KEYS:
            file_name = glob(os.path.join(navi_file, key + "*.txt"))
            if len(file_name):
                structural_data[key] = self.load_structural_txt_info(
                    file_name[0], key
                )
            else:
                structural_data[key] = []

        if save_mode == "local":
            return lcf_timestamp, structural_data
        elif save_mode == "lmdb":
            cur_struct = {}
            cur_struct["key"] = lcf_timestamp
            cur_struct["value"] = json.dumps(structural_data)
            return cur_struct
        else:
            raise ValueError(
                f"Undefined save mode {save_mode} for navigation information."
            )

    @staticmethod
    def load_structural_txt_info(
        file_name: str, key: str, out_type="list"
    ) -> List:
        """Load structural information from txt files.

        Args:
            file_name: the path to the structural information txt file.
            key: the key of the structural road element.
            out_type (str, optional): the output type of the data arrays.
                Defaults to "list". This value must be "list" when the mode
                is "lmdb". If the user uses the "local" mode, this value could
                be "numpy".

        Returns:
            polylines: the polylines.
        """
        with open(file_name, "r") as f:
            data = f.read()
        data = data.split("\n")
        data = [i for i in data if i != ""]
        i = 0
        polylines = []
        cur_polyline = []
        while i <= len(data):
            if i == len(data) or key in data[i]:
                if len(cur_polyline):
                    if out_type == "numpy":
                        polylines.append(np.stack(cur_polyline))
                    else:
                        polylines.append(cur_polyline)
                    cur_polyline = []
            elif "p1" in data[i]:
                str1 = data[i]
                str2 = data[i + 1]
                i += 1
                assert "p2" in str2, "p2 error in polyline vectors."
                poly_vector = np.stack(
                    [
                        np.array(str1.split(" ")[1:]).astype("float"),
                        np.array(str2.split(" ")[1:]).astype("float"),
                    ],
                    axis=1,
                )
                if out_type != "numpy":
                    poly_vector = poly_vector.tolist()
                cur_polyline.append(poly_vector)
            else:
                raise ValueError("p1 error in polyline vectors.")
            i += 1
        return polylines

    def get_offline_structural_road_info(self):  # noqa: D205,D400
        """Traverse the dataset and get the structural road information
        for each samples.
        """
        all_results = []
        with ProcessPoolExecutor(max_workers=16) as pool:
            with tqdm(
                desc="Getting structural road info",
                total=len(self.df_indices),
                unit="samples",
            ) as progress_bar:
                for _, item in self.df_indices.items():
                    result = pool.submit(
                        self.get_structural_road_information,
                        item,
                        self.struct_road_save_mode,
                    )
                    all_results.append(result)
                # Check progress.
                for _ in as_completed(all_results):
                    progress_bar.update(n=1)

        if self.struct_road_save_mode == "local":
            self.struct_road_info = {}
            for res in all_results:
                lctx_stamp, struct_road = res.result()
                self.struct_road_info[lctx_stamp] = struct_road
        elif self.struct_road_save_mode == "lmdb":
            struct_road_info = [res.result() for res in all_results]
            lmdb_path = os.path.join(
                self.navi_file_dir, self.struct_road_lmdb_path
            )
            assert lmdb_path is not None
            os.makedirs(lmdb_path, exist_ok=True)
            packer = Navi2LMDB(struct_road_info, lmdb_path)
            packer()


class Navi2LMDB(Packer):
    """Navi2LMDB is used for converting extracted navi info to LMDB."""

    def __init__(
        self,
        data: Dict,
        target_data_dir: str,
    ):
        """Initialize method.

        Args:
            data: the navigation information.
            target_data_dir: Path for LMDB file.
        """
        nums = len(data)
        self.all_sample = data
        lmdb_kwargs = {
            "map_size": 1099511627776 * 2,
            "meminit": nums,
            "map_async": True,
        }
        super(Navi2LMDB, self).__init__(
            uri=target_data_dir,
            max_data_num=len(self.all_sample),
            pack_type="lmdb",
            num_workers=1,
            **lmdb_kwargs,
        )

    def _write(self, idx: int, data: object):
        """Write data to target format file.

        Args:
            idx: Idx for writing.
            data : Processed data for writing to target PackType.
        """
        idx = data["key"]
        data = data["value"].encode()
        return super()._write(idx, data)

    def pack_data(self, idx: int):
        """Read original data from Folder with some process.

        Args:
            idx: Idx for reading.

        Returns:
            Processed data for pack.
        """
        return self.all_sample[idx]


@OBJECT_REGISTRY.register
class ConcatTdtDataset(torch.utils.data.ConcatDataset):  # noqa: D205,D400
    """A wrapper Dataset to enable getting the global dataset index
    from the concatenation of multiple datasets.

    The datasets to be concatenated here must take TDT (Trajectory Dataset
    Template) format csv files as the input dataset.

    When building a ConcatTdtDataset instance by more than one datasets,
    this class can automatically assign the index of each dataset. Please
    make sure all the dataset instances have the function
    `set_concat_dataset_index`, otherwise, the indices can not be
    sucessfully assigned and remain the default value (0).
    """

    TRAJ_PRED_DATASET_MAPPING = {
        "base": BaseTrajDataset,
        "autourban": AutoMultiAgentDataset,
        "autourban_navi": AutoMultiAgentNaviDataset,
    }

    def __init__(
        self,
        prefix: PathLike,
        files: list,
        dataset_name: str,
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
        *args,
        **kwargs,
    ):
        """Initialize method.

        Args:
            prefix: prefix of dataset path.
            files (list of dict): the list of the dataset to concatenate,
                each item is a dict with the following keys:
                'df': the name of the TDT file.
                'traj_indices' (list of TrajGroupIndex): the group indices
                    of the TDT file.
            dataset_name: the name of the used dataset. It should be
                one key in self.TRAJ_PRED_DATASET_MAPPING.
            seq_length: length of the sequence w.r.t the original
                un-sampled frame index.
            seq_period: sampling period of the sequence.
            sample_step: sample step_size for the generated list of
                sequences. This value usually changes between training and
                validation.
            context_frames: number of historical frames to use as
                temporal context for prediction.
            veh_type_id: the type id of the vehicle class in the input
                dataframe.
            transforms: torch-style transform. Defaults to
                None.
            allow_incomplete_traj (bool, optional): whether or not to allow the
                trajectory to stretch outside of the minimum and maximum frame
                index range and become incomplete. Defaults to False
            ego_as_obs (bool, optional): whether to add an additional row in
                the DataFrame of each sample to treat ego vehicle as one of the
                dynamic objects.
            veh_type_id: the type id of the vehicle class in
                the input dataframe. Default to 8.
            have_index_col (bool, optional): whether the csv file have the
                index column. Default to False.
            load_df_func: the function to load
                dataframes from the TDT csv in `df_path`. Default to None.
                If it is None, the class will call the internal loading
                function `_load_df_func`. The `load_df_func` takes `df_path`
                and `have_index_col` as input.
            args, kwargs: this is a general concat class for TDT dataset
                (based on `BaseTrajDataset` or its sub classes). Only common
                parameters (of `BaseTrajDataset`) are listed above. The other
                parameters of subclass should input by args or kwargs,
                according to the specific class definitions.
        """
        assert (
            dataset_name in self.TRAJ_PRED_DATASET_MAPPING
        ), f"Undefined dataset class {dataset_name}."
        dataset_class = self.TRAJ_PRED_DATASET_MAPPING[dataset_name]
        datasets = []
        for file_info in files:
            df_name = file_info["df"]
            traj_group_indices = file_info["traj_indices"]

            dataset = dataset_class(
                df_path=os.path.join(prefix, df_name),
                traj_group_indices=traj_group_indices,
                seq_length=seq_length,
                seq_period=seq_period,
                sample_step=sample_step,
                context_frames=context_frames,
                veh_type_id=veh_type_id,
                transforms=transforms,
                allow_incomplete_traj=allow_incomplete_traj,
                ego_as_obs=ego_as_obs,
                have_index_col=have_index_col,
                load_df_func=load_df_func,
                *args,
                **kwargs,
            )
            datasets.append(dataset)
        super(ConcatTdtDataset, self).__init__(datasets)
        for idx, d in zip([0] + self.cumulative_sizes[:-1], self.datasets):
            if hasattr(d, "set_concat_dataset_index"):
                d.set_concat_dataset_index(idx)

    def __repr__(self):
        return "ConcatTdtDataset"


@OBJECT_REGISTRY.register
class PickledTdtDataset(torch.utils.data.Dataset):
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
        if type(self.loaded_ds) is ConcatTdtDataset:
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

    def __getitem__(self, index: int) -> Dict:
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
        return "PickledTdtDataset"


@OBJECT_REGISTRY.register
class TrajPredLMDBDateset(torch.utils.data.Dataset):
    """A wrapper Dataset to enable loading traj pred data from LMDB.

    This utility class is useful when you find a dataframe is too large
    to load in memory when training on cluster. We load bev images and
    necessary labels (e.g. seq_df, seq_index) from LMDB.

    For better data compatiblity, we pack bev image and trajectory labels
    from two lmdbs, and use unified keys to query them. Thus the directory
    'lmdb_root' should contain one file (named as param 'key_file_name') and
    two subdirectories (named as param 'image_dir_name' and 'label_dir_name').

    The file 'key_file_name' is used to save all keys of lmdb samples, it is
    stored like: \
        "1638774515500
         1638774516000
         ...
        " \
    The two lmdb subdirectories is generated by class 'BevImage2LMDB' and
    'Json2LMDB'. Note you need to run 'BaseTrajDataset' or
    'AutoMultiAgentDataset' first to get seq_df and seq_index before running
    'Json2LMDB'. More details please refer to the unit test.
    """

    def __init__(
        self,
        lmdb_root: PathLike,
        key_file_name: str,
        image_dir_name: str,
        label_dir_name: str,
        transforms: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            lmdb_root: data path, should contains a file named as
                "key file name", two subdirectories named as "image_dir_name"
                and "label_dir_name".
            key_file_name: the name of file which contains all lmdb
                samples' keys.
            image_dir_name: the name of subdirectory which contains bev
                image lmdb files generated by "BevImage2LMDB".
            label_dir_name: the name of subdirectory which contains label
                lmdb files generated by "Json2LMDB".
            transforms: torch-style transform. Defaults to
                None.
        """
        data_list_path = os.path.join(lmdb_root, key_file_name)
        assert os.path.exists(data_list_path)
        with open(data_list_path, "r") as f:
            all_data = f.readlines()

        self.bev_lmdb_path = os.path.join(lmdb_root, image_dir_name)
        assert os.path.exists(self.bev_lmdb_path)
        self.label_lmdb_path = os.path.join(lmdb_root, label_dir_name)
        assert os.path.exists(self.label_lmdb_path)

        self.label_lmdb = Lmdb(self.label_lmdb_path, False, readonly=True)
        self.bev_lmdb = Lmdb(self.bev_lmdb_path, False, readonly=True)
        self.transforms = transforms
        self.data_list = []

        for data in all_data:
            self.data_list.append(data.split("\n")[0])

        required_keys = [
            "seq_df",
            "map_id",
            "date",
            "data_num",
            "data_version",
            "center_car_id",
            "frame_slice_start",
            "frame_slice_stop",
            "frame_slice_step",
            "last_context_frame",
            "lcf_timestamp",
        ]
        self.required_keys = set(required_keys)

    def __getstate__(self):
        state = self.__dict__
        state["bev_lmdb"] = None
        state["label_lmdb"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.label_lmdb = Lmdb(self.label_lmdb_path, False, readonly=True)
        self.bev_lmdb = Lmdb(self.bev_lmdb_path, False, readonly=True)

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, index):
        """Return a sample according to a integer index.

        Args:
            index: an integer index.

        Returns:
            sample: a dict which contains information of a
                trajectory sample.
        """
        key = self.data_list[index]
        bev_image = self.bev_lmdb.read(key)
        raw_label = self.label_lmdb.read(key)

        raw_image = msgpack.unpackb(bev_image, raw=False)
        bev_image = np.frombuffer(raw_image, dtype=np.uint8)
        bev_image = cv2.imdecode(bev_image, cv2.IMREAD_GRAYSCALE)

        raw_label = json.loads(raw_label.decode())
        assert self.required_keys == raw_label.keys()

        seq_df = pd.DataFrame(raw_label["seq_df"])
        frame_slice = slice(
            raw_label["frame_slice_start"],
            raw_label["frame_slice_stop"],
            raw_label["frame_slice_step"],
        )
        traj_group_index = TrajGroupIndex(
            raw_label["map_id"],
            raw_label["date"],
            raw_label["data_num"],
            raw_label["data_version"],
            raw_label["center_car_id"],
        )
        seq_index = SeqIndex(traj_group_index, frame_slice)
        sample = {
            "road_map": bev_image,
            "seq_df": seq_df,
            "seq_index": seq_index,
            "last_context_frame_id": raw_label["last_context_frame"],
            "lcf_timestamp": raw_label["lcf_timestamp"],
            "dataset_index": index,
        }
        if self.transforms:
            return self.transforms(sample)
        else:
            return sample


class BevImage2LMDB(Packer):
    """BevImage2LMDB is used for converting bev image to lmdb.

    We use 'bev_json_file' which saved all images'
    absolute paths to generate a lmdb file which named as
    'target_data_dir/target_lmdb_name'.

    In order to avoid loading error, we additionally generate
    a new file named as 'target_key_name' to save all reasonable
    image names using their lmdb keys.
    """

    def __init__(
        self,
        bev_json_file: PathLike,
        target_data_dir: PathLike,
        target_key_name: str,
        target_lmdb_name: str,
        map_height=512,
        map_width=512,
    ):
        """Initialize method.

        Args:
        bev_json_file: the path of a json file which
            saved a dict, the dict keys should be also generated
            lmdb's keys, the dict values should be corresponding
            image absolute paths.
        target_data_dir: the path of lmdb root. Make sure
            target_data_dir is not a bucket path or gpfs path.
        target_key_name: the name of file which saved all
            reasonable lmdb keys.
        target_lmdb_name: the name of generated lmdb directory.
        map_height: bev image height.
        map_width: bev image width.
        """
        self.all_sample = []
        data_list = []
        assert os.path.exists(bev_json_file)
        image_dict = json.load(open(bev_json_file, "rb"))
        for key, value in image_dict.items():
            if not os.path.exists(value):
                logger.warning(f"Image file {value} does not exist!")
                continue
            else:
                im = cv2.imread(value, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    logger.warning(f"Image file {value} does not exist!")
                    continue
                if im.shape != (map_height, map_width):
                    logger.warning(
                        f"Image {value} shape {im.shape} doesn't correct!"
                    )
                    continue
            data_list.append(key)
            pack_img = cv2.imencode(".JPEG", im)[1]
            pack_img = np.asarray(pack_img).astype(np.uint8).tobytes()
            cur_sample = {}
            cur_sample["key"] = key
            cur_sample["value"] = msgpack.packb(pack_img, use_bin_type=True)
            self.all_sample.append(cur_sample)

        # save lmdb keys
        with open(os.path.join(target_data_dir, target_key_name), "w") as f:
            for dk in data_list:
                f.write(dk)
                f.write("\n")

        lmdb_kwargs = {
            "map_size": 1099511627776 * 2,
            "meminit": False,
            "map_async": True,
        }
        super(BevImage2LMDB, self).__init__(
            uri=os.path.join(target_data_dir, target_lmdb_name),
            max_data_num=len(self.all_sample),
            pack_type="lmdb",
            num_workers=1,
            **lmdb_kwargs,
        )

    def _write(self, idx, data):
        idx = data["key"]
        data = data["value"]
        return super()._write(idx, data)

    def pack_data(self, idx):
        return self.all_sample[idx]


@OBJECT_REGISTRY.register
class TrajPredFPVPedDataset(torch.utils.data.Dataset):
    """The FPV dataset class.

    For design detail, please visit:
    https://horizonrobotics.feishu.cn/docs/doccn836Zn3uhlP32FXxMN0rdCp#.
    https://horizonrobotics.feishu.cn/minutes/obcnh41kam1r26nlw1311pmm.
    """

    IMPLEMENTATION_FEATURE_TYPE = ["bbox"]
    OBS_TYPE = ["ped", "cyclist"]
    STAGE = ["train", "val", "test"]

    def __init__(
        self,
        dataset_root_paths: list,
        regen_data_cache: bool,
        feature_type: list,
        obs_type: dict,
        enable_relative_tar_coord: bool = True,
        enable_cvae: bool = True,
        stage: str = "train",
        enc_step: int = 3,
        dec_step: int = 8,
        stride: int = 1,
        overlap_ratio: float = 0.5,
        bbox_type: str = "cxcywh",
        normalize_type: str = "zero-one",
        max_bbox_size: tuple = (1920, 1080, 1920, 1080),
        down_sample_ratio: int = 3,
        feature_dim: int = 4,
        k_value: int = 1,
        cvae_mu: float = 0.0,
        cvae_sigma: float = 0.1,
        latent_dim: int = 32,
    ):
        """Initialize method.

        Args:
            dataset_root_paths: a list of dataset paths that store the
                raw csv data, each of it has the directoty structure as
                follows:
                    dataset_root_path:
                        ---data
                            ---train
                            ---val
                            ---test
                        ---data_cache
                        ---images
                        ---split
            regen_data_cache: if regenerate the pickle files in data_cache
                directory.
            feature_type: the feature type used, only 'bbox' feature is
                used if feature_type = ['bbox'], the 'key_point' feature
                and 'image' feature will be used in future.
            obs_type: the obstacle type used, only pedestrian is used if
                obs_type = {"ped":2}, if you want to use cyclist, it
                should be {"ped":2, "cyclist":18}.
            enable_relative_tar_coord: the target_traj will be coordinates
                relative to the last step input_traj if it is True.
            enable_cvae: the cvae_rand_seed will be generated from torch.
                normal() if it is True, otherwise the cvae_rand_seed will
                be torch.zeros((latent_dim, 1, k_value)).
            stage: the HAT stage, it can be 'train', 'val' or 'test'.
            enc_step: the observe length of a sample.
            dec_step: thr predict length of a sample.
            stride: the stride to generate sample.
            overlap_ratio: the overlap ratio betweeen two neighboring
                sapmles.
            bbox_type: the bbox type, 'cxcywh' default, if is None, the bbox
                will not be converted from x1y1x2y2 to cxcywh.
            normalize_type: the normalize type, 'zero-one' default, if it is
                None, the bbox will not be normalized.
            max_bbox_size: the maximum size of ped bbox in (cx, cy ,w, h).
            down_sample_ratio: the down sample rate to generate samples.
            feature_dim: the ped feature type, default 4.
            k_value: the number of decoder trajectories.
            cvae_mu: the mean value to generate cvae_rand_seed.
            cvae_sigma: the variance to generate cvae_rand_seed.
            latent_dim: the size of the latent variable in cvae module.
        """
        # sanity check.
        for feat in feature_type:
            assert (
                feat in self.IMPLEMENTATION_FEATURE_TYPE
            ), f"Not implementation feature type: {feat}."

        for obs in obs_type.keys():
            assert (
                obs in self.OBS_TYPE
            ), f"Only ped and cyclist are supported now. But found: {obs}."

        assert (
            stage in self.STAGE
        ), f"stage must be train, val or test. But found: {stage}."

        if bbox_type is not None:
            assert (
                bbox_type == "cxcywh"
            ), f"Only cxcywh is supported now. But found: {bbox_type}."

        if normalize_type is not None:
            assert (
                normalize_type == "zero-one"
            ), f"Only zero-one norm is supported now: {normalize_type}."
        for path_ in dataset_root_paths:
            assert os.path.isdir(
                path_
            ), f"The input path: {path_} is invalid, please check it."
        self.dataset_root_paths = dataset_root_paths
        self.regen_data_cache = regen_data_cache
        self.feature_type = feature_type
        self.obs_type = list(map(str, obs_type.values()))
        self.enable_relative_tar_coord = enable_relative_tar_coord
        self.enable_cvae = enable_cvae
        self.stage = stage
        self.enc_step = enc_step
        self.dec_step = dec_step
        self.stride = stride
        self.overlap_ratio = overlap_ratio
        self.bbox_type = bbox_type
        self.normalize_type = normalize_type
        self.max_bbox_size = max_bbox_size
        self.down_sample_ratio = down_sample_ratio
        self.feature_dim = feature_dim
        self.k_value = k_value
        self.cvae_mu = cvae_mu
        self.cvae_sigma = cvae_sigma
        self.latent_dim = latent_dim

        if self.stage == "test":
            self.min_sample_length = int(
                self.down_sample_ratio * self.enc_step
            )
        else:
            self.min_sample_length = int(
                self.down_sample_ratio * (self.enc_step + self.dec_step)
            )

        for idx, path_ in enumerate(dataset_root_paths):
            data_name = path_.split("/")[-1]
            logging.info(f"{data_name} begin")
            if idx == 0:
                self.dataset = self.generate_dataset(path_)
            else:
                dataset = self.generate_dataset(path_)
                for key in self.dataset.keys():
                    self.dataset[key] = self.dataset[key] + dataset[key]
            logging.info(f"{data_name} end")
        logging.info(f"{self.stage} sample numbers: {self.__len__()}")

    def __getitem__(self, index: int):
        """__getitem__ method.

        Args:
            index: the index of a sample.

        Returns:
            ret: a dict type sample.
        """
        # [enc_step, dim] -> [dim, enc_step] -> [dim, 1, enc_step]
        bbox = self.dataset["bbox"][index]
        bbox = torch.FloatTensor(bbox).permute(1, 0).unsqueeze(1)

        raw_bbox = self.dataset["raw_bbox"][index]
        raw_bbox = torch.FloatTensor(raw_bbox).permute(1, 0).unsqueeze(1)

        rand_cvae_seed = torch.zeros((self.latent_dim, 1, self.k_value))
        if self.enable_cvae or self.k_value > 1:
            rand_cvae_seed = torch.normal(
                self.cvae_mu,
                self.cvae_sigma,
                (self.latent_dim, 1, self.k_value),
            )

        target = torch.zeros(self.feature_dim, 1, self.dec_step)
        raw_target = torch.zeros(self.feature_dim, 1, self.dec_step)

        if self.dataset["target"][index] != []:
            target = self.dataset["target"][index]
            target = torch.FloatTensor(target).permute(1, 0).unsqueeze(1)
            raw_target = self.dataset["raw_target"][index]
            raw_target = (
                torch.FloatTensor(raw_target).permute(1, 0).unsqueeze(1)
            )

        stamp = self.dataset["stamp"][index]
        date_token = self.dataset["date_token"][index]
        obstacle_id = self.dataset["id"][index]
        obstacle_type = self.dataset["type"][index]

        position = self.dataset["position"][index]
        position = np.array(position, dtype=float)
        height = self.dataset["height"][index]
        height = np.array(height, dtype=float)
        vcs_vel = self.dataset["vcs_vel"][index]
        vcs_vel = np.array(vcs_vel, dtype=float)
        global_vel = self.dataset["global_vel"][index]
        global_vel = np.array(global_vel, dtype=float)
        ret = {
            "input_traj": bbox,
            "raw_input_traj": raw_bbox,
            "rand_cvae_seed": rand_cvae_seed,
            "target_traj": target,
            "raw_target_traj": raw_target,
            "position": position,
            "height": height,
            "vcs_vel": vcs_vel,
            "global_vel": global_vel,
            "stamp": stamp,
            "date_token": date_token,
            "id": obstacle_id,
            "type": obstacle_type,
            "stage": self.stage,
            "enable_relative": self.enable_relative_tar_coord,
        }

        return ret

    def __len__(self):
        """__len__ method.

        Return the length of the dataset.
        """
        return len(self.dataset[list(self.dataset.keys())[0]])

    def generate_dataset(self, data_path: PathLike):
        """Train and val dataset generate function.

        NOTE[zhanbo01.li]: In the train and val dataset, each sample
        contains the input_x and the corresponding target_y.

        Args:
            data_path: dataset_root_path: the path that store the row
                csv data, it has the directoty structure like this:
                data_path:
                ---data
                   ---train
                   ---val
                   ---test
                ---data_cache
                ---images
                ---split

        Returns:
            samples: the dataset dict contains all samples, and the samples
                are downsampled.
        """
        self.gen_dataset = self.load_fpv_raw_dataset(data_path)
        if self.stage == "test":
            split_sample = self.gen_raw_samples_for_test(data_path)
        else:
            split_sample = self.gen_raw_samples_for_train_val(data_path)

        split_sample["raw_bbox"] = deepcopy(split_sample["bbox"])
        split_sample["bbox"] = self.normalize_bbox(split_sample["bbox"])

        # The final samples are downsampled from split_sample and are
        # stored in downsample_split dict, the length of each downsampled
        # sample is enc_step+dec_step.
        downsample_split = {}

        start = self.down_sample_ratio - 1
        observe_length = int(self.down_sample_ratio * self.enc_step)
        end = start + observe_length
        tar_begin = end
        pred_length = int(self.down_sample_ratio * self.dec_step)
        tar_end = tar_begin + pred_length
        step = self.down_sample_ratio

        for key in split_sample.keys():
            downsample_split[key] = []
            if key == "bbox":
                downsample_split["target"] = []
                for sample in split_sample[key]:
                    observe = np.array(sample[start:end:step])
                    if len(sample) >= (observe_length + pred_length):
                        target = sample[tar_begin:tar_end:step]
                        if self.enable_relative_tar_coord:
                            target = np.array(target) - observe[-1]
                        else:
                            target = np.array(target)
                    else:
                        target = []
                    downsample_split[key].append(observe)
                    downsample_split["target"].append(target)
            elif key == "raw_bbox":
                downsample_split["raw_target"] = []
                for sample in split_sample[key]:
                    observe = sample[start:end:step]
                    downsample_split[key].append(observe)
                    if len(sample) >= (observe_length + pred_length):
                        target = sample[tar_begin:tar_end:step]
                        target = np.array(target)
                    else:
                        target = []
                    downsample_split["raw_target"].append(target)
            else:
                downsample_split[key].extend(
                    sample[start:end:step] for sample in split_sample[key]
                )

        samples = {
            "id": downsample_split["id"],
            "type": downsample_split["type"],
            "date_token": downsample_split["date_token"],
            "stamp": downsample_split["stamp"],
            "bbox": downsample_split["bbox"],
            "raw_bbox": downsample_split["raw_bbox"],
            "target": downsample_split["target"],
            "raw_target": downsample_split["raw_target"],
            "height": downsample_split["height"],
            "position": downsample_split["position"],
            "vcs_vel": downsample_split["vcs_vel"],
            "global_vel": downsample_split["global_vel"],
        }

        return samples

    def load_fpv_raw_dataset(self, data_path: PathLike):
        """Load the fpv raw dataset.

        Args:
            data_path: dataset_root_path: the path that store the row
                csv data, it has the directoty structure like this:
                data_path:
                ---data
                   ---train
                   ---val
                   ---test
                ---data_cache
                ---images
                ---split

        Returns:
            dataset: the fpv raw dataset.
        """
        data_cache_path = os.path.join(
            data_path, "data_cache", self.stage + "_cache.pkl"
        )
        if os.path.exists(data_cache_path) and not self.regen_data_cache:
            with open(data_cache_path, "rb") as data_cache:
                try:
                    dataset = pickle.load(data_cache)
                except pickle.PickleError:
                    dataset = pickle.load(data_cache, encoding="bytes")
        else:
            dataset = {}
            for feat_type in self.feature_type:
                feat_type = feat_type + "_feature"
                dataset[feat_type] = self.get_feature(data_path, feat_type)
            with open(data_cache_path, "wb") as data_cache:
                pickle.dump(dataset, data_cache, 4)

        return dataset

    def get_feature(self, data_path: PathLike, feat_type: str):
        """Entrence function to get different features.

        NOTE[zhanbo01.li]: only the "bbox" type is supported now.

        Args:
            data_path: dataset_root_path: the path that store the row
                csv data, it has the directoty structure like this:
                data_path:
                ---data
                   ---train
                   ---val
                   ---test
                ---data_cache
                ---images
                ---split
            type: thr feature type, it can be 'bbox_feature',
                'key_point_feature' or 'image_feature'.

        Returns:
            feature: the corresponding feature.
        """
        if feat_type == "bbox_feature":
            feature = self.get_bbox_feature(data_path)

        elif feat_type in ["key_point_feature", "image_feature"]:
            logging.error("Not implementation feature type")

        return feature

    def get_bbox_feature(self, data_path: PathLike):
        """Bbox feature extraction function.

        Args:
            data_path: dataset_root_path: the path that store the row
                csv data, it has the directoty structure like this:
                data_path:
                ---data
                   ---train
                   ---val
                   ---test
                ---data_cache
                ---images
                ---split

        Returns:
            bbox_feature: the bbox feature generated from the raw csv files.
        """
        bbox_path = os.path.join(data_path, "data", self.stage)
        bbox_csv_files = os.listdir(bbox_path)
        bbox_feature = {}
        for bbox_csv_file in bbox_csv_files:
            if ".csv" in bbox_csv_file:
                feature_per_csv = {}
                csv_file = os.path.join(bbox_path, bbox_csv_file)
                df_ped = self.read_perception_csv(csv_file, self.obs_type)
                if len(df_ped) == 0:
                    continue
                if self.stage == "train" or self.stage == "val":
                    for ped_id in df_ped.id.unique():
                        df_ped_id = df_ped[df_ped["id"] == ped_id]
                        feature_per_csv[ped_id] = {
                            "stamp": df_ped_id["stamp"].values.tolist(),
                            "bbox": df_ped_id["bbox"].values.tolist(),
                            "type": df_ped_id["type"].values.tolist(),
                            "id": df_ped_id["id"].values.tolist(),
                            "position": df_ped_id[
                                ["vcs_x", "vcs_y"]
                            ].values.tolist(),
                            "height": df_ped_id["height"].values.tolist(),
                            "vcs_vel": df_ped_id[
                                ["vcs_vx", "vcs_vy"]
                            ].values.tolist(),
                            "global_vel": df_ped_id[
                                ["global_vx", "global_vy"]
                            ].values.tolist(),
                        }
                elif self.stage == "test":
                    for ped_stamp in df_ped.stamp.unique():
                        feature_per_csv[ped_stamp] = {}
                        df_ped_stamp = df_ped[df_ped["stamp"] == ped_stamp]
                        for ped_id in df_ped_stamp.id.unique():
                            df_ped_id = df_ped[df_ped["id"] == ped_id]
                            feat = {
                                "stamp": df_ped_id["stamp"].values.tolist(),
                                "bbox": df_ped_id["bbox"].values.tolist(),
                                "type": df_ped_id["type"].values.tolist(),
                                "id": df_ped_id["id"].values.tolist(),
                                "position": df_ped_id[
                                    ["vcs_x", "vcs_y"]
                                ].values.tolist(),
                                "height": df_ped_id["height"].values.tolist(),
                                "vcs_vel": df_ped_id[
                                    ["vcs_vx", "vcs_vy"]
                                ].values.tolist(),
                                "global_vel": df_ped_id[
                                    ["global_vx", "global_vy"]
                                ].values.tolist(),
                            }
                            feature_per_csv[ped_stamp][ped_id] = feat

                token = bbox_csv_file.split(".")[0].split("_")[1:3]
                date_token = token[0] + "_" + token[1]
                bbox_feature[date_token] = feature_per_csv

        return bbox_feature

    def read_perception_csv(self, csv_file: PathLike, obs_type: List):
        """Read the perception csv files.

        Args:
            csv_file: the csv file path, it is generated from the
                perception pack.
            obs_type: a list of the obstacle types used, e.g. ["2", "18"].

        Returns:
            df_ped: a pd.DataFrame storing the features of the selected
                types of obstacle.
        """
        with open(csv_file, "r", encoding="utf-8-sig") as csv_input:
            data = []
            for line in csv_input.readlines():
                line = line.rstrip().replace(", ", "。").split(",")
                for i in range(len((line))):
                    line[i] = line[i].replace("。", ",")
                data.append(line)
            if len(data) > 3:
                df = pd.DataFrame(data[3:], columns=data[2])
                df_ped = df[df["type"].isin(obs_type)]
                df_ped["bbox"] = df_ped["bbox"].apply(
                    lambda x: x.replace('"', "")
                )
                df_ped["bbox"] = [eval(i) for i in df_ped["bbox"].values]
                df_ped["id"] = df_ped["id"].astype("int")
                df_ped["stamp"] = df_ped["stamp"].astype("int")
                df_ped["type"] = df_ped["type"].astype("int")
                if "vcs_x" in df_ped.columns:
                    df_ped["vcs_x"] = df_ped["vcs_x"]
                    df_ped["vcs_y"] = df_ped["vcs_y"]
                else:
                    df_ped["vcs_x"] = None
                    df_ped["vcs_y"] = None
            else:
                df_ped = pd.DataFrame()

        return df_ped

    def normalize_bbox(self, all_bbox: list):
        """Bbox normalize function.

        NOTE[zhanbo01.li]: The input box type is x1y1x2y2 in original
        resolution. The all_bbox are converted to cxcywh first and then
        be normalized. This process may cause quantization accuracy problem
        beacuse the w and h feature is not uniform distribution in
        [0, 1920] and [0, 1080]. If this issue arises, you can optimize
        from here.

        Args:
            all_bbox: the raw bboxes.

        Returns:
            all_bbox: the normalized bboxex.
        """
        for i in range(len(all_bbox)):
            if len(all_bbox[i]) == 0:
                continue
            bbox = np.array(all_bbox[i])
            # x1y1x2y2 to cxcywh
            if self.bbox_type == "cxcywh":
                bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                bbox[..., [0, 1]] += bbox[..., [2, 3]] / 2
            # Normalize bbox
            if self.normalize_type == "zero-one":
                _max = np.array(self.max_bbox_size)
                bbox = bbox / _max
            all_bbox[i] = bbox
        return all_bbox

    def gen_raw_samples_for_test(self, data_path: PathLike):
        """Generate raw samples for test stage.

        NOTE[zhanbo01.li]: these samples genarated here is not downsampled,
        so they are called raw_sample.

        Args:
            data_path: dataset_root_path: the path that store the row
                csv data, it has the directoty structure like this:
                data_path:
                ---data
                   ---train
                   ---val
                   ---test
                ---data_cache
                ---images
                ---split

        Returns:
            split_sample: the raw samples corresponding to the stage split.
        """
        split_sample = {
            "id": [],
            "type": [],
            "date_token": [],
            "stamp": [],
            "bbox": [],
            "position": [],
            "height": [],
            "vcs_vel": [],
            "global_vel": [],
        }

        if "bbox_feature" in self.gen_dataset.keys():
            bbox_featrue = self.gen_dataset["bbox_feature"]
            split_data_files = self.get_split_csv_files(data_path)
            for date_token in split_data_files:
                date_token = date_token.rstrip()
                if date_token not in bbox_featrue.keys():
                    continue
                feature = bbox_featrue[date_token]
                for stamp in feature.keys():
                    for ped_id in feature[stamp].keys():
                        num_history_stamp = 0
                        num_future_stamp = 0
                        for i in feature[stamp][ped_id]["stamp"]:
                            if i <= stamp:
                                num_history_stamp += 1
                            else:
                                num_future_stamp += 1

                        need_his_len = int(
                            self.min_sample_length * self.stride
                        )
                        if num_history_stamp < need_his_len:
                            continue

                        index = feature[stamp][ped_id]["stamp"].index(stamp)
                        start = index - need_his_len + 1

                        need_fut_len = int(
                            self.dec_step
                            * self.down_sample_ratio
                            * self.stride
                        )
                        if num_future_stamp >= need_fut_len:
                            end = index + need_fut_len + 1
                        else:
                            end = index + 1

                        split_sample["id"].append(
                            feature[stamp][ped_id]["id"][
                                start : end : self.stride
                            ]
                        )
                        split_sample["type"].append(
                            feature[stamp][ped_id]["type"][
                                start : end : self.stride
                            ]
                        )
                        split_sample["date_token"].append(
                            [date_token] * len(split_sample["id"][-1])
                        )
                        split_sample["stamp"].append(
                            feature[stamp][ped_id]["stamp"][
                                start : end : self.stride
                            ]
                        )
                        split_sample["bbox"].append(
                            feature[stamp][ped_id]["bbox"][
                                start : end : self.stride
                            ]
                        )
                        split_sample["position"].append(
                            feature[stamp][ped_id]["position"][
                                start : end : self.stride
                            ]
                        )
                        split_sample["height"].append(
                            feature[stamp][ped_id]["height"][
                                start : end : self.stride
                            ]
                        )
                        split_sample["vcs_vel"].append(
                            feature[stamp][ped_id]["vcs_vel"][
                                start : end : self.stride
                            ]
                        )
                        split_sample["global_vel"].append(
                            feature[stamp][ped_id]["global_vel"][
                                start : end : self.stride
                            ]
                        )

        return split_sample

    def gen_raw_samples_for_train_val(self, data_path: PathLike):
        """Generate raw samples for train and val stage.

        NOTE[zhanbo01.li]: these samples genarated here is not downsampled,
        so they are called raw_sample.

        Args:
            data_path: dataset_root_path: the path that store the row
                csv data, it has the directoty structure like this:
                data_path:
                ---data
                   ---train
                   ---val
                   ---test
                ---data_cache
                ---images
                ---split

        Returns:
            split_sample: the raw samples corresponding to the stage split.
        """
        split_dataset = {
            "id": [],
            "type": [],
            "date_token": [],
            "stamp": [],
            "bbox": [],
            "position": [],
            "height": [],
            "vcs_vel": [],
            "global_vel": [],
        }
        if "bbox_feature" in self.gen_dataset.keys():
            bbox_featrue = self.gen_dataset["bbox_feature"]
            split_data_files = self.get_split_csv_files(data_path)
            for date_token in split_data_files:
                date_token = date_token.rstrip()
                if date_token not in bbox_featrue.keys():
                    continue
                feature = bbox_featrue[date_token]
                for ped_id in feature.keys():
                    obs_len = int(len(feature[ped_id]["bbox"]) / self.stride)
                    if obs_len < self.min_sample_length:
                        continue
                    split_dataset["id"].append(
                        feature[ped_id]["id"][:: self.stride]
                    )
                    split_dataset["type"].append(
                        feature[ped_id]["type"][:: self.stride]
                    )
                    split_dataset["date_token"].append(
                        [date_token] * len(split_dataset["id"][-1])
                    )
                    split_dataset["stamp"].append(
                        feature[ped_id]["stamp"][:: self.stride]
                    )
                    split_dataset["bbox"].append(
                        feature[ped_id]["bbox"][:: self.stride]
                    )
                    split_dataset["position"].append(
                        feature[ped_id]["position"][:: self.stride]
                    )
                    split_dataset["height"].append(
                        feature[ped_id]["height"][:: self.stride]
                    )
                    split_dataset["vcs_vel"].append(
                        feature[ped_id]["vcs_vel"][:: self.stride]
                    )
                    split_dataset["global_vel"].append(
                        feature[ped_id]["global_vel"][:: self.stride]
                    )

        # The samples generated from split_dataset are stored in
        # split_sample dict, the length of each sample are
        # min_sample_length.
        split_sample = {}
        observe_length = int(self.down_sample_ratio * self.enc_step)
        overlap = max(int(observe_length * self.overlap_ratio), 1)
        min_length = self.min_sample_length
        for key in split_dataset.keys():
            tracks = []
            for track in split_dataset[key]:
                track_ = []
                for i in range(0, len(track) - min_length + 1, overlap):
                    track_.append(track[i : i + min_length])
                tracks.extend(track_)
            split_sample[key] = tracks

        return split_sample

    def get_split_csv_files(self, data_path: PathLike):
        """Get the csv files name.

        Args:
            data_path: the path that store the row csv data, it
                has the directoty structure like this:
                data_path:
                ---data
                   ---train
                   ---val
                   ---test
                ---data_cache
                ---images
                ---split

        Returns:
            split_data_files: a list contains all the csv file's names.
        """
        split_data = os.path.join(
            data_path,
            "split",
            self.stage + ".txt",
        )
        with open(split_data, "rt") as file:
            split_data_files = file.readlines()

        return split_data_files


@OBJECT_REGISTRY.register
class TrajPredJaadDataset(torch.utils.data.Dataset):
    """Build JAAD interface.

    NOTE[xiaoming.zhang]: This class is for JAAD dataset to be
        compatible with the output of TrajPredFPVPedDataset.
    We use a class of jaad to implement this dataset.

    Args:
        data_root: the cached pickle which contains the whole datasets,and
            its structure detail is in
            https://horizonrobotics.feishu.cn/wiki/wikcnRgBi2RKd0rwBPhjx5OQHyb
        enc_steps: observed frames
        dec_steps: predict frames
        split: 'train', 'val', 'test', their split files is in the
                data_root/default named 'train.txt', 'val.txt', 'test.txt'
                respectively with video_ids
        normalize: the normalize type, 'zero-one' default, if it is
                None, the bbox will not be normalized
        bbox_type: the bbox type, 'cxcywh' default, if is None, the bbox
                will not be converted from  ltrb to cxcywh
        FPS: the frame rate of input squence
        overlap: the stride when sampling the frames, when overlap is
        larger, the sampled data becomes less
        latent_dim: the size of the latent variable in cvae module.
        k_value: the number of decoder trajectories.
        box_size: the maximum size of ped bbox in (cx, cy ,w, h)
        enable_cvae: the cvae_rand_seed will be generated from torch
        cvae_mu: the mean value to generate cvae_rand_seed
        cvae_sigma: the variance to generate cvae_rand_seed
        data_token: token for pack, but here default 0
        type: type of obstacle for pack, but here default 2

    Returns:
        split_sample: the raw samples corresponding to the stage split.
    """

    def __init__(
        self,
        data_root: str,
        enc_steps: int,
        dec_steps: int,
        split: str,
        normalize: str = "zero-one",
        bbox_type: str = "cxcywh",
        FPS: int = 5,
        overlap: int = 1,
        latent_dim: int = 32,
        k_value: int = 1,
        box_size: tuple = (1920, 1080, 1920, 1080),
        enable_cvae: bool = True,
        cvae_mu: float = 0.0,
        cvae_sigma: float = 0.1,
        data_token: int = 0,
        type_: int = 2,
    ):
        self.split = split  # trian val test
        self.root = data_root
        self.stage = self.split
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps
        self.FPS = FPS
        self.overlap_stride = overlap
        self.latent_dim = latent_dim
        self.k_value = k_value
        self.cvae_mu = cvae_mu
        self.cvae_sigma = cvae_sigma
        self.enable_cvae = enable_cvae
        self.normalize = normalize
        self.bbox_type = bbox_type
        self.max_bbox_size = box_size
        self.downsample_step = int(30 / self.FPS)
        self.data_token = (data_token,)
        self.ped_type = type_
        data_opts = {
            "fstride": 1,
            "data_split_type": "default",
            "min_track_size": (self.enc_steps + self.dec_steps)
            * self.downsample_step
            + 1,
        }
        traj_model_opts = {
            "observe_length": self.enc_steps,
            "enc_input_type": ["bbox"],
            "dec_input_type": [],
            "prediction_type": ["bbox"],
        }
        jaad = JAAD(data_path=self.root)
        beh_seq = jaad.generate_data_trajectory_sequence(
            self.split, **data_opts
        )
        self.data = self.get_data(beh_seq, **traj_model_opts)

    def __getitem__(self, index: int) -> Dict:
        """__getitem__ method.

        Args:
            index: the index of a sample.

        Returns:
            ret: return the dict of data samples
        """
        obs_bbox = (
            torch.FloatTensor(self.data["obs_bbox"][index])
            .permute(1, 0)
            .unsqueeze(1)
        )
        obs_bbox_non_norm = (
            torch.FloatTensor(self.data["obs_bbox_non_norm"][index])
            .permute(1, 0)
            .unsqueeze(1)
        )
        pred_bbox = torch.FloatTensor(self.data["pred_bbox"][index]).permute(
            2, 0, 1
        )
        pred_bbox_non_norm = torch.FloatTensor(
            self.data["pred_bbox_non_norm"][index]
        ).permute(2, 0, 1)
        obs_pid = itertools.chain.from_iterable(self.data["obs_pid"][index])
        pids = list(obs_pid)
        cur_image_file = self.data["obs_image"][index][-1]
        stage = self.stage
        enable_relative = True
        rand_cvae_seed = torch.zeros((self.latent_dim, 1, self.k_value))
        if self.enable_cvae or self.k_value > 1:
            rand_cvae_seed = torch.normal(
                self.cvae_mu,
                self.cvae_sigma,
                (self.latent_dim, 1, self.k_value),
            )
        ret = {
            "input_traj": obs_bbox,
            "raw_input_traj": obs_bbox_non_norm,
            "target_traj": pred_bbox,
            "raw_target_traj": pred_bbox_non_norm,
            "cur_image_file": cur_image_file,
            "id": pids,
            "stage": stage,
            "enable_relative": enable_relative,
            "rand_cvae_seed": rand_cvae_seed,
        }

        cur_time = int(cur_image_file.split("/")[-1].split(".")[0])
        ret["stamp"] = [
            cur_time + self.downsample_step * i for i in range(self.enc_steps)
        ]
        ret["date_token"] = [self.data_token for _ in range(self.enc_steps)]
        ret["type"] = [self.ped_type for _ in range(self.enc_steps)]
        ret["position"] = [[None, None] for _ in range(self.enc_steps)]
        ret["cur_position"] = [None, None]

        return ret

    def __len__(self):
        """__len__ method.

        Returns:
            int: return the length of the dataset.
        """
        return len(self.data[list(self.data.keys())[0]])

    def normalize_bbox(
        self, all_bbox: list, bbox_type: str, normalize_type: str
    ) -> np.ndarray:
        """Bbox normalize function.

        NOTE[xiaoming.zhang]: In order to adapt this function to the
        interface of the Jaad dataset, the original bbox won't be directly
        rewritten, but return its copy instead.

        Args:
            all_bbox: the raw bboxes.
            bbox_type: 'cxcywh': ltrb to cxcywh
            normalize_type: 'zero-one': normalization between 0-1, or 'None'

        Returns:
            all_bbox : the normalized bboxex.
        """
        duplicate_all_boxes = deepcopy(all_bbox)
        for i in range(len(all_bbox)):
            if len(all_bbox[i]) == 0:
                continue
            bbox = np.array(all_bbox[i])
            # x1y1x2y2 to cxcywh
            if bbox_type == "cxcywh":
                bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                bbox[..., [0, 1]] += bbox[..., [2, 3]] / 2
            # Normalize bbox
            if normalize_type == "zero-one":
                _max = np.array(self.max_bbox_size)
                bbox = bbox / _max

            duplicate_all_boxes[i] = bbox
        return duplicate_all_boxes

    def get_tracks(self, dataset: Dict, data_types: List) -> Dict:
        """Generate tracks by sampling from pedestrian sequences.

        Args:
            dataset: load the generated jaad dataset
            data_types: default 'bbox'

        Returns:
            data_dict: A dictinary containing sampled tracks for
                each data modality
        """
        #  Calculates the overlap in terms of number of frames
        observe_length = self.enc_steps
        predict_length = self.dec_steps
        down_sample = self.downsample_step
        overlap_stride = self.overlap_stride
        seq_length = (observe_length + predict_length) * down_sample
        #  Check the validity of keys selected by user as data type
        data_dict = {}
        for dt in data_types:
            data_dict[dt] = dataset[dt]

        data_dict["image"] = dataset["image"]
        data_dict["pid"] = dataset["pid"]
        data_dict["resolution"] = dataset["resolution"]
        data_dict["bbox_non_norm"] = {}

        #  Sample tracks from sequneces
        for key in data_dict.keys():
            tracks = []
            for track in data_dict[key]:
                tracks.extend(
                    [
                        track[i : i + seq_length]
                        for i in range(
                            0, len(track) - seq_length + 1, overlap_stride
                        )
                    ]
                )
            data_dict[key] = tracks

        if down_sample != 1:
            for key in data_dict.keys():
                track_down = []
                for track in data_dict[key]:
                    track_down.append(track[::down_sample])
                data_dict[key] = track_down

        #  Normalize tracks using FOL paper method,
        data_dict["bbox_non_norm"] = self.normalize_bbox(
            data_dict["bbox"], self.bbox_type, "none"
        )
        data_dict["bbox"] = self.normalize_bbox(
            data_dict["bbox"], self.bbox_type, self.normalize
        )

        return data_dict

    def get_data(self, data: Dict, **model_opts) -> Dict:
        """Generate the data for training/testing.

        Args:
            dataset: data: The raw data
            model_opts: Control parameters for data generation

        Returns: ret: A dictinary containing sampled tracks for each
                data modality
        """

        opts = {
            "observe_length": self.enc_steps,
            "enc_input_type": ["bbox"],
            "dec_input_type": [],
            "prediction_type": ["bbox"],
            "down_samlpe": self.downsample_step,
            "overlap": self.overlap_stride,
            "data_types": ["bbox"],
        }
        for key, value in model_opts.items():
            assert key in opts.keys(), "wrong data parameter %s" % key
            opts[key] = value

        observe_length = opts["observe_length"]
        data_tracks = self.get_tracks(data, opts["data_types"])
        obs_slices = {}
        pred_slices = {}

        #  Generate observation/prediction sequences from the tracks
        for keys in data_tracks.keys():
            obs_slices[keys] = []
            pred_slices[keys] = []
            # NOTE[xiaoming.zhang]: Add downsample function
            down = 1
            if keys in ["bbox", "bbox_non_norm"]:
                gen_target = (
                    self.gen_relative_target
                    if keys == "bbox"
                    else self.gen_raw_target
                )
                start = down - 1
                end = start + observe_length
                observe_list = []
                target_list = []
                for sample in data_tracks[keys]:
                    target = gen_target(sample, end)
                    target_list.append(target)
                    observe = sample[start:observe_length:down]
                    observe_list.append(observe)
                obs_slices[keys].extend(observe_list)
                pred_slices[keys].extend(target_list)

            else:
                obs_slices[keys].extend(
                    [
                        sample[down - 1 : observe_length : down]
                        for sample in data_tracks[keys]
                    ]
                )
        ret = {
            "obs_image": obs_slices["image"],
            "obs_pid": obs_slices["pid"],
            "obs_resolution": obs_slices["resolution"],
            "pred_image": pred_slices["image"],
            "pred_pid": pred_slices["pid"],
            "pred_resolution": pred_slices["resolution"],
            "obs_bbox": np.array(obs_slices["bbox"]),
            "obs_bbox_non_norm": np.array(obs_slices["bbox_non_norm"]),
            "pred_bbox": np.array(pred_slices["bbox"]),
            "pred_bbox_non_norm": np.array(pred_slices["bbox_non_norm"]),
            "model_opts": opts,
        }

        return ret

    def gen_relative_target(self, session: np.ndarray, end: int):
        """Generate future target with the relative coordinate.

        Args:
            session: the sample stride with all the length of a pid
            end: the index that end the observe

        Returns:
            target: predict target shape of
                [1, self.dec_steps, 4]
        """
        predict_length = self.dec_steps
        target = np.zeros((1, predict_length, session.shape[-1]))
        target_start = end
        target[0, :, :] = np.asarray(
            session[target_start : target_start + predict_length, :]
            - session[target_start - 1 : target_start, :]
        )
        return target

    def gen_raw_target(self, session: np.ndarray, end: int) -> np.ndarray:
        """Generate future target without the relative coordinate.

        Args:
            session: the sample stride with all the length of a pid
            end: the index that end the observe

        Returns:
            target: predict target shape of
                [1, self.dec_steps, 4]
        """
        predict_length = self.dec_steps
        raw_target = np.zeros((1, predict_length, session.shape[-1]))
        target_start = end
        raw_target[0, :, :] = np.asarray(
            session[target_start : target_start + predict_length, :]
        )

        return raw_target


@OBJECT_REGISTRY.register
class BaseTrajDatasetV2(torch.utils.data.Dataset):  # noqa: D205,D400
    """Torch based trajectory dataset for TDT (Trajectory Dataset Template)
    format datasets.

    This class is responsible for building a pytorch map-style dataset
    which returns a dictionary containing an index and a pandas DataFrame
    with the __getitem__ method. This class is designed to deal with
    one file at a time, and it serves as the building block for more
    complex datasets.

    This class load customized TDT csv file. Each row of the csv records
    the information of an obstacle in a certain time stamp. The csv file
    has columns for obstacle information (in `OBS_COLS`) and columns for
    odometry of the ego vehicle (in `EGO_COLS`).

    Note: TDT is a csv format to save trajectory datasets. It must have
    all the columns in `BASIC_COLS`, `EGO_COLS` and `OBS_COLS`.
    """

    # DataFrame columns for the ego vehicle and the obstacles.
    BASIC_COLS = [
        "timestamp",
        "frame_id",
    ]
    EGO_COLS = [
        "width",
        "length",
        "pos_x",
        "pos_y",
        "pos_z",
        "yaw",
    ]
    OBS_COLS = [
        "classification",
        "obs_width",
        "obs_length",
        "x",
        "y",
        "z",
        "obs_yaw",
        "abs_vx_global",
        "abs_vy_global",
        "abs_ax_global",
        "abs_ay_global",
    ]
    UNI_COLS = [
        "track_id",
    ]
    STRUCTURAL_KEYS = [
        "roadedge",
        "stopline",
        "crosswalk",
        "solid_lane",
        "logical_lane",
        "virtuallanelines",
        "virtual_lane",
        "zone",
        "parking_slot",
    ]

    # If we add ego vehicle to the obstacle list to predict, we use
    # -ANSWER as its `track_id`.
    ANSWER = 42

    def __init__(
        self,
        df_path: PathLike,
        traj_group_indices: List[TrajGroupIndex],
        path_prefix: str,
        date_token: str,
        last_context_frame: int = 3,
        transforms: Optional[Callable] = None,
        have_index_col: bool = False,
        load_df_func: Optional[Callable] = None,
        struct_road_lmdb_path: Optional[Dict] = None,
        use_vision: bool = True,
        use_lidar: bool = False,
    ):
        """Initialize method.

        Args:
            df_path: the path to a TDT-format DataFrame.
            traj_group_indices (List[TrajGroupIndex]): a list of TrajGroupIndex
                obtained from dataset config file.
            path_prefix: the plate for a pack.
            date_token: the token for a pack.
            last_context_frame: the last context frame index of a sample.
            transforms: torch-style transform. Defaults to
                None.
            have_index_col (bool, optional): whether the csv file have the
                index column. Default to False.
            load_df_func: the function to load
                dataframes from the TDT csv in `df_path`. Default to None.
                If it is None, the class will call the internal loading
                function `_load_df_func`. The `load_df_func` takes `df_path`
                and `have_index_col` as input.
            use_vision: whether to use vision data.
            use_lidar: whether to use lidar data.
        """
        super().__init__()

        self.traj_group_indices = traj_group_indices
        self.transforms = transforms
        self.path_prefix = path_prefix
        self.date_token = date_token
        self.concat_dataset_idx = 0
        self.struct_road_lmdb_path = struct_road_lmdb_path
        self.last_context_frame = last_context_frame
        self.df_path = df_path
        assert os.path.exists(
            df_path
        ), f"The dataframe csv file in the path {df_path} is not exist."
        self.use_vision = use_vision
        self.use_lidar = use_lidar
        self.load_df_func = (
            self.default_load_df_func if load_df_func is None else load_df_func
        )  # noqa: E501
        ori_df = self.load_df_func(df_path, have_index_col)
        df_cols = ori_df.columns
        for col in (
            self.BASIC_COLS + self.EGO_COLS + self.OBS_COLS + self.UNI_COLS
        ):
            assert col in df_cols, f"The input dataframe misses column {col}."
        self.df = self.split_lidar_vision_df(ori_df)

        df_dirpath = os.path.dirname(self.df_path)
        obs_to_vec_ts_json_path = os.path.join(
            df_dirpath, "static_to_dynamic_ts.json"
        )
        assert os.path.exists(
            obs_to_vec_ts_json_path
        ), f"{obs_to_vec_ts_json_path} not exists!"
        with open(obs_to_vec_ts_json_path, "r", encoding="UTF-8") as f:
            obs_to_map = json.load(f)
        self.sample_map_dict = {}
        for k, v in obs_to_map.items():
            self.sample_map_dict[int(k)] = int(v)

    @staticmethod
    def default_load_df_func(df_path, have_index_col: DataFrame) -> DataFrame:
        """Load the dataframe.

        Args:
            df_path: the path to a TDT-format DataFrame.
            have_index_col (bool, optional): whether the csv file have the
                index column.

        Returns:
            df: the dataframe.
        """
        index_col = 0 if have_index_col else None
        df = pd.read_csv(df_path, index_col=index_col)
        return df

    def split_lidar_vision_df(self, df: DataFrame) -> DataFrame:
        """Split lidar and vision dataframe from df.

        Args:
            df: the dataframe that has both lidar and vision info.

        Returns:
            df: the dataframe that has vision or lidar info.
        """
        vision_col_names = (
            self.BASIC_COLS + self.EGO_COLS + self.OBS_COLS + self.UNI_COLS
        )
        lidar_col_names = [
            "lidar_" + col_name for col_name in self.EGO_COLS + self.OBS_COLS
        ]
        df.dropna(inplace=True)
        vision_df = df.loc[:, vision_col_names]
        # use vision for both input and gt
        if self.use_vision and not self.use_lidar:
            return vision_df
        elif self.use_lidar:
            assert "lidar_pos_x" in df, "lidar info not in dataframe!"
            lidar_col_names += self.BASIC_COLS
            lidar_col_names += self.UNI_COLS
            lidar_df = df.loc[:, lidar_col_names]
            lidar_rename = {}
            for col_name in self.EGO_COLS + self.OBS_COLS:
                lidar_rename["lidar_" + col_name] = col_name
            lidar_df.rename(columns=lidar_rename, inplace=True)
            # use lidar for both input and gt
            if not self.use_vision:
                return lidar_df
            # use vision for input and lidar for gt
            else:
                fut_frames = []
                for seq_index in self.traj_group_indices:
                    fut_frames += seq_index[self.last_context_frame + 1 :]
                fut_mask = vision_df["frame_id"].isin(np.unique(fut_frames))
                replace_cols = self.EGO_COLS + self.OBS_COLS
                replace_cols.remove("classification")
                vision_df.loc[fut_mask, replace_cols] = lidar_df.loc[
                    fut_mask, replace_cols
                ]
                return vision_df
        else:
            raise ValueError(
                "Unsupported for 'use_vision = False and use_lidar = False!'"
            )

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset.
        """
        return len(self.traj_group_indices)

    def _get_basic_sample_dict(self, index: int) -> dict:
        """Return the basic sample dictionary according to a integer index.

        Args:
            index: an integer index.

        Returns:
            sample: a dict which contains complete information of a
                trajectory sample. It mainly contains a DataFrame and a
                SeqIndex. The accompanying SeqIndex contains complete
                information of how we can get the sample DataFrame from the
                original big DataFrame read during initialization. We can also
                utilize this SeqIndex to slice the output DataFrame to extract
                DataFrame corresponding to a specific frame_id.
        """
        # Get the sequence index of this sample and its last context frame.
        seq_index = self.traj_group_indices[index]
        try:
            last_context_frame = seq_index[self.last_context_frame]
        except AttributeError:
            last_context_frame = seq_index[3]
        # Filter out a DataFrame that correspond to the current sequence.
        seq_df = self.filter_df_by_seq_index(self.df, seq_index)
        # Add one additional row for ego vehicle as obs if requested.
        # Select first row in case of possible duplicate rows.
        seq_df = seq_df.groupby(["frame_id", "track_id"]).first().reset_index()
        # Sort seq_df by frame id and track_id.
        seq_df.sort_values(by=["frame_id", "track_id"], inplace=True)
        # Build the sample, record the dataset index for later data retrieval.
        sample = {
            "concat_dataset_index": self.concat_dataset_idx,
            "dataset_index": index,
            "seq_index": seq_index,
            "seq_df": seq_df,
            "last_context_frame_id": last_context_frame,
        }
        return sample

    def __getitem__(self, index: int) -> dict:
        """Return a sample according to a integer index.

        Args:
            index: an integer index.

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

        sample["dataset_prefix"] = self.path_prefix + "_"
        sample["date_token"] = self.date_token

        last_context_frame = sample["last_context_frame_id"]
        seq_lcf_mask = sample["seq_df"]["frame_id"] == last_context_frame
        lcf_timestamp = int(
            sample["seq_df"].loc[seq_lcf_mask, "timestamp"].tolist()[0]
        )

        try:
            map_stamp = int(self.sample_map_dict[lcf_timestamp])
        except (NameError, KeyError, AttributeError):
            map_stamp = lcf_timestamp

        struct_road_lmdb_reader = Lmdb(
            self.struct_road_lmdb_path, False, readonly=True
        )
        struct_road = struct_road_lmdb_reader.read(map_stamp)
        if struct_road:
            struct_road = json.loads(struct_road.decode())
            sample["struct_road"] = struct_road

            if struct_road.get("navi"):
                navi_info = struct_road["navi"]
                for _, data in navi_info.items():
                    if data.get("bounding_box"):
                        bbox = np.array(data["bounding_box"])
                        if len(bbox.shape) == 4:
                            bbox = np.concatenate(
                                [bbox[0, :, :2, 0], bbox[0, -1:, :2, 1]]
                            )
                        elif len(bbox.shape) == 1:
                            bbox = bbox.reshape(-1, 3)
                            bbox = bbox[:, :2]
                        data["bounding_box"] = bbox
                    if data.get("drive_line"):
                        drive_line = np.array(data["drive_line"])
                        if len(drive_line.shape) == 4:
                            drive_line = np.concatenate(
                                [
                                    drive_line[0, :, :2, 0],
                                    drive_line[0, -1:, :2, 1],
                                ]
                            )
                        elif len(drive_line.shape) == 1:
                            drive_line = drive_line.reshape(-1, 3)
                            drive_line = drive_line[:, :2]
                        data["drive_line"] = drive_line
                sample["navi_info"] = navi_info
            else:
                sample["navi_info"] = {}
        else:
            sample["struct_road"] = {}
            sample["navi_info"] = {}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def set_concat_dataset_index(self, idx: int):
        """Set the id of the current instance in the concatenated dataset.

        When we use multiple sub-datasets (BaseTrajDataset) to build an
        overall data iterator, each BaseTrajDataset instance does not record
        its index in the concatenated dataset. Therefore, it is unable to
        locate the original dataframe through the output information.

        Here, we enable the users to manually assign this index if some
        `trace back` operations are needed.

        Args:
            idx: the index of the current BaseTrajDataset instance
                in the concatenated dataset.
        """
        self.concat_dataset_idx = idx

    @staticmethod
    def filter_df_by_seq_index(
        df: DataFrame, seq_index: SeqIndex
    ) -> DataFrame:
        """Slice out a trajectory DataFrame according to a SeqIndex.

        Args:
            df: a pandas DataFrame which contains traj_group_index
                columns and frame_id column.
            seq_index: a SeqIndex, (traj_group_index, frame_slice).

        Returns:
            DataFrame: a sliced DataFrame whose traj_group_index columns
                correspond to the input SeqIndex.
        """
        # Slice out the DataFrame for the current trajectory group.

        out_df = df[df["frame_id"].isin(seq_index)]

        return out_df


@OBJECT_REGISTRY.register
class ConcatTdtDatasetV2(torch.utils.data.ConcatDataset):  # noqa: D205,D400
    """A wrapper Dataset to enable getting the global dataset index
    from the concatenation of multiple datasets.

    The datasets to be concatenated here must take TDT (Trajectory Dataset
    Template) format csv files as the input dataset.

    When building a ConcatTdtDataset instance by more than one datasets,
    this class can automatically assign the index of each dataset. Please
    make sure all the dataset instances have the function
    `set_concat_dataset_index`, otherwise, the indices can not be
    sucessfully assigned and remain the default value (0).
    """

    TRAJ_PRED_DATASET_MAPPING = {
        "base": BaseTrajDatasetV2,
        "autourban": AutoMultiAgentDataset,
        "autourban_navi": AutoMultiAgentNaviDataset,
    }

    def __init__(
        self,
        files: list,
        dataset_name: str,
        last_context_frame: int = 3,
        transforms: Optional[Callable] = None,
        have_index_col: bool = False,
        load_df_func: Optional[Callable] = None,
        use_vision: bool = True,
        use_lidar: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize method.

        Args:
            files (list of dict): the list of the dataset to concatenate,
                each item is a dict with the following keys:
                'df': the name of the TDT file.
                'traj_indices' (list of TrajGroupIndex): the group indices
                    of the TDT file.
            dataset_name: the name of the used dataset. It should be
                one key in self.TRAJ_PRED_DATASET_MAPPING.
            last_context_frame: the last context frame index of a sample.
            transforms: torch-style transform. Defaults to
                None.
            have_index_col (bool, optional): whether the csv file have the
                index column. Default to False.
            load_df_func: the function to load
                dataframes from the TDT csv in `df_path`. Default to None.
                If it is None, the class will call the internal loading
                function `_load_df_func`. The `load_df_func` takes `df_path`
                and `have_index_col` as input.
            use_vision: whether to use vision data.
            use_lidar: whether to use lidar data.
            args, kwargs: this is a general concat class for TDT dataset
                (based on `BaseTrajDataset` or its sub classes). Only common
                parameters (of `BaseTrajDataset`) are listed above. The other
                parameters of subclass should input by args or kwargs,
                according to the specific class definitions.
        """
        assert (
            dataset_name in self.TRAJ_PRED_DATASET_MAPPING
        ), f"Undefined dataset class {dataset_name}."
        dataset_class = self.TRAJ_PRED_DATASET_MAPPING[dataset_name]
        datasets = []
        for file_info in files:
            df_name = file_info["df"]
            traj_group_indices = file_info["traj_indices"]
            struct_road_lmdb_path = file_info["struct_road_lmdb_path"]
            dataset_prefix = file_info["dataset_prefix"]
            date_token = file_info["date_token"]
            dataset = dataset_class(
                df_path=df_name,
                traj_group_indices=traj_group_indices,
                path_prefix=dataset_prefix,
                last_context_frame=last_context_frame,
                date_token=date_token,
                transforms=transforms,
                have_index_col=have_index_col,
                load_df_func=load_df_func,
                struct_road_lmdb_path=struct_road_lmdb_path,
                use_vision=use_vision,
                use_lidar=use_lidar,
                *args,
                **kwargs,
            )
            datasets.append(dataset)
        super(ConcatTdtDatasetV2, self).__init__(datasets)
        for idx, d in zip([0] + self.cumulative_sizes[:-1], self.datasets):
            if hasattr(d, "set_concat_dataset_index"):
                d.set_concat_dataset_index(idx)

    def __repr__(self):
        return "ConcatTdtDatasetV2"


@OBJECT_REGISTRY.register
class PickledTdtDatasetV2(torch.utils.data.Dataset):
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
        transforms: Optional[Callable] = None,
        indices: Optional[Sequence] = None,
        wrap_func: Optional[List] = None,
    ):
        """Initialize method.

        Args:
            pkl_path: path to a dumped tdt dataset.
            transforms: torch-style transform. Defaults to
                None.
            indices: indices in the whole set selected
                for subset. Default to None.
            wrap_func: functions that will be called after
                loading pickle dataset. Default to None.
        """
        with open(pkl_path, "rb") as fp:
            self.loaded_ds = pickle.load(fp)

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
        return "PickledTdtDatasetV2"


@OBJECT_REGISTRY.register
class TrajPredBehavDataset(torch.utils.data.Dataset):
    """Torch based behavior dataset for offline saved post-transform behavior data.

    This class is responsible for building a pytorch map-style dataset
    which returns a dictionary containing fully-prepared behavior data
    as the input of vectornet model with the __getitem__ method.
    """

    def __init__(
        self,
        dataset_pkl: PathLike,
        road_scale: List[float],
        traj_scale: List[float],
        stage: str = "train",
        down_ratio: float = 1,
        transforms: List[dict] = None,
        seed: float = 0,
    ):
        """Initialize method.

        Args:
            dataset_pkl: the path to pickle file which saves
                the groundtruth information of valid behavior samples and
                their corresponding tokens.
            road_scale (List[float]): the coefficents for scaling the
                struct_road_feats.
            traj_sacle (List[float]): the coefficents for scaling the
                struct_traj_feats.
            stage: train or val stage of model.
            down_ratio (float): the ratio of downsampling for the keeplane
                samples. Defaults to be 1, which means no downsampling.
            seed (float): the random seed for the downsampling process.
                Defaults to be 0.
        """
        self.dataset_pkl = dataset_pkl
        self.road_scale = np.array(road_scale)
        self.traj_scale = np.array(traj_scale)
        self.transforms = transforms
        np.random.seed(seed)

        # load behavior dataset
        with open(self.dataset_pkl, "rb") as f:
            origin_ds = pickle.load(f)

        # downsample the keeplane samples
        self.dataset = []
        self.count = [0, 0, 0]
        for sample in origin_ds:
            if sample["lat_behaviors"] == 0 and np.random.rand() > down_ratio:
                continue
            self.count[int(sample["lat_behaviors"])] += 1
            self.dataset.append(sample)
        logger.info(
            f"valid number of {stage} behavior samples: {len(self.dataset)}. "
            f"keeplane: {self.count[0]} | "
            f"leftchange: {self.count[1]} | "
            f"rightchange: {self.count[2]}"
        )

    def __getitem__(self, index: int) -> dict:
        """Return a sample according to an integer index.

        Args:
            index: an integer index.

        Returns:
            sample: a dict which contains complete information of a
                behavior sample. It mainly contains a set of post-transform
                features and corresponding behavior labels. The features can
                be directly input into the vectornet model and are compatible
                with both single-task and multi-task training.
        """
        sample = self.dataset[index]
        obj_id = sample["obj_id"]
        lmdb_path = sample["lmdb_path"]
        last_context_frame_id = int(sample["scene_token"].split("_")[-1])
        if "/bucket/output" in lmdb_path:
            lmdb_path = lmdb_path.replace("/bucket/output", "/horizon-bucket")
        if "02_user/wenke.wang" in lmdb_path:
            lmdb_path = lmdb_path.replace(
                "02_user/wenke.wang", "03_pack_package/wenke.wang"
            )
        # Load behavior lmdb.
        behav_lmdb = Lmdb(lmdb_path, False, readonly=True)
        saved_feats = behav_lmdb.get(sample["scene_token"].encode())
        # Load seq_df only if enable online transform
        if self.transforms:
            seq_df_bytes = behav_lmdb.get(
                f"{sample['scene_token']}_seq_df".encode()
            )
        behav_lmdb.close()
        behav_lmdb = None
        load_feats = json.loads(saved_feats.decode())
        for key, value in load_feats.items():
            load_feats[key] = np.array(value).astype(np.float32)
        # Extract input feat of the current obstacle.
        for key in load_feats.keys():
            sample[key] = load_feats[key][obj_id]
        # Scale the input features
        sample["struct_road_feats"] *= self.road_scale
        sample["struct_traj_feats"] *= self.traj_scale
        sample["behav_state_vectors"] = sample["behav_state_vectors"].reshape(
            -1, 1, 1
        )
        # Online historical feature engineer
        if self.transforms:
            # Restore sequence dataframe
            seq_df = pd.read_json(seq_df_bytes.decode())
            sample["seq_df"] = seq_df
            sample["last_context_frame_id"] = last_context_frame_id
            seq_df["timestamp"] = [i.value for i in seq_df["timestamp"]]
            seq_df["timestamp"] = seq_df["timestamp"].astype("float")
            sample = self.transforms(sample)
            sample["lcf_timestamp"] = seq_df[
                seq_df["frame_id"] == last_context_frame_id
            ]["timestamp"].values[0]
            # Default collate func in torch cannot accept pandas.DataFrame
            sample.pop("seq_df")

        return sample

    def __len__(self) -> int:
        """__len__ method.

        Returns:
            int: return the length of the dataset.
        """
        return len(self.dataset)
