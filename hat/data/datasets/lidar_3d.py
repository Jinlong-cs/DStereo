# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.lmdb import LmdbReadList

__all__ = ["PointCloudLmdbDataset"]


@OBJECT_REGISTRY.register
class PointCloudLmdbDataset(Dataset):
    """
    A dataset using key to read point cloud from Lmdb file.

    Args:
        uri: path to lmdb file, support str and list[str].
        load_dim: point cloud data dimension to be loaded, example: 4.
        data_type: point cloud data type to be loaded, default: np.float32.
        keep_dim: point cloud dimension to be used, default: 4.
        map_size: map size of default lmdb settings.
        parse_raw_buffer: whether to parse raw buffer to numpy data.
            Default: True.
    """

    def __init__(
        self,
        uris: Union[str, Sequence[str]],
        load_dim: int,
        data_type: np.dtype = np.float32,
        keep_dim: Optional[int] = None,
        map_size: int = 1024 ** 2 * 100,
        parse_raw_buffer: bool = True,
    ) -> None:

        self.uris = uris
        self.data_type = data_type
        self.load_dim = load_dim
        if keep_dim is None:
            keep_dim = load_dim
        self.keep_dim = keep_dim
        assert keep_dim <= load_dim, "Used dim should smaller than loaded dim."
        self.map_size = map_size
        self.parse_raw_buffer = parse_raw_buffer

        self.init_lst_lmdb()

    def init_lst_lmdb(
        self,
    ):
        self.lmdb_dataset = LmdbReadList(self.uris, map_size=self.map_size)

    def __getstate__(self):
        state = self.__dict__
        state["lmdb_dataset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.init_lst_lmdb()

    def __len__(self):
        """Get the length."""
        return len(self.lmdb_dataset)

    def __getitem__(self, key):
        buffer_point = self.lmdb_dataset.read(key)
        if not self.parse_raw_buffer:
            return buffer_point
        encode_point = np.frombuffer(buffer_point, self.data_type)
        point = encode_point.reshape((-1, self.load_dim))
        point = point[:, : self.keep_dim]
        return point
