# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
from typing import Sequence

from hat.data.datasets.data_packer import Packer
from hat.data.datasets.img_dataset import _read_lst
from hat.utils.apply_func import _as_list

__all__ = ["FileList2LMDB", "RecLst2LMDB", "Json2LMDB"]

logger = logging.getLogger(__name__)


class FileList2LMDB(Packer):
    """
    FileList2LMDB is used for converting file list to to LMDB format.

    NOTE: Make sure target_data_dir is not a bucket path.
    Otherwise it will be very slow.

    Args:
        file_list_path: The dir of original imagenet data.
        target_data_dir: Path for LMDB file.
    """

    def __init__(self, file_list_path: str, target_data_dir: str):
        with open(file_list_path, "r") as f:
            self.lines = f.read().splitlines()
        nums = len(self.lines)
        assert nums > 0, f"line nums in {file_list_path} must >0"
        lmdb_kwargs = {
            "map_size": 1099511627776 * 2,
            "meminit": nums,
            "map_async": True,
        }
        super(FileList2LMDB, self).__init__(
            uri=target_data_dir,
            max_data_num=len(self.lines),
            pack_type="lmdb",
            num_workers=1,
            **lmdb_kwargs,
        )

    def pack_data(self, idx):
        return self.lines[idx].encode()


class RecLst2LMDB(Packer):
    """RecLst2LMDB is used for converting rec lst files to LMDB.

    NOTE: Make sure target_data_dir is not a bucket path or gpfs path.

    Args:
        lst_file_list: List contains json files for packing.
        target_data_dir: Path for LMDB file.
    """

    def __init__(
        self,
        lst_file_list: Sequence,
        target_data_dir: str,
    ):
        lst_file_list = _as_list(lst_file_list)
        nums = len(lst_file_list)
        assert nums > 0, "nums of lst file(s) must >0"
        all_lst_data = {}
        for lst_file in lst_file_list:
            all_lst_data.update(_read_lst(lst_file))

        self.all_sample = []
        for key, value in all_lst_data.items():
            cur_sample = {}
            cur_sample["key"] = key
            cur_sample["value"] = json.dumps(value)
            self.all_sample.append(cur_sample)

        lmdb_kwargs = {
            "map_size": 1099511627776 * 2,
            "meminit": nums,
            "map_async": True,
        }
        super(RecLst2LMDB, self).__init__(
            uri=target_data_dir,
            max_data_num=len(self.all_sample),
            pack_type="lmdb",
            num_workers=1,
            **lmdb_kwargs,
        )

    def _write(self, idx, data):
        idx = data["key"]
        data = data["value"].encode()
        return super()._write(idx, data)

    def pack_data(self, idx):
        return self.all_sample[idx]


class Json2LMDB(Packer):
    """Json2LMDB is used for converting json files to LMDB.

    The format of each json file should be like:{key_1: val_1,
    key_2: val_2 ...}, for different json file, all the keys must
    be unique.

    NOTE: Make sure target_data_dir is not a bucket path or gpfs path.

    Args:
        json_file_list: List contains json files for packing.
        target_data_dir: Path for LMDB file.
    """

    def __init__(
        self,
        json_file_list: Sequence,
        target_data_dir: str,
    ):
        json_file_list = _as_list(json_file_list)
        nums = len(json_file_list)
        assert nums > 0, "nums of json file(s) must >0"
        all_json_data = {}
        for json_file in json_file_list:
            with open(json_file, "rb") as fin:
                json_data = json.load(fin)
            all_json_data.update(json_data)

        self.all_sample = []
        for key, value in all_json_data.items():
            cur_sample = {}
            cur_sample["key"] = key
            cur_sample["value"] = json.dumps(value)
            self.all_sample.append(cur_sample)

        lmdb_kwargs = {
            "map_size": 1099511627776 * 2,
            "meminit": nums,
            "map_async": True,
        }
        super(Json2LMDB, self).__init__(
            uri=target_data_dir,
            max_data_num=len(self.all_sample),
            pack_type="lmdb",
            num_workers=1,
            **lmdb_kwargs,
        )

    def _write(self, idx, data):
        idx = data["key"]
        data = data["value"].encode()
        return super()._write(idx, data)

    def pack_data(self, idx):
        return self.all_sample[idx]
