import itertools
import os
from typing import Literal, Optional

import yaml
from hatbc.aidi.env import is_running_on_aidi


class DataYamlParser:
    """Data yaml parser.

    ..note ::
        For more details about the rules, ask @chunyu.bi.

    Args:
        task: task name.
        collection_name: collection name.
        data_yaml: data.yaml file name.
        version_yaml: data_version.yaml file path.
        root_dir: root directory of data. Default: /horizon-bucket/SuperParking
    """

    def __init__(
        self,
        task: str,
        collection_name: str,
        data_yaml: str = "data.yaml",
        version_yaml: str = "data_version.yaml",
        root_dir: Optional[str] = None,
    ):
        self.task = task
        self.collection_name = collection_name
        self.data_yaml = data_yaml
        self.version_yaml = version_yaml
        self.root_dir = self.cast_path(
            root_dir or "/horizon-bucket/SuperParking"
        )

    def read_data_collection(self) -> dict:
        """Read data collection from data.yaml."""

        data_yaml = yaml.safe_load(open(self.data_yaml, "r"))
        data_yaml = self._fullfill_data_path(data_yaml)
        return data_yaml

    def select_collection_names(self, dtype: Literal["train", "val"]) -> tuple:
        """Selection to use train or validation dataset."""

        versions = yaml.safe_load(open(self.version_yaml, "r"))
        names = versions[self.task][self.collection_name][dtype][
            "data_version"
        ]
        return tuple(names)

    def get_lmdb_paths(self, dtype: Literal["train", "val"]) -> tuple:
        collection = self.read_data_collection()
        names = self.select_collection_names(dtype)
        paths = [collection[self.task][name][dtype]["lmdb"] for name in names]
        if not paths:
            return paths

        if isinstance(paths[0], (tuple, list)):
            paths = itertools.chain.from_iterable(paths)

        return tuple(paths)

    @classmethod
    def cast_path(cls, path: str) -> str:
        if is_running_on_aidi():
            true_path = path.replace("/horizon-bucket", "/bucket/input")
        else:
            true_path = path.replace("/bucket/input", "/horizon-bucket")
        return true_path

    def _fullfill_data_path(self, data_paths: dict):
        """Cat each data path to the root dir."""
        complete_data_paths = {}

        if isinstance(data_paths, dict):
            complete_data_paths = {
                k: self._fullfill_data_path(v) for k, v in data_paths.items()
            }
        elif isinstance(data_paths, list):
            return [
                os.path.join(self.root_dir, p) if "/" in p else p
                for p in data_paths
            ]

        return complete_data_paths
