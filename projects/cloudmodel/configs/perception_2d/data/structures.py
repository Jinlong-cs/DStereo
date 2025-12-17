"""
Basic data structs used in cloudmodel configs.
"""  # noqa

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

from dataclasses_json import DataClassJsonMixin
from hatbc.utils import Enum

from hat.data.datasets.densebox_dataset import DenseboxDataset
from hat.data.datasets.densebox_from_lmdb_dataset import (
    DenseboxFromLMDBDataset,
)
from projects.cloudmodel.configs.perception_2d.tasks.all_tasks import (
    CloudModelTask,
)

__all__ = [
    "SourceDataProject",
    "CloudModelDatasetItem",
    "DataSplit",
    "DenseboxSource",
    "MulticlassDenseboxSource",
    "LMDBSource",
    "AidiEval2DSource",
]


# ------------------------- dataset structure ------------------------------- #
class SourceDataProject(Enum):
    mono = "mono"
    pilot = "pilot"
    canno = "canno"
    unknown = "unknown"


class DataSplit(Enum):
    train = "train"
    val = "val"
    test = "test"


@dataclass
class DenseboxSource:
    rec_path: str
    anno_path: str
    rec_path_md5sum: Optional[str] = None
    anno_path_md5sum: Optional[str] = None
    roi_list: Optional[str] = None


@dataclass
class MulticlassDenseboxSource:
    rec_path: str
    anno_path: str
    rec_path_md5sum: Optional[str] = None
    anno_path_md5sum: Optional[str] = None
    roi_list: Optional[str] = None


@dataclass
class LMDBSource:
    data_path: str


@dataclass
class AidiEval2DSource:
    leaderboard_id: int
    root_dir: Optional[str] = None


@dataclass
class CloudModelDatasetItem(DataClassJsonMixin):
    data_source: Union[DenseboxSource, LMDBSource, AidiEval2DSource]
    project: SourceDataProject
    task: CloudModelTask
    split: Optional[str] = DataSplit.train.value
    sample_weight: Optional[float] = None
    length: Optional[int] = None
    img_shape: Optional[List[List]] = None
    tags: Optional[List[str]] = field(default_factory=lambda: [])
    # for validation
    coco_anno_json: Optional[str] = None

    # TODO ugly
    def get_default_dataset_config(self, transforms, **kwargs):
        if isinstance(self.data_source, DenseboxSource):
            if self.task.task_type.value in ["instanceseg"]:
                dataset_config = dict(
                    type="InstSegDenseboxDataset",
                    task_type=self.task.task_type.value,
                    data_path=self.data_source.rec_path,
                    anno_path=self.data_source.anno_path,
                    return_orig_img=self.split == DataSplit.val.value,
                    return_orig_gt_seg=self.split == DataSplit.val.value,
                    ignore_hard=self.split != DataSplit.train.value,
                    # additional args
                    transforms=transforms,
                    to_rgb=kwargs.get("to_rgb", False),
                    use_ignore=kwargs.get("use_ignore", False),
                )
                return dataset_config
            elif self.task.task_type.value in [
                "detection",
                "segmentation",
                "classification",
            ]:
                dataset_config = dict(
                    type="DenseboxDataset",
                    task_type=self.task.task_type.value,
                    data_path=self.data_source.rec_path,
                    anno_path=self.data_source.anno_path,
                    return_orig_img=self.split == DataSplit.val.value,
                    return_orig_gt_seg=self.split == DataSplit.val.value,
                    ignore_hard=self.split != DataSplit.train.value,
                    # additional args
                    transforms=transforms,
                    remove_det_duplicate=kwargs.get(
                        "remove_det_duplicate", False
                    )
                    if self.task.task_type.value != "classification"
                    else False,
                    abandon_other_category=kwargs.get(
                        "abandon_other_category", False
                    ),
                    to_rgb=kwargs.get("to_rgb", False),
                    use_ignore=kwargs.get("use_ignore", False),
                )
                if self.task.task_type.value == "detection":
                    assert "ins_id2category" in kwargs
                    ins_id2category = kwargs["ins_id2category"]
                    dataset_config.update(
                        {
                            "class_id": list(ins_id2category.keys())[0],
                            "category": list(ins_id2category.values())[0],
                        }
                    )
                return dataset_config
            else:
                raise ValueError()
        elif isinstance(self.data_source, MulticlassDenseboxSource):
            dataset_config = dict(
                type="MulticlassDenseboxDataset",
                task_type=self.task.task_type.value,
                data_path=self.data_source.rec_path,
                anno_path=self.data_source.anno_path,
                ignore_hard=self.split != DataSplit.train.value,
                transforms=transforms,
                to_rgb=kwargs.get("to_rgb", False),
                use_ignore=kwargs.get("use_ignore", False),
            )
            if self.task.task_type.value == "detection":
                assert "ins_id2category" in kwargs
                dataset_config.update(
                    {
                        "class_id_map": kwargs["ins_id2category"],
                    }
                )
            return dataset_config
        elif isinstance(self.data_source, AidiEval2DSource):
            assert self.split == DataSplit.test.value
            dataset_config = dict(
                type="Auto2dFromImage",
                data_path=self.data_source.root_dir,
                to_rgb=kwargs.get("to_rgb", False),
                transforms=transforms,
                return_orig_img=kwargs.get("return_orig_img", True),
            )
            return dataset_config
        elif isinstance(self.data_source, LMDBSource):
            if self.task.task_type.value in ["detection", "segmentation"]:
                dataset_config = dict(
                    type="DenseboxFromLMDBDataset",
                    idx_path=os.path.join(self.data_source.data_path, "idx"),
                    img_path=os.path.join(self.data_source.data_path, "img"),
                    anno_path=os.path.join(self.data_source.data_path, "anno"),
                    decode_img=True,
                    task_type=self.task.task_type.value,
                    to_rgb=kwargs.get("to_rgb", True),
                    ignore_hard=self.split != DataSplit.train.value,
                    use_ignore=kwargs.get("use_ignore", True),
                    transform_ignore_bboxes=kwargs.get(
                        "transform_ignore_bboxes", True
                    ),
                    return_orig_img=self.split == DataSplit.val.value,
                    return_orig_gt_seg=self.split == DataSplit.val.value,
                    remove_det_duplicate=kwargs.get(
                        "remove_det_duplicate", True
                    ),
                    abandon_other_category=kwargs.get(
                        "abandon_other_category", True
                    ),
                    transforms=transforms,
                )

                if self.task.task_type.value == "detection":
                    assert "ins_id2category" in kwargs
                    ins_id2category = kwargs["ins_id2category"]
                    dataset_config.update(
                        {
                            "class_id": list(ins_id2category.keys())[0],
                            "category": list(ins_id2category.values())[0],
                        }
                    )

                return dataset_config

            else:
                raise NotImplementedError

    def get_single_dataset_length(self):
        if isinstance(self.data_source, DenseboxSource):
            densebox_dataset = DenseboxDataset(
                data_path=self.data_source.rec_path,
                anno_path=self.data_source.anno_path,
            )
            dataset_length = len(densebox_dataset)
            self.length = dataset_length
            return dataset_length
        elif isinstance(self.data_source, LMDBSource):
            lmdb_dataset = DenseboxFromLMDBDataset(
                idx_path=os.path.join(self.data_source.data_path, "idx"),
                img_path=os.path.join(self.data_source.data_path, "img"),
                anno_path=os.path.join(self.data_source.data_path, "anno"),
            )
            dataset_length = len(lmdb_dataset)
            self.length = dataset_length
            return dataset_length
        else:
            raise ValueError(f"{self.data_source} does not supported exist.")
