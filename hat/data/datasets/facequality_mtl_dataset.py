from typing import List, Optional, Union

import cv2
import numpy as np
import yaml
from easydict import EasyDict
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.pack_type.mxrecord import unpack_img
from .data_packer import PackTypeMapper

__all__ = ["FaceQualityDataset"]


@OBJECT_REGISTRY.register
class FaceQualityDataset(Dataset):
    """Dataset for face quality, which only process one .rec file at once.

    Return cropped face roi and label for selected task from mxnet rec files.
    Note: Each image must contain labels for all categories!

    Args:
        rec_path: the path of face quality .rec file.
        idx_path: the path of face quality .idx file.
        label_path: .npy file, the label which shares index with .rec.
        info_path: yaml file path, has task name to index mapping information.
        task_name: the tasks which need to be processed.
        need_flag: whether to group each piece of data,
            similar to the role of labels, default to False.
        need_index: whether to record the index of the image, default to False.
        transfroms: transfroms of data before using.
    """

    def __init__(
        self,
        rec_path: str,
        idx_path: str,
        label_path: str,
        info_path: str,
        task_name: Union[str, List],
        need_flag: bool = False,
        need_index: bool = False,
        transforms: Optional[List] = None,
    ):
        super(FaceQualityDataset, self).__init__()
        self.label_path = label_path
        self.info_path = info_path
        self.task_name = _as_list(task_name)
        self.need_flag = need_flag
        self.need_index = need_index
        if self.need_flag:
            assert (
                len(self.task_name) == 1
            ), "Flag can only be used when there is just one task."
        self.pack_type = PackTypeMapper["mxrecord"]
        self.pack_file = self.pack_type(
            uri=rec_path, idx_path=idx_path, writable=False
        )
        self.pack_file.open()
        self.transforms = transforms

        # mapping the tasks to idx
        self._task_to_idx()
        # process the label, and save the labels of selected tasks
        self._process_label()

    def _task_to_idx(self):
        with open(self.info_path, "r") as infile:
            info_dict = EasyDict(
                yaml.load(infile.read(), Loader=yaml.SafeLoader)
            )
        self._task2idx_version = info_dict["face_label_version"]
        self._task2idx = info_dict["label_index"]

    def _process_label(self):
        self._label_info = {}
        # To save memory, load only mappings
        label_tmp = np.load(self.label_path, mmap_mode="r")
        self._length = len(label_tmp)
        for t in self.task_name:
            self._label_info[t] = label_tmp[:, int(self._task2idx[t])]
        if self.need_flag:
            self.flag = np.array(self._label_info[self.task_name[0]])
            # To save memory, map 0.5 to 255, -1 to 254
            self.flag[self.flag == 0.5] = 255
            self.flag[self.flag == -1] = 254
            self.flag = self.flag.astype(np.uint8)

    def __getitem__(self, index: int):
        label = {k: float(v[index]) for k, v in self._label_info.items()}
        s = self.pack_file.read(index)
        _, img = unpack_img(s, cv2.IMREAD_COLOR)  # img: bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = {"img": img, "gt_face_quality": label, "layout": "hwc"}
        if self.need_index:
            data.update({"index": index})
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return self._length
