import json
import os
from typing import List, Optional

import cv2
import numpy as np
import torch.utils.data as data
from scipy.interpolate import splev, splprep

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXIndexedRecordIO, unpack_img


@OBJECT_REGISTRY.register
class HppDataset(data.Dataset):
    def __init__(
        self,
        img_path: str,
        anno_path: str,
        read_mode: str = "rec",
        filter_invalid: bool = True,
        interpolated: bool = True,
        transforms: Optional[List] = None,
    ):
        """HPP Dataset.

        Args:
            img_path: image path
            anno_path: label path
            read_mode: read mode of image path, [rec/dir].
                Defaults to "rec".
            filter_invalid: whether filter invalid json or
                not. The lane json with empty `lanes` or `h_samples` will be
                regarded as invalid json. Defaults to True.
            interpolated: whether resample original gt points or not.  # noqa
                Defaults to True.
            transforms: transforms of data before using.  # noqa
                Defaults to None.

        """
        self.img_path = img_path
        self.interpolated = interpolated
        self.read_mode = read_mode
        self.auto_data = []
        with open(anno_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.auto_data.append(jsonString)
        if filter_invalid:
            self.auto_data = self.filter_invalid_json()
        self.size_train = len(self.auto_data)
        self.transforms = transforms

    def filter_invalid_json(self):
        # filter json that the length of x == 0 or y == 0
        new_auto_data = []
        for _, json_file in enumerate(self.auto_data):
            if (
                len(json_file["lanes"][0]) == 0
                or len(json_file["h_samples"]) == 0
            ):
                continue
            new_auto_data.append(json_file)
        return new_auto_data

    def read_list(self, imglst_path):
        img_list = {}
        with open(imglst_path) as fin:
            for line in iter(fin.readline, ""):
                line = line.strip().split("\t")
                img_list[line[-1]] = int(line[0])
        return img_list

    def interpolate_odometry(self, x, y, smoothness=0, knots=50):
        # Spline interpolation of a lane. Used on the train stage
        assert len(x) == len(y)
        # cubic spline are recommended here, k \in [3, point_length]
        tck, _ = splprep([x, y], s=smoothness, t=knots, k=min(3, len(x) - 1))
        u = np.linspace(0.0, 1.0, knots)
        return np.array(splev(u, tck)).T

    def __getitem__(self, idx):
        data = self.auto_data[idx]
        if self.read_mode == "rec":
            img_list_path = self.img_path.replace(".rec", ".lst")
            img_list = self.read_list(img_list_path)
            img_idx_path = self.img_path.replace(".rec", ".idx")
            imgrec = MXIndexedRecordIO(
                img_idx_path, self.img_path, "r"
            )  # noqa
            # H x W x C
            _, temp_image = unpack_img(
                imgrec.read_idx(img_list[data["raw_file"]]), cv2.IMREAD_COLOR
            )  # noqa
        elif self.read_mode == "dir":
            temp_image = cv2.imread(
                os.path.join(self.img_path, data["raw_file"])
            )
        else:
            raise KeyError(f"do not support {self.read_mode} read mode")
        # only one odometry per frame
        x = np.array(data["lanes"][0]) * 1.0
        y = np.array(data["h_samples"]) * 1.0
        coordinates = np.concatenate([x[:, None], y[:, None]], axis=1)
        if self.interpolated:
            coordinates = self.interpolate_odometry(
                coordinates[:, 0], coordinates[:, 1]
            )
        data_dict = {
            "img": temp_image,
            "img_path": self.img_path,
            "points": coordinates,
            "layout": "hwc",
            "img_name": data["raw_file"],
            "color_space": "bgr",
        }
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        return data_dict

    def __len__(self):
        return len(self.auto_data)
