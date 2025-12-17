import os

import numpy as np
import pytest
import torch

try:
    import mxnet as mx
except ImportError:
    mx = None

from hat.data.datasets.smoke_cls_dataset import SmokeClsDataset

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root_dir = os.path.join(
    bucket_root, "HDLTAlgorithm/data/orig_data/action/video/data"
)  # noqa
rec_list = ["2020092122_small_image_train.rec"]
all_rec_info_list = [
    "anno_phone_refresh_inter_newRule_recInfo_0605.json",
    "anno_phone_recInfo_0530_newRule.json",
]
all_rec_imgIdx_list = [
    "anno_phone_refresh_inter_newRule_summary_0605.json",
    "anno_phone_oldVal_summary_0524.json",
    "anno_phone_imgIdx_0530_newRule.json",
]

rec_list = [os.path.join(root_dir, i) for i in rec_list]
all_rec_info_list = [os.path.join(root_dir, i) for i in all_rec_info_list]
all_rec_imgIdx_list = [os.path.join(root_dir, i) for i in all_rec_imgIdx_list]


@pytest.mark.parametrize(
    ["rec_list", "all_rec_info_list", "all_rec_imgIdx_list"],
    [
        pytest.param(rec_list, all_rec_info_list, all_rec_imgIdx_list),
    ],
)
@pytest.mark.skipif(
    not mx or not os.path.exists(root_dir), reason="data path doesn't exists."
)
def test_SmokeClsDataset(rec_list, all_rec_info_list, all_rec_imgIdx_list):
    dataset = SmokeClsDataset(rec_list, all_rec_info_list, all_rec_imgIdx_list)
    data = dataset[0]
    img = data["img"]

    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) == 88
    assert isinstance(data, dict)
    assert isinstance(img, np.ndarray)
