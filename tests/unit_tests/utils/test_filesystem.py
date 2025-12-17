import os
import pickle

import fsspec
import pytest
import torch

from hat.utils.filesystem import (
    file_load,
    get_filesystem,
    join_path,
    pickle_load,
)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(
            "/horizon-bucket/HDLTAlgorithm/pipeline_test_data/ut_checkpoint/float-checkpoint-best-ef67e7d8.pth.tar"  # noqa: E501
        ),
        pytest.param(
            "http://fm-mengyang-duan.train.hogpu.cc/HAT_TEST/ut_checkpoint/float-checkpoint-best-ef67e7d8.pth.tar"  # noqa: E501
        ),
    ],
)
def test_get_filesystem(url):
    fs = get_filesystem(url)
    assert isinstance(fs, fsspec.AbstractFileSystem)
    with fs.open(url, "rb") as f:
        checkpoint = torch.load(f)
        assert "state_dict" in checkpoint


def test_join_path():
    prefix_path = "/home/users/"
    fake_path = "data"
    fake_group_path = {"group1": "data1", "group2": "data2"}
    ignore_key = ["group1"]
    input = dict(
        path1=fake_path,
        path2=[fake_path, fake_group_path],
        path3=fake_group_path,
        v=2,
    )
    result = join_path(prefix_path, input, ignore_key)

    assert result["path1"] == os.path.join(prefix_path, fake_path)
    assert result["path2"][0] == os.path.join(prefix_path, fake_path)
    assert result["path2"][1]["group1"] == fake_group_path["group1"]
    assert result["path2"][1]["group2"] == os.path.join(
        prefix_path, fake_group_path["group2"]
    )
    assert result["path3"]["group1"] == fake_group_path["group1"]
    assert result["path3"]["group2"] == os.path.join(
        prefix_path, fake_group_path["group2"]
    )
    assert result["v"] == 2


def test_pickle_load():
    temp_dict = {"a": "1"}
    temp_dict = pickle.dumps(temp_dict)
    with open(".tmp.pkl", "ab") as f:
        f.write(temp_dict)
        f.close()
    with open(".tmp.pkl", "rb") as f:
        result = pickle_load(f)
    assert result == {"a": "1"}
    os.remove(".tmp.pkl")


def test_file_load():
    temp_dict = {"a": "1"}
    temp_dict = pickle.dumps(temp_dict)
    with open(".tmp.pkl", "ab") as f:
        f.write(temp_dict)
        f.close()
    for result in file_load(".tmp.pkl"):
        assert result == {"a": "1"}
    os.remove(".tmp.pkl")
