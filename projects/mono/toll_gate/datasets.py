import os
from collections import namedtuple

import yaml
from common import mount_root
from easydict import EasyDict
from hatbc import filestream as fs

from hat.utils.apply_func import _as_list

hdfs_root = "hdfs://hobot-bigdata-aliyun"
batch_size_map = {"tollgate_4pe": 10}

dataset_fn = os.path.split(os.path.realpath(__file__))[0]
dataset_fn = "%s/datasets.yaml" % dataset_fn
with fs.io.reader(dataset_fn) as fread:
    dataset_map = yaml.safe_load(fread)

datapaths = dict()
for cls_name, batch_size in batch_size_map.items():
    assert (
        cls_name in dataset_map.keys()
    ), f"{cls_name} not in {dataset_map.keys()}"

    def _get_root(path):
        if path.startswith("/user"):
            return hdfs_root
        else:
            return mount_root

    # batch_size as sample_weight, do normalize
    sum_bs = float(sum([i["batch_size"] for i in dataset_map[cls_name]]))

    datapaths[cls_name.replace("_day", "").replace("_night", "")] = dict(
        train_batch_size_per_ctx=batch_size,
        train_data_paths=[
            dict(
                rec_path=_get_root(i["train_rec"]) + i["train_rec"],
                anno_path=_get_root(i["train_json"]) + i["train_json"],
                sample_weight=i["batch_size"] / sum_bs * batch_size,
            )
            for i in dataset_map[cls_name]
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=_get_root(i["val_rec"]) + i["val_rec"],
                anno_path=_get_root(i["val_json"]) + i["val_json"],
                sample_weight=1,
            )
            for i in dataset_map[cls_name]
            if i["val_rec"] is not None
        ],
    )
datapaths = EasyDict(datapaths)

DataDesc = namedtuple(
    "DataDesc",
    [
        "rec_path",
        "anno_path",
        "pb_anno_path",
        "batch_size_per_ctx",
    ],
)


def _check_path(path, expected_ext=None):
    for path_i in _as_list(path):
        assert fs.io.exists(path_i), "%s does not exists" % path_i
        if expected_ext is not None:
            assert os.path.splitext(path_i)[-1] in _as_list(
                expected_ext
            ), "%s does not endswith ext %s" % (path_i, expected_ext)


def parse_dataset(datapaths, take_classnames, check_path=True):
    """
    Parse data paths and take specified classes.

    Parameters
    ----------
    datapaths : dict
        Data paths.
    take_classnames : list/tuple of str
        Taked class names
    check_path : bool, optional
        Whether check path exists or not, by default True
    """

    assert isinstance(datapaths, dict)

    results = dict(
        train_data_desc=dict(
            batch_size_per_ctx=0,
            rec_paths=[],
            anno_paths=[],
            sample_weights=[],
        ),
        val_data_desc=dict(
            batch_size_per_ctx=0,
            rec_paths=[],
            anno_paths=[],
            sample_weights=[],
        ),
    )

    for name_i in _as_list(take_classnames):

        def _update(src, dst):
            def _flat(val, key):
                if val is None:
                    return []
                else:
                    return [val_i.get(key, None) for val_i in _as_list(val)]

            dst["train_data_desc"]["batch_size_per_ctx"] += src[
                "train_batch_size_per_ctx"
            ]  # noqa
            dst["train_data_desc"]["rec_paths"].extend(
                _flat(src["train_data_paths"], "rec_path")
            )  # noqa
            dst["train_data_desc"]["anno_paths"].extend(
                _flat(src["train_data_paths"], "anno_path")
            )  # noqa
            dst["train_data_desc"]["sample_weights"].extend(
                _flat(src["train_data_paths"], "sample_weight")
            )  # noqa

            dst["val_data_desc"]["batch_size_per_ctx"] += src[
                "val_batch_size_per_ctx"
            ]  # noqa
            dst["val_data_desc"]["rec_paths"].extend(
                _flat(src["val_data_paths"], "rec_path")
            )  # noqa
            dst["val_data_desc"]["anno_paths"].extend(
                _flat(src["val_data_paths"], "anno_path")
            )  # noqa
            dst["val_data_desc"]["sample_weights"].extend(
                _flat(src["val_data_paths"], "sample_weight")
            )  # noqa

        _update(datapaths[name_i], results)

    def _check(paths, exts=None):
        for path_i in _as_list(paths):
            if path_i is not None:
                _check_path(path_i, exts)

    if check_path:
        _check(
            results["train_data_desc"]["rec_paths"],
            [".pb_rec", ".rec", ".record"],
        )
        _check(results["train_data_desc"]["anno_paths"], [".pb_rec", ".json"])
        _check(
            results["val_data_desc"]["rec_paths"],
            [".pb_rec", ".rec", ".record"],
        )
        _check(results["val_data_desc"]["anno_paths"], [".pb_rec", ".json"])

    return EasyDict(results)
