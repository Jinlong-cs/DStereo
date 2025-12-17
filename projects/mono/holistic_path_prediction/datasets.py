import os
from collections import namedtuple

import torch
import yaml
from easydict import EasyDict
from hatbc import filestream as fs

from hat.data.collates.collates import collate_2d
from hat.utils.apply_func import _as_list

batch_size_map = {"hpp": 12}

dataset_fn = os.path.split(os.path.realpath(__file__))[0]
dataset_fn = "%s/datasets.yaml" % dataset_fn
with fs.io.reader(dataset_fn) as fread:
    dataset_map = yaml.safe_load(fread)


datapaths = dict()
for cls_name, batch_size in batch_size_map.items():
    assert (
        cls_name in batch_size_map.keys()
    ), f"{cls_name} not in {batch_size_map.keys()}"

    # batch_size as sample_weight, do normalize
    sum_bs = float(sum([i["batch_size"] for i in dataset_map[cls_name]]))
    datapaths[cls_name] = dict(
        train_batch_size_per_ctx=batch_size,
        train_data_paths=[
            dict(
                rec_path=i["train_rec"],
                anno_path=i["train_json"],
                sample_weight=i["batch_size"] / sum_bs * batch_size,
            )
            for i in dataset_map[cls_name]
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=[
            dict(
                rec_path=i["val_rec"],
                anno_path=i["val_json"],
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
    """Parse data paths and take specified classes

    Args:
        datapaths (dict): Data paths
        take_classnames (list[str]/tuple[str]):
            Taked class names
        check_path (bool, optional):
            Whether check path existence or not. Defaults to True.
    """
    assert isinstance(
        datapaths, dict
    ), f"expect dict \
        type but get {type(datapaths)} type"
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
        _check(results["train_data_desc"]["rec_paths"], [".rec"])
        _check(results["train_data_desc"]["anno_paths"], [".json"])
        # current val dataset is a directory, not rec
        # _check(results["val_data_desc"]["rec_paths"], [".rec"])
        _check(results["val_data_desc"]["anno_paths"], [".json"])

    return EasyDict(results)


classnames = ["hpp"]

# ------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------
data_desc = parse_dataset(datapaths, classnames[0])
train_data_desc = data_desc.train_data_desc
train_batch_size = train_data_desc.batch_size_per_ctx
train_rec_paths = train_data_desc.rec_paths
train_anno_paths = train_data_desc.anno_paths
train_sample_weights = train_data_desc.sample_weights
train_datasets = [
    dict(
        type="HppDataset",
        read_mode="rec",
        interpolated=True,
        img_path=rec_path_i,
        anno_path=anno_path_i,
        transforms=[
            dict(
                type="RandomHorizontalFilpWithPoints",
                flip_prob=0.25,
            ),
            dict(
                type="RandomTranslationWithPointsV2",
                prob=0.25,
                translate_x=[(-25, -15), (15, 25)],
                translate_y=[(0, 0)],
            ),
            dict(
                type="HPPPerspectiveTransform",
                prob=0.25,
            ),
            dict(type="GaussianNoiseWithChannel", prob=0.5, mean=0, sigma=20),
            dict(type="ChangeIntensity", prob=0.5, rescale=60),
            dict(type="Shadow", prob=0.5, min_alpha=0.5, max_alpha=0.75),
            dict(type="HPPPoint2Map", output_stride=8),
            dict(type="AlignOdometryLength", length=100),
            dict(type="ToTensor", to_yuv=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    )
    for rec_path_i, anno_path_i in zip(
        train_rec_paths[:-1], train_anno_paths[:-1]
    )
]

train_datasets.append(
    dict(
        type="HppDataset",
        read_mode="rec",
        interpolated=True,
        img_path=train_rec_paths[-1],
        anno_path=train_anno_paths[-1],
        transforms=[
            dict(type="GaussianNoiseWithChannel", prob=0.25, mean=0, sigma=20),
            dict(type="ChangeIntensity", prob=0.25, rescale=60),
            dict(type="Shadow", prob=0.25, min_alpha=0.5, max_alpha=0.75),
            dict(type="HPPPoint2Map", output_stride=8),
            dict(type="AlignOdometryLength", length=100),
            dict(type="ToTensor", to_yuv=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    )
)

val_data_desc = data_desc.val_data_desc
val_batch_size = val_data_desc.batch_size_per_ctx
val_rec_paths = val_data_desc.rec_paths
val_anno_paths = val_data_desc.anno_paths
val_sample_weights = val_data_desc.sample_weights
val_datasets = [
    dict(
        type="HppDataset",
        read_mode="dir",
        interpolated=False,
        img_path=rec_path_i,
        anno_path=anno_path_i,
        filter_invalid=False,
        transforms=[
            dict(type="AlignOdometryLength", length=200),
            dict(type="ToTensor", to_yuv=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    )
    for rec_path_i, anno_path_i in zip(val_rec_paths, val_anno_paths)
]
# ------------------------------------------------------------------------------
# dataloader
# ------------------------------------------------------------------------------
train_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        datasets=train_datasets,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_2d,
    batch_size=train_batch_size,
    num_workers=18,
    pin_memory=True,
    persistent_workers=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        datasets=val_datasets,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_2d,
    batch_size=16,
    num_workers=8,
    pin_memory=True,
)
