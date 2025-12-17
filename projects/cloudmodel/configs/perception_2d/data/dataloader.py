import copy
from functools import partial

import torch

from hat.data.collates.collates import collate_2d, collate_2d_replace_empty
from projects.cloudmodel.configs.perception_2d.common import pipeline_test

collate_2d_re = partial(collate_2d_replace_empty, prob=0.67)
collate_2d = partial(collate_2d, verbose=pipeline_test)


def build_singletask_dataloader(
    dataset,
    sample_weights=None,
    mode="train",
    batch_size_per_gpu=1,
    num_workers=1,
    is_iterabledataset=False,
    collate_fn=collate_2d_re,
):
    # According to PyTorch documentation, when sampler is set, shuffle should
    # not be specified in dataloader.
    base_dataloader = dict(
        type=torch.utils.data.DataLoader,
        batch_size=batch_size_per_gpu,
        collate_fn=collate_fn if mode == "train" else collate_2d,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        sampler=dict(
            type=torch.utils.data.DistributedSampler,
            shuffle=False if mode == "test" else True,
            drop_last=False if mode == "test" else True,
        ),
        drop_last=True,
        pin_memory=True,
    )
    if is_iterabledataset:
        base_dataloader.update(
            sampler=None, drop_last=False, collate_fn=collate_2d
        )
        if sample_weights:
            assert len(dataset) == len(sample_weights)
        else:
            sample_weights = [1.0 for _ in dataset]
    if mode == "train":
        # In "train" mode, we build one dataloader for a "ConcatDataset" that
        # contains multiple datasets.
        if not is_iterabledataset:
            if len(dataset) > 1:
                dataset = dict(type="ConcatDataset", datasets=dataset)
            else:
                dataset = dataset[0]
        else:
            dataset = dict(
                type="DistributedComposeRandomDataset",
                datasets=dataset,
                sample_weights=sample_weights,
                multi_sample_output=True,
            )
        base_dataloader.update(dataset=dataset)
        dataloader = copy.deepcopy(base_dataloader)
    elif mode == "val" or mode == "test":
        # In "val" or "test" mode, we build a dataloader for each dataset so
        # that we can do validation and AIDI evaluation on multiple datasets.
        dataloader = []
        assert isinstance(
            dataset, list
        ), f"In {mode} mode, dataset should be a list."
        for ds in dataset:
            if is_iterabledataset:
                ds = dict(
                    type="DistributedComposeRandomDataset",
                    datasets=[ds],
                    sample_weights=[1],
                    multi_sample_output=True,
                    shuffle=False,
                )
            base_dataloader.update(dataset=ds)
            dataloader.append(copy.deepcopy(base_dataloader))
    else:
        raise ValueError(mode)

    return dataloader
