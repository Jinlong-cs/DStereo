import torch

from hat.data.collates.collates import collate_2d


def get_train_dataloader(
    lmdbs,
    nums,
    transforms,
    batch_size_per_gpu,
    num_workers,
    prefetch_factor,
):
    train_dataset = [
        dict(
            type="Auto2dFromLMDB",
            data_path=lmdb,
            num_samples=num,
            to_rgb=False,
            transforms=transforms,
            infer_model_type="resize",
        )
        for lmdb, num in zip(lmdbs, nums)
    ]

    train_datasets = dict(
        type="ConcatDataset",
        datasets=train_dataset,
    )

    train_dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=train_datasets,
        sampler=dict(type=torch.utils.data.DistributedSampler),
        collate_fn=collate_2d,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    return train_dataloader


def get_train_chunk_dataloader(
    lmdbs,
    nums,
    transforms,
    batch_size_per_gpu,
    num_workers,
    prefetch_factor,
):
    train_dataset = [
        dict(
            type="Auto2dFromLMDB",
            data_path=train_lmdb,
            num_samples=train_num,
            to_rgb=False,
            transforms=transforms,
            infer_model_type="resize",
        )
        for train_lmdb, train_num in zip(lmdbs, nums)
    ]

    train_datasets = dict(
        type=torch.utils.data.ChainDataset,
        datasets=[
            dict(
                type="ChunkShuffleDataset",
                dataset=dataset,
                chunk_size_in_worker=1024,
                drop_last=True,
            )
            for dataset in train_dataset
        ],
    )

    train_dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=train_datasets,
        collate_fn=collate_2d,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return train_dataloader


def get_val_dataloader(
    lmdbs,
    nums,
    transforms,
    batch_size_per_gpu,
    num_workers,
    prefetch_factor,
):
    val_dataset = [
        dict(
            type="Auto2dFromLMDB",
            data_path=lmdb,
            num_samples=num,
            to_rgb=False,
            transforms=transforms,
            infer_model_type="resize",
        )
        for lmdb, num in zip(lmdbs, nums)
    ]

    val_datasets = dict(
        type="ConcatDataset",
        datasets=val_dataset,
    )

    val_dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=val_datasets,
        batch_size=batch_size_per_gpu,
        collate_fn=collate_2d,
        sampler=dict(
            type=torch.utils.data.DistributedSampler,
            shuffle=False,
            drop_last=False,
        ),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return val_dataloader
