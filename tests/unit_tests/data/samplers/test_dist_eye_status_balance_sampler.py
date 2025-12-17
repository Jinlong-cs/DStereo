import torch
import torchvision

from hat.data.transforms.detection import ToTensor
from hat.registry import build_from_registry


def test_eye_status_distributed_balance_sampler():
    transforms = torchvision.transforms.Compose([ToTensor()])
    dataset = dict(
        type="EyeStatusDataset",
        rec_paths=[
            "./tmp_orig_data/face/eye_status/train_02.rec",
            "./tmp_orig_data/face/eye_status/train_03.rec",
        ],
        transforms=transforms,
    )

    sampler = dict(
        type="DistributedEyeStatusBalanceSampler",
        dataset=dataset,
        rank=0,
        num_replicas=1,
        drop_last=True,
        shuffle=True,
    )

    _ = build_from_registry(sampler)
    max_iter_time = 5
    sampler_iter = iter(sampler)
    for _ in range(max_iter_time):
        _ = sampler_iter.__next__()


def test_eye_status_distributed_balance_sampler_dataloader():
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="EyeStatusDataset",
            rec_paths=[
                "./tmp_orig_data/face/eye_status/train_02.rec",
                "./tmp_orig_data/face/eye_status/train_03.rec",
            ],
            transforms=[
                dict(
                    type="CropRecROI",
                    crop_type="scale",
                    target_shape=(96, 160, 3),
                    base_roi=[112, 186, 208, 346],
                    crop_jitter_range=0.1,
                    center_shift_range=0,
                ),
                dict(
                    type="RandomShiftRotateScale",
                    max_shift_range=(-0.1, 0.1),
                ),
                dict(
                    type="ToTensor",
                ),
            ],
        ),
        sampler=dict(
            type="DistributedEyeStatusBalanceSampler",
            rank=0,
            num_replicas=1,
            drop_last=True,
            shuffle=True,
        ),
    )

    dataloader = build_from_registry(data_loader)
    max_iter_time = 5
    iter_times = 0
    for _ in dataloader:
        if iter_times > max_iter_time:
            break
        iter_times += 1
