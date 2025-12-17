import os

import torch

from hat.registry import build_from_registry


def test_perf_transforms():
    perf_transform = dict(
        type="PerfTransforms",
        transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="RandomResizedCrop",
                size=224,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            dict(
                type="TorchVisionAdapter",
                interface="RandomResizedCrop",
                size=224,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
        ],
        profiler=dict(
            type="SimpleProfiler",
            dirpath="./",
            filename="simple_profiler",
        ),
        perf_data_len=5,
    )
    data = dict(img=torch.randn(1, 3, 224, 224))
    perf_transform = build_from_registry(perf_transform)
    for _ in range(5):
        perf_transform(data)
    assert os.path.exists("./simple_profiler.txt")
