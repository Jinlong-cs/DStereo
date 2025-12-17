# Copyright (c) Horizon Robotics. All rights reserved.
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hat.models.losses.dist_faceid_softmax import DistFCCrossEntropyLoss
from hat.models.losses.faceid import ArcFace
from hat.utils.distributed import find_free_port


def _main_worker(rank, world_size):
    # init process group
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    batch_size = 2
    total_pred = torch.FloatTensor(
        [
            [0.4475, 1.6851, -1.6667, 0.7341, -2.0608],
            [0.0953, -0.3291, 0.3174, 0.6713, 1.3140],
            [0.4475, 1.6851, -1.6667, 0.7341, -2.0608],
            [0.0953, -0.3291, 0.3174, 0.6713, 1.3140],
        ]
    ).to("cuda:{}".format(rank))
    total_target = torch.LongTensor([3, 1, 4, 2]).to("cuda:{}".format(rank))
    pred = total_pred[
        rank * batch_size : (rank + 1) * batch_size,
    ]
    target = total_target[
        rank * batch_size : (rank + 1) * batch_size,
    ]

    target_loss = torch.FloatTensor([64.485]).to("cuda:{}".format(rank))

    dist_cross_entropy_loss = DistFCCrossEntropyLoss(
        resume=False,
        margin_loss=ArcFace(),
        loss_name="dist_softmax_loss",
        num_classes=5,
        embedding_size=5,
        gpu_per_device=2,
        seed=123,
    )

    loss = dist_cross_entropy_loss(pred, target)
    assert torch.abs(loss["dist_softmax_loss"] - target_loss) < 1e-3
    dist.destroy_process_group()


@pytest.mark.serial_task
def test_with_dist_mode():
    num_workers = 2
    host_name = "localhost"
    port = find_free_port()
    os.environ["MASTER_ADDR"] = host_name
    os.environ["MASTER_PORT"] = str(port)

    mp.spawn(_main_worker, nprocs=num_workers, args=(num_workers,))
