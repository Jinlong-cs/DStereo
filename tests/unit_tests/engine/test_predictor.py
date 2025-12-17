import random

import torch

from hat.callbacks.monitor import StatsMonitor
from hat.data.datasets.rand_dataset import RandDataset
from hat.engine import Predictor
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.engine.processors.processor import MultiBatchProcessor
from tests.data.toy_modules import ToyBackbone, ToyHead, ToyLoss, ToyModel

predictor = Predictor(
    model=ToyModel(
        backbone=ToyBackbone(strides=(1, 2), channels=(3, 8)),
        head=ToyHead(
            in_channels=8,
            fc_filter=16,
            num_classes=10,
            with_dequant=True,
        ),
        loss=ToyLoss(),
    ),
    data_loader=torch.utils.data.DataLoader(
        dataset=RandDataset(
            length=4,
            example=(torch.randn((3, 14, 14)), random.randint(0, 10 - 1)),
            clone=True,
        ),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    ),
    batch_processor=MultiBatchProcessor(
        need_grad_update=False,
        loss_collector=collect_loss_by_regex("^.*loss.*"),
    ),
    device=0,
    callbacks=[
        StatsMonitor(log_freq=1),
    ],
)


def test_predictor_fit():
    predictor.fit()
