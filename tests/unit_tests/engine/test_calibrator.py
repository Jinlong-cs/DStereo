import random

import pytest
import torch

from hat.callbacks.monitor import StatsMonitor
from hat.data.datasets.rand_dataset import RandDataset
from hat.engine import Calibrator
from hat.engine.processors.processor import MultiBatchProcessor
from hat.models.model_convert.converters import Float2Calibration
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.utils.package_helper import check_packages_available
from tests.data.toy_modules import ToyBackbone, ToyHead, ToyLoss, ToyModel


def test_calibrator_fit():
    calibrator = Calibrator(
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
        model_convert_pipeline=ModelConvertPipeline(
            qat_mode="fuse_bn", converters=[Float2Calibration()]
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
            pin_memory=True,
        ),
        batch_processor=MultiBatchProcessor(
            need_grad_update=False,
        ),
        device=0,
        callbacks=[
            StatsMonitor(log_freq=1),
        ],
        num_steps=1,
    )
    calibrator.fit()


@pytest.mark.skipif(
    check_packages_available(
        "horizon_plugin_pytorch<2.0.1",
        raise_exception=False,
    ),
    reason="not supported version",
)
@pytest.mark.parametrize("preload_data", [True, False])
def test_auto_calibration(preload_data):
    calibrator = Calibrator(
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
        model_convert_pipeline=ModelConvertPipeline(
            qat_mode="fuse_bn", converters=[Float2Calibration()]
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
            pin_memory=True,
        ),
        batch_processor=MultiBatchProcessor(
            need_grad_update=False,
        ),
        device=0,
        callbacks=[
            StatsMonitor(log_freq=1),
        ],
        num_steps=1,
        auto_calibration=True,
        auto_calibration_config=dict(preload_data=preload_data),
    )
    calibrator.fit()


@pytest.mark.skipif(
    check_packages_available(
        "horizon_plugin_pytorch<2.0.1",
        raise_exception=False,
    ),
    reason="not supported version",
)
@pytest.mark.parametrize("preload_data", [True, False])
def test_weight_reconstruction(preload_data):
    calibrator = Calibrator(
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
        model_convert_pipeline=ModelConvertPipeline(
            qat_mode="fuse_bn", converters=[Float2Calibration()]
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
            pin_memory=True,
        ),
        batch_processor=MultiBatchProcessor(
            need_grad_update=False,
        ),
        device=0,
        callbacks=[
            StatsMonitor(log_freq=1),
        ],
        num_steps=1,
        weight_reconstruction=True,
        weight_reconstruction_config=dict(preload_data=preload_data),
    )
    calibrator.fit()
