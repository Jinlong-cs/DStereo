import horizon_plugin_pytorch as horizon
import pytest
import torch

import hat
from hat.models.backbones.mobilenetv1 import MobileNetV1
from hat.models.model_convert.converters import TorchCompile
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.models.structures.classifier import Classifier


def test_inference():
    device = 0
    model = Classifier(
        backbone=MobileNetV1(num_classes=10, bn_kwargs={}, flat_output=False),
        losses=None,
    )
    march = horizon.march.March.BAYES
    runner = hat.engine.Inference(device, model, march)

    data = {"img": torch.randn(1, 3, 224, 224)}
    runner.forward(data)


@pytest.mark.skip(reason="CI env lack of TensorRT")
def test_tensorrt_inference():
    import torchdynamo

    from hat.utils.compile_backends import tensorRT_backend

    device = 0
    model = Classifier(
        backbone=MobileNetV1(num_classes=10, bn_kwargs={}, flat_output=False),
        losses=None,
    )
    march = horizon.march.March.BAYES
    data = {"img": torch.randn(1, 3, 224, 224)}
    # using official backend
    runner = hat.engine.Inference(
        device,
        model,
        march,
        model_convert_pipeline=ModelConvertPipeline([TorchCompile("fx2trt")]),
    )
    runner.forward(data)
    torchdynamo.reset()
    # using custom backend
    # fp32
    runner = hat.engine.Inference(
        device,
        model,
        march,
        model_convert_pipeline=ModelConvertPipeline(
            [TorchCompile(tensorRT_backend(fp16=False))]
        ),
    )
    runner.forward(data)
    torchdynamo.reset()
    # fp16
    runner = hat.engine.Inference(
        device,
        model,
        march,
        model_convert_pipeline=ModelConvertPipeline(
            [TorchCompile(tensorRT_backend(fp16=True))]
        ),
    )
    torchdynamo.reset()
    # fp16 + load group norm extension
    runner = hat.engine.Inference(
        device,
        model,
        march,
        model_convert_pipeline=ModelConvertPipeline(
            [
                TorchCompile(
                    tensorRT_backend(fp16=True), load_extensions=["group_norm"]
                )
            ]
        ),
    )
    runner.forward(data)
