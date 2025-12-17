import pytest
import torch

from hat.models.task_modules.output_module import OutputModule
from tests.data.toy_modules import (
    ToyHead,
    ToyHeadParser,
    ToyLoss,
    ToyPostProcess,
    ToyTarget,
)

head = ToyHead(
    in_channels=512,
    fc_filter=128,
    num_classes=1000,
    with_dequant=True,
)

head_parser = ToyHeadParser()

features = torch.randn((1, 512, 7, 7))
label = torch.randint(0, 1000, (1,))


def test_output_module_train():
    mod = OutputModule(
        head=head,
        head_parser=head_parser,
        target=ToyTarget(),
        loss=ToyLoss(),
        prefix="toy",
    )

    assert mod.has_target
    assert mod.with_loss
    assert not mod.with_postprocess

    mod.train()
    result = mod(features, label=label)

    assert isinstance(result, dict)
    assert "toy_predict" in result
    assert "toy_ToyLoss" in result
    assert "toy_target" in result


@pytest.mark.parametrize(
    ["postprocess"],
    [
        pytest.param(ToyPostProcess()),
        pytest.param([ToyPostProcess(), ToyPostProcess()]),
    ],
)
def test_output_module_test(postprocess):
    mod = OutputModule(
        head=head,
        head_parser=head_parser,
        postprocess=postprocess,
        prefix="toy",
    )

    assert not mod.with_loss
    assert not mod.has_target
    assert mod.with_postprocess

    pred = mod(features)
    assert isinstance(pred, dict)
    assert "toy_predict" in pred
    assert len(pred) == 1
