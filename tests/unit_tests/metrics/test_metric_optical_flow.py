import pytest

from hat.metrics.metric_optical_flow import EndPointError
from tests.utils import gen_fake_torch_randn_data


@pytest.mark.parametrize(
    ["use_mask"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_metric_optical_flow(use_mask):
    metric = EndPointError("test_epe", use_mask=use_mask)
    preds = gen_fake_torch_randn_data((10, 2, 384, 512)).float()
    labels = gen_fake_torch_randn_data((10, 2, 384, 512)).float()
    masks = labels > 0.5
    metric.update(labels, preds, masks)
    _, result = metric.get()
    assert result is not None
