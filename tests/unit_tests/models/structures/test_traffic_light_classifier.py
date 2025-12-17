import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test

try:
    import hatbc
except ImportError:
    hatbc = None


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.parametrize(
    ["input_preprocess", "mask_in_bpu", "num_classes"],
    [
        pytest.param(
            dict(
                type="TrafficLightPreprocess",
                mask_in_bpu=True,
                transpose_hw=False,
                node_name="input_preprocess",
            ),
            True,
            3,
        ),
        pytest.param(None, False, 3),
    ],
)
def test_traffic_light_classifier(input_preprocess, mask_in_bpu, num_classes):

    config = dict(
        type="TrafficLightClassifier",
        input_preprocess=input_preprocess,
        backbone_extra=torch.nn.Identity(),
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
            alpha=0.5,
            group_base=8,
            include_top=False,
            disable_quanti_input=mask_in_bpu,
            input_resize_scale=None,
            channel_list=[32, 32, 64, 128, 256],
            node_name="backbone",
        ),
        prediction_head=dict(
            type="TinyVarGNetV2ClassificationHead",
            input_channels=128,
            disable_quanti_input=True,
            num_classes=num_classes,
            gc_group_base=8,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
            alpha=0.5,
            cls_pooling_stride=2,
            cls_pooling_padding=0,
            factor=2,
            node_name="prediction_head",
        ),
        losses=None,
        mask_in_bpu=mask_in_bpu,
    )
    traffic_model = build_from_registry(config)
    x = dict(
        img=torch.rand(1, 3, 96, 96),
        mask_width=torch.rand(1, 1, 1, 96),
        mask_height=torch.rand(1, 1, 96, 1),
    )
    batch_out = traffic_model(x)
    assert batch_out["pred_cls"].shape[-1] == num_classes
    qat_test(traffic_model, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
