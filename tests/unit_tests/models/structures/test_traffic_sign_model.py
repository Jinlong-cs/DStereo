import pytest
import torch
from horizon_plugin_pytorch.march import March

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    ["training_step", "march"],
    [
        pytest.param("float", March.BERNOULLI2),
        pytest.param("int_infer", March.BERNOULLI2),
        pytest.param("int_infer", March.BAYES),
    ],
)
def test_traffic_sign_classifier_multitask(training_step, march):
    num_classes = 258
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    backbone = dict(
        type="TinyVargNetV2",
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        alpha=0.5,
        group_base=8,
        include_top=False,
        extend_features=False,
        input_resize_scale=None,
        channel_list=[32, 32, 64, 128, 256],
    )
    cfg = dict(
        type="TrafficSignClassifierMultitask",
        backbone=backbone,
        backbone_extra=torch.nn.Identity(),
        prediction_head=dict(
            type="TinyVarGNetV2ClassificationHead",
            input_channels=int(
                backbone["channel_list"][-1] * backbone["alpha"]
            ),
            disable_quanti_input=True,
            num_classes=num_classes,
            gc_group_base=8,
            bn_kwargs=bn_kwargs,
            alpha=0.5,
            cls_pooling_stride=2,
            factor=2,
            export_model=True if training_step == "int_infer" else False,
            output_by_argmax=False if march == March.BAYES else True,
        ),
        losses=dict(
            type="CEWithLabelSmooth",
            smooth_alpha=0.01,
            ignore_index=-1,
        ),
        desc=None,
    )
    traffic_sign_model = build_from_registry(cfg)
    x = dict(
        img=torch.rand(1, 3, 64, 64),
    )
    traffic_sign_model(x)
    qat_test(traffic_sign_model, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
