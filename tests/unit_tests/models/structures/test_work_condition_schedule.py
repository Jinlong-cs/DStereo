import torch

from hat.registry import build_from_registry


def test_classifier_multitask():
    num_classes = 6
    width = 64
    input_hw = (234, 456)

    config = dict(
        type="WorkConditionClassifier",
        backbone=dict(
            type="WorkConditionResNet",
            layers=(3, 4, 6, 3),
            output_dim=1024,
            heads=32,
            input_resolution=input_hw,
            channel_list=[width // 2, width, width * 2, width * 4, width * 8],
            bn_kwargs={},
            use_attnpool=False,
        ),
        backbone_extra=torch.nn.Identity(),
        prediction_head=dict(
            type="WorkConditionClsHead",
            output_dim=1024,
            bn_kwargs={},
            num_classes=num_classes,
            in_channel=2048,
        ),
        losses=dict(
            type="CEWithLabelSmooth",
            smooth_alpha=0.01,
        ),
    )

    classifier_multitask = build_from_registry(config)

    x = {}
    x["img"] = torch.zeros((1, 3, *input_hw))
    x["labels"] = torch.zeros((1,)).long()

    y = classifier_multitask(x)

    assert isinstance(y, dict)
    assert y["preds"].shape[1] == num_classes
    print("ll")


test_classifier_multitask()
