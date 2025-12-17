import torch

from hat.models.task_modules.pointpillars.head import PointPillarsHead


def test_pointpillars_head():

    head = PointPillarsHead(
        in_channels=128,
        num_classes=1,
        anchors_num_per_class=2,
        use_direction_classifier=True,
        num_direction_bins=2,
        box_code_size=7,
    )

    input_features = torch.randn(1, 128, 320, 320)
    output = head(input_features)

    assert isinstance(output, (list, tuple)) and len(output) == 3

    assert output[0].shape == (1, 14, 320, 320)  # box
    assert output[1].shape == (1, 2, 320, 320)  # cls
    assert output[2].shape == (1, 4, 320, 320)  # dir
