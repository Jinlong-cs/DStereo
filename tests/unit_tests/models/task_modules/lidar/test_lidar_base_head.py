# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from hat.models.task_modules.lidar.lidar_base_head import (
    LidarBaseHead,
    SepHead,
)


def test_sep_head():
    """Test ShareHead module. Assert output shapes."""
    # hm head has 20 classes and 2 conv layers
    # reg head has 4 classes and 4 conv layers
    heads = {
        "hm": (20, 2),
        "reg": (4, 4),
    }

    input_features = torch.randn(1, 64, 128, 128)

    sep_head = SepHead(
        in_channels=64,
        heads=heads,
        head_conv=64,
        bn=True,
        upsample_factor=4,  # Final output will be upsampled by 4
    )
    assert hasattr(sep_head, "hm") and hasattr(sep_head, "reg")

    output = sep_head(input_features)

    assert isinstance(output, dict)
    assert output["hm"].shape == (1, 20, 512, 512)
    assert output["reg"].shape == (1, 4, 512, 512)


def test_lidar_base_head():
    """Test lidar base head. Assert output shapes."""
    tasks = [
        dict(num_class=3, class_names=["VEHICLE", "PEDESTRIAN", "CYCLIST"])
    ]  # 1 task, 3 classes
    weight = 2.0
    code_weights = [1.0] * 8
    common_heads = {"reg": (2, 2), "dim": (3, 2)}

    head = LidarBaseHead(
        in_channels=64,
        tasks=tasks,
        weight=weight,
        code_weights=code_weights,
        common_heads=common_heads,
    )

    input_features = torch.randn(1, 64, 128, 128)
    output = head(input_features)

    assert isinstance(output, list) and len(output) == len(tasks)
    output = output[0]
    assert output["reg"].shape == (1, 2, 128, 128)
    assert output["dim"].shape == (1, 3, 128, 128)
