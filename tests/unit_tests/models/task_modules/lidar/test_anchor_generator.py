from typing import List, Tuple

from hat.models.task_modules.lidar import Anchor3DGeneratorStride


def test_anchor3dgeneratorstride():

    anchor_generator = Anchor3DGeneratorStride(
        anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
        anchor_strides=[[0.32, 0.32, 0.0]],  # noqa B006
        anchor_offsets=[[0.16, -39.52, -1.78]],  # noqa B006
        rotations=[[0, 1.57]],  # noqa B006
        class_names=["Car"],
        match_thresholds=[0.6],
        unmatch_thresholds=[0.45],
    )

    feature_maps_size = [1, 20, 30]

    anchors = anchor_generator.generate_anchors(feature_maps_size)

    assert isinstance(anchors, (List, Tuple))
    assert len(anchors) == 1
    assert anchors[0].shape == (1, 20, 30, 2, 7)
