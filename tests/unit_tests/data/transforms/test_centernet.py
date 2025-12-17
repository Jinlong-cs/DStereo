import numpy as np

from hat.data.transforms.centernet import CenterNetTargetGenerator


def test_centernet_target_generator():
    head_channels = dict(hm=1, wh=2)
    centernet_target_generator = CenterNetTargetGenerator(  # noqa: F841
        (960, 512),
        head_channels,
        down_stride=4,
    )
    img = np.zeros([960, 512, 3], dtype=np.uint8)
    gt_classes = np.random.randint(-1, 1, size=(5), dtype=np.int32)
    gt_bboxes = np.array(
        [
            [10, 10, 240, 280],
            [8, 6, 78, 23],
            [220, 10, 480, 230],
            [46, 53, 88, 99],
            [500, 20, 700, 320],
        ],
        dtype=np.float32,
    )
    data = {"img": img, "gt_classes": gt_classes, "gt_bboxes": gt_bboxes}
    output = centernet_target_generator(data)
    assert output["labels"]["hm"].shape == (1, 128, 240)
    assert output["labels"]["wh"].shape == (2, 128, 240)
    assert output["labels"]["ignore_mask"].shape == (1, 128, 240)
