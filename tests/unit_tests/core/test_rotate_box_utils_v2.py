import numpy as np

bbox1 = np.array(
    [
        [0, 0, 4, 1.5, -np.pi / 4],
        [1.54, -3.57, 4, 1.25, -np.pi / 3],
        [1, 1.2, 4, 1.5, -np.pi / 4],
        # [34.8982048, -4.01927567, 4.15128033, 1.70010416, 0.0371456],
    ]
)
bbox2 = np.array(
    [
        [0, 0, 4, 2, 0],
        [2.33, 1.65, 3.6, 2.01, 0.736],
        [1, 1.2, 4, 1.5, 0],
        # [34.84272, -3.9025896, 4.104194, 1.6739343, -0.06501262],
    ]
)


def test_rotate_iou_v2():

    from hat.core.rotate_box_utils_v2 import rotate_iou_v2

    expected_iou = np.array(
        [
            [0.43085998, 0.02505888, 0.20547356],
            [0.0, 0.0, 0.0],
            [0.22546186, 0.25240228, 0.36084977],
            # [0, 0, 0, 0.8301288],
        ]
    )
    iou = rotate_iou_v2(bbox1, bbox2, -1)
    assert np.all(abs(iou - expected_iou) < 1e-6)
