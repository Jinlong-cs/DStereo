import numpy as np

from hat.core.circle_nms_jit import circle_nms


def test_circle_nms():

    box_x = np.random.uniform(0, 100, (10, 1))
    box_y = np.random.uniform(0, 100, (10, 1))
    box_score = np.random.uniform(0, 1.0, (10, 1))

    boxes = np.concatenate([box_x, box_y, box_score], axis=1)

    keep = circle_nms(boxes, thresh=0.5)
    assert len(keep) <= boxes.shape[0]
    assert box_score[keep[0]] >= box_score[keep[1]]
