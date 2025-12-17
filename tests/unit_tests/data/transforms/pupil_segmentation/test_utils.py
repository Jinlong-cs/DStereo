import numpy as np

from hat.data.transforms.pupil_segmentation import Ellipse


def test_ellipse():
    param = [
        43.043212890625,
        23.693267822265625,
        9.960553169250488,
        13.157154083251953,
        56.03654861450195,
    ]
    param[-1] = np.deg2rad(param[-1])
    ellipse = Ellipse(tuple(param))
    p = ellipse.mat2param(ellipse.mat)[:-1]
    for i in range(5):
        assert abs(param[i] - p[i]) < 1e-5
