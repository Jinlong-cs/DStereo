# cannot support viz in ci now.
import numpy as np

from hat.visualize.seg import colorize, semantic_segmentation

COLOR = np.random.randint(0, 255, (255, 3))


def test_colorize():

    roi_img = np.random.randint(0, 255, (512, 512))
    img = colorize(roi_img, COLOR)

    assert img.shape == (512, 512, 3)
    assert (img[0][0] == COLOR[roi_img[0][0]]).all


def test_semantic_segmentation():

    roi_img = np.random.randint(0, 255, (3, 512, 512))
    img = semantic_segmentation(roi_img, COLOR)

    assert img.shape == (512, 512, 3)
