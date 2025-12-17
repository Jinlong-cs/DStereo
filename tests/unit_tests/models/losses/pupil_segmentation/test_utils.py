import numpy as np
import pytest
import torch

from hat.models.losses.pupil_segmentation.utils import (
    create_meshgrid,
    get_mask,
    norm_pts,
    pupil_center_from_mask,
)


@pytest.mark.parametrize(
    "normalized_coordinates",
    [True, False],
)
def test_create_meshgrid(normalized_coordinates):
    height, width = 3, 3
    gt_norm = np.array(
        [
            [
                [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
                [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
            ]
        ]
    )
    gt_without_norm = np.array(
        [
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]],
            ]
        ]
    )
    res = create_meshgrid(height, width, normalized_coordinates)
    if normalized_coordinates:
        assert (res.numpy() - gt_norm).sum() == 0
    else:
        assert (res.numpy() - gt_without_norm).sum() == 0


def test_get_mask():
    mesh = torch.Tensor(
        [
            [
                [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
                [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
            ]
        ]
    )
    x = torch.Tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    y = torch.Tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gt_posmask_1 = torch.sigmoid(
        64 * ((x / 1) ** 2 + (y / 1) ** 2 - 1)
    ).numpy()
    gt_negmask_1 = torch.sigmoid(
        64 * (1 - (x / 1) ** 2 - (y / 1) ** 2)
    ).numpy()

    gt_posmask_2 = torch.sigmoid(
        64 * ((x / 0.5) ** 2 + (y / 0.5) ** 2 - 1)
    ).numpy()
    gt_negmask_2 = torch.sigmoid(
        64 * (1 - (x / 0.5) ** 2 - (y / 0.5) ** 2)
    ).numpy()

    ellipse_param = torch.Tensor([[0, 0, 1, 1, 0], [0, 0, 0.5, 0.5, 0]])
    posmask, negmask = get_mask(mesh, ellipse_param)
    assert abs(posmask[0].numpy() - gt_posmask_1).sum() < 1e-4
    assert abs(posmask[1].numpy() - gt_posmask_2).sum() < 1e-4
    assert abs(negmask[0].numpy() - gt_negmask_1).sum() < 1e-4
    assert abs(negmask[1].numpy() - gt_negmask_2).sum() < 1e-4


def test_norm_pts():
    sz = (3, 3)
    pts = np.array([[1.0, 1.0], [0.0, 0.0]])
    notm_pts = norm_pts(pts, sz)
    assert (
        abs(notm_pts - np.array([[-1 / 3, -1 / 3], [-1.0, -1.0]])).sum() < 1e-4
    )


def test_pupil_center_from_mask():
    mask = torch.Tensor(
        [
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [[0, 1, 0], [0.5, 1.0, 0.5], [0.5, 0.5, 0.5]],
        ]
    )
    c = pupil_center_from_mask(mask)
    assert abs(c.numpy() - np.array([[0.0, 0.0], [0.0, 0.0145]])).sum() < 1e-4
