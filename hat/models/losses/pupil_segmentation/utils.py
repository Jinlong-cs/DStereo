import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def create_meshgrid(
    height: int, width: int, normalized_coordinates: Optional[bool] = True
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys])).transpose(
        1, 2
    )  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def pupil_center_from_mask(mask: torch.Tensor, temperature: int = 1):
    # Custom function to find the center of mass to get detected pupil
    # mask: BXHXW - multiple channel corresponding to pupil predictions
    B, H, W = mask.shape
    wt_map = F.softmax(mask.view(B, -1) * temperature, dim=1)  # [B, HXW]

    xy_grid = create_meshgrid(H, W, normalized_coordinates=True)  # 1xHxWx2

    xloc = xy_grid[0, :, :, 0].reshape(-1).to(mask.device)
    yloc = xy_grid[0, :, :, 1].reshape(-1).to(mask.device)

    xpos = torch.sum(wt_map * xloc, -1, keepdim=True)
    ypos = torch.sum(wt_map * yloc, -1, keepdim=True)
    pred_pts = torch.cat([xpos, ypos], dim=1)
    return pred_pts


def soft_heaviside(x, sc, mode):
    # Given an input and a scaling factor (default 64), the soft heaviside \
    # function approximates the behavior of a 0 or 1 operation in a \
    # differentiable manner.

    # Note the max values in the heaviside function are scaled to 0.9. \
    # This scaling is for convenience and stability with bCE loss.

    sc = torch.tensor([sc]).to(torch.float32).to(x.device)
    if mode == 1:
        # Original soft-heaviside
        # Try sc = 64
        return 0.9 / (1 + torch.exp(-sc / x))
    elif mode == 2:
        # Some funky shit but has a nice gradient
        # Try sc = 0.001
        return 0.45 * (1 + (2 / np.pi) * torch.atan2(x, sc))
    elif mode == 3:
        # Good ol' scaled sigmoid. FUTURE: make sc free parameter
        # Try sc = 8
        return torch.sigmoid(sc * x)
    else:
        raise Exception("Mode undefined,  which can only be 1,2,3")


def get_mask(mesh, ellipse_param):
    # posmask: Positive outside the ellipse
    # negmask: Positive inside the ellipse
    # mesh: 1xHxWx2 or HxWx2,
    X = (mesh[..., 0] - ellipse_param[:, 0].reshape(-1, 1, 1)) * torch.cos(
        ellipse_param[:, -1]
    ).reshape(-1, 1, 1) + (
        mesh[..., 1] - ellipse_param[:, 1].reshape(-1, 1, 1)
    ) * torch.sin(
        ellipse_param[:, -1].reshape(-1, 1, 1)
    )
    Y = -(mesh[..., 0] - ellipse_param[:, 0].reshape(-1, 1, 1)) * torch.sin(
        ellipse_param[:, -1].reshape(-1, 1, 1)
    ) + (mesh[..., 1] - ellipse_param[:, 1].reshape(-1, 1, 1)) * torch.cos(
        ellipse_param[:, -1].reshape(-1, 1, 1)
    )
    posmask = (
        (X / ellipse_param[:, 2].reshape(-1, 1, 1)) ** 2
        + (Y / ellipse_param[:, 3].reshape(-1, 1, 1)) ** 2
        - 1
    )
    negmask = (
        1
        - (X / ellipse_param[:, 2].reshape(-1, 1, 1)) ** 2
        - (Y / ellipse_param[:, 3].reshape(-1, 1, 1)) ** 2
    )
    posmask = soft_heaviside(posmask, sc=64, mode=3)
    negmask = soft_heaviside(negmask, sc=64, mode=3)
    return posmask, negmask


def norm_pts(pts, sz):
    pts_o = copy.deepcopy(pts)
    res = pts_o.shape
    pts_o = pts_o.reshape(-1, 2)
    pts_o[:, 0] = 2 * (pts_o[:, 0] / sz[1]) - 1
    pts_o[:, 1] = 2 * (pts_o[:, 1] / sz[0]) - 1
    pts_o = pts_o.reshape(res)
    return pts_o
