from typing import Optional, Sequence, Union

import torch


# reference: https://github.com/kornia/kornia/blob/master/kornia/geometry/conversions.py  # noqa: E501
def convert_points_from_homogeneous(
    points: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Convert points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

    Examples:
        >>> input = torch.tensor([[4., 4., 2.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[2., 2.]])
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(
            f"Input must be at least a 2D tensor. Got {points.shape}"
        )

    # we check for points at max_val
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: torch.Tensor = torch.abs(z_vec) > eps

    # TODO(xudong.he): BPU op Limited.
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """Convert points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(
            f"Input must be at least a 2D tensor. Got {points.shape}"
        )

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def _convert_affinematrix_to_homography_impl(A: torch.Tensor):
    H: torch.Tensor = torch.nn.functional.pad(
        A, [0, 0, 0, 1], "constant", value=0.0
    )
    H[..., -1, -1] += 1.0
    return H


def convert_affinematrix_to_homography(A: torch.Tensor):
    """Convert batch of affine matrices.

    Args:
        A: the affine matrix with shape :math:`(B,2,3)`.

    Returns:
        the homography matrix with shape of :math:`(B,3,3)`.

    Examples:
        >>> A = torch.tensor([[[1., 0., 0.],
        ...                    [0., 1., 0.]]])
        >>> convert_affinematrix_to_homography(A)
        tensor([[[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]]])
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(A)}")

    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError(f"Input matrix must be a Bx2x3 tensor. Got {A.shape}")

    return _convert_affinematrix_to_homography_impl(A)


def convert_affinematrix_to_homography3d(A: torch.Tensor) -> torch.Tensor:
    """Convert batch of 3d affine matrices.

    Args:
        A: the affine matrix with shape :math:`(B,3,4)`.

    Returns:
        the homography matrix with shape of :math:`(B,4,4)`.

    Examples:
        >>> A = torch.tensor([[[1., 0., 0., 0.],
        ...                    [0., 1., 0., 0.],
        ...                    [0., 0., 1., 0.]]])
        >>> convert_affinematrix_to_homography3d(A)
        tensor([[[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]]])
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(A)}")

    if not (len(A.shape) == 3 and A.shape[-2:] == (3, 4)):
        raise ValueError(f"Input matrix must be a Bx3x4 tensor. Got {A.shape}")

    return _convert_affinematrix_to_homography_impl(A)


def create_meshgrid(
    batch_size: int,
    height: int,
    width: int,
    device: Optional[Union[torch.device, str, None]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the
    pytorch function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        batch_size: batch size of coord grid.
        height: the image height (rows).
        width: the image width (cols).
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

    """
    xs: torch.Tensor = torch.linspace(
        0, width - 1, width, device=device, dtype=dtype
    )
    ys: torch.Tensor = torch.linspace(
        0, height - 1, height, device=device, dtype=dtype
    )
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys], indexing="ij"), dim=-1
    )  # WxHx2
    base_grid = base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2
    base_grid = base_grid.repeat(batch_size, 1, 1, 1)  # BxHxWx2
    return base_grid


def pad_reshape_length(input_shape: Sequence, target_length: int):
    """Pad the shape of input to target shape by pad length.

    Args:
        input_shape: input shape, e.g. (B, N, N)
        target_length: target shape length, e.g. 4

    Returns:
        tuple: padded shape
    """
    if len(input_shape) > target_length:
        raise ValueError(
            "The dimension of input shape should be less than or equal to "
            "the dimension of target shape."
        )
    assert (
        len(input_shape) > 1
    ), "The dimension of input shape should be greater than 1."
    if len(input_shape) == target_length:
        return input_shape
    new_shape = [1] * target_length
    new_shape[0] = input_shape[0]
    if len(input_shape) == 2:
        new_shape[-1] = input_shape[1]
    else:
        new_shape[-2:] = input_shape[-2:]
    return new_shape
