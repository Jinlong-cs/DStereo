from torch import Tensor

from hat.models.losses.hand_bmcloss import H3DBMCLoss
from hat.models.losses.smooth_l1_loss import SmoothL1Loss

try:
    from pytorch3d.loss import (
        chamfer_distance,
        mesh_edge_loss,
        mesh_laplacian_smoothing,
        mesh_normal_consistency,
    )
    from pytorch3d.structures import Meshes
except Exception:
    chamfer_distance = None
    mesh_edge_loss = None
    mesh_laplacian_smoothing = None
    mesh_normal_consistency = None
    Meshes = Tensor


# Index of index finger joints
# fmt: off
INDEX_LIST = [
    48, 49, 56, 57, 58, 59, 62, 65, 86, 87, 127, 128, 132,
    133, 134, 135, 136, 137, 138, 139, 140, 144, 150, 156,
    164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
    175, 176, 177, 186, 189, 194, 195, 212, 213, 225, 258,
    259, 260, 261, 262, 263, 274, 280, 46, 47, 155, 221,
    222, 223, 224, 226, 237, 238, 245, 272, 273, 281, 282,
    283, 294, 295, 296, 297, 298, 299, 300, 301, 302, 316,
    321, 330, 331, 340, 341, 342, 344, 303, 304, 305, 306,
    307, 308, 309, 310, 311, 312, 313, 314, 315, 317, 318,
    319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 332,
    333, 334, 335, 336, 337, 338, 339, 343, 345, 346, 347,
    348, 349, 350, 351, 352, 353, 354, 355,
]
# fmt: on


def loss_smoothl1(
    pred: Tensor,
    target: Tensor,
    normlier: float = 1,
    sigma: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    smooth_l1_loss = SmoothL1Loss(
        beta=1.0 / (sigma ** 2),
        reduction=reduction,
    )
    while len(normlier.shape) < len(pred.shape):
        normlier = normlier.unsqueeze(-1)
    min_len = min(pred.shape[-1], target.shape[-1])
    return smooth_l1_loss(
        pred[..., :min_len], target[..., :min_len], weight=normlier
    )


# ----------------------------- regular related loss -------------------------
def loss_bmc_init(
    lambda_bl: float, lambda_rb: float, lambda_a: float
) -> Tensor:
    bmc_loss = H3DBMCLoss(
        lambda_bl=lambda_bl, lambda_rb=lambda_rb, lambda_a=lambda_a
    )

    def bmc_loss_forward(*args, **kwargs):
        _, bmc_loss_v = bmc_loss.compute_loss(*args, **kwargs)
        return bmc_loss_v

    return bmc_loss_forward


# ----------------------------- mesh related loss -----------------------------
def loss_mesh_edge(rerender_mesh: Meshes) -> Tensor:
    return mesh_edge_loss(rerender_mesh)


def loss_mesh_normal_consistency(rerender_mesh: Meshes) -> Tensor:
    return mesh_normal_consistency(rerender_mesh)


def loss_mesh_laplacian_smoothing(
    rerender_mesh: Meshes, method: str = "uniform"
) -> Tensor:
    return mesh_laplacian_smoothing(rerender_mesh, method=method)


def loss_mesh_chamfer_distance(
    verts_pred: Tensor, verts_gt: Tensor, normlier: float = 1, mode="FULL"
) -> Tensor:
    assert mode in ["FULL", "INDEX"]

    if mode == "INDEX":
        return chamfer_distance(
            verts_pred[:, INDEX_LIST, :] * normlier.reshape((-1, 1, 1)),
            verts_gt[:, INDEX_LIST, :] * normlier.reshape((-1, 1, 1)),
        )[0]
    else:
        return chamfer_distance(
            verts_pred * normlier.reshape((-1, 1, 1)),
            verts_gt * normlier.reshape((-1, 1, 1)),
        )[0]


# ----------------------------- render related loss -------------------------
def loss_ms_ssim_l1(
    pred: Tensor,
    target: Tensor,
    foreground_mask: Tensor = None,
    sigma: float = 10,
    normlier: float = 1,
    reduction: str = "mean",
    evaluate_in_rgb_space: bool = True,
    evaluate_with_color_var: bool = True,
) -> Tensor:

    raw_image_masked = (
        target[:, :3, :, :] * foreground_mask
        if foreground_mask is not None
        else target[:, :3, :, :]
    )
    if evaluate_in_rgb_space:
        raw_image_masked = raw_image_masked * 128 + 128
        pred *= 256

    smooth_l1_loss = SmoothL1Loss(
        beta=1.0 / (sigma ** 2),
        reduction=reduction,
    )
    ms_value = smooth_l1_loss(
        pred,
        raw_image_masked,
        weight=normlier.view(-1, 1, 1, 1),
    )

    if evaluate_with_color_var:
        ms_var = smooth_l1_loss(
            pred.var(dim=[2, 3]),
            raw_image_masked.var(dim=[2, 3]),
            weight=normlier.view(-1, 1, 1, 1),
        )
        ms_value += ms_var

    return ms_value
