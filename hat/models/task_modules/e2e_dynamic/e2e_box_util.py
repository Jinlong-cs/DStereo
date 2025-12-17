import math

import torch
from torchvision.ops.boxes import box_area


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=1e-9)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union.clamp(min=1e-9)
    return iou, union


def matched_boxlist_iou(boxes1, boxes2) -> torch.Tensor:
    """Compute pairwise intersection over union (IOU).

    using in two sets of matched boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal,
    elements of the matrix

    Args:
        boxes1: (Boxes) bounding boxes, sized [N,4].
        boxes2: (Boxes) bounding boxes, sized [N,4].
    Returns:
        Tensor: iou, sized [N].
    """

    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    box1, box2 = boxes1, boxes2
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/.

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def distance_box_iou(boxes1, boxes2):
    """
    Distance IoU below paper.

    https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1911.08287.pdf

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    rows, cols = boxes1.shape[0], boxes2.shape[0]
    device = boxes1.device
    dious = torch.zeros((rows, cols)).to(device)
    if rows * cols == 0:
        return dious
    iou, _ = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    outer_diag = torch.sum(wh ** 2, dim=-1, keepdim=False)  # [N, M]
    centers1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2.0
    centers2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2.0
    centers_diag = centers1[:, None] - centers2[None]  # [N, M, 2]
    centers_diag = torch.sum(centers_diag ** 2, dim=-1, keepdim=False)
    return iou - centers_diag / (outer_diag + 1e-9)


def get_rdiou(boxes1, boxes2, yaws1=None, yaws2=None):
    """Calculate Rotation-Decoupled IoU.

    Refering paper: `Rethinking IoU-based Optimization for Single-stage 3D
    Object Detection`.

    boxes: N x 4, [x0, y0, x1, y1]
    yaws: N x 2, [cos(yaw), sin(yaw)], k=1 in default
    """
    k = 1
    area1 = box_area(boxes1) * k
    area2 = box_area(boxes2) * k
    inner_lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    inner_rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inner_wh = (inner_rb - inner_lt).clamp(min=1e-9)  # [N,M,2]
    if yaws1 is not None:
        assert yaws2 is not None
        tmp_yaws = (
            yaws1.unsqueeze(1) * yaws2.unsqueeze(0)[..., [1, 0]]
        )  # [N, M, 2]
        delta_yaws = (
            torch.min(tmp_yaws + k / 2, dim=-1)[0]
            - torch.max(tmp_yaws - k / 2, dim=-1)[0]
        )  # [N, M]
    else:
        assert yaws2 is None
        tmp_yaws = torch.zeros_like(inner_wh)
        delta_yaws = torch.ones_like(inner_wh[..., 1])
        k = 0
    inter = inner_wh[..., 0] * inner_wh[..., 1] * delta_yaws.clamp(min=1e-9)
    union = area1[:, None] + area2 - inter
    riou = inter / union.clamp(min=1e-9)
    return riou, union, tmp_yaws, k


def complete_box_iou(boxes1, boxes2, yaws1=None, yaws2=None):
    """Complete IoU of boxes.

    The boxes should be in [x0, y0, x1, y1] format

    boxes: N x 4, [x0, y0, x1, y1]
    yaws: N x 2, [cos(yaw), sin(yaw)]
    """
    assert (yaws1 is None) == (yaws2 is None)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    rows, cols = boxes1.shape[0], boxes2.shape[0]
    device = boxes1.device
    dious = torch.zeros((rows, cols)).to(device)
    if rows * cols == 0:
        return dious

    riou, _, tmp_yaws, k = get_rdiou(boxes1, boxes2, yaws1, yaws2)
    outer_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    outer_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    outer_wh = (outer_rb - outer_lt).clamp(min=1e-5)  # [N,M,2]
    outer_diag = torch.sum(outer_wh ** 2, dim=-1, keepdim=False)  # [N, M]
    delta_yaws = (
        torch.max(tmp_yaws + k / 2, dim=-1)[0]
        - torch.min(tmp_yaws - k / 2, dim=-1)[0]
    )  # [N, M]
    outer_diag = outer_diag + delta_yaws ** 2

    centers1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2.0
    centers2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2.0
    centers_diag = centers1[:, None] - centers2[None]  # [N, M, 2]
    centers_diag = torch.sum(centers_diag ** 2, dim=-1, keepdim=False)
    centers_diag = (
        centers_diag + (tmp_yaws[..., 0] - tmp_yaws[..., 1]) ** 2
    )  # [N, M]

    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 0]
    v = (4 / (math.pi ** 2)) * torch.pow(
        (
            torch.atan(w2 / (h2 + 1e-5))[None]
            - torch.atan(w1 / (h1 + 1e-5))[:, None]
        ),
        2,
    )
    with torch.no_grad():
        S = (riou > 0.5).float()
        alpha = S * v / torch.clamp(1 - riou + v, min=1e-9)
    cious = riou - centers_diag / (outer_diag + 1e-9) - alpha * v
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    return cious
