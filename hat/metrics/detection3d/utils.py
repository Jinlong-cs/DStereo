# type: ignore[no-untyped-def]

import numpy as np

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.utils.package_helper import require_packages


@require_packages("pycocotools")
def mioa_ignore(bbox2d, ignore_mask, thresh, decode_mask=True):
    if ignore_mask is None:
        return False
    bbox2d = np.array(bbox2d)
    if decode_mask:
        ignore_mask = coco_mask.decode(ignore_mask).astype(np.uint8)
    ignore_mask = ignore_mask.astype(np.uint8)
    mask = np.zeros_like(ignore_mask).astype(np.uint8)
    bbox2d = np.clip(
        bbox2d,
        [0, 0, 0, 0],
        [mask.shape[1], mask.shape[0], mask.shape[1], mask.shape[0]],
    )  # noqa
    mask[
        int(bbox2d[1]) : int(bbox2d[3] + 1),
        int(bbox2d[0]) : int(bbox2d[2] + 1),
    ] = 1  # noqa
    mask = mask * ignore_mask
    if (
        len(np.where(mask == 1)[0])
        > (bbox2d[2] - bbox2d[0] + 1) * (bbox2d[3] - bbox2d[1] + 1) * thresh
    ):  # noqa
        return True
    else:
        return False
