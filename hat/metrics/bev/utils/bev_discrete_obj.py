from typing import List

import numpy as np
import torch
from shapely.geometry import Polygon


def get_topk_min_value(
    data: List,
    topk_norm: float,
) -> float:
    """
    Get the minimum value of the list topk.

    Args:
        data: list
            [x1, x2, ... xn]
        topk_norm: range [0, 1], if 0.05, means
            take 0.05 * data_length values
    """
    if not len(data):
        return 0
    data = torch.tensor(data)
    len_topk = int(topk_norm * len(data))
    len_topk = max(1, len_topk)
    min_value = torch.topk(data, len_topk)[0][-1].item()
    return min_value


def cal_iou_by_polygon(g, p):

    g = tuple(tuple(x) for x in g)
    p = tuple(tuple(x) for x in p)

    if np.isnan(g).any() or np.isnan(p).any():
        return 0
    g = Polygon(g)
    p = Polygon(p)
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union
