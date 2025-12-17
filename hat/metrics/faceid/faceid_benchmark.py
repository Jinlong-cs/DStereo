# -*- coding: utf-8 -*-
import logging
from typing import Callable

import numpy as np

from .metric import compute_leuclidean_distance, find_metric_threshold

logger = logging.getLogger(__name__)


class FeatSet(object):
    """FeatSet computes faceid GAR.

    Args:
        id : faceid label, shape: [N]
        feat : faceid feature, shape: [N, emb_size]
    """

    def __init__(self, id: np.ndarray, feat: np.ndarray):
        assert id.shape[0] == feat.shape[0]
        self._id = id
        self._feat = feat
        self._len = id.shape[0]

    @property
    def len(self):
        return self._len

    @property
    def feat(self):
        return self._feat

    @property
    def id(self):
        return self._id

    def GAR(
        self,
        L: Callable = None,
        dump_dist: bool = False,
        dump_intra_path: str = None,
        dump_inter_path: str = None,
        output_path: str = None,
    ):
        if L is None:
            dist = compute_leuclidean_distance(self._feat, self._feat)
        else:
            dist = L(self._feat, self._feat)

        return find_metric_threshold(
            self._feat,
            self._id,
            self._feat,
            self._id,
            dist=dist,
            dump_dist=dump_dist,
            dump_intra_path=dump_intra_path,
            dump_inter_path=dump_inter_path,
            output_path=output_path,
        )
