#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import logging
from torch.utils.data.dataset import Dataset, ConcatDataset
import traceback

logger = logging.getLogger(__name__)
__all__ = ["CatRandomDataset"]


class CatRandomDataset(ConcatDataset):
    # def __init__(self, datasets, debug=False, max_size=12800):
    #     super(CatRandomDataset, self).__init__()
    #     self.datasets = datasets
    #     self.debug = debug
    #     self.max_size = max_size
    #     logger.info("###################### init CatRandomDataset done ######################")

    # def __len__(self):
    #     if self.debug:
    #         return 32
    #     else:
    #         n = 0
    #         for d in self.datasets:
    #             n += len(d)
    #         if self.max_size is None:
    #             return n
    #         else:
    #             return min(n, self.max_size)

    def __getitem__(self, item):
        while True:
            try:
                # dataset = self.datasets[i]
                # j = random.randint(0, len(dataset) - 1)
                # print(i,j)
                data = super().__getitem__(item)
                if not data["mask_flag"]:
                    item = random.randint(0, len(self) - 1)  # 295, 711  295, 400
                    continue
                return data
            except Exception as e:
                logger.info(traceback.format_exc())
                logger.info(e)
                item = random.randint(0, len(self) - 1)
                continue
