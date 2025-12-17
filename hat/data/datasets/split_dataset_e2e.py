import math
from typing import List

import numpy as np
import torch

from hat.data.datasets.dataset_wrappers import ConcatDataset
from hat.data.datasets.split_dataset import (
    RankSplitDataLoader,
    RankSplitDataLoaderIter,
    RankSplitDataset,
    find_index,
)
from hat.registry import OBJECT_REGISTRY, build_from_registry
from hat.utils.distributed import get_dist_info

__all__ = ["ANCRankSplitE2EDataset", "ANCRankSplitE2EDataLoader"]


@OBJECT_REGISTRY.register
class ANCRankSplitE2EDataset(RankSplitDataset):
    """RankSplitE2EDataset is used for split concat E2E dataset on different rank.

    RankSplitE2EDataset must be used with RankSplitE2EDataLoader and it can
    avoids consuming excessive MEM and Virtual MEM resources,
    get more information from
    https://horizonrobotics.feishu.cn/wiki/wikcnALBu8Y6CkqQewtFIhANGGe.

    .. note::
        给定:
            batch_size=3, sub_clip_num=4,
            world_size=2, len(dataset) = 420

        这里dataset的长度已经在auto3dv的sync_info中被扩大了sub_clip_num倍,
        所以len(dataset)必定能被sub_clip_num整除.

        首先根据len(dataset)计算clip数量:
            total_clip_len = len(dataset) // sub_clip_num = 105

        然后将total_clip_len分成world_size, 计算每个rank含有多少个clip:
            world_size = math.ceil(
                (total_clip_len - world_size) / float(world_size)
                ) = 52

        然后根据给定的batch_size进行取整:
            world_step = math.ceil(
                (world_step - batch_size) / float(batch_size)
                ) * batch_size = 51

        最后根据计算结果计算每个rank含有的sub_clip_num数量:
            self.world_step = 51 * sub_clip_num = 204

    Args:
        datasets: A list of dataset cfg.
        with_flag: Whether to concatenate datasets flags.
            If True, concatenate all datasets flag (all datasets must has
            flag attribute in this case). Default to False.
        sub_clip_num: The number of sub_clip in a clip.
        batch_size: Batch size of every iteration.
    """

    def __init__(
        self,
        datasets: List[dict],
        with_flag: bool = False,
        sub_clip_num: int = 1,
        batch_size: int = 1,
    ):
        self.sub_clip_num = sub_clip_num
        self.batch_size = batch_size
        self.len_list = []
        flag_list = []
        for dataset in datasets:
            dataset = build_from_registry(dataset)
            self.len_list.append(len(dataset))

            if with_flag:
                assert hasattr(dataset, "flag"), "dataset must has group flag"
                assert isinstance(
                    dataset.flag, np.ndarray
                ), "dataset flag must is numpy array instance"
                assert (
                    len(dataset) == dataset.flag.shape[0]
                ), "dataset flag length at axis 0 must equal to the dataset length"  # noqa: E501
                flag_list.append(dataset.flag)

            del dataset
        if with_flag:
            self.flag_list = np.concatenate(flag_list)

        self.rank, self.world_size = get_dist_info()

        # the lens of input datasets has been enlarged by sub_clip_num
        # times in the SyncInfo of Auto3dv,
        # so sum(self.len_list) % self.sub_clip_num == 0.
        self.total_clip_len = sum(self.len_list) // self.sub_clip_num

        # Split datasets with world_size,
        # To balance the number of dataitem in different ranks, we split
        # datasets by clip lens instead of dataset instance.
        if self.total_clip_len % self.world_size != 0:
            world_step = math.ceil(
                (self.total_clip_len - self.world_size)
                / float(self.world_size)
            )

        else:
            world_step = math.ceil(
                self.total_clip_len / float(self.world_size)
            )
        # 每个rank中的clip数量必须保证能被batch_size整除，否则会导致clip不完整。
        if world_step % self.batch_size != 0:
            world_step = (
                math.ceil(
                    (world_step - self.batch_size) / float(self.batch_size)
                )
                * self.batch_size
            )

        # sub_clip_num for every world
        self.world_step = world_step * self.sub_clip_num

        # find datasets by split index
        sub_index = list(range(sum(self.len_list)))[
            self.world_step * self.rank : self.world_step * (self.rank + 1)
        ]

        # find datasets by split index
        ds_begin, _ = find_index(self.len_list, sub_index[0])
        ds_end, _ = find_index(self.len_list, sub_index[-1])
        ds_end += 1

        datasets_build = []

        for i in range(ds_begin, ds_end):
            datasets_build.append(build_from_registry(datasets[i]))
        self.datasets = ConcatDataset(datasets_build)

    def __len__(self):
        # this is fake length, real length will be sampled in Sampler.
        # return the number of sub_clip in all rank.
        return self.world_step * self.world_size


class ANCRankSplitE2EDataLoaderIter(RankSplitDataLoaderIter):
    def __init__(self, *args, **kwargs):
        super(ANCRankSplitE2EDataLoaderIter, self).__init__(*args, **kwargs)

    def _try_put_index(self):
        assert (
            self._tasks_outstanding < self._prefetch_factor * self._num_workers
        )

        try:
            index = self._next_index()
        except StopIteration:
            return

        # hardcode batch_size, its error while batchsize < len(dataset)
        base_seed = 0

        if self._send_idx == 0:
            self._real_send_idx = self._send_idx
            self._batch_size = len(index)
            self._generator = torch.Generator()
            self._generator.manual_seed(base_seed)
            # base_seed += 1

            # split dataset
            rank = self._dataset.rank
            sub_clip_num = self._dataset.sub_clip_num
            world_step = self._dataset.world_step

            sub_index = range(sum(self._dataset.len_list))[
                world_step * rank : world_step * (rank + 1)
            ]

            global_index = world_step * rank
            _, offset = find_index(self._dataset.len_list, global_index)
            self._offset = offset

            if hasattr(self._dataset, "flag_list"):
                self._flag = self._dataset.flag_list[sub_index]

                # calculate rounded number of sub_clip with different flag
                # in this rank.
                group_sizes = np.bincount(self._flag)
                num_samples = 0
                for size in group_sizes:
                    num_samples += (
                        int(
                            math.ceil(
                                size * 1.0 / self._batch_size / sub_clip_num
                            )
                        )
                        * self._batch_size
                        * sub_clip_num
                    )

                perm_indices = []
                for i, size in enumerate(group_sizes):
                    if size <= 0:
                        continue

                    indice = np.where(self._flag == i)[0]
                    assert len(indice) == size

                    # squeeze for sampling, perm_indice is the first indice of
                    # sub_clip of every clip.
                    perm_indice = indice[::sub_clip_num]
                    perm_size = size // sub_clip_num

                    # shuffle the indice.
                    # add .numpy() to avoid bug when selecting indice in parrots.  # noqa
                    perm_indice = perm_indice[
                        list(
                            torch.randperm(
                                int(perm_size), generator=self._generator
                            ).numpy()
                        )
                    ].tolist()

                    perm_extra = int(
                        math.ceil(perm_size * 1.0 / self._batch_size)
                    ) * self._batch_size - len(perm_indice)

                    # pad indice
                    perm_tmp = perm_indice.copy()
                    for _ in range(perm_extra // perm_size):
                        perm_indice.extend(perm_tmp)
                    perm_indice.extend(perm_tmp[: perm_extra % perm_size])
                    perm_indices.extend(perm_indice)

                perm_indices = [
                    perm_indices[j]
                    for i in list(
                        torch.randperm(
                            len(perm_indices) // self._batch_size,
                            generator=self._generator,
                        )
                    )
                    for j in range(
                        i * self._batch_size, (i + 1) * self._batch_size
                    )
                ]

                # unsqueeze for get
                random_index = []
                for i in range(len(perm_indices) // self._batch_size):
                    for j in range(sub_clip_num):
                        random_index += (
                            np.array(
                                perm_indices[
                                    i
                                    * self._batch_size : (i + 1)
                                    * self._batch_size
                                ]
                            )
                            + j
                        ).tolist()
            else:
                perm_indices = np.arange(world_step)[::sub_clip_num]
                perm_indices = perm_indices[
                    list(
                        torch.randperm(
                            perm_indices.shape[0], generator=self._generator
                        ).numpy()
                    )
                ].tolist()
                random_index = []
                for i in range(len(perm_indices) // self._batch_size):
                    for j in range(sub_clip_num):
                        random_index += (
                            np.array(
                                perm_indices[
                                    i
                                    * self._batch_size : (i + 1)
                                    * self._batch_size
                                ]
                            )
                            + j
                        ).tolist()
            self._random_indexs = random_index

        for _ in range(
            self._num_workers
        ):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        # find worker idx of current idx
        cur_offset = self._send_idx

        begin = int(cur_offset * self._batch_size)
        end = begin + self._batch_size

        cur_index = [
            idx + self._offset for idx in self._random_indexs[begin:end]
        ]

        self._index_queues[worker_queue_idx].put((self._send_idx, cur_index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1


@OBJECT_REGISTRY.register
class ANCRankSplitE2EDataLoader(RankSplitDataLoader):
    """RankSplitDataset must be used with RankSplitDataloader.

    RankSplitDataloader only supports DistributedSampler
    and DistributedGroupSampler now.
    """

    def __init__(self, *args, **kwargs):
        super(ANCRankSplitE2EDataLoader, self).__init__(*args, **kwargs)
        assert self.num_workers > 0
        assert self.drop_last
        if hasattr(self.sampler, "drop_last"):
            assert self.sampler.drop_last

    def _get_iterator(self):
        self.check_worker_number_rationality()
        return ANCRankSplitE2EDataLoaderIter(self)
