import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from hat.data.samplers.dist_group_sampler import DistributedGroupSampler
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info, get_local_process_group

__all__ = [
    "ANCDistributedGroupClipSampler",
    "ANCDistributedClipValSampler",
]


@OBJECT_REGISTRY.register
class ANCDistributedGroupClipSampler(
    DistributedGroupSampler
):  # noqa: D205,D400
    """Sampler that restricts sequential clip data (multi frames) to
    a subset of the dataset.

    在端到端任务中，数据是以clip的形式打包的，每个clip包含多个sub_clip。
    每个step输出一个sub_clip的数据。

    Example:
        一个clip包含四个连续的帧: [[0, 1], [2, 3]],
        其中0,1,2,3表示帧的indices.
        这里[0, 1]和[2, 3]是两个sub_clip，分别包含了两帧数据。

    在每个step中，数据indices表示了sub_clip的位置。每个indices相对于上一个
    batch的相同位置的indice是连续的。只有当一个clip结束时，indices才会从下
    一个clip继续开始。

    Example:
        每个clip 包含了两个sub_clip, 即indices 0，1表示第一个clip中的两个
        subclip；2,3表示第二个clip中的两个subclip。
        在batch size = 2的情况下，一个合理的数据输出indices应该为：
        step1: [0, 100], 因为bs=2，每个step输出两个indices。
        step2: [1, 101], step 1-2 表示经过了完整的一个clip。
        step3: [50, 200], 开始新的clip。
        step4: [51, 201], step 3-4 表示经过了另一个完成的clip。

    参考文档：https://horizonrobotics.feishu.cn/docx/Ry0KdbLJcoBNECxhx07czJqznfd

    .. note::
        数据集的大小应该是固定的，并且包含flag这个属性。flag的长度等于数据
        集的大小，不同flag的值表示不同的groups。比如，数据含有不同长宽比时，
        可以用flag=0表示h / w>=1的样本，用1表示h/ w <1的样本。flag必须是
        numpy array类型，且dtype必须是np.uint8。

    Args:
        dataset: Dataset used for sampling.
        samples_per_gpu: Number samplers for each gpu.
            Default is 1.
        num_replicas: Number of processes participating in
            distributed training.
        rank: Rank of the current process within num_replicas.
        shuffle: If ``True`` (default), sampler will shuffle the indices.
        seed: random seed used in torch.Generator().
            This number should be identical across all
            processes in the distributed group. Default: 0.
        sub_clip_num: the number of sub_clip in a clip.
    """

    def __init__(
        self,
        dataset,
        samples_per_gpu: Optional[int] = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        sub_clip_num: int = 1,
    ):
        if num_replicas is None or rank is None:
            rank, num_replicas = get_dist_info(get_local_process_group())

        super(ANCDistributedGroupClipSampler, self).__init__(
            dataset,
            samples_per_gpu=samples_per_gpu,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )
        self.sub_clip_num = sub_clip_num
        self.shuffle = shuffle
        self.samples_per_gpu = samples_per_gpu

        # calculate rounded number of sub_clip with different flag in
        # this rank. the size of dataset is num_clip * sub_clip_num,
        # therefore, the self.num_samples should consider this.
        self.num_samples = 0
        for size in self.group_sizes:
            self.num_samples += (
                int(
                    math.ceil(
                        size
                        * 1.0
                        / self.samples_per_gpu
                        / self.num_replicas
                        / self.sub_clip_num
                    )
                )
                * self.samples_per_gpu
                * self.sub_clip_num
            )
        self.total_size = self.num_replicas * self.num_samples

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)

        perm_indices = []
        for i, size in enumerate(self.group_sizes):
            if size <= 0:
                continue

            indice = np.where(self.flag == i)[0]
            assert len(indice) == size

            # squeeze for sampling, perm_indice is the first indice of
            # every clip.
            perm_indice = indice[:: self.sub_clip_num]
            perm_size = size // self.sub_clip_num

            if self.shuffle:
                perm_indice = perm_indice[
                    list(torch.randperm(int(perm_size), generator=g).numpy())
                ].tolist()
            else:
                perm_indice = list(perm_indice)

            perm_extra = int(
                math.ceil(
                    perm_size * 1.0 / self.samples_per_gpu / self.num_replicas
                )
            ) * self.samples_per_gpu * self.num_replicas - len(perm_indice)

            # pad indice
            perm_tmp = perm_indice.copy()
            for _ in range(perm_extra // perm_size):
                perm_indice.extend(perm_tmp)
            perm_indice.extend(perm_tmp[: perm_extra % perm_size])
            perm_indices.extend(perm_indice)

        assert len(perm_indices) == (self.total_size // self.sub_clip_num)

        if self.shuffle:
            perm_indices = [
                perm_indices[j]
                for i in list(
                    torch.randperm(
                        len(perm_indices) // self.samples_per_gpu, generator=g
                    )
                )
                for j in range(
                    i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu
                )
            ]

        # subsample
        perm_clip_num = self.num_samples // self.sub_clip_num
        perm_offset = perm_clip_num * self.rank
        perm_indices = perm_indices[perm_offset : perm_offset + perm_clip_num]
        assert len(perm_indices) == perm_clip_num

        for ind in perm_indices:
            assert not (
                ind % self.sub_clip_num
            ), f"{ind} do not match {self.sub_clip_num}"

        # unsqueeze for get
        clip_indices = []
        for i in range(len(perm_indices) // self.samples_per_gpu):
            for j in range(self.sub_clip_num):

                clip_indices += (
                    np.array(
                        perm_indices[
                            i
                            * self.samples_per_gpu : (i + 1)
                            * self.samples_per_gpu
                        ]
                    )
                    + j
                ).tolist()

        assert len(clip_indices) == self.num_samples

        return iter(clip_indices)


@OBJECT_REGISTRY.register
class ANCDistributedClipValSampler(DistributedSampler):  # noqa: D205,D400
    """Sampler that restricts end2end sequential clip data (multi frames)
    to a subset of the dataset in the validation process.

    在模型验证推理过程中，同一个pack中的帧需要在同一个GPU上进行顺序推理。因此，
    需要找到各个pack在验证集中的初始位置，然后根据这些位置，将不同的indice分配给
    不同的GPU。

    在验证数据集打包过程中，pack的打包顺序是按照一个clip20帧，不重复打包的，即，
    需要对当前clip进行完整遍历，然后再infer下一个clip。这里需要考虑sync_info中
    的split_num_per_sample以及对dataset的扩充倍数。

    参考文档：https://horizonrobotics.feishu.cn/docx/Ry0KdbLJcoBNECxhx07czJqznfd

    Note: Only support bs = 1 as the moment.

    Args:
        dataset: Dataset used for sampling.
        num_replicas: Number of processes participating in
            distributed validation.
        rank: Rank of the current process within num_replicas.
        seed: random seed used in torch.Generator().
            This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        if num_replicas is None or rank is None:
            rank, num_replicas = get_dist_info(get_local_process_group())

        super(ANCDistributedClipValSampler, self).__init__(
            dataset,
            num_replicas,
            rank,
            shuffle=False,
            seed=seed,
        )

        pack_dir_lists = []
        pack_start_index = np.zeros(0)
        for ds in dataset.datasets:
            # for every dataset, obtain the pack list and cusum of the frames
            # number of every pack.
            sync_info = ds.dataset.sync_info

            # sub_clip_pack_dir contains the pack_dir of every sub_clip.
            sub_clip_pack_dir = []
            # val_dataset has been repeat by sub_clip_num times.
            for i in range(len(sync_info)):
                sub_clip_pack_dir_temp = [
                    sync_info_sample["pack_dir"]
                    for sync_info_sample in sync_info[i][0]
                ]
                sub_clip_pack_dir.append(sub_clip_pack_dir_temp[0])

            # all pack_dir in this dataset.
            ds_pack_dir_lists = list(set(sub_clip_pack_dir))

            # start index for every pack
            pack_start_index_temp = [
                sub_clip_pack_dir.index(pack_name)
                for pack_name in ds_pack_dir_lists
            ]
            pack_start_index_temp.sort()

            # append the length of whole validation dataset for
            # split and index.
            pack_start_index_temp.append(len(sub_clip_pack_dir))
            pack_start_index_temp = np.array(pack_start_index_temp)
            if pack_start_index.shape[0] == 0:
                pack_start_index = pack_start_index_temp
            else:
                pack_start_index_temp += pack_start_index[-1]
                pack_start_index = np.concatenate(
                    [pack_start_index, pack_start_index_temp[1:]]
                )

            pack_dir_lists.extend(ds_pack_dir_lists)

        # assign packs and relative indexes to different processors.
        packs_list_on_rank = [
            v[0]
            for v in np.array_split(
                np.arange(len(pack_dir_lists)), num_replicas
            )
        ] + [len(pack_dir_lists)]
        self.start_idx = pack_start_index[packs_list_on_rank[rank]]
        self.end_idx = pack_start_index[packs_list_on_rank[rank + 1]]

    def __iter__(self):
        indices = range(self.start_idx, self.end_idx)
        return iter(indices)

    def __len__(self):
        return self.end_idx - self.start_idx
