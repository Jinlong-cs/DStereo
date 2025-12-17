import copy
import os

import numpy as np
import pytest

from hat.data.datasets.dataset_wrappers import ConcatDataset, RepeatDataset
from hat.data.datasets.facequality_mtl_dataset import FaceQualityDataset
from hat.data.datasets.roidb_act_dataset import RoidbActDataset
from hat.data.samplers.dist_proportion_sampler import (
    DistributedProportionSampler,
)


@pytest.mark.parametrize(
    "expect_distribution, repeat_times, expect_res",
    [
        (
            {"0": 0.45, "1": 0.55},
            [1, 1],
            {
                "label_0": 31,
                "label_1": 40,
                "ind_info": {"0": np.arange(32), "1": np.arange(32, 64)},
            },
        ),
        (
            {"0": 0.2, "1": 0.8},
            [1, 2],
            {
                "label_0": 32,
                "label_1": 128,
                "ind_info": {"0": np.arange(32), "1": np.arange(32, 96)},
            },
        ),
    ],
)
def test_dist_proportion_sampler(
    expect_distribution, repeat_times, expect_res
):
    data_info_1 = {
        "rec_path": "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "glass_ir_n32_4ut.rec",
        "idx_path": "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "glass_ir_n32_4ut.idx",
        "label_path": "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "glass_ir_n32_4ut.label.npy",
        "info_path": "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "label_idx_map_v1.0.yaml",
    }
    data_info_2 = {
        "rec_path": "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "hat_ir_n32_4ut.rec",
        "idx_path": "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "hat_ir_n32_4ut.idx",
        "label_path": "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "hat_ir_n32_4ut.label.npy",
        "info_path": "./tmp_orig_data/face/face_quality/mx-record/mtl/"
        "label_idx_map_v1.0.yaml",
    }
    dataset1 = FaceQualityDataset(
        data_info_1["rec_path"],
        data_info_1["idx_path"],
        data_info_1["label_path"],
        data_info_1["info_path"],
        "glass",
        True,
        None,
    )
    dataset2 = FaceQualityDataset(
        data_info_2["rec_path"],
        data_info_2["idx_path"],
        data_info_2["label_path"],
        data_info_2["info_path"],
        "glass",
        True,
        None,
    )
    # Modify flag for easy testing
    dataset1.flag = np.zeros(32)
    dataset2.flag = np.ones(32)
    dataset1 = RepeatDataset(dataset1, repeat_times[0])
    dataset2 = RepeatDataset(dataset2, repeat_times[1])
    datasets = [dataset1, dataset2]
    dataset = ConcatDataset(datasets, with_flag=True, record_index=True)
    sampler = DistributedProportionSampler(
        dataset, expect_distribution, num_replicas=1, rank=0, drop_last=False
    )
    # sampler._ind_info
    inds_info = sampler._ind_info
    for k, v in inds_info.items():
        v = np.sort(v)
        assert np.sum(v - expect_res["ind_info"][k]) == 0
    assert sampler.expect_distribution["0"] == expect_res["label_0"]
    assert sampler.expect_distribution["1"] == expect_res["label_1"]
    label_0 = 0
    label_1 = 0
    with pytest.raises(StopIteration):
        sampler_iter = iter(sampler)
        while True:
            ind = sampler_iter.__next__()
            if ind >= 0 and ind <= 31:
                label_0 += 1
            else:
                label_1 += 1
    assert label_0 == expect_res["label_0"]
    assert label_1 == expect_res["label_1"]


@pytest.mark.parametrize(
    "ref_num",
    [
        pytest.param(1000),
        pytest.param(13000),
    ],
)
def test_dist_proportion_sampler_for_gesture(ref_num):
    train_valid_classes = ["0"]
    balance_proportion = [3.0]

    scale_ref = sum(balance_proportion)
    balance_proportion = [_ / scale_ref for _ in balance_proportion]
    ref_num = ref_num * scale_ref

    expect_distribution = {
        f"{key}": val
        for key, val in zip(train_valid_classes, balance_proportion)
    }
    repeat_times = [1, 2]

    def _get_dataset_per_rec():
        # "/horizon-bucket/HDLTAlgorithm/data/orig_data/action/gesture"
        path_data = "./tmp_orig_data/action/gesture/hgr_dac_dag/"
        is_read_rec = True
        dataset = RoidbActDataset(
            rec_path=os.path.join(
                path_data,
                "img_rec/CD569_hand_IFRC_IFARC_220121_220126_testset_dynamic.rec",  # noqa: E501
            ),
            roidb_path=os.path.join(
                path_data,
                "roidbs/CD569_hand_IFRC_IFARC_220121_220126_testset_det_kps_v2.0_ldmk2.5dv3.0.3_roidb.pkl",  # noqa: E501
            ),
            roidb_seq_path=os.path.join(
                path_data,
                "roidbs/CD569_hand_IFRC_IFARC_220121_220126_testset_act_v2.0_ldmk2.5dv3.0.3_roidb.pkl",  # noqa: E501
            ),
            is_read_rec=is_read_rec,
        )
        return dataset

    dataset1 = _get_dataset_per_rec()
    dataset2 = copy.deepcopy(dataset1)

    dataset1 = RepeatDataset(dataset1, repeat_times[0])
    dataset2 = RepeatDataset(dataset2, repeat_times[1])
    datasets = [dataset1, dataset2]
    dataset = ConcatDataset(datasets, with_flag=True)
    sampler = DistributedProportionSampler(
        dataset,
        expect_distribution,
        task_name="gesture",
        num_reference=ref_num,
        num_replicas=1,
        rank=0,
        drop_last=False,
    )

    max_iter_time = 5
    sampler_iter = iter(sampler)
    for _ in range(max_iter_time):
        _ = sampler_iter.__next__()
