import pytest

from hat.registry import build_from_registry


def test_faceid_distributed_balance_sampler():
    dataset = dict(
        type="DeepInsightRecordDataset",
        rec_path="./tmp_orig_data/face/recognition/mx-record/"
        "face_recognition_test_10cls_122img/"
        "face_recognition_test_10cls_122img.rec",
        idx_path="./tmp_orig_data/face/recognition/mx-record/"
        "face_recognition_test_10cls_122img/"
        "face_recognition_test_10cls_122img.idx",
        unpack64=True,
    )

    sampler = dict(
        type="DistributedFaceIDBalanceSampler",
        dataset=dataset,
        bounds=[8, 10, 8],
        replace=False,
        drop_last=True,
        shuffle=True,
    )

    with pytest.raises(RuntimeError):
        _ = build_from_registry(sampler)
        max_iter_time = 5
        sampler_iter = iter(sampler)
        for _ in range(max_iter_time):
            _ = sampler_iter.__next__()
