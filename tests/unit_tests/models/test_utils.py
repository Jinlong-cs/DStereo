import pytest
import torch

from hat.models.utils import _check_strides, _take_features, multi_class_nms
from hat.utils.package_helper import check_packages_available


def test_check_strides():
    _check_strides([4, 8], [4, 8, 16, 32])
    with pytest.raises(AssertionError):
        _check_strides([64], [4, 8, 16, 32])


def test_take_features():
    assert _take_features([1, 2, 3, 4], [4, 8, 16, 32], [32]) == [4]
    with pytest.raises(AssertionError):
        _take_features([1, 2, 3, 4], [4, 8, 16, 32, 64], [32])


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_multi_class_nms():
    bbox = torch.FloatTensor(
        [(187, 82, 337, 317), (188, 83, 338, 318), (246, 121, 368, 304)]
    )
    score = torch.FloatTensor([0.9, 0.75, 0.8])
    cate_id = torch.FloatTensor([0, 0, 1])

    keep_idx = multi_class_nms(
        bbox=bbox, score=score, category=cate_id, iou_threshold=0.7
    )
    assert keep_idx.tolist() == [0, 2]
