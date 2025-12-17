import pytest
import torch

from hat.registry import build_from_registry

config = dict(
    type="BevFormerHungarianAssigner3D",
    cls_cost=dict(type="FocalLossCost", weight=2.0),
    reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
)


def gen_data():

    bbox_pred = torch.randn((900, 10))
    cls_pred = torch.randn((900, 10))
    gt_bboxes = torch.randn((17, 9))
    gt_labels = torch.randint(high=10, size=(17,))

    return bbox_pred, cls_pred, gt_bboxes, gt_labels


@pytest.mark.serial_task
def test_bevformerhungarianassigner3d():
    model = build_from_registry(config)
    data = gen_data()
    model.assign(*data)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
