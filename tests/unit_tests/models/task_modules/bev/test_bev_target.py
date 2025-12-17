import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    ["gt_name"],
    [
        pytest.param("gt_bev_seg"),
        pytest.param(["gt_bev_seg", "gt_bev_om"]),
    ],
)
def test_bev_target(gt_name):
    bs, bev_height, bev_width = 2, 512, 512
    gt_shape = (bs, 1, 512, 512)

    new_gt_shape = (bs, 512, 512)

    config = dict(
        type="ANCBEVTarget",
        gt_name=gt_name,
        bev_height=bev_height,
        bev_width=bev_width,
        gt_shape=new_gt_shape,
    )
    bev_target = build_from_registry(config)

    pred_dict, label_dict = dict(), dict()
    if isinstance(gt_name, list):
        gt_name = gt_name[0]

    label_dict[gt_name] = torch.randn(gt_shape)
    pred_dict["bev_rot_mat"] = torch.randn((bs, 3, 3))
    label_dict = bev_target(label_dict, pred_dict)

    assert label_dict[gt_name].shape == new_gt_shape
