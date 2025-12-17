import pytest
import torch

from hat.models.task_modules.sparse4d import SparseBox3DTarget


@pytest.mark.parametrize(
    ["bs", "num_pred", "num_gt", "num_classes", "state_dims"],
    [
        pytest.param(10, 0, 20, 1, 10),
        pytest.param(10, 0, 20, 10, 10),
        pytest.param(10, 10, 0, 10, 10),
        pytest.param(10, 10, 20, 10, 10),
    ],
)
def test_sparse_box3d_target(bs, num_pred, num_gt, num_classes, state_dims):
    target = SparseBox3DTarget(
        reg_weights=[1.0] * state_dims,
        num_dn_groups=3,
        add_neg_dn=True,
        max_dn_gt=30,
    )
    for _ in range(5):
        cls_pred = torch.randn((bs, num_pred, num_classes))
        box_pred = torch.randn((bs, num_pred, state_dims))

        cls_target = [
            torch.randn((num_gt)).to(dtype=torch.int64) for i in range(bs)
        ]
        box_target = [torch.ones((num_gt, state_dims - 1)) for i in range(bs)]
        gt_ignore = [torch.rand((num_gt)) < 0 for i in range(bs)]
        (
            output_cls_target,
            output_box_target,
            output_cls_weights,
            output_reg_weights,
        ) = target(cls_pred, box_pred, cls_target, box_target, gt_ignore)

        assert output_cls_target.shape == (bs, num_pred)
        assert output_box_target.shape == (bs, num_pred, state_dims)
        assert output_reg_weights.shape == (bs, num_pred, state_dims)
        assert output_cls_weights.shape == (bs, num_pred)

        (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
        ) = target.get_dn_anchors(cls_target, box_target, gt_ignore)
        max_dn_gt = min(target.max_dn_gt, num_gt) * target.num_dn_groups
        if target.add_neg_dn:
            max_dn_gt = max_dn_gt * 2
        assert dn_anchor.shape == (bs, max_dn_gt, state_dims)
        assert dn_box_target.shape == (bs, max_dn_gt, state_dims)
        assert dn_cls_target.shape == (bs, max_dn_gt)
        assert attn_mask.shape == (max_dn_gt, max_dn_gt)
        assert valid_mask.shape == dn_cls_target.shape
