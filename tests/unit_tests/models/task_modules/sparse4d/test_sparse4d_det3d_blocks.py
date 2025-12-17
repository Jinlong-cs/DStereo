import numpy as np
import pytest
import torch

from hat.models.task_modules.sparse4d import (
    SparseBox3DEncoder,
    SparseBox3DKeyPointsGenerator,
    SparseBox3DRefinementModule,
)
from hat.models.task_modules.sparse4d.target import (
    COS_YAW,
    SIN_YAW,
    VX,
    VY,
    X,
    Y,
    Z,
)


@pytest.mark.parametrize(
    ["output_dim", "normalize_yaw", "refine_yaw", "with_cls_branch"],
    [
        pytest.param(11, True, True, True),
        pytest.param(10, True, True, True),
        pytest.param(8, True, True, True),
        pytest.param(10, False, False, False),
    ],
)
def test_sparse_box3d_refinement_module(
    output_dim, normalize_yaw, refine_yaw, with_cls_branch
):
    embed_dims = 20
    num_cls = 5
    refine_module = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=output_dim,
        num_cls=num_cls,
        normalize_yaw=normalize_yaw,
        refine_yaw=refine_yaw,
        with_cls_branch=with_cls_branch,
    )

    for shape in ((2,), (2, 4)):
        input = dict(
            instance_feature=torch.randn(*shape, embed_dims),
            anchor=torch.randn(*shape, output_dim),
            anchor_embed=torch.randn(*shape, embed_dims),
            time_interval=torch.tensor([1] * shape[0]),
            return_cls=with_cls_branch,
        )
        output, cls, centerness, yawness = refine_module(**input)
        assert output.shape == (*shape, output_dim)
        if cls is not None:
            assert cls.shape == (*shape, num_cls)
        else:
            assert not with_cls_branch


@pytest.mark.parametrize(
    ["num_learnable_pts", "num_fix_pts"],
    [
        pytest.param(0, 7),
        pytest.param(7, 1),
        pytest.param(7, 7),
    ],
)
def test_sparse_box3d_key_points_generator(num_learnable_pts, num_fix_pts):
    embed_dims = 20
    fix_scale = np.random.uniform(size=[num_fix_pts, 3]) * 2 - 1
    kps_generator = SparseBox3DKeyPointsGenerator(
        embed_dims=embed_dims,
        num_learnable_pts=num_learnable_pts,
        fix_scale=fix_scale,
    )
    bs = 2
    num_anchor = 10
    for state_dim in (8, 10, 11):
        input = dict(
            anchor=torch.randn(bs, num_anchor, state_dim),
            instance_feature=torch.randn(bs, num_anchor, embed_dims),
        )
        key_points = kps_generator(**input)
        assert key_points.shape == (
            bs,
            num_anchor,
            num_learnable_pts + num_fix_pts,
            3,
        )

        seq_length = 3
        input.update(
            dict(
                cur_timestamp=torch.tensor([1.0] * bs),
                temp_timestamps=[torch.tensor([2.0] * bs)] * seq_length,
            )
        )
        T_cur2temp_list = []
        for _ in range(seq_length):
            mat = torch.randn(bs, 4, 4)
            mat[:, -1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
            T_cur2temp_list.append(mat)
        input["T_cur2temp_list"] = T_cur2temp_list
        key_points, temp_key_points_list = kps_generator(**input)
        assert key_points.shape == (
            bs,
            num_anchor,
            num_learnable_pts + num_fix_pts,
            3,
        )
        assert len(temp_key_points_list) == seq_length
        for i in range(seq_length):
            assert temp_key_points_list[i].shape == (
                bs,
                num_anchor,
                num_learnable_pts + num_fix_pts,
                3,
            )

        temp_anchors = kps_generator.anchor_projection(
            input["anchor"],
            input["T_cur2temp_list"],
            input["cur_timestamp"],
            input["temp_timestamps"],
        )
        for i in range(seq_length):
            assert temp_anchors[i].shape == input["anchor"].shape
            inv = np.linalg.inv(input["T_cur2temp_list"][i].cpu().numpy())
            inv = input["T_cur2temp_list"][i].new_tensor(inv)
            _anchor = kps_generator.anchor_projection(
                temp_anchors[i],
                [inv],
                input["temp_timestamps"][i],
                [input["cur_timestamp"]],
            )[0]
            error = torch.abs(_anchor - input["anchor"])
            error[..., [SIN_YAW, COS_YAW]] = 0.0
            if state_dim == 10:
                error[..., [X, Y, Z, VX, VY]] = 0.0
            print(error.flatten(end_dim=1).max(dim=0))
            assert torch.all(error < 1e-2)


@pytest.mark.parametrize(
    ["vel_dims"],
    [
        pytest.param(0),
        pytest.param(2),
        pytest.param(3),
    ],
)
def test_sparse_box3d_encoder(vel_dims):
    embed_dims = 20
    encoder = SparseBox3DEncoder(embed_dims=embed_dims, vel_dims=vel_dims)
    for shape in ((2,), (2, 10)):
        input = torch.randn(*shape, vel_dims + 8)
        output = encoder(input)
        assert output.shape == (*shape, embed_dims)
