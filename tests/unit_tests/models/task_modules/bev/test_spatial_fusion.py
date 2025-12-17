import pytest
import torch

from hat.models.task_modules.bev.spatial_fusion import (
    ANCBEVFusionModule,
    ANCBEVMultiFusionModule,
)
from hat.registry import build_from_registry


@pytest.mark.parametrize(
    "vcs_plane_nums",
    [
        1,
        4,
    ],
)
def test_bev_fusion(vcs_plane_nums):
    bev_bs, views = 2, 8
    ipm_output_size = (512, 512)
    grid_quant_scale = 0.03125
    homo_offset = torch.randn(
        (views * vcs_plane_nums,) + ipm_output_size + (2,)
    )
    homography = torch.randn((bev_bs, views * vcs_plane_nums, 3, 3))

    bev_fusion = ANCBEVFusionModule(
        grid_quant_scale=grid_quant_scale,
        random_rotation_cfg=build_from_registry(
            {
                "type": "RandomRotation",
                "angles": (0, 90, 180, 270),
                "height": ipm_output_size[0],
                "width": ipm_output_size[1],
                "grid_quant_scale": grid_quant_scale,
                "mode": "bilinear",
                "padding_mode": "zeros",
                "use_horizon_grid_sample": True,
            }
        ),
        drop_view_prob=0.2,
        views=views,
        ipm_output_size=ipm_output_size,
        vcs_plane_nums=vcs_plane_nums,
    )

    input_w, input_h = 960, 512
    feats = [
        torch.randn((bev_bs * views, 8, input_h // 4, input_w // 4)),
    ]

    meta = {
        "meta_info": {"homo_offset": homo_offset, "homography": homography}
    }
    rot_mat, bev_input = bev_fusion(feats, meta)

    assert bev_input.shape == (bev_bs, 8 * vcs_plane_nums, *(ipm_output_size))


@pytest.mark.parametrize(
    "vcs_plane_nums",
    [
        1,
        4,
    ],
)
def test_bev_twice_fusion(vcs_plane_nums):
    bev_bs, views = 2, 11
    ipm_output_size = (512, 512)
    grid_quant_scale = 0.03125

    homo_offset = torch.randn(
        (views * vcs_plane_nums,) + ipm_output_size + (2,)
    )
    homography = torch.randn((bev_bs, views * vcs_plane_nums, 3, 3))
    bev_fusion = ANCBEVMultiFusionModule(
        grid_quant_scale=grid_quant_scale,
        random_rotation_cfg=build_from_registry(
            {
                "type": "RandomRotation",
                "angles": (0, 90, 180, 270),
                "height": ipm_output_size[0],
                "width": ipm_output_size[1],
                "grid_quant_scale": grid_quant_scale,
                "mode": "bilinear",
                "padding_mode": "zeros",
                "use_horizon_grid_sample": True,
            }
        ),
        drop_view_prob=0.2,
        views=views,
        fusion_idx_lst=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10]],
        ipm_output_size=ipm_output_size,
        vcs_plane_nums=vcs_plane_nums,
    )

    input_w, input_h = 960, 512
    feats = [
        torch.randn((bev_bs * views, 8, input_h // 4, input_w // 4)),
    ]

    meta = {
        "meta_info": {"homo_offset": homo_offset, "homography": homography}
    }
    rot_mat, bev_input = bev_fusion(feats, meta)

    assert bev_input.shape == (bev_bs, 8 * vcs_plane_nums, *(ipm_output_size))
