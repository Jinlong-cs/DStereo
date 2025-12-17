import pytest
import torch

from hat.visualize.elevation import vis_parallax


@pytest.mark.parametrize(
    ["vis_type"],
    [
        pytest.param("gamma"),
        pytest.param("depth"),
        pytest.param("height"),
    ],
)
def test_vis_parallax(vis_type):
    pred = torch.randn((1, 1, 512, 960)).float()
    img = torch.randn((1, 3, 512, 960)).float()
    vis_img = vis_parallax(
        pred=pred,
        img=img,
        save=False,
        path="",
        timestamp="",
        vis_type=vis_type,
    )
    assert vis_img.shape == (256, 480, 3)
