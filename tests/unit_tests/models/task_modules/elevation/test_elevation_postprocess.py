import torch

from hat.models.task_modules.elevation import ElevationPostprocess


def test_elevation_postprocess():
    color_imgs = [
        torch.rand(2, 3, 512, 960),
        torch.rand(2, 3, 512, 960),
        torch.rand(2, 3, 512, 960),
    ]
    intrinsics = torch.rand(2, 3, 3)
    camera_high = [torch.rand(2, 1)]
    ground_norm = [torch.rand(2, 3, 1)]
    pred = {
        "pred_gammas_frame0": [torch.rand(2, 1, 512, 960)],
    }
    label = {
        "color_imgs": color_imgs,
        "intrinsics": intrinsics,
        "camera_high": camera_high,
        "ground_norm": ground_norm,
    }
    elevation_postprocess = ElevationPostprocess(
        gamma_scale=1000.0,
        gt_size=(512, 960),
        output_name="pred_gammas",
        postprocess_frame_idx=[0],
    )

    pred = elevation_postprocess(pred, label)
    assert pred["pred_depth_frame0"].shape == (2, 1, 512, 960)
    assert pred["pred_height_frame0"].shape == (2, 1, 512, 960)
