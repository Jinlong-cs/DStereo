import torch

from hat.registry import build_from_registry


def test_pupil_segmentation_losses():
    loss_weights = {
        "seg2pt": 1,
        "ellipse": 1,
        "seg": 1,
        "selfcorr": 1,
        "surface_loss_ratio": 0.5,
    }
    pupil_segmentation_losses = build_from_registry(
        dict(
            type="PupilSegLoss",
            loss_weights=loss_weights,
            do_self_corr=True,
            mask_shape=(3, 3),
        )
    )
    mask_pred = torch.Tensor(
        [[[0.5, 0.4, 1.0], [0.1, 0.1, 0.1], [0.1, 0.5, 1.0]]]
    ).reshape(1, 1, 3, 3)
    ellipse_param_pred = torch.Tensor([[0.1, 0.1, 0.5, 0.5, 0.0]])
    model_out = {
        "mask_pred": mask_pred,
        "ellipse_param_pred": ellipse_param_pred,
    }
    gt_pupil_center = torch.Tensor([[0.0, 0.0]])
    gt_norm_pupil_ellipse_param = torch.Tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    gt_pupil_mask = torch.Tensor(
        [[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]
    ).long()
    spat_weights = torch.Tensor(
        [[[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]]]
    )
    dist_map = torch.Tensor(
        [[[0.5, 1.0, 0.5], [1.0, 0.5, 0.1], [0.5, 0.5, 1.0]]]
    )
    data = {
        "gt_pupil_center": gt_pupil_center,
        "gt_norm_pupil_ellipse_param": gt_norm_pupil_ellipse_param,
        "gt_pupil_mask": gt_pupil_mask,
        "spat_weights": spat_weights,
        "dist_map": dist_map,
    }
    losses = pupil_segmentation_losses(model_out, data)
    assert abs(losses["seg2pt_loss"].item() - 1.3590) < 1e-4
    assert abs(losses["ellipse_loss"].item() - 0.1909) < 1e-4
    assert abs(losses["seg_loss"].item() - 1.5487) < 1e-4
    assert abs(losses["selfcorr_loss"].item() - 0.0716) < 1e-4
