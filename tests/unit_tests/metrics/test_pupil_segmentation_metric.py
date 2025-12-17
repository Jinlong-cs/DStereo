import pytest
import torch

from hat.metrics.pupil_segmentation_metric import EllipseParamError


@pytest.mark.parametrize(
    "do_norm",
    [True, False],
)
def test_ellipse_param_error(do_norm):
    pred_ellipse_param = torch.Tensor(
        [
            [1, 1, 2, 2, 30],
            [2, 2, 10, 12, 10],
        ]
    )
    gt_ellipse_param = torch.Tensor(
        [
            [2, 2, 3, 4, 50],
            [4, 5, 12, 16, 30],
        ]
    )
    gt_metric_with_norm = [
        (2 ** (1 / 2) / 5 + 13 ** (1 / 2) / 20) / 2,
        (5 ** (1 / 2) / 5 + 20 ** (1 / 2) / 20) / 2,
        20,
    ]
    gt_metric_without_norm = [
        (2 ** (1 / 2) + 13 ** (1 / 2)) / 2,
        (5 ** (1 / 2) + 20 ** (1 / 2)) / 2,
        20,
    ]

    error_metric = EllipseParamError(do_norm=do_norm)
    error_metric.update(gt_ellipse_param, pred_ellipse_param)
    _, val = error_metric.get()
    if do_norm:
        assert abs(val[0] - gt_metric_with_norm[0]) < 1e-4
        assert abs(val[1] - gt_metric_with_norm[1]) < 1e-4
        assert abs(val[2] - gt_metric_with_norm[2]) < 1e-4
    else:
        assert abs(val[0] - gt_metric_without_norm[0]) < 1e-4
        assert abs(val[1] - gt_metric_without_norm[1]) < 1e-4
        assert abs(val[2] - gt_metric_without_norm[2]) < 1e-4
