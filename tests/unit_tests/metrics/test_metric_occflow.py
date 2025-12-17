import pytest
import torch

from hat.metrics.metric_occflow import OccFlowMetrics

# [N, C, H, W]
occ_preds = torch.Tensor(
    [
        [
            [
                [0.2791, 0.3981, 0.9743, 0.9195],
                [0.5459, 0.0296, 0.7328, 0.6588],
                [0.7980, 0.8432, 0.6973, 0.8424],
                [0.3557, 0.9700, 0.9131, 0.7267],
            ],
            [
                [0.1759, 0.1576, 0.4143, 0.9252],
                [0.2392, 0.4409, 0.3439, 0.2581],
                [0.6797, 0.6978, 0.3670, 0.7892],
                [0.9076, 0.1986, 0.0377, 0.9558],
            ],
        ]
    ]
)
# [N, C, H, W]
flow_preds = torch.Tensor(
    [
        [
            [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [1, 0, 1, 1]],
        ]
    ]
)
# [N, C, H, W]
occ_target = torch.Tensor(
    [
        [
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 0]],
            [[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 0]],
        ]
    ]
)
# [N, C, H, W]
flow_target = torch.Tensor(
    [
        [
            [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1]],
            [[1, 0, 1, 1], [1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0]],
        ]
    ]
)
# [N, C, H, W]
flow_origin = torch.Tensor(
    [[[[1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]]]]
)
# result
result = {
    "vehicles_observed_auc": 0.648,
    "vehicles_occluded_auc": 0.648,
    "vehicles_observed_iou": 0.44,
    "vehicles_occluded_iou": 0.44,
    "vehicles_flow_epe": 0.795,
    "vehicles_flow_warped_occupancy_auc": 0.788,
    "vehicles_flow_warped_occupancy_iou": 0.474,
}

predictions = {
    "occ_preds": occ_preds,
    "flow_preds": flow_preds,
}
ground_truth = {
    "occupancy_waypoints": occ_target,
    "flow_waypoints": flow_target,
    "flow_origin_occupancy_waypoints": flow_origin,
}


RESULT_KEYS = [
    "vehicles_observed_auc",
    "vehicles_occluded_auc",
    "vehicles_observed_iou",
    "vehicles_occluded_iou",
    "vehicles_flow_epe",
    "vehicles_flow_warped_occupancy_auc",
    "vehicles_flow_warped_occupancy_iou",
]


@pytest.mark.parametrize(
    ["preds", "target", "expect_result"],
    [
        pytest.param(predictions, ground_truth, result),
    ],
)
def test_acc(preds, target, expect_result):
    # transform preds and target
    for k, v in preds.items():
        v = v.tile((1, 8, 1, 1))
        preds[k] = v.to(torch.float32)
    for k, v in target.items():
        N, C, H, W = v.shape
        v = v.tile((1, 8, 1, 1)).view(N, C, 8, H, W)
        v = v.permute([0, 3, 4, 1, 2]).unsqueeze(-2)
        target[k] = v.to(torch.float32)

    occ_metric = OccFlowMetrics(cls_index=[0])
    occ_metric.config.grid_height_cells = 4
    occ_metric.config.grid_width_cells = 4
    occ_metric.update(preds, target)
    name, metric_result = occ_metric.get()
    assert name == "OccFlowMetrics"
    for key in RESULT_KEYS:
        assert key in metric_result
        assert abs(metric_result[key] - expect_result[key]) < 1e-6
