import torch

from hat.models.losses.occflow_waymo_loss import (
    BaseOccflowLossWaymo,
    ProbabilisticLossWaymo,
    flow_warp,
)

# Test BaseOccflowLossWaymo will also test FocalLossWaymo, FlowLoss,
# and FlowTracedLoss.

num_class = 1
num_waypoints = 8


def get_fake_pred():
    predictions = {}
    predictions["occ_preds"] = torch.zeros(
        (1, 2 * num_waypoints * num_class, 256, 256)
    )
    predictions["flow_preds"] = torch.zeros(
        (1, 2 * num_waypoints * num_class, 256, 256)
    )
    return predictions


def get_fake_gt():
    ground_truth = {}
    ground_truth["occupancy_waypoints"] = torch.zeros(
        (1, 256, 256, 2, num_class, num_waypoints)
    )
    ground_truth["flow_waypoints"] = torch.zeros(
        (1, 256, 256, 2, num_class, num_waypoints)
    )
    ground_truth["flow_origin_occupancy_waypoints"] = torch.zeros(
        (1, 256, 256, 1, num_class, num_waypoints)
    )
    return ground_truth


def get_fake_latent_pred():
    output = {}
    output["present_mu"] = [torch.zeros((1, 128, 256, 256))]
    output["present_log_sigma"] = [torch.zeros((1, 128, 256, 256))]
    output["future_mu"] = [torch.zeros((1, 128, 256, 256))]
    output["future_log_sigma"] = [torch.zeros((1, 128, 256, 256))]
    predictions = {"output_distributions": output}
    return predictions


def test_BaseOccflowLossWaymo():
    pred = get_fake_pred()
    target = get_fake_gt()
    loss = BaseOccflowLossWaymo(
        compute_loss_cls_index=[0],
        loss_weights=[500, 1, 500, 500],
    )
    result = loss(pred, target)
    loss_keys = [
        "occupancy_loss",
        "flow_loss",
        "traced_bce_loss",
        "traced_focal_loss",
        "combined_loss",
    ]
    for key in loss_keys:
        assert key in result
        assert isinstance(result[key], torch.Tensor)


def test_ProbabilisticLossWaymo():
    pred = get_fake_latent_pred()
    loss = ProbabilisticLossWaymo(scale_weights=(1.0,))
    result = loss(pred, {})
    assert isinstance(result, torch.Tensor)


def test_flow_warp():
    current_occupancy = torch.zeros((1, 256, 256, 1, 8))
    pred_flows = torch.zeros((1, 256, 256, 2, 1, 8))

    warped = flow_warp(current_occupancy, pred_flows)
    assert warped.shape == (1, 256, 256, 1, 1, 8)
