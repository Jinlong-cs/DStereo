import torch

from hat.models.losses.seg_loss import MultiStrideLosses, SegEdgeLoss
from hat.models.losses.smooth_l1_loss import SmoothL1Loss
from hat.utils.seed import seed_everything


def test_edge_loss():

    seed_everything(17)
    batch_size = 5
    num_classes = 4
    height = 640
    width = 960
    target_loss = 17.4019
    pred = torch.rand(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    edge_graph = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
    ]
    edge_loss = SegEdgeLoss(edge_graph, loss_name=None)
    edge_loss_value = edge_loss(pred, target)
    assert edge_loss_value - target_loss <= 1e-4


test_edge_loss()


def test_multi_stride_loss():
    pred = torch.Tensor(
        [
            [
                [0.9140, 0.3939, 0.8436, 0.8709, 0.3387],
                [0.7222, 0.1991, 0.7227, 0.0187, 0.8135],
                [0.5644, 0.7513, 0.0774, 0.3974, 0.2366],
            ],
            [
                [0.8459, 0.2419, 0.1675, 0.3813, 0.7049],
                [0.2227, 0.8864, 0.8276, 0.9101, 0.4410],
                [0.6678, 0.2443, 0.5320, 0.6253, 0.0654],
            ],
        ]
    )
    target = torch.Tensor(
        [
            [
                [0.4135, 0.6855, 0.8427, 0.3257, 0.9094],
                [0.1176, 0.4234, 0.7032, 0.2256, 0.4893],
                [0.4502, 0.8952, 0.6433, 0.0331, 0.3308],
            ],
            [
                [0.9196, 0.7138, 0.6535, 0.2618, 0.0073],
                [0.3535, 0.3894, 0.0118, 0.0079, 0.0777],
                [0.0233, 0.7960, 0.5167, 0.4476, 0.1777],
            ],
        ]
    )

    pred_list = [pred, pred]
    target_list = [target, target]

    target_loss = torch.FloatTensor([0.4415]).squeeze()
    out_strides = [2, 4]
    loss_weights = [2, 1]

    multi_stride_loss = MultiStrideLosses(
        num_classes=2,
        out_strides=out_strides,
        loss=SmoothL1Loss(
            beta=1 / 9.0,
            reduction="mean",
        ),
        loss_weights=loss_weights,
    )

    loss = multi_stride_loss(pred_list, target_list)

    for stride, weight in zip(out_strides, loss_weights):
        assert f"stride_{stride}_loss" in loss
        stride_loss = loss.get(f"stride_{stride}_loss", None)
        assert torch.abs(stride_loss - target_loss * weight) < 1e-4
