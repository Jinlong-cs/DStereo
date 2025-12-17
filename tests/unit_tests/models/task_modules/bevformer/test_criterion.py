import pytest
import torch

from hat.registry import build_from_registry


def gen_data():

    preds_dicts = {
        "bev_embed": torch.randn(2500, 1, 256),
        "all_cls_scores": torch.randn(6, 1, 900, 10),
        "all_bbox_preds": torch.randn(6, 1, 900, 10),
        "enc_cls_scores": None,
        "enc_bbox_preds": None,
    }

    data = {}
    seq_meta = {}
    ego_bboxes_labels = torch.randn((1, 10))
    ego_bboxes_labels[0, 9] = 2.0
    seq_meta["ego_bboxes_labels"] = [ego_bboxes_labels]
    data["seq_meta"] = [seq_meta]
    return preds_dicts, data


@pytest.mark.serial_task
def test_bevformercriterion():
    config = dict(
        type="BevFormerCriterion",
        assigner=dict(
            type="BevFormerHungarianAssigner3D",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
        ),
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=10 + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            reduction="mean",
        ),
        loss_bbox=dict(
            type="L1Loss",
            loss_weight=0.25,
        ),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    )
    model = build_from_registry(config)
    data = gen_data()
    model(*data)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
