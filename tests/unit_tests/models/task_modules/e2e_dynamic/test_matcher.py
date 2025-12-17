import pytest
import torch

from hat.models.task_modules.e2e_dynamic.set_criterion import (
    E2EClipMatcher,
    E2EHungarianMatcher,
)


def get_fake_data():
    gt_instances = {
        "bev_tracking": [
            {
                "labels": torch.randint(0, 2, (50,)),
                "boxes": torch.rand((50, 4)),
                "yaws": torch.randn((50, 2)),
                "obj_idxes": torch.randperm(50),
            },
            {
                "labels": torch.randint(0, 2, (30,)),
                "boxes": torch.rand((30, 4)),
                "yaws": torch.randn((30, 2)),
                "obj_idxes": torch.randperm(30),
            },
        ]
    }
    outputs = {
        "pred_logits": torch.rand((300, 2)),
        "pred_boxes": torch.rand((300, 4)),
        "pred_yaws": torch.rand((300, 2)),
        "pred_zheights": torch.rand((300, 2)),
        "aux_outputs": [
            {
                "obj_idxes": torch.ones((300,)) * -1,
                "pred_logits": torch.rand((300, 2)),
                "pred_boxes": torch.rand((300, 4)),
                "pred_yaws": torch.rand((300, 2)),
                "pred_zheights": torch.rand((300, 2)),
                "matched_gt_idxes": torch.ones((300)).long() * -1,
            }
        ],
    }
    track_instances = {
        "matched_gt_idxes": torch.ones((300)).long() * -1,
        "obj_idxes": torch.ones((300,)) * -1,
    }
    return gt_instances, outputs, track_instances


@pytest.mark.parametrize(
    ["iou_type", "match_indices_each_layer"],
    [
        pytest.param("ciou", True),
        pytest.param("ciou", False),
        pytest.param("diou", False),
        pytest.param("giou", False),
    ],
)
def test_e2e_matcher(iou_type, match_indices_each_layer):
    gt_instances, outputs, track_instances = get_fake_data()
    matcher = E2EHungarianMatcher(iou_loss_type=iou_type)
    clipMatcher = E2EClipMatcher(
        matcher=matcher, match_indices_each_layer=match_indices_each_layer
    )
    clipMatcher.initialize_for_single_clip(gt_instances)
    pred_instances, output_for_losses = clipMatcher.match_for_single_frame(
        outputs, track_instances
    )
    output_key = "frame_0_output_layer"
    assert output_key in output_for_losses
    assert "pred_output" in output_for_losses[output_key]
    assert "gt_instances" in output_for_losses[output_key]
    assert "match_indices" in output_for_losses[output_key]
    assert "pre_matched_indices" in output_for_losses[output_key]

    assert "pred_logits" in output_for_losses[output_key]["pred_output"]
    assert "matched_gt_idxes" in output_for_losses[output_key]["pred_output"]
    pred_logits = output_for_losses[output_key]["pred_output"]["pred_logits"]
    assert pred_logits.size(0) == 300 and pred_logits.size(-1) == 2
    match_indices = output_for_losses[output_key]["match_indices"][0]
    assert (match_indices[0] < 0).sum() == 0
    assert (match_indices[1] < 0).sum() == 0
    pre_match_indices = output_for_losses[output_key]["pre_matched_indices"]
    assert (pre_match_indices[0] < 0).sum() == 0
    assert (pre_match_indices[1] < 0).sum() == 0

    outputs.update(pred_instances)
    outputs.update(
        {
            "aux_outputs": [
                {
                    "obj_idxes": torch.ones((300,)) * -1,
                    "pred_logits": torch.rand((300, 2)),
                    "pred_boxes": torch.rand((300, 4)),
                    "pred_yaws": torch.rand((300, 2)),
                    "pred_zheights": torch.rand((300, 2)),
                    "matched_gt_idxes": torch.ones((300)).long() * -1,
                }
            ],
        }
    )

    clipMatcher._step()

    _, output_for_losses = clipMatcher.match_for_single_frame(
        outputs, pred_instances
    )
    output_key = "frame_1_output_layer"
    assert output_key in output_for_losses
    assert "pred_output" in output_for_losses[output_key]
    assert "gt_instances" in output_for_losses[output_key]
    assert "match_indices" in output_for_losses[output_key]
    assert "pre_matched_indices" in output_for_losses[output_key]

    assert "pred_logits" in output_for_losses[output_key]["pred_output"]
    assert "matched_gt_idxes" in output_for_losses[output_key]["pred_output"]
    pred_logits = output_for_losses[output_key]["pred_output"]["pred_logits"]
    assert pred_logits.size(0) == 300 and pred_logits.size(-1) == 2
    match_indices = output_for_losses[output_key]["match_indices"][0]
    assert (match_indices[0] < 0).sum() == 0
    assert (match_indices[1] < 0).sum() == 0
    pre_match_indices = output_for_losses[output_key]["pre_matched_indices"]
    assert (pre_match_indices[0] < 0).sum() == 0
    assert (pre_match_indices[1] < 0).sum() == 0
