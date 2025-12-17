import torch

from hat.models.task_modules.landmark.human_pose_decoder import (
    HumanPoseDecoder,
)


def test_human_pose_decoder():
    decoder = HumanPoseDecoder(
        num_ldmk=2,
        pos_distance=0.1,
        pos_distance_scale=4.0,
        ldmk_transform_scores=False,
    )
    ldmk_pred = torch.randn((8, 45, 16, 16))
    batch_rois = torch.randn((2, 4, 4))
    head_out = {
        "ldmk_pred": ldmk_pred,
    }
    r = decoder(batch_rois, head_out)
    assert "pred_ldmk" in r
