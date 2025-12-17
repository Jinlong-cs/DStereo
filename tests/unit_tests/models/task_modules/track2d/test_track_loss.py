from copy import deepcopy

import pytest
import torch

from hat.models.losses.hinge_loss import ElementwiseL2HingeLoss
from hat.models.task_modules.track2d.roi_track_loss import TrackLoss


def test_roi_track_loss():
    num_seq = 1
    seq_len = 2
    num_rois_per_frame = 10
    emb_len = 64
    num_cls = 1

    # track feat set to same
    track_feat = torch.ones((num_seq * seq_len * num_rois_per_frame, emb_len))
    preds = dict(
        track_feat=track_feat,
    )

    # target track id set to same
    track_id = torch.ones((num_seq * seq_len * num_rois_per_frame, num_cls))

    labels = dict(
        track_id=track_id,
        num_seq=[num_seq for _ in range(num_seq)],
        seq_len=[seq_len for _ in range(num_seq)],
    )

    match_loss = ElementwiseL2HingeLoss(
        reduction="mean",
        norm_type="positive_loss_elt",
    )
    track_loss = TrackLoss(
        match_loss=match_loss,
    )
    loss = track_loss(deepcopy(preds), deepcopy(labels))
    # loss must be 0.0
    assert torch.allclose(loss["rcnn_track_loss"], torch.tensor(0.0))

    # set each seq first frame rois track id to 2,
    # which is different with the second frame in each seq.
    track_id[0::seq_len, ...] = 2
    # loss should be 0.25
    loss = track_loss(deepcopy(preds), deepcopy(labels))
    assert torch.allclose(loss["rcnn_track_loss"], torch.tensor(0.25))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
