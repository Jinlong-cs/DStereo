import pytest
import torch
from easydict import EasyDict

from hat.models.task_modules.motr import MotrCriterion


def generate_empty_tracks(num_queries, queries_dim):
    """Genrate empty track instance for detection new object."""
    track_instances = EasyDict()
    track_instances.output_embedding = torch.zeros(
        (num_queries, queries_dim),
    )
    track_instances.obj_idxes = torch.full(
        (num_queries,),
        -1,
        dtype=torch.long,
    )
    track_instances.matched_gt_idxes = torch.full(
        (num_queries,),
        -1,
        dtype=torch.long,
    )
    track_instances.disappear_time = torch.zeros(
        (num_queries,),
        dtype=torch.long,
    )
    track_instances.iou = torch.zeros(
        (num_queries,),
        dtype=torch.float,
    )
    track_instances.scores = torch.zeros(
        (num_queries,),
        dtype=torch.float,
    )
    track_instances.track_scores = torch.zeros(
        (num_queries,),
        dtype=torch.float,
    )
    track_instances.pred_boxes = torch.zeros(
        (num_queries, 4),
        dtype=torch.float,
    )
    track_instances.pred_boxes_unsigmoid = torch.zeros(
        (num_queries, 4),
        dtype=torch.float,
    )
    track_instances.pred_logits = torch.zeros(
        (num_queries, 1),
        dtype=torch.float,
    )
    track_instances.mask_query = torch.ones(
        (num_queries,),
        dtype=torch.float,
    )
    track_instances.ref_pts = torch.randn((num_queries, 4))
    track_instances.query_pos = torch.randn((num_queries, queries_dim))
    return track_instances


def test_motr_criterion():

    criterion = MotrCriterion(num_classes=1)
    outputs_classes = []
    outputs_coords = []
    out_hs = torch.randn((1, 512, 256))
    for _ in range(6):
        outputs_classes.append(torch.randn((1, 512, 1)))
        outputs_coords.append(torch.randn((1, 512, 4)))
    track_instance = generate_empty_tracks(512, 256)

    targets = EasyDict(
        gt_bboxes=torch.tensor(
            [[10, 10, 20, 20], [20, 20, 40, 40]], dtype=torch.float
        ),
        gt_classes=torch.tensor([0, 0], dtype=torch.float),
        gt_ids=torch.tensor([0, 1], dtype=torch.float),
    )
    criterion.initialize_for_single_clip()
    criterion.match_for_single_frame(
        track_instance,
        outputs_classes,
        outputs_coords,
        out_hs,
        targets,
    )
    losses = criterion()
    assert len(losses) == 18


if __name__ == "__main__":
    pytest.main(["-s", __file__])
