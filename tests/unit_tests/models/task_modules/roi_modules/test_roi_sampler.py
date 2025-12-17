# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.task_modules.roi_modules.roi_sampler import (
    RoIHardProposalSampler,
    RoIRandomSampler,
)

batch = 2
num_bbox = 4

ig_flag1 = torch.zeros((batch, num_bbox, 1))


@pytest.mark.parametrize("ig_flag", [ig_flag1, None])
def test_random_sampler(ig_flag):
    # gt
    gt_bboxes = torch.ones((batch, num_bbox, 4))
    for i in range(num_bbox):
        width, height = 100, 100
        gt_bboxes[:, i, 0] = i * width
        gt_bboxes[:, i, 1] = i * height
        gt_bboxes[:, i, 2] = gt_bboxes[:, i, 0] + width
        gt_bboxes[:, i, 3] = gt_bboxes[:, i, 1] + height
    gt_box_num = torch.zeros(batch)
    for i in range(batch):
        gt_box_num[i] = num_bbox

    # rois
    boxes = gt_bboxes + 1
    boxes[:, num_bbox // 2 :, :] *= 100
    match_pos_flag = torch.zeros((batch, num_bbox))
    match_gt_id = -1 * torch.ones((batch, num_bbox))
    for i in range(num_bbox // 2):
        match_pos_flag[:, i] = True
        match_gt_id[:, i] = i

    sample_num = num_bbox // 2
    pos_fraction = 0.5
    random_sampler = RoIRandomSampler(
        num=sample_num,
        pos_fraction=pos_fraction,
    )

    (
        boxes_ret,
        match_pos_flag_ret,
        match_gt_id_ret,
        ig_flag_ret,
    ) = random_sampler.forward(
        boxes=boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
        ig_flag=ig_flag,
        gt_boxes=gt_bboxes,
        gt_boxes_num=gt_box_num,
    )
    # check sampled result
    assert (match_pos_flag_ret > 0).sum() == batch * sample_num * pos_fraction
    assert (match_pos_flag_ret == 0).sum() == batch * sample_num * (
        1 - pos_fraction
    )

    # set half neg roi label to ignore
    if ig_flag is not None:
        ig_flag[:, -1, :] = 1
    (
        boxes_ret,
        match_pos_flag_ret,
        match_gt_id_ret,
        ig_flag_ret,
    ) = random_sampler.forward(
        boxes=boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
        ig_flag=ig_flag,
        gt_boxes=gt_bboxes,
        gt_boxes_num=gt_box_num,
    )
    # check sampled result
    assert (match_pos_flag_ret > 0).sum() == batch * sample_num * pos_fraction
    assert (match_pos_flag_ret == 0).sum() == batch * sample_num * (
        1 - pos_fraction
    )
    if ig_flag is not None:
        # need sampled one neg rois in each image,
        # and only boxes in index 2 are no ignored neg roi
        # so sampled result neg roi must be boxes in index 2.
        assert torch.allclose(
            boxes_ret[match_pos_flag_ret == 0], boxes[:, 2, :]
        )


def test_random_sampler_with_whole_image_ignore():
    ig_flag = torch.ones((batch, num_bbox, 1))
    # gt
    gt_bboxes = torch.ones((batch, num_bbox, 4))
    for i in range(num_bbox):
        width, height = 100, 100
        gt_bboxes[:, i, 0] = i * width
        gt_bboxes[:, i, 1] = i * height
        gt_bboxes[:, i, 2] = gt_bboxes[:, i, 0] + width
        gt_bboxes[:, i, 3] = gt_bboxes[:, i, 1] + height
    gt_box_num = torch.zeros(batch)
    for i in range(batch):
        gt_box_num[i] = num_bbox

    # rois
    boxes = gt_bboxes + 1
    boxes[:, num_bbox // 2 :, :] *= 100
    match_pos_flag = torch.zeros((batch, num_bbox))
    match_gt_id = -1 * torch.ones((batch, num_bbox))
    for i in range(num_bbox // 2):
        match_pos_flag[:, i] = True
        match_gt_id[:, i] = i

    sample_num = num_bbox // 2
    pos_fraction = 0.5
    random_sampler = RoIRandomSampler(
        num=sample_num,
        pos_fraction=pos_fraction,
    )

    (
        boxes_ret,
        match_pos_flag_ret,
        match_gt_id_ret,
        ig_flag_ret,
    ) = random_sampler.forward(
        boxes=boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
        ig_flag=ig_flag,
        gt_boxes=gt_bboxes,
        gt_boxes_num=gt_box_num,
    )
    # check sampled result
    assert (match_pos_flag_ret > 0).sum() == batch * sample_num * pos_fraction
    # because whole image ignore, no negative in the sampled rois
    assert (match_pos_flag_ret == 0).sum() == 0


ig_flag2 = torch.zeros((batch, num_bbox, 1))


@pytest.mark.parametrize("ig_flag", [ig_flag2, None])
def test_hard_proposal_sampler(ig_flag):
    # gt
    gt_bboxes = torch.ones((batch, num_bbox, 4))
    for i in range(num_bbox):
        width, height = 100, 100
        gt_bboxes[:, i, 0] = i * width
        gt_bboxes[:, i, 1] = i * height
        gt_bboxes[:, i, 2] = gt_bboxes[:, i, 0] + width
        gt_bboxes[:, i, 3] = gt_bboxes[:, i, 1] + height
    gt_box_num = torch.zeros(batch)
    for i in range(batch):
        gt_box_num[i] = num_bbox

    # rois
    boxes = gt_bboxes + 1
    boxes[:, num_bbox // 2 :, :] *= 100
    match_pos_flag = torch.zeros((batch, num_bbox))
    match_gt_id = -1 * torch.ones((batch, num_bbox))
    for i in range(num_bbox // 2):
        match_pos_flag[:, i] = True
        match_gt_id[:, i] = i

    sample_num = num_bbox // 2
    pos_fraction = 0.5
    hard_proposal_sampler = RoIHardProposalSampler(
        num=sample_num,
        pos_fraction=pos_fraction,
        bottom_pos_fraction=0.5,
        top_neg_fraction=0.5,
    )

    (
        boxes_ret,
        match_pos_flag_ret,
        match_gt_id_ret,
        ig_flag_ret,
    ) = hard_proposal_sampler.forward(
        boxes=boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
        ig_flag=ig_flag,
        gt_boxes=gt_bboxes,
        gt_boxes_num=gt_box_num,
    )
    # check sampled result
    assert (match_pos_flag_ret > 0).sum() == batch * sample_num * pos_fraction
    assert (match_pos_flag_ret == 0).sum() == batch * sample_num * (
        1 - pos_fraction
    )
    # need sampled one neg/pos rois in each image,
    # so sampled result neg roi must be boxes in index 2 and
    # sampled result pos roi must be boxes in index 1
    assert torch.allclose(boxes_ret[match_pos_flag_ret == 0], boxes[:, 2, :])
    assert torch.allclose(boxes_ret[match_pos_flag_ret > 0], boxes[:, 1, :])

    # set half neg roi label to ignore
    if ig_flag is not None:
        ig_flag[:, -1, :] = 1
    (
        boxes_ret,
        match_pos_flag_ret,
        match_gt_id_ret,
        ig_flag_ret,
    ) = hard_proposal_sampler.forward(
        boxes=boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
        ig_flag=ig_flag,
        gt_boxes=gt_bboxes,
        gt_boxes_num=gt_box_num,
    )
    # check sampled result
    assert (match_pos_flag_ret > 0).sum() == batch * sample_num * pos_fraction
    assert (match_pos_flag_ret == 0).sum() == batch * sample_num * (
        1 - pos_fraction
    )
    if ig_flag is not None:
        # need sampled one neg rois in each image,
        # and only boxes in index 2 are no ignored neg roi
        # so sampled result neg roi must be boxes in index 2.
        assert torch.allclose(
            boxes_ret[match_pos_flag_ret == 0], boxes[:, 2, :]
        )

    # set all rois to ignore
    if ig_flag is not None:
        ig_flag[...] = 1
    (
        boxes_ret,
        match_pos_flag_ret,
        match_gt_id_ret,
        ig_flag_ret,
    ) = hard_proposal_sampler.forward(
        boxes=boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
        ig_flag=ig_flag,
        gt_boxes=gt_bboxes,
        gt_boxes_num=gt_box_num,
    )
    # check sampled result
    assert (match_pos_flag_ret > 0).sum() == batch * sample_num * pos_fraction
    if ig_flag is not None:
        # all rois is ignored, no neg in sampled result
        assert (match_pos_flag_ret == 0).sum() == 0


ig_flag3 = torch.zeros((batch, num_bbox, 1))


def test_hard_proposal_sampler_with_no_gt_rois(ig_flag=ig_flag3):
    # gt
    gt_bboxes = torch.ones((batch, num_bbox, 4))
    for i in range(num_bbox):
        width, height = 100, 100
        gt_bboxes[:, i, 0] = i * width
        gt_bboxes[:, i, 1] = i * height
        gt_bboxes[:, i, 2] = gt_bboxes[:, i, 0] + width
        gt_bboxes[:, i, 3] = gt_bboxes[:, i, 1] + height
    gt_box_num = torch.zeros(batch)
    for i in range(batch):
        gt_box_num[i] = num_bbox

    # rois
    boxes = gt_bboxes + 1
    boxes[:, :, :] *= 1000
    match_pos_flag = torch.zeros((batch, num_bbox))
    match_gt_id = torch.zeros((batch, num_bbox))

    sample_num = num_bbox // 2
    pos_fraction = 0.5
    hard_proposal_sampler = RoIHardProposalSampler(
        num=sample_num,
        pos_fraction=pos_fraction,
        bottom_pos_fraction=0.5,
        top_neg_fraction=0.5,
    )

    (
        boxes_ret,
        match_pos_flag_ret,
        match_gt_id_ret,
        ig_flag_ret,
    ) = hard_proposal_sampler.forward(
        boxes=boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
        ig_flag=ig_flag,
        gt_boxes=gt_bboxes,
        gt_boxes_num=gt_box_num,
    )
    # check sampled result
    assert torch.all(match_pos_flag_ret == 0)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
