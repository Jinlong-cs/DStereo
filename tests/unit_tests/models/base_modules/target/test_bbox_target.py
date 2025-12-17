import pytest
import torch

from hat.models.base_modules.label_encoder import (
    MatchLabelSepEncoder,
    OneHotClassEncoder,
    XYWHBBoxEncoder,
)
from hat.models.base_modules.matcher import IgRegionMatcher, MaxIoUMatcher
from hat.models.base_modules.target import BBoxTargetGenerator, ProposalTarget


def pseudo_sampler(
    boxes, match_pos_flag, match_gt_id, ig_flag, gt_boxes, gt_boxes_num
):
    return boxes, match_pos_flag, match_gt_id, ig_flag


@pytest.mark.parametrize("add_gt_bbox_to_proposal", [False, True])
@pytest.mark.parametrize("sampler", [None, pseudo_sampler])
def test_bbox_target_generator(add_gt_bbox_to_proposal, sampler):
    # only check some interface behavior. Computational
    # details are checked in member classes

    num_classes = 2

    matcher = MaxIoUMatcher(
        pos_iou=0.6,
        neg_iou=0.45,
        allow_low_quality_match=True,
        low_quality_match_iou=0.3,
        legacy_bbox=True,
    )

    label_encoder = MatchLabelSepEncoder(
        class_encoder=OneHotClassEncoder(
            num_classes=num_classes,
            class_agnostic_neg=False,
            exclude_background=True,
        ),
        bbox_encoder=XYWHBBoxEncoder(
            legacy_bbox=True,
        ),
        cls_use_pos_only=False,
        cls_on_hard=False,
        reg_on_hard=False,
    )

    ig_matcher = IgRegionMatcher(
        num_classes=num_classes,
        ig_region_overlap=0.5,
        legacy_bbox=True,
        exclude_background=True,
    )

    target0 = BBoxTargetGenerator(matcher, label_encoder, sampler=sampler)
    assert not target0.with_ig_region_matcher

    target1 = BBoxTargetGenerator(matcher, label_encoder, sampler=sampler)

    assert not target1.with_ig_region_matcher

    target2 = BBoxTargetGenerator(
        matcher, label_encoder, ig_region_matcher=ig_matcher, sampler=sampler
    )
    assert target2.with_ig_region_matcher

    target3 = ProposalTarget(
        matcher,
        label_encoder,
        ig_region_matcher=ig_matcher,
        sampler=sampler,
        add_gt_bbox_to_proposal=add_gt_bbox_to_proposal,
    )
    assert target3.with_ig_region_matcher

    # make sure padding works

    boxes = torch.randn(4, 20, 4)

    gt_nums = torch.randint(low=10, high=20, size=(4,))
    gt_boxes = [torch.randn(num, 4) for num in gt_nums]
    gt_boxes = [
        torch.hstack([box, torch.randint(num_classes, size=(num, 1))])
        for box, num in zip(gt_boxes, gt_nums)
    ]

    ig_nums = torch.randint(low=10, high=20, size=(4,))
    ig_regions = [torch.randn(num, 4) for num in ig_nums]
    ig_regions = [
        torch.hstack([ig_region, torch.randint(num_classes, size=(num, 1))])
        for ig_region, num in zip(ig_regions, ig_nums)
    ]

    target2(boxes, gt_boxes, ig_regions=ig_regions)

    target3(boxes, gt_boxes, ig_regions=ig_regions)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
