# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.core.box_utils import bbox_overlaps
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class RoIRandomSampler(nn.Module):
    """Random sample positive and negative roi.

    Args:
        num: number of proposals after sampled.
        pos_fraction: fraction of positive proposals. pos_num in the sampler
            result will be ``min(num * pos_fraction, all_pos_roi_num)``.
        neg_pos_ub: Upper bound ratio of neg/pos, maximum neg_pos_up * pos_num
            limit in the neg sampler. If neg_pos_up >= 0, neg_num
            in the sampler result will be
            ``min(num - pos_num, neg_pos_up * pos_num, all_neg_roi_num)``,
            otherwise will be ``min(num - pos_num, all_neg_roi_num)``.
            Default to -1.
        neg_num_when_no_pos: Max neg number when image has no positive rois.
            Default to -1, means no this limit.
        pad_with_neg: Whether to pad the sampler index with random negative
            when sampled number of pos + neg < num.
            Default to False, mean pad with ignore index.
        resample_fg: Whether to resample fore ground when the sample
            is not enough. Defaults to False.
        sampler_mode: Choice from ["query_rois" or "mask_rois"].
            "query_rois" mode: query selected rois, the boxes tensor
            shape will changed, usually used in roi_module (rcnn);
            "mask_rois" mode: mask the no selected boxes tensor to
            ignored and don't changed the tensor shape, usually used in
            rpn.
    """

    def __init__(
        self,
        num: int,
        pos_fraction: float,
        neg_pos_ub: float = -1,
        neg_num_when_no_pos: float = -1,
        pad_with_neg: bool = False,
        resample_fg: bool = False,
        sampler_mode: str = "query_rois",
    ):
        super(RoIRandomSampler, self).__init__()

        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.neg_num_when_no_pos = neg_num_when_no_pos
        self.pad_with_neg = pad_with_neg
        self.resample_fg = resample_fg

        assert not (
            (self.neg_pos_ub >= 0) and (self.pad_with_neg)
        ), f"conflic set pad_with_neg {pad_with_neg} \
            and neg_pos_ub {self.neg_pos_ub}"

        assert sampler_mode in [
            "query_rois",
            "mask_rois",
        ], f"sampler_mode {sampler_mode} is not support"
        self.sampler_mode = sampler_mode

    def random_choice(self, gallery: torch.Tensor, num: int) -> torch.Tensor:
        """Random select some elements from the gallery.

        `gallery` is a Tensor, the returned indices also is a Tensor.

        Args:
            gallery: indices pool.
            num: expected sample num.

        Returns:
            sampled indices.
        """
        assert len(gallery) >= num
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        return rand_inds

    def _sample_pos(
        self, match_pos_flag, match_gt_id, ig_flag, max_overlaps, num_expected
    ):
        """Randomly sample some positive samples."""
        # only ingore the negative witch matched to the ignore_region
        # pos_inds = torch.nonzero(
        #     (match_pos_flag > 0) & (ig_flag == 0), as_tuple=False
        # )
        pos_inds = torch.nonzero(match_pos_flag > 0, as_tuple=False)
        pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            # resample fg if needed
            if self.resample_fg and pos_inds.numel() != 0:
                resample_index = torch.randint(
                    0,
                    pos_inds.numel(),
                    (num_expected - pos_inds.numel(),),
                    device=pos_inds.device,
                )  # noqa
                resample_pos_inds = pos_inds[resample_index]
                pos_inds = torch.cat([pos_inds, resample_pos_inds])
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(
        self, match_pos_flag, match_gt_id, ig_flag, max_overlaps, num_expected
    ):
        """Randomly sample some negative samples.."""
        if ig_flag is None:
            neg_inds = torch.nonzero(match_pos_flag == 0, as_tuple=False)
        else:
            neg_inds = torch.nonzero(
                (match_pos_flag == 0) & (ig_flag == 0), as_tuple=False
            )
        neg_inds = neg_inds.squeeze(1)
        if neg_inds.numel() <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    def sample(self, match_pos_flag, match_gt_id, ig_flag, overlaps):
        if overlaps.shape[1] == 0:
            max_overlaps = np.zeros(overlaps.shape[0])
        else:
            max_overlaps = overlaps.max(dim=1)[0].detach().cpu().numpy()
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self._sample_pos(
            match_pos_flag,
            match_gt_id,
            ig_flag,
            max_overlaps,
            num_expected_pos,
        )

        if not self.resample_fg:
            pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        if (self.neg_num_when_no_pos >= 0) and (num_sampled_pos == 0):
            num_expected_neg = max(self.neg_num_when_no_pos, num_expected_neg)
        neg_inds = self._sample_neg(
            match_pos_flag,
            match_gt_id,
            ig_flag,
            max_overlaps,
            num_expected_neg,
        )

        if self.pad_with_neg and neg_inds.numel() > 0:
            num_sampled_neg = neg_inds.numel()
            num_expected_pad = self.num - num_sampled_pos - num_sampled_neg
            while num_expected_pad > 0:
                gap = min(num_expected_pad, neg_inds.numel())
                neg_pad = self.random_choice(neg_inds, gap)
                neg_inds = torch.cat([neg_inds, neg_pad])
                num_expected_pad = (
                    self.num - num_sampled_pos - neg_inds.numel()
                )
            ig_inds = None
        else:
            neg_inds = neg_inds.unique()
            num_sampled_neg = neg_inds.numel()
            num_expected_ig = self.num - num_sampled_pos - num_sampled_neg
            if num_expected_ig > 0:
                all_set = set(np.where(max_overlaps >= 0)[0])
                ig_inds = torch.tensor(
                    list(
                        all_set
                        - set(pos_inds.cpu().numpy())
                        - set(neg_inds.cpu().numpy())
                    ),
                    dtype=pos_inds.dtype,
                    device=pos_inds.device,
                )
                ig_inds = self.random_choice(ig_inds, num_expected_ig)
            else:
                ig_inds = None

        return torch.cat([pos_inds, neg_inds]), ig_inds

    @autocast(enabled=False)
    def forward(
        self,
        boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
        ig_flag: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_boxes_num: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # noqa: D205,D400,E501
        """
        Args:
            boxes: Box tensor with shape (B, N, 4). B stands for
                batch size, N the number of boxes for each sample.
            match_pos_flag: flag tensor with shape (B, N). Entries with value
                1 represents positive in matching, 0 for neg and -1
                for ignore.
            matched_gt_id: matched_gt_id tensor in (B, N).
                The best matched gt box id. -1 means unavailable.
            ig_flag: Flag tensor with shape (B, N, self._num_classes - 1) when
                self._exclude_background is True, or otherwise
                (B, N, self._num_classes). The range of the output is {0, 1}.
                Entries with value 1 are matched with ignore regions.
            gt_boxes: GT box tensor with shape (B, M1, 5+), In one sample,
                if the number of gt boxes is less than M1, the first M1
                entries should be filled with real data, and others
                padded with arbitrary values.
            gt_box_num: If provided, it is the gt box num tensor with shape
                (B,), the actual number of  gt boxes of each sample. Cannot
                be greater than M1.

        Returns:
            (boxes, match_pos_flag, match_gt_id, ig_flag): Tuple of sub-sampled
                tensors.
        """

        assert (
            isinstance(boxes, torch.Tensor) and len(boxes.shape) == 3
        ), "boxes must be torch tensor and shape must be (B, N, 4)"

        # set invalid bbox to ignore
        invalid_inds = torch.logical_or(
            (boxes[:, :, 2] <= boxes[:, :, 0]),
            (boxes[:, :, 3] <= boxes[:, :, 1]),
        )
        match_pos_flag[invalid_inds] = -1
        match_gt_id[invalid_inds] = -1
        if ig_flag is not None:
            ig_flag[invalid_inds] = 1

        if self.sampler_mode == "query_rois":
            boxes_ret_list = []
            match_pos_flag_ret_list = []
            match_gt_id_ret_list = []
            if ig_flag is not None:
                ig_flag_ret_list = []
        elif self.sampler_mode == "mask_rois":
            select_index_mask = torch.zeros(
                (boxes.shape[0], boxes.shape[1]),
                dtype=torch.bool,
                device=boxes.device,
            )
        else:
            raise NotImplementedError(
                f"sampler_mode {self.sampler_mode} is not implemented"
            )

        num_imgs = gt_boxes.shape[0]
        for i in range(num_imgs):
            boxes_i = boxes[i]
            if gt_boxes_num is not None:
                num_gt_i = int(gt_boxes_num[i])
                gt_boxes_i = gt_boxes[i, 0:num_gt_i]

            match_pos_flag_i = match_pos_flag[i]
            match_gt_id_i = match_gt_id[i]
            if ig_flag is not None:
                ig_flag_i = ig_flag[i]
            overlaps_i = bbox_overlaps(boxes_i, gt_boxes_i[:, 0:4])
            sample_inds, ig_inds = self.sample(
                match_pos_flag_i,
                match_gt_id_i,
                ig_flag_i.sum(dim=1).to(torch.bool)
                if ig_flag is not None
                else None,
                overlaps_i,
            )
            if ig_inds is not None:
                sample_inds = torch.cat([sample_inds, ig_inds])
                if ig_flag is not None:
                    ig_flag_i[ig_inds] = 1
                # NOTE: don't forget set ig_inds in the match_pos_flag to -1,
                # otherwise, the ignore state in the label encoder
                # will lose efficacy.
                match_pos_flag_i[ig_inds] = -1
                match_gt_id_i[ig_inds] = -1

            if self.sampler_mode == "query_rois":
                boxes_ret_list.append(boxes_i[sample_inds])
                match_pos_flag_ret_list.append(match_pos_flag_i[sample_inds])
                match_gt_id_ret_list.append(match_gt_id_i[sample_inds])
                if ig_flag is not None:
                    ig_flag_ret_list.append(ig_flag_i[sample_inds])
            elif self.sampler_mode == "mask_rois":
                select_index_mask[i][sample_inds] = True
                match_pos_flag[i][~select_index_mask[i]] = -1
                match_gt_id[i][~select_index_mask[i]] = -1
                if ig_flag is not None:
                    ig_flag[i][~select_index_mask[i]] = 1
            else:
                raise NotImplementedError(
                    f"sampler_mode {self.sampler_mode} is not implemented"
                )

        if self.sampler_mode == "query_rois":
            boxes_ret = torch.stack(boxes_ret_list, dim=0)
            match_pos_flag_ret = torch.stack(match_pos_flag_ret_list, dim=0)
            match_gt_id_ret = torch.stack(match_gt_id_ret_list, dim=0)
            if ig_flag is not None:
                ig_flag_ret = torch.stack(ig_flag_ret_list, dim=0)
            else:
                ig_flag_ret = None
        elif self.sampler_mode == "mask_rois":
            boxes_ret = boxes
            match_pos_flag_ret = match_pos_flag
            match_gt_id_ret = match_gt_id
            ig_flag_ret = ig_flag
        else:
            raise NotImplementedError(
                f"sampler_mode {self.sampler_mode} is not implemented"
            )

        return (
            boxes_ret,
            match_pos_flag_ret,
            match_gt_id_ret,
            ig_flag_ret,
        )

    def set_qconfig(self):
        self.qconfig = None


@OBJECT_REGISTRY.register
class RoIHardProposalSampler(RoIRandomSampler):
    """Sampler hard positive & negative rois on the proposals.

    Sampling hard positive rois on the bottom of all positive rois, and
    negative rois on the top of all negative rois, all rois are the out of rpn
    postprocess (already sorted by the NMS). `bottom_pos_fraction` of needed
    positive RoIs are sampled on the bottom. `top_neg_fraction` of needed
    negative RoIs are sampled on the top. The others are sampled randomly.

    Args:
        num: Same as RoIRandomSampler.
        pos_fraction: Same as RoIRandomSampler.
        neg_pos_ub: Same as RoIRandomSampler.
        neg_num_when_no_pos: Same as RoIRandomSampler.
        pad_with_neg: Same as RoIRandomSampler.
        bottom_pos_fraction: Sampling positive fraction of proposals on the
            bottom of proposals.
        top_neg_fraction: Sampling negative fraction of proposals on the top of
            proposals.
    """

    def __init__(
        self,
        num: int,
        pos_fraction: float,
        neg_pos_ub: float = -1,
        neg_num_when_no_pos: float = -1,
        pad_with_neg: bool = False,
        bottom_pos_fraction: float = 0,
        top_neg_fraction: float = 0,
    ):
        super(RoIHardProposalSampler, self).__init__(
            num,
            pos_fraction,
            neg_pos_ub,
            neg_num_when_no_pos,
            pad_with_neg,
        )

        assert 0 <= top_neg_fraction <= 1
        self.top_neg_fraction = top_neg_fraction

        assert 0 <= bottom_pos_fraction <= 1
        self.bottom_pos_fraction = bottom_pos_fraction

    def sample_on_top(
        self, gallery: torch.Tensor, num_expected: int
    ) -> torch.Tensor:
        """Sample on top.

        Args:
            gallery: indices of boxes gallery
            num_expected: Number of expected samples

        Returns:
            Indices of samples
        """
        if gallery.numel() <= num_expected:
            return gallery
        else:
            return gallery[0:num_expected]

    def sample_on_bottom(
        self, gallery: torch.Tensor, num_expected: int
    ) -> torch.Tensor:
        """Sample on bottom.

        Args:
            gallery: indices of boxes gallery
            num_expected: Number of expected samples

        Returns:
            Indices of samples
        """
        if gallery.numel() <= num_expected:
            return gallery
        else:
            return gallery[-num_expected:]

    def _sample_neg(
        self, match_pos_flag, match_gt_id, ig_flag, max_overlaps, num_expected
    ):
        """Sample negative boxes."""
        if self.top_neg_fraction <= 0:
            return super()._sample_neg(
                match_pos_flag,
                match_gt_id,
                ig_flag,
                max_overlaps,
                num_expected,
            )
        if ig_flag is None:
            neg_inds = torch.nonzero(match_pos_flag == 0, as_tuple=False)
        else:
            neg_inds = torch.nonzero(
                (match_pos_flag == 0) & (ig_flag == 0), as_tuple=False
            )
        neg_inds = neg_inds.squeeze(1)

        if neg_inds.numel() <= num_expected:
            return neg_inds
        else:
            # sampling on the negative top
            num_expected_top_sampling = max(
                1, int(num_expected * self.top_neg_fraction)
            )
            top_sampled_inds = self.sample_on_top(
                neg_inds,
                num_expected_top_sampling,
            )

            floor_neg_inds = neg_inds[top_sampled_inds.numel() :]
            num_expected_floor = num_expected - top_sampled_inds.numel()

            if floor_neg_inds.numel() > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor
                )
            else:
                sampled_floor_inds = floor_neg_inds
            sampled_inds = torch.cat([top_sampled_inds, sampled_floor_inds])
            return sampled_inds

    def _sample_pos(
        self, match_pos_flag, match_gt_id, ig_flag, max_overlaps, num_expected
    ):
        """Sample positive boxes."""
        if self.bottom_pos_fraction <= 0:
            return super()._sample_pos(
                match_pos_flag,
                match_gt_id,
                ig_flag,
                max_overlaps,
                num_expected,
            )
        # only ingore the negative witch matched to the ignore_region
        # pos_inds = torch.nonzero(
        #     (match_pos_flag > 0) & (ig_flag == 0), as_tuple=False
        # )
        pos_inds = torch.nonzero(match_pos_flag > 0, as_tuple=False)
        pos_inds = pos_inds.squeeze(1)

        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            # sampling on the negative top
            pos_set = set(pos_inds.cpu().numpy())

            num_expected_bottom_sampling = max(
                1, int(num_expected * self.bottom_pos_fraction)
            )

            # maybe gt boxes as proposal rois, so filter these
            hard_set = set(np.where(max_overlaps < 1)[0])
            pos_hard_inds = torch.tensor(
                list(pos_set & hard_set),
                dtype=pos_inds.dtype,
                device=pos_inds.device,
            )

            bottom_sampled_inds = self.sample_on_bottom(
                pos_hard_inds,
                num_expected_bottom_sampling,
            )

            ceil_pos_inds = torch.tensor(
                list(pos_set - set(bottom_sampled_inds.cpu().numpy())),
                dtype=pos_inds.dtype,
                device=pos_inds.device,
            )

            num_expected_ceil = num_expected - bottom_sampled_inds.numel()

            if len(ceil_pos_inds) > num_expected_ceil:
                sampled_ceil_inds = self.random_choice(
                    ceil_pos_inds, num_expected_ceil
                )
            else:
                sampled_ceil_inds = ceil_pos_inds
            sampled_inds = torch.cat([bottom_sampled_inds, sampled_ceil_inds])
            return sampled_inds
