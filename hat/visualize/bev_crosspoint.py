# Copyright (c) Horizon Robotics. All rights reserved.

import os
from typing import Callable, Dict

import cv2
import numpy as np

from hat.core.affine import get_vcs2bev_img_mat
from hat.registry import OBJECT_REGISTRY
from hat.visualize.utils import ANCBEVImgStitcher

category2vis_name = {
    "merge_start": "m_st",
    "merge_stop": "m_sp",
    "split_start": "s_st",
    "split_stop": "s_sp",
    "u_turn": "u_t",
    "other": "o",
    "changepoint": "cp",
}


def get_bev_pts(cross_pts, mat_vcs2bev):
    """Convert crosspoint from vcs to bev.

    Args:
        cross_pts: crosspoint result, [x, y, cls_id, score].
        mat_vcs2bev: transfer matrix from vcs coord to bev pixel.

    """
    if not cross_pts:
        return []
    cross_pts = np.array(cross_pts, np.float)
    pts = np.array([[pt[0], pt[1], 1] for pt in cross_pts]).transpose()
    bev_pts = (mat_vcs2bev @ pts).transpose()
    bev_pts[:, 0] = bev_pts[:, 0] / bev_pts[:, 2]
    bev_pts[:, 1] = bev_pts[:, 1] / bev_pts[:, 2]
    cross_pts[:, :2] = bev_pts[:, :2]
    return cross_pts


def image_cover(cover_mask, img, alpha):
    img_copy = img.copy()
    mask1 = cover_mask[:, :, 0] != 0
    mask2 = cover_mask[:, :, 1] != 0
    mask3 = cover_mask[:, :, 2] != 0
    mask = mask1 | mask2 | mask3
    img_copy[mask, :] = (1.0 - alpha) * img_copy[mask, :] + alpha * cover_mask[
        mask, :
    ]
    return img_copy


@OBJECT_REGISTRY.register
class CrossPointVisualize(object):
    """
    Visualize class for Bev-crosspoint.

    Args:
        stride: the stride of module output.
        cls_group_map: Output categories of each group.
        bev_size: the size of bird eye view.
        vcs_range: visbile range of bev, (bottom, right, top, left) in order.
        reformat_fn: the function for reformat the task results.
        extra_img: whether need extra 6v imgs.
        vcs range mode, included "wide" and "small".
        prefix_name: prefix name of model output.
    """

    def __init__(
        self,
        stride: int,
        target_categorys: Dict,
        bev_size: tuple,
        vcs_range: tuple,
        reformat_fn: Callable = None,
        extra_img: ANCBEVImgStitcher = None,
        range_mode: str = "wide",
        prefix_name: str = None,
    ):
        self.stride = stride
        self.target_categorys = target_categorys
        self.reformat_fn = reformat_fn
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.extra_img = extra_img
        self._mat_vcs2bev = None
        self.range_mode = range_mode
        self.prefix_name = prefix_name
        if range_mode == "wide":
            self.prefix = (
                "bev_crosspoint_bev_stage2_crosspoint_head_predict_crosspt_predict"  # noqa
                if prefix_name is None
                else f"{prefix_name}_head_predict_crosspt_predict"
            )
        elif range_mode == "small":
            self.prefix = (
                "bev_crosspoint_bev_stage2_crosspoint_small_head_predict_crosspt_predict"  # noqa
                if prefix_name is None
                else f"{prefix_name}_small_head_predict_crosspt_predict"
            )
        assert self.range_mode in ["wide", "small"]

    @property
    def mat_vcs2bev(self):
        if self._mat_vcs2bev is None:
            self._mat_vcs2bev = get_vcs2bev_img_mat(
                self.vcs_range, self.bev_size
            )
        return self._mat_vcs2bev

    @staticmethod
    def draw_crosspoint(
        cross_pts,
        bev_h,
        bev_w,
        mat_vcs2bev,
        target_categorys,
        text=None,
        thickness=1,
        pt_thickness=None,
        img_bev=None,
        is_gt=False,
        ignore_mask=None,
    ):
        """Crosspoint result visulization.

        Args:
            cross_pts: crosspoint result, [x, y, cls_id, score].
            bev_h: bev image height.
            bev_w: bev image width.
            mat_vcs2bev: transfer matrix from vcs coord to bev pixel.
            target_categorys: crosspoint output category.
            text: annotation text.
            thickness: draw pt thickness for backup.
            pt_thickness: draw pt thickness.
            img_bev: image in bev view.
            is_gt: is ground truth or prediction.
            ignore_mask: ignore mask in gt.

        """
        pt_thickness = thickness * 2 if pt_thickness is None else pt_thickness
        if img_bev is None:
            img_bev = np.zeros((bev_h, bev_w, 3))
        else:
            h, w, _ = img_bev.shape
            assert h == bev_h and w == bev_w
        if text:
            cv2.putText(
                img_bev,
                text,
                (50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2,
            )

        bev_ego_pts = (mat_vcs2bev @ np.array([[0], [0], [1]])).transpose()
        bev_ego_pts[:, 0] = bev_ego_pts[:, 0] / bev_ego_pts[:, 2]
        bev_ego_pts[:, 1] = bev_ego_pts[:, 1] / bev_ego_pts[:, 2]
        bev_ego_pts = bev_ego_pts[0, :2]
        cv2.rectangle(
            img_bev,
            (int(bev_ego_pts[0]) - 5, int(bev_ego_pts[1]) - 10),
            (int(bev_ego_pts[0]) + 5, int(bev_ego_pts[1]) + 10),
            color=(0, 255, 0),
            thickness=thickness,
        )

        if ignore_mask is not None:
            img_bev = image_cover(ignore_mask, img_bev, alpha=0.8)
        bev_pts = get_bev_pts(cross_pts, mat_vcs2bev)
        for pt in bev_pts:
            color = [0, 0, 255] if not is_gt else [0, 255, 0]
            cv2.circle(
                img_bev,
                (int(round(pt[0])), int(round(pt[1]))),
                radius=1,
                color=color,
                thickness=pt_thickness,
            )
            cv2.circle(
                img_bev,
                (int(round(pt[0])), int(round(pt[1]))),
                radius=7,
                color=color,
                thickness=1,
            )

            # find key from value
            cls_name = [k for k, v in target_categorys.items() if v == pt[2]][
                0
            ]

            font_scale = 0.5
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            cls_text = category2vis_name[cls_name]
            size, _ = cv2.getTextSize(
                cls_text, font_face, font_scale, pt_thickness
            )
            height_offset = 3
            cv2.putText(
                img_bev,
                cls_text,
                (
                    int(round(pt[0]) - size[0] / 2),
                    int(round(pt[1])) + size[1] + height_offset,
                ),
                fontFace=font_face,
                fontScale=font_scale,
                color=color,
                thickness=pt_thickness,
            )
        return img_bev

    def __call__(self, batch, results, task, save_dir):
        if self.prefix_name is not None:
            save_dir = save_dir + "_" + self.prefix_name
            os.makedirs(save_dir, exist_ok=True)

        if self.reformat_fn:
            results = self.reformat_fn(results)

        if isinstance(batch, tuple):  # multidataloader
            batch = batch[0]

        prefix = self.prefix
        pred_pts_batch = results[prefix]

        batch_size = batch["img"][0].shape[0]
        for ind in range(batch_size):
            pred_pts = pred_pts_batch[ind]
            ret_img = self.draw_crosspoint(
                pred_pts,
                self.bev_size[0],
                self.bev_size[1],
                self.mat_vcs2bev,
                self.target_categorys,
            )

            if self.extra_img:
                assert "img_vis" in batch
                batch_extra_imgs = [
                    i[ind].cpu().numpy().squeeze().transpose(1, 2, 0)
                    for i in batch["img_vis"][0]
                ]
                ret_img = self.extra_img(
                    multiview_imgs=batch_extra_imgs, bev_img=ret_img
                )

            name = list(batch["timestamp"].cpu().numpy())[ind]
            cv2.imwrite(f"{save_dir}/{name}.png", ret_img)
