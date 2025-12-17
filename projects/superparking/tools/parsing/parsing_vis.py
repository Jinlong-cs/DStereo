"""psd validation script.

Author: @chunyu.bi

TODO: This script is used to run ipm int evalution successfully. It shoule be
deprecated after the ipm adopts the new evaluation method.
"""
import copy
import os

import cv2
import numpy as np
import torch

ipm_8cls_color_list = [
    [0, 255, 0],  # 0: Green
    [0, 0, 0],  # 1: Black
    [255, 255, 255],  # 2: White
    [0, 0, 255],  # 3: Blue
    [255, 97, 0],  # 4: Orange
    [255, 255, 0],  # 5: Yellow
    [255, 0, 0],  # 6: Red
    [160, 32, 240],  # 7: Purple
    [115, 74, 18],  # 8: Brown
    [192, 192, 192],  # 9: Shallow gray
]

super_parking_ipm_color_list = [
    [128, 64, 128],  # 0: road
    # [200,200,128], # 1: Sidewalk
    [230, 150, 140],  # 2: lane_line
    [18, 145, 170],  # 3: parking_line
    [0, 0, 230],  # 4: parking——slot
    [220, 220, 0],  # 5: arrow
    [220, 20, 60],  # 6: guide_line
    [70, 130, 180],  # 7: crosswalk_line
    [0, 0, 110],  # 8: no_parking_sign_line
    [0, 80, 100],  # 9: stop_line
    [190, 153, 153],  # 10: speed_bump
    [224, 35, 232],  # 11. other
    [70, 70, 70],  # 12. parking_lock_open
    [129, 187, 89],  # 13. parking_lock_closed
    [153, 153, 153],  # 14. traffic_cone
    [230, 123, 34],  # 15. parking_rod
    [34, 237, 242],  # 16. curb
    [102, 102, 156],  # 17. cement_column
    [150, 100, 100],  # 18. immovable_obstacle
    [111, 74, 0],  # 19. movable_obstacle
    [193, 17, 101],  # 20. background
    [200, 200, 128],  # 21. sidewalk
    # [0, 0, 0],
    # [200, 200, 128]
]

woodscape_color_list = [
    [0, 80, 170],
    [128, 64, 128],
    [220, 20, 60],
    [250, 0, 0],
    [0, 0, 142],
    [119, 11, 32],
    [34, 237, 242],
    [152, 251, 152],
    [153, 153, 153],
    [220, 220, 0],
    [190, 153, 153],
    [0, 128, 128],
    [70, 130, 180],
]

super_parking_fisheye_color_list = [
    [128, 64, 128],
    [0, 0, 0],
    [244, 35, 232],
    [70, 130, 180],
    [107, 142, 35],
    [190, 153, 153],
    [152, 251, 152],
    [153, 153, 153],
    [220, 220, 0],
    [0, 0, 142],
    [0, 0, 255],
    [119, 11, 32],
    [220, 20, 60],
    [237, 162, 13],
    [111, 74, 0],
    [70, 70, 70],
    [150, 120, 120],
]

mono_lane_parsing_5cls_color_list = [
    [60, 60, 60],
    [0, 0, 255],
    [0, 255, 0],
    [255, 255, 0],
    [0, 255, 255],
]


def draw_all(
    im_draw,
    pred_result,
    image_name,
    save_path,
    gt=None,
    save_gt=False,
    color_list=super_parking_ipm_color_list,
):
    if isinstance(im_draw, torch.Tensor):
        im_draw = im_draw.cpu().numpy()
    im_draw = cv2.cvtColor(im_draw, cv2.COLOR_RGB2BGR)

    # output for seg
    if isinstance(pred_result, torch.Tensor):
        pred_result = pred_result.cpu().numpy()

    img_h, img_w, _ = im_draw.shape
    pred_h, pred_w = pred_result.shape
    if pred_h != img_h or pred_w != img_w:
        pred_result = cv2.resize(
            pred_result, dsize=(img_w, img_h), interpolation=cv2.INTER_NEAREST
        )

    roi_im = im_draw.copy()
    seg_im = attach_color_to_seg(pred_result, color_list)
    if color_list == mono_lane_parsing_5cls_color_list:
        mask = pred_result > 0
    else:
        mask = pred_result >= 0
    scale = 0.4
    roi_im[mask, :] = scale * roi_im[mask, :] + (1 - scale) * seg_im[mask, :]

    if not os.path.join(save_path):
        os.makedirs(save_path)
    pred_dir = os.path.join(save_path, "preds")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    label_dir = os.path.join(save_path, "labels")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    cv2.imwrite(os.path.join(label_dir, image_name + ".png"), pred_result)
    cv2.imwrite(os.path.join(pred_dir, image_name + ".jpg"), roi_im)

    if gt is not None and save_gt:
        if isinstance(gt, torch.Tensor):
            gt = gt.squeeze().cpu().numpy()
        if pred_h != img_h or pred_w != img_w:
            gt = cv2.resize(
                gt, dsize=(img_w, img_h), interpolation=cv2.INTER_NEAREST
            )
        gt_draw = copy.deepcopy(gt)

        gt_dir = os.path.join(save_path, "gt")
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)

        roi_im = im_draw.copy()
        gt_im = attach_color_to_seg(gt_draw, color_list)
        gt_mask = gt_draw >= 0
        roi_im[gt_mask, :] = (
            scale * roi_im[gt_mask, :] + (1 - scale) * gt_im[gt_mask, :]
        )
        cv2.imwrite(os.path.join(gt_dir, image_name + ".jpg"), roi_im)


def attach_color_to_seg(seg, seg_color=None):
    if seg_color is None:
        seg_im = cv2.applyColorMap(
            (seg * 15).astype(np.uint8), cv2.COLORMAP_JET
        )
    else:
        seg_im = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        min_ind = int(seg.min())
        max_ind = int(seg.max())
        for i in range(min_ind, max_ind + 1):
            if i == 255:
                continue
            color = seg_color[i % len(seg_color)]
            seg_im[seg == i, 0] = color[2]
            seg_im[seg == i, 1] = color[1]
            seg_im[seg == i, 2] = color[0]
    return seg_im
