import copy
import os
from typing import Dict

import cv2
import numpy as np
import torch

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.core.box3d_utils import (
    compute_box_3d,
    corners_to_local_rot_y,
    project_to_image,
)
from hat.utils.package_helper import require_packages


def draw_projected_box3d(
    image,
    qs,
    color=(0, 255, 0),
    thickness=2,
    show_arrow=True,
):
    try:
        qs = qs.astype(np.int32)
    except BaseException:
        return image
    for k in range(0, 4):
        # Ref:
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html  # noqa
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)  # noqa
        cv2.line(
            image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
        )
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(
            image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
        )

        i, j = k, k + 4
        cv2.line(
            image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
        )

    if show_arrow:
        # 4,5,6,7
        p1 = (qs[0, :] + qs[1, :] + qs[2, :] + qs[3, :]) / 4
        p2 = (qs[0, :] + qs[1, :]) / 2
        p3 = p2 + (p2 - p1) * 0.5

        p1 = p1.astype(np.int32)
        p3 = p3.astype(np.int32)
        cv2.line(
            image,
            (p1[0], p1[1]),
            (p3[0], p3[1]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def draw_projected_box2d(
    image, qs, color=(0, 255, 0), thickness=2, show_arrow=True
):
    try:
        qs = qs.astype(np.int32)
    except BaseException:
        return image
    cv2.line(
        image, (qs[0, 0], qs[0, 1]), (qs[1, 0], qs[1, 1]), color, thickness
    )
    cv2.line(
        image, (qs[1, 0], qs[1, 1]), (qs[2, 0], qs[2, 1]), color, thickness
    )
    cv2.line(
        image, (qs[2, 0], qs[2, 1]), (qs[3, 0], qs[3, 1]), color, thickness
    )
    cv2.line(
        image, (qs[3, 0], qs[3, 1]), (qs[0, 0], qs[0, 1]), color, thickness
    )
    if show_arrow:
        p1 = np.mean(qs, axis=0)
        p2 = (qs[0, :] + qs[3, :]) / 2
        p3 = p2 + (p2 - p1) * 0.5

        p1 = p1.astype(np.int32)
        p3 = p3.astype(np.int32)
        cv2.line(
            image,
            (p1[0], p1[1]),
            (p3[0], p3[1]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def compute_2d_points(
    dim, location, yaw, calib, dist_coeff=None, fisheye=False, pitch=0.0
):
    corners3d = compute_box_3d(dim, location, yaw, pitch)
    corners3d[:, 2][corners3d[:, 2] < 0] = 0.1
    corners3d_proj = project_to_image(
        corners3d, calib, dist_coeff=dist_coeff, fisheye=fisheye
    )
    corners3d_proj = corners3d_proj.reshape(-1, 2).astype(np.int32)
    return corners3d_proj


def get_3dboxcorner_in_cam(dim, location, rotation_y):
    """Convert from KITTI label to 3dbox in velo."""
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
        3, 1
    )

    return corners_3d.transpose()


def project_camera_to_velo(loc_c, dim_c, yaw_c, r_cam2vel, T_vel2cam):
    points = get_3dboxcorner_in_cam(dim_c, loc_c, yaw_c)
    points_vel = np.dot(r_cam2vel, (points - T_vel2cam).T)
    points_vel_ = points_vel[:, (0, 3, 2, 1, 4, 7, 6, 5)]
    yaw_vel = corners_to_local_rot_y(points_vel_.T)
    return yaw_vel


def camera2velo(loc, yaw, Tr_vel2cam):
    Tr_vel2cam = np.array(Tr_vel2cam)
    T_vel2cam = Tr_vel2cam[:3, -1]
    r_vel2cam = Tr_vel2cam[:3, :3]
    r_cam2vel = np.linalg.inv(r_vel2cam)
    loc_vel = np.dot(r_cam2vel, loc - T_vel2cam)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    yaw_vel = np.arctan2(
        np.dot(r_cam2vel, R)[0, 2], np.dot(r_cam2vel, R)[0, 0]
    )
    return loc_vel, yaw_vel


def compute_2d_box_V2(
    dim,
    location,
    yaw,
    calib,
    distCoeffs=None,
    fisheye=False,
    pitch=0.0,
    img_w=2048,
    img_h=1024,
    ratio=0.5,
):
    points_cam_input = compute_box_3d(dim, location, yaw, pitch)
    points_cam = copy.deepcopy(points_cam_input)
    VALID_Z = 0.1
    EMinX = -ratio * img_w
    EMinY = -ratio * img_h
    EMaxX = (1 + ratio) * img_w
    EMaxY = (1 + ratio) * img_h
    z_valid_index = np.squeeze(np.argwhere(points_cam[:, 2] > VALID_Z))
    if z_valid_index.size <= 2:
        return None, None, 1
    z_valid_points = points_cam[z_valid_index, :]
    image_pts = project_to_image(z_valid_points, calib, distCoeffs, fisheye)
    valid_point_index = np.squeeze(
        np.argwhere(
            np.logical_and(
                np.logical_and(
                    image_pts[:, 0] > EMinX, image_pts[:, 0] < EMaxX
                ),
                np.logical_and(
                    image_pts[:, 1] > EMinY, image_pts[:, 1] < EMaxY
                ),
            )
        )
    )
    if valid_point_index.size <= 2:
        return None, None, 1
    if valid_point_index.size == 8:
        rate_truncate = 0
    else:
        box_h = np.max(points_cam[:, 1]) - np.min(points_cam[:, 1])
        valid_point_original = z_valid_index[valid_point_index]
        valid_point_cam = points_cam[valid_point_original, :]
        valid_points_indx_set = set(valid_point_original)
        if np.pi * 2 / 5 < np.abs(yaw) < np.pi * 3 / 5:
            delta_index = 2
        else:
            delta_index = 0
        for i in range(points_cam.shape[0]):
            if i not in valid_points_indx_set:
                pp = points_cam[i, :]
                candi_anchors = valid_point_cam[
                    np.abs(pp[1] - valid_point_cam[:, 1]) < box_h * 0.4, :
                ]
                if candi_anchors.shape[0] == 0:
                    print("Error, there should be at least one valid point")
                    return None, None, 1
                elif candi_anchors.shape[0] == 1:
                    pp_nearest = candi_anchors[0, :]
                else:
                    dist = np.sum(
                        np.square(candi_anchors[:, [0, 2]] - pp[[0, 2]]),
                        axis=-1,
                    )
                    pp_nearest = candi_anchors[np.argsort(dist)[-2]]
                ll = pp_nearest[delta_index]
                rr = pp[delta_index]
                edge_p = np.array([pp[0], pp[1], pp_nearest[2]]).reshape(
                    (1, 3)
                )
                while np.abs(ll - rr) > 1e-1:
                    mid = (ll + rr) / 2.0
                    if delta_index == 0:
                        edge_p[0, 0] = mid
                        edge_p[0, 2] = (pp[2] - pp_nearest[2]) * (
                            mid - pp[0]
                        ) / (pp[0] - pp_nearest[0]) + pp[2]
                    else:
                        edge_p[0, 0] = (pp[0] - pp_nearest[0]) * (
                            mid - pp[2]
                        ) / (pp[2] - pp_nearest[2]) + pp[0]
                        edge_p[0, 2] = mid
                    if edge_p[0, 2] < VALID_Z:
                        rr = mid
                        continue
                    edge_p_image = project_to_image(edge_p, calib, distCoeffs)
                    if len(edge_p_image.shape) == 1:
                        edge_p_image = np.expand_dims(edge_p_image, 0)
                    if (
                        EMinX < edge_p_image[0, 0] < EMaxX
                        or EMinY < edge_p_image[0, 1] < EMaxY
                    ):
                        ll = mid
                    else:
                        rr = mid
                points_cam[i, :] = edge_p[0, :]

        image_pts = project_to_image(points_cam, calib, distCoeffs, fisheye)
        Fc = np.mean(points_cam[[0, 1, 4, 5], :], axis=0)
        Bc = np.mean(points_cam[[2, 3, 6, 7], :], axis=0)
        Fc_ori = np.mean(points_cam_input[[0, 1, 4, 5], :], axis=0)
        Bc_ori = np.mean(points_cam_input[[2, 3, 6, 7], :], axis=0)
        l_ori = np.sqrt(np.sum(np.square(Bc_ori - Fc_ori)))
        l_truncated = np.sqrt(np.sum(np.square(Bc - Fc)))
        rate_truncate = (l_ori - l_truncated) / l_ori

    corners3d_proj = image_pts.reshape(-1, 2).astype(np.int32)
    bbox2d = np.concatenate(
        [np.min(corners3d_proj, axis=0), np.max(corners3d_proj, axis=0)]
    )
    return corners3d_proj, bbox2d, rate_truncate


def compute_2d_box(dim, location, yaw, calib, dist_coeff=None, fisheye=False):
    corners3d = compute_box_3d(dim, location, yaw)
    corners3d_proj = project_to_image(
        corners3d, calib, dist_coeff=dist_coeff, fisheye=fisheye
    )
    corners3d_proj = corners3d_proj.reshape(-1, 2).astype(np.int32)
    bbox2d = np.concatenate(
        [np.min(corners3d_proj, axis=0), np.max(corners3d_proj, axis=0)]
    )
    return bbox2d


class Real3DVisualize(object):
    def __init__(
        self,
        save_path,
        score_threshold,
        anno_show,
        draw_lidar,
        show_ignore_mask,
        pred_show=True,
        save_raw_img=False,
        nms=None,
    ):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.score_threshold = score_threshold
        self.anno_show = anno_show
        self.pred_show = pred_show
        self.draw_lidar = draw_lidar
        self.save_raw_img = save_raw_img
        self.show_ignore_mask = show_ignore_mask
        self.nms = nms

        self.color_map = [(0, 0, 255), (255, 0, 0), (226, 43, 138), (0, 0, 0)]

    def save_imgs(self, output: Dict, batch: Dict):
        output_np = {}
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                output_np[k] = v.cpu().numpy()
            else:
                output_np[k] = v

        for ind in range(len(output_np["image_id"])):  # batch size
            annos = batch["annotations"][ind]
            preds = [
                output_np["dim"][ind],
                output_np["category_id"][ind],
                output_np["score"][ind],
                output_np["bbox"][ind],
                output_np["location"][ind],
                output_np["dep"][ind],
                output_np["alpha"][ind],
                output_np["rotation_y"][ind],
                output_np["nms_keep"][ind]
                if "nms_keep" in output_np
                else None,
            ]

            calibration = batch["calibration"][ind].cpu().numpy()
            dist_coeffs = batch["dist_coeffs"][ind].cpu().numpy()
            Tr_vel2cam = batch["Tr_vel2cam"][ind].cpu().numpy()
            valid = batch["valid"][ind]
            raw_img = batch["raw_img"][ind].cpu().numpy()
            if self.save_raw_img:
                raw_image_name = (
                    os.path.splitext(batch["image_name"][ind])[0] + "_raw.jpg"
                )
                cv2.imwrite(
                    os.path.join(self.save_path, raw_image_name), raw_img
                )

            if not self.pred_show and not self.anno_show:
                continue

            if self.anno_show:
                raw_img = self.draw_camera_anno_bbox(
                    raw_img, annos, calibration, dist_coeffs, Tr_vel2cam, valid
                )

            final_img = self.draw_camera_pred_bbox(
                raw_img,
                preds,
                calibration,
                dist_coeffs,
                Tr_vel2cam,
            )

            image_name = (
                os.path.splitext(batch["image_name"][ind])[0] + "_eval.jpg"
            )
            # plot the area of input roi
            M = batch["image_transform"]["M"][ind]
            resize_ratio = M[0, 0]
            offset_x = M[0, 2]
            offset_y = M[1, 2]
            input_w = batch["image_transform"]["input_size"][0]
            input_h = batch["image_transform"]["input_size"][1]
            x1 = abs(offset_x) / resize_ratio
            y1 = abs(offset_y) / resize_ratio
            x2 = x1 + input_w / resize_ratio
            y2 = y1 + input_h / resize_ratio

            cv2.rectangle(
                final_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=(0, 0, 0),
                thickness=2,
            )
            cv2.putText(
                final_img,
                "input_roi:(%.1f,%.1f,%.1f,%.1f)" % (x1, y1, x2, y2),
                (int(x1), int(y2 + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
            )
            cv2.imwrite(os.path.join(self.save_path, image_name), final_img)

    @require_packages("pycocotools")
    def draw_camera_anno_bbox(
        self, img, annos, calib, distCoeffs, Tr_vel2cam, valid
    ):
        for anno in annos:
            if self.show_ignore_mask:
                ignore_mask = coco_mask.decode(anno["ignore_mask"]).astype(
                    np.uint8
                )
                mask = (ignore_mask > 0).astype(np.uint8)
                mask2 = np.where(mask > 0)
                img[mask2] = np.clip(
                    img[mask2] + np.array([0, 0, 200], dtype=np.int32), 0, 255
                ).astype(np.uint8)

            if not valid:
                continue

            dim = anno["dim"]
            location = anno["location"]
            yaw = anno["rotation_y"]
            points_cam = compute_box_3d(dim, location, yaw)
            points_img = project_to_image(points_cam, calib, distCoeffs)
            if points_img is not None:
                points_img = points_img.astype(np.int32)
                draw_projected_box3d(img, points_img, color=(0, 255, 0))

        return img

    def draw_camera_pred_bbox(
        self, img, pred, calib, distCoeffs, Tr_vel2cam=None
    ):
        for ind in range(len(pred[0])):
            dim = pred[0][ind]
            location = pred[4][ind]
            score = pred[2][ind]
            if score < self.score_threshold:
                continue
            yaw = pred[7][ind]
            points_cam = compute_box_3d(dim, location, yaw)
            points_img = project_to_image(points_cam, calib, distCoeffs)
            category_id = pred[1][ind]
            if points_img is not None:
                points_img = points_img.astype(np.int32)
                draw_projected_box3d(
                    img, points_img, color=self.color_map[int(category_id)]
                )

            bbox_2d = list(map(lambda x: int(round(x)), pred[3][ind]))
            cv2.rectangle(
                img,
                (bbox_2d[0], bbox_2d[1]),
                (bbox_2d[2], bbox_2d[3]),
                color=(255, 255, 255),
                thickness=2,
            )

        return img
