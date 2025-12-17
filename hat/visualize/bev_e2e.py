import logging
import math
import os
import random
from typing import Sequence

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from hat.visualize.bev_3d import Bev3DVisualize, init_bev

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "cyclist": (255, 0, 0),  # blue
    "vehicle": (0, 255, 0),  # green
    "pedestrian": (0, 0, 255),  # red
    "vrumerge": (255, 255, 0),  # yellow
}


@OBJECT_REGISTRY.register
class ANCBevE2EVisualize(Bev3DVisualize):
    """BEVE2E Visualize tool.

    Args:
        save_path: absolute path to save the visualize results.
        camera_view_names: camera view names.
        bev_size: bev image map size.
        vcs_range: vcs visible range.
        score_threshold: score_threshold to draw bboxes.
        anno_show: whether to show the GT boxes.
            Defaults to True.
        project_bbox_to_cameras: whether project the bev3d to images.
            Defaults to True.
        bev_ratio: the bev img size ratio when concat image.
        vis_tracking: whether draw the tracking results. default False
        vis_trajectory: whether draw the predicted traj. default False
        max_draw_traj_num: Only used when vis_trajectory is True.
            default 5.
        vis_velocity: whether draw the predicted velo. default False.
        is_bev_horizon: indicates whether the BEV image occupies the bottom
            row or the rightmost column.
        num_classes: number of predicted classes, used to distinguish the
            color index in colormap between the gt and the prediction

    """

    def __init__(
        self,
        save_path: str,
        camera_view_names: Sequence,
        bev_size: Sequence,
        vcs_range: Sequence,
        score_threshold: float,
        anno_show: bool,
        project_bbox_to_cameras: bool,
        bev_ratio: float = 1.0,
        vis_tracking: bool = False,
        vis_trajectory: bool = False,
        max_draw_traj_num: int = 5,
        vis_velocity: bool = False,
        is_bev_horizon: bool = True,
        num_classes: int = 3,
        **kwargs,
    ):
        super().__init__(
            save_path=save_path,
            camera_view_names=camera_view_names,
            bev_size=bev_size,
            vcs_range=vcs_range,
            score_threshold=score_threshold,
            anno_show=anno_show,
            project_bbox_to_cameras=project_bbox_to_cameras,
            bev_ratio=bev_ratio,
            is_bev_horizon=is_bev_horizon,
            camera_layouts=[
                ["camera_front_left", "camera_front", "camera_front_right"],
                ["camera_rear_left", "camera_rear", "camera_rear_right"],
            ],
        )
        self.vis_tracking = vis_tracking
        self.vis_trajectory = vis_trajectory
        self.vis_velocity = vis_velocity
        self.max_draw_traj_num = max_draw_traj_num
        self.num_classes = num_classes

    @staticmethod
    def draw_bev_boxes(
        bev_img,
        instances: dict,
        vcs_range: Sequence,
        bev_size: Sequence,
        vis_velocity: bool = False,
        vis_trajectory: bool = False,
        max_draw_traj_num: int = 5,
    ):
        """Draw e2e results on bev.

        Args:
            bev_img: input bev img.
            instances: object instances to be plot, Keys as below

                .. code-block:: none

                    scores: instance scores, shape: N.
                    boxes: instance bev location and shape, shape: N * 4,
                        in the format of [x, y, l, w].
                    obj_idxes: the index of instance, shape: N.
                    yaws: instance rotaton, shape: N * 2,
                        in the format of [cos(yaw), sin(yaw)].
                    labels: instance lalesl, shape: N.
                    bev_loc_z: the z height of instance, shape: N.
                    appear_time: optional, the number of frame instance exists,
                        shape: N.

            vcs_range: vcs visible range, (bottom, right, top, left)
                in order.
            bev_size: bev image map size, (height, width) in order.
            vis_velocity: whether draw the predicted velo.
            vis_trajectory: whether draw the predicted traj.
            max_draw_traj_num: Only used when vis_trajectory is True.
        """
        vcs_x_y_range = (
            vcs_range[2] - vcs_range[0],
            vcs_range[3] - vcs_range[1],
        )
        vcs_max_range_idx = np.argmax(vcs_x_y_range)
        world_width = int(vcs_x_y_range[vcs_max_range_idx])
        bev_pixel_meter = (
            bev_size[vcs_max_range_idx] / world_width
        )  # 1 meter -> x pixels
        obj_num = len(instances["scores"])
        track_ids = instances["obj_idxes"]  # N
        # vcs 2 bev_img coord
        cxcys = np.stack(
            (
                (vcs_range[3] - instances["boxes"][:, 1]) * bev_pixel_meter,
                (vcs_range[2] - instances["boxes"][:, 0]) * bev_pixel_meter,
            ),
            axis=-1,
        )

        whs = instances["boxes"][:, [3, 2]] * bev_pixel_meter
        yaws = instances["yaws"]  # N x 2
        # generate vcs pred_bbox
        fake_z = np.zeros_like(instances["boxes"][:, 0:1])
        fake_height = np.ones_like(instances["boxes"][:, 0:1])
        vcs_yaw = np.arctan2(yaws[:, 1:], yaws[:, 0:1])
        # vcs_x,vcs_y,vcs_z, h, w ,l, yaw
        bboxes = np.concatenate(
            (
                instances["boxes"][:, :2],
                fake_z,
                fake_height,
                instances["boxes"][:, 2:][..., ::-1],
                vcs_yaw,
            ),
            axis=-1,
        )
        velocities = instances.get("velocities", np.zeros((obj_num, 3)))
        velocities[:, 0] = velocities[:, 0] * bev_pixel_meter
        labels = instances["labels"]
        appear_time = (
            instances["appear_time"]
            if ("appear_time" in instances)
            else [0] * obj_num
        )

        if vis_trajectory:
            if "gt_traj_regs" in instances:
                trajs_points = -instances["gt_traj_regs"][
                    :, np.newaxis
                ]  # [N, 1, traj_len, 2]
                trajs_mask = instances["gt_traj_masks"][:, np.newaxis]
                trajs_probs = np.ones_like(trajs_points[:, :, 0, 0])  # [N, 1]
            else:
                trajs_points = -instances["TrajRegs"]  # [N, K, traj_len, 2]
                trajs_probs = instances["TrajScores"]  # [N, K]
                trajs_mask = np.ones_like(trajs_points)
                trajs_points, trajs_probs = select_top_k_trajs(
                    trajs_points, trajs_probs, max_draw_traj_num
                )
            zeros_p = np.zeros_like(trajs_points[:, :, :1])  # [N, K, 1, 2]
            # transfer the vcs coord to the BEV coord, draw the trajectory
            trajs_points = (
                np.concatenate([zeros_p, trajs_points], 2) * bev_pixel_meter
                + cxcys[:, np.newaxis, np.newaxis, [1, 0]]
            )
            trajs_mask = np.concatenate(
                [np.ones_like(trajs_mask[:, :, :1, :]), trajs_mask], 2
            )
        else:
            trajs_points = cxcys[:, np.newaxis, np.newaxis, [1, 0]]
            trajs_probs = np.zeros_like(trajs_points[:, :, 0, 0])
            trajs_mask = np.zeros_like(trajs_points)
        bev_img_traj = bev_img.copy()
        for (
            cxcy,
            wh,
            yaw,
            bbox,
            track_id,
            velocity,
            cur_traj,
            cur_prob,
            cur_mask,
            _,
            cur_appear_time,
        ) in zip(
            *(
                cxcys,
                whs,
                yaws,
                bboxes,
                track_ids,
                velocities,
                trajs_points,
                trajs_probs,
                trajs_mask,
                labels,
                appear_time,
            )
        ):
            angle = np.arctan2(yaw[1], yaw[0]) / math.pi * 180
            rect = [cxcy, wh, -angle]
            if cur_appear_time == 1:
                bev_img = plot_box(
                    rect,
                    bbox,
                    bev_img,
                    bev_size,
                    vcs_range,
                    color=get_specific_color(track_id),
                    line_thickness=-1,
                    draw_direction=False,
                    # label=str(int(track_id)),
                )  #
                bev_img_traj = plot_box(
                    rect,
                    bbox,
                    bev_img_traj,
                    bev_size,
                    vcs_range,
                    color=get_specific_color(track_id),
                    line_thickness=-1,
                    draw_direction=False,
                    # label=str(int(track_id)),
                )  #
            else:
                bev_img = plot_box(
                    rect,
                    bbox,
                    bev_img,
                    bev_size,
                    vcs_range,
                    color=get_specific_color(track_id),
                    line_thickness=1,
                    draw_direction=True,
                    # label=str(int(track_id)),
                )
                bev_img_traj = plot_box(
                    rect,
                    bbox,
                    bev_img_traj,
                    bev_size,
                    vcs_range,
                    color=get_specific_color(track_id),
                    line_thickness=1,
                    draw_direction=True,
                    # label=str(int(track_id)),
                )

            if vis_trajectory:
                if np.any(np.abs(cur_traj - cur_traj[:, :1]) > 1e-2):
                    bev_img_traj = plot_obj_trajs(
                        bev_img_traj,
                        cur_traj,
                        cur_prob,
                        cur_mask,
                        trajconfThr=0.2,
                    )  # , cur_modal)
            if vis_velocity:
                bev_img = plot_velocity(
                    rect,
                    bev_img,
                    velocity,
                    color=get_specific_color(track_id),
                )
        return bev_img, bev_img_traj

    def save_imgs(
        self, output, annotations, imgs_data, timestamps, sub_dir_name=""
    ):
        """Save bev e2e results.

        Args:
            output: output from model.
            annotations: GT annos
            imgs_data: other input's data contains imgs and homo info
            timestamps: batch timestamps
        """
        save_path = os.path.join(self.save_path, sub_dir_name)
        pack_dir = imgs_data["pack_dir"]
        for frame_idx, timestamp in enumerate(timestamps):
            timestamp = str(timestamp)
            if self.project_bbox_to_cameras:
                # get the lidar calib and camera imgs
                attribute_file = os.path.join(pack_dir, "attribute.json")
                if not os.path.exists(attribute_file):
                    attribute_file = os.path.join(
                        imgs_data["meta_info"]["calib_path"][0],
                        "calibration.json",
                    )
                assert os.path.exists(
                    attribute_file
                ), f"Please check the pack: {pack_dir}'s attribute file"
                self.update_calib_dict(attribute_file)

            # initial the bev map,
            bev_img = init_bev(
                world_width=self.world_width,
                init_bev_size=self.bev_size,
                center_loc=self.center_loc,
                bev_pixel_meter=self.bev_pixel_meter,
            )

            gt_bev_img = bev_img  # 448 x 512 x 3
            pred_bev_img = bev_img.copy()  # 448 x 512 x 3

            gt_img, gt_img_traj = self.draw_bev_boxes(
                gt_bev_img,
                annotations[frame_idx],
                self.vcs_range,
                self.bev_size,
                self.vis_velocity,
                self.vis_trajectory,
                self.max_draw_traj_num,
            )
            pred_img, pred_img_traj = self.draw_bev_boxes(
                pred_bev_img,
                output[frame_idx],
                self.vcs_range,
                self.bev_size,
                self.vis_velocity,
                self.vis_trajectory,
                self.max_draw_traj_num,
            )
            if gt_img_traj is not None:
                bev_boxes_img = np.hstack(
                    (
                        gt_img_traj,
                        pred_img_traj,
                        gt_img,
                        pred_img,
                    )
                )
                task_str = "traj_gt / traj_pred / velo_gt / velo_pred"
            else:
                bev_boxes_img = np.hstack(
                    (
                        gt_img,
                        pred_img,
                    )
                )
                task_str = "gt / pred"

            origin_imgs = [
                np.asarray(img) for img in imgs_data["origin_imgs"][frame_idx]
            ]
            anno = {
                "bev3d_ct": annotations[frame_idx]["boxes"][..., :2],
                "bev3d_dim": np.concatenate(
                    (
                        annotations[frame_idx]["heights"][:, None],
                        annotations[frame_idx]["boxes"][:, [3, 2]],
                    ),
                    axis=-1,
                ),
                "bev3d_loc_z": annotations[frame_idx]["bev_loc_z"],
                "bev3d_rot": np.arctan2(
                    annotations[frame_idx]["yaws"][:, 1],
                    annotations[frame_idx]["yaws"][:, 0],
                ),
                "bev3d_score": annotations[frame_idx]["scores"],
                "bev3d_cls_id": annotations[frame_idx]["labels"],  # NOTE
            }
            pred = {
                "bev3d_ct": output[frame_idx]["boxes"][:, :2],
                "bev3d_dim": np.concatenate(
                    (
                        output[frame_idx]["heights"][:, None],
                        output[frame_idx]["boxes"][:, [3, 2]],
                    ),
                    axis=-1,
                ),
                "bev3d_loc_z": output[frame_idx]["bev_loc_z"],
                "bev3d_rot": np.arctan2(
                    output[frame_idx]["yaws"][:, 1],
                    output[frame_idx]["yaws"][:, 0],
                ),
                "bev3d_score": output[frame_idx]["scores"],
                # TODO: 下次发版(08/07)前，完成HAT可视化适配，pred和gt 给两个colormap
                "bev3d_cls_id": output[frame_idx]["labels"] + self.num_classes,
            }

            img_all = self.get_format_camera_imgs(
                origin_imgs,
                bev_boxes_img,
                anno,
                pred,
                attribute_file=attribute_file
                if (self.project_bbox_to_cameras)
                else None,
            )

            cv2.putText(
                img_all,
                timestamp + f" {task_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                color=(255, 255, 255),
                thickness=2,
            )

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            savefile = os.path.join(save_path, timestamp + ".jpg")
            cv2.imwrite(savefile, img_all)


def get_specific_color(track_id):
    return (
        int(track_id * 20 + 20) % 255,
        int(track_id * 20 + 200) % 255,
        int(track_id * 20 + 100) % 255,
    )


def select_top_k_trajs(trajs_points, trajs_probs, top_k):
    # trajs_points(Array): [N, K, traj_len, 2]
    # trajs_probs(Array): [N, K]
    # top_k(int):
    obj_num, traj_num = trajs_probs.shape[:2]
    if (1 == traj_num) or (obj_num < 1):
        return trajs_points, trajs_probs
    if traj_num <= top_k:
        top_k = traj_num

    out_trajs, out_prob = [], []
    for idx in range(obj_num):  # obtain top-k trajs one obj-by-one obj.
        cur_trajs = trajs_points[idx]
        cur_probs = trajs_probs[idx]
        sort_idxes = np.argsort(cur_probs)[::-1][:top_k]
        cur_trajs = cur_trajs[sort_idxes]
        cur_probs = cur_probs[sort_idxes]
        if (
            np.sum(cur_probs) > 1e-2
        ):  # if is tracking agent, cur_probs should all > 0
            cur_probs = cur_probs / np.sum(cur_probs)  # prob normalize.
        out_trajs.append(cur_trajs[np.newaxis])
        out_prob.append(cur_probs[np.newaxis])
    out_trajs = np.concatenate(out_trajs, 0)
    out_prob = np.concatenate(out_prob, 0)
    # Here, reversing the order helps to draw traces with higher confidence
    # in the outermost layer, avoiding high-confidence tracks being
    # overwritten by low-confidence trajectories
    return out_trajs[:, ::-1], out_prob[:, ::-1]


def plot_box(
    rect,
    bbox,
    img,
    bev_size,
    vcs_range,
    color=None,
    label=None,
    line_thickness=None,
    draw_direction=False,
):
    """Plot one box on the img.

    rect: list, [[cx, cy], [w, h], angle(degree)]
    """
    color = color or [random.randint(0, 255) for _ in range(3)]
    box = cv2.boxPoints(tuple(rect))
    box = np.int64(box)
    if draw_direction:
        # rotate pts: bl->tl->tr->br
        # only draw 3 line, since we need direction.
        for box_idx in range(3):
            cv2.line(
                img,
                tuple(box[box_idx]),
                tuple(box[box_idx + 1]),
                color,
                line_thickness,
            )
    else:
        cv2.drawContours(img, [box], 0, color, line_thickness)
    cx, cy = np.int64(rect[0])
    if label:
        cv2.putText(
            img,
            label,
            (cx + 5, cy + 5),
            0,
            1 / 3,
            [225, 255, 255] if label != "E" else [0, 0, 255],
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return img


def plot_velocity(
    rect,
    img,
    velocities,
    color=None,
    label=None,
    score=None,
    line_thickness=None,
):
    """Plot one box on the img.

    rect: list, [[cx, cy], [w, h], angle(degree)]
    """
    tl = line_thickness or 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    cx, cy = np.int64(rect[0])
    start_point = (cx, cy)
    end_point1 = (cx - velocities[1], cy)
    end_point2 = (cx, cy - velocities[0])
    ex1, ey1 = np.int64(end_point1)
    ex2, ey2 = np.int64(end_point2)
    end_point1 = (ex1, ey1)
    end_point2 = (ex2, ey2)
    cv2.line(img, start_point, end_point1, color, tl)
    cv2.line(img, start_point, end_point2, color, tl)
    return img


def plot_obj_trajs(
    img,
    trajs,
    probs,
    masks,
    modal=None,
    Color1=(0, 255, 255),
    Color2=(36, 44, 81),
    trajconfThr=1e-3,
):
    Color1 = np.array(Color1)
    Color2 = np.array(Color2)
    trajs = trajs.astype(np.int)
    if modal is not None:
        if modal >= 0:
            cv2.putText(
                img,
                str(modal),
                (trajs[0, 0, 1] + 2, trajs[0, 0, 0] + 2),
                0,
                0.3,
                [225, 255, 255],
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    for cur_traj, cur_prob, cur_mask in zip(trajs, probs, masks):
        cur_traj = np.reshape(cur_traj[cur_mask >= 0.5], [-1, 2])
        if cur_prob < trajconfThr:
            continue
        Colortmp = tuple(
            [
                int(xxx * cur_prob + yyy * (1 - cur_prob))
                for xxx, yyy in zip(Color1, Color2)
            ]
        )
        for jjj in range(1, len(cur_traj)):  # render lines.
            cv2.line(
                img,
                (cur_traj[jjj - 1, 1], cur_traj[jjj - 1, 0]),
                (cur_traj[jjj, 1], cur_traj[jjj, 0]),
                Colortmp,
                1,
            )
    return img
