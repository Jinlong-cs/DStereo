import logging

import numpy as np

from hat.core.box3d_utils import get_3dbox_corners
from hat.core.utils_3d import compute_box_parametric
from hat.core.virtual_camera import PinholeCamera
from hat.utils.apply_func import convert_numpy as to_numpy

logger = logging.getLogger(__name__)

VALID_Z = 1.29


def points_to_bbox(points):
    left, top = np.min(points, axis=0)
    right, bottom = np.max(points, axis=0)
    return np.asarray([left, top, right, bottom], dtype=np.float32)


def clip_bbox_and_filter(bbox2d, img_wh: tuple, border_ignore: int = 60):
    if bbox2d is None:
        return None
    if bbox2d[0] >= img_wh[0] or bbox2d[2] < 0:
        return None
    if bbox2d[1] >= img_wh[1] or bbox2d[3] < 0:
        return None
    bbox2d[0] = max(0, bbox2d[0])
    bbox2d[1] = max(0, bbox2d[1])
    bbox2d[2] = min(bbox2d[2], img_wh[0] - 1)
    bbox2d[3] = min(bbox2d[3], img_wh[1] - 1)
    if bbox2d[2] < border_ignore or img_wh[0] - bbox2d[0] < border_ignore:
        return None
    return bbox2d


def reformat_bev3d_to_det3d_aidi_eval(
    batch_data,
    batch_outputs,
    obj_key,
    dump_obj_key=None,
    idx2cam=None,
    camera_view_names=None,
    class_key_id_map=None,
    score_thresh=0.1,
    center_type="bottom_center",
):
    out_key = dump_obj_key if dump_obj_key is not None else obj_key

    batch_timestamps = batch_data["img_metas"]["timestamp"].cpu().numpy()
    batch_timestamps = [
        str(int(_bs_time * 1e3)) for _bs_time in batch_timestamps
    ]
    batch_size = len(batch_timestamps)

    cam2idx = {v: k for k, v in idx2cam.items()}

    line_res_v2_list = []
    for bi in range(batch_size):
        pred_mask = batch_outputs[bi]["scores_3d"] >= score_thresh
        pred_bboxes_3d = batch_outputs[bi]["boxes_3d"][pred_mask]
        pred_labels_3d = batch_outputs[bi]["labels_3d"][pred_mask]
        pred_scores_3d = batch_outputs[bi]["scores_3d"][pred_mask]

        num_objs = pred_bboxes_3d.shape[0]
        ts = batch_timestamps[bi]

        for cam_i, cam in enumerate(camera_view_names):
            image_key = f"{ts}__{cam}__{cam2idx[cam]}"

            ori_image_size = (
                batch_data["img_metas"]["ori_image_size"][bi, cam_i]
                .cpu()
                .numpy()
            )
            ori_camera_matrix = (
                batch_data["img_metas"]["ori_camera_matrix"][bi, cam_i]
                .cpu()
                .numpy()
            )
            ori_vcs2cam = (
                batch_data["img_metas"]["ori_vcs2cam"][bi, cam_i].cpu().numpy()
            )
            ori_distcoeffs = (
                batch_data["img_metas"]["ori_distcoeffs"][bi, cam_i]
                .cpu()
                .numpy()
            )
            if np.allclose(ori_camera_matrix, np.identity(3)):
                logger.info(f"ts {ts}, {cam} not in calibs")
                continue

            virtual_camera = PinholeCamera.init_cam_param_by_matrix(
                ori_image_size,
                ori_camera_matrix,
                ori_vcs2cam,
                ori_distcoeffs,
                False,
            )

            img_h, img_w = ori_image_size[1], ori_image_size[0]

            line_res_v2 = {out_key: [], "image_key": image_key + ".jpg"}

            for j in range(num_objs):
                dim = to_numpy(pred_bboxes_3d[j][3:6]).tolist()

                class_id = int(to_numpy(pred_labels_3d[j], dtype="int16"))
                score = float(to_numpy(pred_scores_3d[j]))
                yaw = float(to_numpy(pred_bboxes_3d[j][6]))
                loc = to_numpy(pred_bboxes_3d[j][:3]).tolist()
                if center_type == "bottom_center":
                    continue
                elif center_type == "cube_center":
                    # convert cube_center to bottom_center
                    loc[2] -= dim[2] * 0.5
                else:
                    raise NotImplementedError(
                        f"not support center_type is {center_type}"
                    )

                if class_id != class_key_id_map[obj_key]:
                    continue

                points_3d = get_3dbox_corners(
                    loc, dim, yaw, coord_system="vcs"
                )
                _, valid_index = virtual_camera.filter_points_by_fov(
                    points_3d, coord_system="vcs"
                )
                if sum(valid_index) < 2:
                    continue

                points_cam = virtual_camera.project_vcs2cam(points_3d)
                z_valid_index = points_cam[:, 2] > VALID_Z
                if sum(z_valid_index) <= 2:
                    continue

                points_img = virtual_camera.project_vcs2pixel(points_3d)

                bbox_2d = points_to_bbox(points_img)
                bbox_2d = clip_bbox_and_filter(
                    bbox_2d, (img_w, img_h), border_ignore=30
                )
                if bbox_2d is None:
                    continue

                l, w, h = dim
                dim_in_cam = (h, w, l)
                loc_in_cam, rot_y = compute_box_parametric(
                    points_cam, "cv_camera"
                )

                line_res_v2[out_key].append(
                    {
                        "location": loc_in_cam.astype(np.float32).tolist(),
                        "depth": loc_in_cam[2],
                        "score": score,
                        "rotation_y": rot_y,
                        "dimensions": dim_in_cam,
                        "bbox_2d": bbox_2d.astype(np.float32).tolist(),
                    }
                )

            line_res_v2_list.append(line_res_v2)

    return line_res_v2_list
