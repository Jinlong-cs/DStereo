import logging
import os
import pickle
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, Sequence

import cv2
import numpy as np

from hat.core.box3d_utils import (
    camera2image_pinhole,
    load_calib,
    project_velo_to_camera,
)
from hat.core.utils_3d import get_3dbox_dense_points
from hat.core.virtual_camera import CameraBase
from hat.core.virtual_camera.utils import ImagePointsInterpolation
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info
from hat.visualize.utils import get_flexiable_thickness

logger = logging.getLogger(__name__)

CLS2COLOR_MAP = {
    0: (255, 80, 80),
    1: (0, 204, 0),
    2: (0, 0, 204),
    3: (255, 128, 0),
    4: (255, 0, 225),
    5: (0, 255, 255),
    6: (102, 51, 0),
    7: (47, 255, 173),
    8: (203, 192, 255),
}


@OBJECT_REGISTRY.register
class Bev3DVisualize(object):
    """BEV3D Visualize tool.

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
        inference_mode: visualize and save pred pkl
            in inference mode. default=False.
        inference_pkl_dir: path to save the inference bev3d results.
        bev_ratio: the bev img size ratio when concat image.
        camera_layouts: the customized layout of cameras, e.g.
            [
                [
                    "camera_front_left",
                    "camera_front",
                    "camera_front_right"
                ],
                [
                    "fisheye_rear",
                    "tmp_img",
                    "fisheye_right"
                ],
            ].
            there are 2 rows and 3 columns, corresponding camera views,
            the "tmp_img" denotes placeholder img with all zeros.
        is_bev_horizon: indicates whether the BEV image occupies the bottom
            row or the rightmost column.
        anno_name: anno name.
        pts_per_line: the number of points per line in the ori image.
        **kwargs: other arguments.
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
        inference_mode: bool = False,
        inference_pkl_dir: str = None,
        bev_ratio: float = 1.5,
        camera_layouts: Sequence = None,
        is_bev_horizon: bool = False,
        anno_name: str = "annos_bev_3d",
        pts_per_line: int = 100,
        **kwargs,
    ):

        self.save_path = save_path
        self.camera_view_names = camera_view_names
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.score_threshold = score_threshold
        self.anno_show = anno_show
        self.project_bbox_to_cameras = project_bbox_to_cameras
        self.inference_mode = inference_mode
        self.bev_ratio = bev_ratio
        self.anno_name = anno_name
        self.camera2index_dict = {}
        for idx, cam in enumerate(self.camera_view_names):
            self.camera2index_dict[cam] = idx

        if camera_layouts is None:
            self.camera_layouts = [
                ["camera_front_left", "camera_front", "camera_front_right"],
                ["fisheye_left", "camera_front_30fov", "fisheye_front"],
                ["fisheye_rear", "placeholder", "fisheye_right"],
                ["camera_rear_left", "camera_rear", "camera_rear_right"],
            ]
        else:
            self.camera_layouts = camera_layouts

        self.is_bev_horizon = is_bev_horizon

        if self.project_bbox_to_cameras:
            self.project_calib = {}
            self.cameras_inst = OrderedDict()
            self.pts_per_line = pts_per_line
            self.sync_info = {}
        if self.inference_mode:
            self.inference_pkl_dir = inference_pkl_dir
            self.inference_reset()

        self.center_loc = [
            int(vcs_range[3] / (vcs_range[3] - vcs_range[1]) * bev_size[1]),
            int(vcs_range[2] / (vcs_range[2] - vcs_range[0]) * bev_size[0]),
        ]
        vcs_x_y_range = (
            vcs_range[2] - vcs_range[0],
            vcs_range[3] - vcs_range[1],
        )
        vcs_max_range_idx = np.argmax(vcs_x_y_range)
        self.world_width = int(vcs_x_y_range[vcs_max_range_idx])
        self.bev_pixel_meter = (
            bev_size[vcs_max_range_idx] / self.world_width
        )  # 1 meter -> x pixels

    def inference_reset(self):
        self.save_res = defaultdict(list)
        rank, word_size = get_dist_info()
        self.pkl_save_path = os.path.join(
            self.inference_pkl_dir,
            f"inference_rank_{rank}.pkl",
        )
        pkl_save_root = os.path.dirname(self.pkl_save_path)
        if not os.path.exists(pkl_save_root):
            os.makedirs(pkl_save_root, exist_ok=True)

    def save_inference_result(self, output: Dict, batch: Dict):
        """Save bev3d inference results.

        Args:
            output: output from model.
            batch: items contains input's data and GT annos.
        """
        if os.path.exists(self.pkl_save_path):
            with open(self.pkl_save_path, "rb") as f:
                exist_pred = pickle.load(f)
            self.save_res.update(exist_pred)

        batch_size, num_objs = output[list(output.keys())[0]].shape[:2]

        if not isinstance(batch["timestamp"], np.ndarray):
            batch_timestamps = np.array(batch["timestamp"].cpu())
        else:
            batch_timestamps = batch["timestamp"]
        batch_timestamps = [
            str(int(_bs_time * 1000)) for _bs_time in batch_timestamps
        ]

        location = np.concatenate(
            (output["bev3d_ct"], np.expand_dims(output["bev3d_loc_z"], -1)),
            axis=-1,
        )
        for i in range(batch_size):
            front_img_timestamp = batch_timestamps[i]
            pack_dir = batch["pack_dir"][i]
            # the default timestamp in auto3dv is camera_front
            key = os.path.join(pack_dir, front_img_timestamp)
            for j in range(num_objs):
                # filter the padded objs in data transform
                if output["bev3d_score"][i][j] > 0.0:
                    self.save_res[key].append(
                        # all the key_names are used to adap to adas_eval
                        {
                            "dimensions": output["bev3d_dim"][i][j].tolist(),
                            "class_id": output["bev3d_cls_id"][i][j],
                            "score": output["bev3d_score"][i][j],
                            "yaw": output["bev3d_rot"][i][j],
                            "location": location[i][j].tolist(),
                            "timestamp": str(front_img_timestamp),
                        }
                    )
            if len(self.save_res[key]) < 1:
                self.save_res[key] = []

        with open(self.pkl_save_path, "wb") as f:
            pickle.dump(self.save_res, f)

    def get_format_camera_imgs(
        self,
        origin_imgs,
        bev_boxes_img,
        gt_bboxes_valid,
        pred_bboxes=None,
        attribute_file=None,
        gt_bboxes_ignore=None,
        cameras_inst=None,
    ):
        """
        Get the imgs with the projected boxes and format them.

        Args:
            origin_imgs: List, origin img stored in list according to the
                camera_view_names.
            bev_boxes_img: numpy, bev img
            gt_bboxes_valid: Dict, the boxes from gt, it should contain
                `bev3d_ct`, `bev3d_dim`, `bev3d_loc_z`, `bev3d_rot`,
                `bev3d_score` and `bev3d_cls_id`.
            pred_bboxes: Dict, the boxes from pred, it should contain
                `bev3d_ct`, `bev3d_dim`, `bev3d_loc_z`, `bev3d_rot`,
                `bev3d_score` and `bev3d_cls_id`. If pred_bboxes is None,
                pred_boxes will not be projected to camera imgs.
            attribute_file: str, the key used to get calib params.
            gt_bboxes_ignore: the boxes from ignored gt, it should contain
                `bev3d_ct`, `bev3d_dim`, `bev3d_loc_z`, `bev3d_rot`,
                `bev3d_score` and `bev3d_cls_id`.
            cameras_inst: Dict, the camera instance used to project boxes.

        """

        camera_boxes_imgs = []
        for cam in self.camera_view_names:
            index = self.camera2index_dict[cam]
            if (
                self.project_bbox_to_cameras
                and attribute_file in self.project_calib
                and cam
                in self.project_calib[attribute_file]["cam_per_view_shape"]
                and self.project_calib[attribute_file]["cam_per_view_shape"][
                    cam
                ]
                != []
            ):
                ori_size = self.project_calib[attribute_file][
                    "cam_per_view_shape"
                ][cam]
                if origin_imgs[index].shape[:2] != ori_size:
                    camera_boxes_imgs.append(
                        cv2.resize(
                            cv2.cvtColor(
                                origin_imgs[index], cv2.COLOR_RGB2BGR
                            ),
                            (ori_size[1], ori_size[0]),
                        )
                    )
                else:
                    camera_boxes_imgs.append(origin_imgs[index])
            else:
                camera_boxes_imgs.append(origin_imgs[index])

        if self.project_bbox_to_cameras:
            project_calib = (
                self.project_calib[attribute_file]
                if attribute_file in self.project_calib
                else None
            )
            if cameras_inst is None:
                cameras_inst = (
                    self.cameras_inst[attribute_file]
                    if attribute_file in self.cameras_inst
                    else None
                )
            if self.anno_show:
                camera_boxes_imgs = self.draw_camera_boxes(
                    bev3d_ct=gt_bboxes_valid["bev3d_ct"],
                    bev3d_dim=gt_bboxes_valid["bev3d_dim"],
                    bev3d_loc_z=gt_bboxes_valid["bev3d_loc_z"],
                    bev3d_rot=gt_bboxes_valid["bev3d_rot"],
                    bev3d_score=gt_bboxes_valid["bev3d_score"],
                    bev3d_cls_id=gt_bboxes_valid["bev3d_cls_id"],
                    project_calib=project_calib,
                    cameras=self.camera_view_names,
                    camera2index=self.camera2index_dict,
                    color_imgs=camera_boxes_imgs,
                    score_threshold=self.score_threshold,
                    color=(0, 255, 0),
                    cameras_inst=cameras_inst,
                    pts_per_line=self.pts_per_line,
                )

                # draw ignore by white
                if gt_bboxes_ignore is not None:
                    camera_boxes_imgs = self.draw_camera_boxes(
                        bev3d_ct=gt_bboxes_ignore["bev3d_ct"],
                        bev3d_dim=gt_bboxes_ignore["bev3d_dim"],
                        bev3d_loc_z=gt_bboxes_ignore["bev3d_loc_z"],
                        bev3d_rot=gt_bboxes_ignore["bev3d_rot"],
                        bev3d_score=gt_bboxes_ignore["bev3d_score"],
                        bev3d_cls_id=gt_bboxes_ignore["bev3d_cls_id"],
                        project_calib=project_calib,
                        cameras=self.camera_view_names,
                        camera2index=self.camera2index_dict,
                        color_imgs=camera_boxes_imgs,
                        score_threshold=self.score_threshold,
                        color=(255, 255, 255),
                        cameras_inst=cameras_inst,
                        pts_per_line=self.pts_per_line,
                    )

            if pred_bboxes is not None:
                camera_boxes_imgs = self.draw_camera_boxes(
                    bev3d_ct=pred_bboxes["bev3d_ct"],
                    bev3d_dim=pred_bboxes["bev3d_dim"],
                    bev3d_loc_z=pred_bboxes["bev3d_loc_z"],
                    bev3d_rot=pred_bboxes["bev3d_rot"],
                    bev3d_score=pred_bboxes["bev3d_score"],
                    bev3d_cls_id=pred_bboxes["bev3d_cls_id"],
                    project_calib=project_calib,
                    cameras=self.camera_view_names,
                    camera2index=self.camera2index_dict,
                    color_imgs=camera_boxes_imgs,
                    score_threshold=self.score_threshold,
                    color=(0, 0, 255),
                    cameras_inst=cameras_inst,
                    pts_per_line=self.pts_per_line,
                )

        # camera views
        color_imgs_dict = {
            view_name: cv2.resize(img, (2048 // 2, 1280 // 2))
            for view_name, img in zip(
                self.camera_view_names, camera_boxes_imgs
            )
        }
        color_imgs_dict["tmp_img"] = np.zeros_like(
            color_imgs_dict[list(color_imgs_dict.keys())[0]]
        )

        def combine_cams(cam_names):
            have_img = False
            output_imgs = []
            for cam_name in cam_names:
                if cam_name not in color_imgs_dict:
                    cam_name = "tmp_img"
                else:
                    have_img = True
                output_imgs.append(color_imgs_dict[cam_name])
            if have_img:
                return np.hstack(output_imgs)
            else:
                return None

        all_imgs = []
        max_columns = max([len(v) for v in self.camera_layouts])
        for row_cams in self.camera_layouts:
            real_row_cams = row_cams + ["placeholder"] * (
                max_columns - len(row_cams)
            )
            row_img = combine_cams(real_row_cams)
            if row_img is not None:
                all_imgs.append(row_img)

        img_all = np.vstack(all_imgs)

        if self.is_bev_horizon:
            bev_ratio = img_all.shape[1] / bev_boxes_img.shape[1]
            pad_w = img_all.shape[1]
            pad_h = int(bev_boxes_img.shape[0] * bev_ratio)
        else:
            bev_ratio = img_all.shape[0] / bev_boxes_img.shape[0]
            pad_w = int(bev_boxes_img.shape[1] * bev_ratio)
            pad_h = img_all.shape[0]

        target_bev_size = (pad_w, pad_h)

        bev_image_pad = np.zeros(
            (
                pad_h,
                pad_w,
                img_all.shape[2],
            ),
            dtype=np.uint8,
        )
        bev_boxes_img = cv2.resize(bev_boxes_img, target_bev_size)
        bev_image_pad[
            (bev_image_pad.shape[0] - bev_boxes_img.shape[0])
            // 2 : (bev_image_pad.shape[0] + bev_boxes_img.shape[0])
            // 2,
            (bev_image_pad.shape[1] - bev_boxes_img.shape[1])
            // 2 : (bev_image_pad.shape[1] + bev_boxes_img.shape[1])
            // 2,
        ] = bev_boxes_img
        if self.is_bev_horizon:
            img_all = np.vstack([img_all, bev_image_pad])
        else:
            img_all = np.hstack([img_all, bev_image_pad])

        return img_all

    def update_calib_dict(self, attribute_file):
        if attribute_file not in self.project_calib:
            (
                lidar2cam_calibs,
                lidar2chassis,
                cam_per_view_shape,
                cameras_inst,
            ) = load_calib(attribute_file, self.camera_view_names)
            self.project_calib[attribute_file] = {
                "lidar2cam_calibs": lidar2cam_calibs,
                "lidar2chassis": lidar2chassis,
                "cam_per_view_shape": cam_per_view_shape,
            }
            self.cameras_inst[attribute_file] = cameras_inst

    def save_imgs(self, output: Dict, batch: Dict):
        """Save bev 3d vis results.

        Args:
            output: output from model.
            batch: items contains input's data and GT annos.
        """
        if not isinstance(batch["timestamp"], np.ndarray):
            batch_timestamps = np.array(batch["timestamp"].cpu())
        else:
            batch_timestamps = batch["timestamp"]
        batch_timestamps = [
            str(int(_bs_time * 1000)) for _bs_time in batch_timestamps
        ]
        output_np = {}
        for k, v in output.items():
            if not isinstance(batch["timestamp"], np.ndarray):
                output_np[k] = v.cpu().numpy()
            else:
                output_np[k] = v

        if self.anno_show:
            batch_np = {}
            annos_bev_3d = batch[self.anno_name]
            for k, v in annos_bev_3d.items():
                if not isinstance(batch["timestamp"], np.ndarray):
                    batch_np[k] = v.cpu().numpy()
                else:
                    batch_np[k] = v

        for bs, _ in enumerate(batch_timestamps):
            # if project the bev3d results to camera image,
            # should read the calibration.
            # cameras_inst inplace geometry transform.
            if self.project_bbox_to_cameras:
                # get the lidar calib and 6V imgs
                attribute_file = os.path.join(
                    batch["meta_info"]["calib_path"][bs], "calibration.json"
                )
                assert os.path.exists(
                    attribute_file
                ), "Please check the attribute file!"
                self.update_calib_dict(attribute_file)

            # initial the bev map
            bev_img = init_bev(
                world_width=self.world_width,
                init_bev_size=self.bev_size,
                center_loc=self.center_loc,
                bev_pixel_meter=self.bev_pixel_meter,
            )
            bev_size = bev_img.shape[:2]

            # Draw the GT boxes.
            ignore_exist = False
            gt_bboxes_valid_dict = None
            gt_bboxes_ignore_dict = None
            if self.anno_show:
                anno_cls_id = batch_np["vcs_cls_"][bs]
                anno_cls_valid = anno_cls_id != -99

                anno_ignore = batch_np["vcs_ignore_"][bs]
                if anno_ignore.sum() > 0:
                    ignore_exist = True

                anno_valid = np.logical_and(
                    anno_cls_valid, np.logical_not(anno_ignore)
                )
                anno_ignore = np.logical_and(anno_cls_valid, anno_ignore)

                anno_cls_id_valid = anno_cls_id[anno_valid][:, None]
                anno_score_valid = np.ones_like(anno_cls_id_valid)
                gt_bboxes_valid = np.concatenate(
                    [
                        batch_np["vcs_loc_"][bs][:, :2][anno_valid],
                        batch_np["vcs_loc_"][bs][:, 2][anno_valid][:, None],
                        batch_np["vcs_dim_"][bs][anno_valid],
                        batch_np["vcs_rot_z_"][bs][anno_valid][:, None],
                    ],
                    axis=1,
                )
                gt_bboxes_valid_dict = {
                    "bev3d_ct": gt_bboxes_valid[..., :2],
                    "bev3d_dim": gt_bboxes_valid[..., 3:6],
                    "bev3d_loc_z": gt_bboxes_valid[..., 2],
                    "bev3d_rot": gt_bboxes_valid[..., -1],
                    "bev3d_score": anno_score_valid[..., 0],
                    "bev3d_cls_id": None,
                }

                bev_img = self.draw_bev_boxes(
                    bev_img=bev_img,
                    pred_bboxes=gt_bboxes_valid,
                    score=anno_score_valid,
                    bev_size=bev_size,
                    bev_range=self.vcs_range,
                    score_threshold=self.score_threshold,
                    thickness=2,
                    color=(0, 255, 0),  # green
                )
                # draw ignore by white color
                if ignore_exist:
                    anno_cls_id_ignore = anno_cls_id[anno_ignore][:, None]
                    anno_score_ignore = np.ones_like(anno_cls_id_ignore)
                    gt_bboxes_ignore = np.concatenate(
                        [
                            batch_np["vcs_loc_"][bs][:, :2][anno_ignore],
                            batch_np["vcs_loc_"][bs][:, 2][anno_ignore][
                                :, None
                            ],
                            batch_np["vcs_dim_"][bs][anno_ignore],
                            batch_np["vcs_rot_z_"][bs][anno_ignore][:, None],
                        ],
                        axis=1,
                    )
                    gt_bboxes_ignore_dict = {
                        "bev3d_ct": gt_bboxes_ignore[..., :2],
                        "bev3d_dim": gt_bboxes_ignore[..., 3:6],
                        "bev3d_loc_z": gt_bboxes_ignore[..., 2],
                        "bev3d_rot": gt_bboxes_ignore[..., -1],
                        "bev3d_score": anno_score_ignore[..., 0],
                        "bev3d_cls_id": None,
                    }

                    bev_img = self.draw_bev_boxes(
                        bev_img=bev_img,
                        pred_bboxes=gt_bboxes_ignore,
                        score=anno_score_ignore,
                        bev_size=bev_size,
                        bev_range=self.vcs_range,
                        score_threshold=self.score_threshold,
                        thickness=2,
                        color=(255, 255, 255),
                    )
            # Draw the pred boxes
            pred_bboxes = np.concatenate(
                [
                    output_np["bev3d_ct"][bs],
                    output_np["bev3d_loc_z"][bs][:, None],
                    output_np["bev3d_dim"][bs],
                    output_np["bev3d_rot"][bs][:, None],
                ],
                axis=1,
            )
            score = output_np["bev3d_score"][bs]
            pred_bboxes_dict = {
                "bev3d_ct": pred_bboxes[..., :2],
                "bev3d_dim": pred_bboxes[..., 3:6],
                "bev3d_loc_z": pred_bboxes[..., 2],
                "bev3d_rot": pred_bboxes[..., -1],
                "bev3d_score": score,
                "bev3d_cls_id": None,  # NOTE
            }

            bev_boxes_img = self.draw_bev_boxes(
                bev_img=bev_img,
                pred_bboxes=pred_bboxes,
                score=score,
                bev_size=bev_size,
                bev_range=self.vcs_range,
                score_threshold=self.score_threshold,
                thickness=2,
                color=(0, 0, 255),  # red
            )
            origin_imgs = [
                np.asarray(img[bs]) for img in batch["origin_imgs"][0]
            ]
            img_all = self.get_format_camera_imgs(
                origin_imgs,
                bev_boxes_img,
                gt_bboxes_valid_dict,
                pred_bboxes_dict,
                attribute_file=attribute_file
                if (self.project_bbox_to_cameras)
                else None,
                gt_bboxes_ignore=gt_bboxes_ignore_dict,
            )

            savefile = os.path.join(
                self.save_path, batch_timestamps[bs] + ".jpg"
            )
            if not self.inference_mode:
                cv2.imwrite(savefile, img_all)
            else:
                return img_all

    @staticmethod
    def draw_bev_boxes(
        bev_img,
        pred_bboxes,
        score,
        bev_size,
        bev_range,
        score_threshold,
        thickness,
        color,
        class_id=None,
    ):
        """Draw the boxes on bev image.

        Args:
            bev_img (np.ndarray): bev image used for draw boxes.
            pred_bboxes: the predict boxes,(N, 7): N means: num_of_objs,
                7 means [x,y,z,h,w,l,yaw].
            class_id (np.ndarray): the predict class_id.
            score (np.ndarray): the predict score.
            bev_size (Sequence): bev image map size. Common to (512, 512).
            bev_range (Sequence): vcs visible range. Common to
                (-30.0, -51.2, 72.4, 51.2).
            score_threshold (float, optional): score_threshold,
                Defaults to 0.2.
            thickness (int): thickness of lines.
            color (tuple): color of boxes.
        """
        vcs_corner = get_3dboxcorner_in_vcs_numpy(pred_bboxes)
        bev_vis_image = draw_color_bbox(
            vcs_corner,
            bev_img,
            score,
            bev_size,
            bev_range,
            color=color,
            cls_id=class_id,
            scor_thr=score_threshold,
            thickness=thickness,
        )
        bev_img = cv2.resize(bev_img, bev_size)
        return bev_vis_image

    @staticmethod
    def draw_camera_boxes(
        bev3d_dim,
        bev3d_ct,
        bev3d_loc_z,
        bev3d_rot,
        bev3d_score,
        project_calib,
        cameras,
        color,
        bev3d_cls_id=None,
        color_imgs=None,
        camera2index=None,
        score_threshold: float = 0.2,
        use_lidar: bool = True,
        cameras_inst: OrderedDict = None,
        pts_per_line: int = 100,
    ):
        """Draw box on camera images."""
        if cameras_inst is None:
            if use_lidar:
                lidar2cam_calibs, lidar2vcs = (
                    project_calib["lidar2cam_calibs"],
                    project_calib["lidar2chassis"],
                )
            else:
                local_calibs, local2vcs = (
                    project_calib["local_calibs"],
                    project_calib["local2chassis"],
                )

        # (1) Getting the attr based on score_thr
        score_mask = bev3d_score > score_threshold
        vcs_loc_xy = bev3d_ct[score_mask]
        box_dimension = bev3d_dim[score_mask]  # h w l
        vcs_loc_z = bev3d_loc_z[score_mask]
        cls_id = None if bev3d_cls_id is None else bev3d_cls_id[score_mask]
        vcs_loc_xyz = np.concatenate(
            (vcs_loc_xy, vcs_loc_z[:, np.newaxis]), axis=1
        )
        # (2) Change the attr to support vis setting
        rot_z = bev3d_rot[score_mask]

        img_list = []
        if isinstance(cameras_inst, Dict) and len(vcs_loc_xyz) > 0:
            # hwl->lwh
            box_dimension = box_dimension[:, (2, 1, 0)]
            for view in cameras:
                camera_view: CameraBase = cameras_inst[view]

                _vcs_loc_xyz = deepcopy(vcs_loc_xyz)
                _vcs_loc_xyz[..., 2] -= box_dimension[..., 2] / 2

                index = camera2index[view]
                image = color_imgs[index]
                thickness = get_flexiable_thickness(image.shape)
                for i in range(_vcs_loc_xyz.shape[0]):
                    vcs_bbox_corners3d = get_3dbox_dense_points(
                        loc=_vcs_loc_xyz[i],
                        dim=box_dimension[i],
                        heading_angle=rot_z[i],
                        coord_system="vcs",
                        front_lines=True,
                        pts_per_line=pts_per_line,
                    )

                    cam_bbox_corners3d = camera_view.project_vcs2cam(
                        vcs_bbox_corners3d.reshape((-1, 3))
                    )
                    pts_in_fov, _ = camera_view.filter_points_by_fov(
                        cam_bbox_corners3d
                    )
                    if len(pts_in_fov) > 0:
                        pixel_in_fov = camera_view.project_cam2pixel(
                            pts_in_fov
                        )
                        cls_color = (
                            color
                            if cls_id is None
                            else CLS2COLOR_MAP.get(int(cls_id[i]), color)
                        )
                        image = ImagePointsInterpolation.draw_points(
                            image,
                            pixel_in_fov,
                            color=cls_color,
                            thickness=thickness,
                        )

                img_list.append(image)

        elif len(vcs_loc_xyz) > 0:
            # hwl->wlh
            box_dimension = box_dimension[:, (1, 2, 0)]
            rot_z = -(rot_z + np.pi / 2)

            if use_lidar:
                # vcs->lidar
                homo_ones = np.ones([len(vcs_loc_xyz), 1])
                vcs_loc_xyz = np.concatenate((vcs_loc_xyz, homo_ones), axis=1)
                lidar_loc = [
                    np.linalg.inv(lidar2vcs) @ _vcs_loc
                    for _vcs_loc in vcs_loc_xyz
                ]
                lidar_loc = np.array(lidar_loc)[:, :3]
                # vis results
                ddd_corners = center_to_corner_box3d(
                    lidar_loc,
                    box_dimension,
                    rot_z,
                    origin=(0.5, 0.5, 0.5),
                )
            for cam in cameras:
                if not use_lidar:
                    # vis results n,8,3
                    ddd_corners = center_to_corner_box3d(
                        vcs_loc_xyz,
                        box_dimension,
                        rot_z,
                        origin=(0.5, 0.5, 0.5),
                        axis=2,
                    )
                    if cam in local2vcs:
                        corners = np.ones((len(ddd_corners), 8, 4))
                        corners[:, :, :3] = ddd_corners
                        # vcs->local
                        vcs2local = np.linalg.inv(local2vcs[cam])
                        ddd_corners = np.ones((len(ddd_corners), 8, 4))
                        for i in range(len(ddd_corners)):
                            for j in range(len(ddd_corners[0])):
                                ddd_corners[i, j] = vcs2local @ corners[i][j]
                index = camera2index[cam]
                img = color_imgs[index]
                calibs = lidar2cam_calibs if use_lidar else local_calibs
                if cam in calibs:
                    if "fisheye" in cam:
                        image = draw_fisheye_detection_box_on_image(
                            img,
                            calibs[cam],
                            ddd_corners,
                            color=color,
                            distort_label=True,
                            draw_lidar=True,
                            cls_id=cls_id,
                        )
                    else:
                        # img = undistort(img, lidar2cam_calibs[cam])
                        image = draw_detection_box_on_image(
                            img,
                            calibs[cam],
                            ddd_corners,
                            color=color,
                            distort_label=True,
                            cls_id=cls_id,
                        )
                else:
                    image = img
                img_list.append(image)
        else:
            for cam in cameras:
                index = camera2index[cam]
                img = color_imgs[index]
                img_list.append(img)
        return img_list

    @staticmethod
    def draw_lidar_boxes(
        bev_map,
        bev3d_dim,
        bev3d_ct,
        bev3d_loc_z,
        bev3d_rot,
        bev3d_score,
        project_calib,
        coors_range,
        color,
        score_threshold: float = 0.2,
    ):
        """Draw box on lidar point-cloud image."""
        _, lidar2chassis = (
            project_calib["lidar2cam_calibs"],
            project_calib["lidar2chassis"],
        )

        # (1) Getting the attr based on score_thr
        score_mask = bev3d_score > score_threshold
        vcs_loc_xy = bev3d_ct[score_mask]
        box_dimension = bev3d_dim[score_mask]  # h w l
        vcs_loc_z = bev3d_loc_z[score_mask]
        vcs_loc_xy = np.concatenate(
            (vcs_loc_xy, vcs_loc_z[:, np.newaxis]), axis=1
        )
        # (2) Change the attr to support vis setting
        box_dimension = box_dimension[:, (1, 2, 0)]
        rot_z = bev3d_rot[score_mask]
        rot_z = -(rot_z + np.pi / 2)
        if len(vcs_loc_xy) > 0:
            # vcs->lidar
            homo_ones = np.ones([len(vcs_loc_xy), 1])
            vcs_loc_xy = np.concatenate((vcs_loc_xy, homo_ones), axis=1)
            lidar_loc = [
                np.linalg.inv(lidar2chassis) @ _vcs_loc
                for _vcs_loc in vcs_loc_xy
            ]
            lidar_loc = np.array(lidar_loc)[:, :3]
            # vis results
            pred_bbox = np.concatenate(
                [lidar_loc, box_dimension, rot_z[:, None]], axis=1
            )

            bev_3d_vis_img = draw_kitti_pred_in_bev(
                pred_bbox, bev_map, coors_range, None, color=color
            )
        else:
            bev_3d_vis_img = bev_map
        return bev_3d_vis_img


def init_bev(world_width, init_bev_size, center_loc, bev_pixel_meter):
    """Intialize a bev map.

    Args:
        world_width (int, optional): The real world size. Defaults to 102m.
        init_bev_size (tuple, optional): Init bev map size. Defaults
            to (1024,1024).
        center_loc (tuple, optional): The pixel location of ego car.
            Defaults to (512, 362 * 2).

    """
    bev = np.zeros((*init_bev_size, 3), dtype=np.uint8)
    # draw ego car
    self_lt = (
        int((center_loc[0] - 1.8 / 2 * bev_pixel_meter)),
        int((center_loc[1] - 4 / 2 * bev_pixel_meter)),
    )  # noqa
    self_rb = (
        int((center_loc[0] + 1.8 / 2 * bev_pixel_meter)),
        int((center_loc[1] + 4 / 2 * bev_pixel_meter)),
    )  # noqa
    front_arrow = (center_loc[0], int(center_loc[1] - 4 * bev_pixel_meter))
    cv2.rectangle(bev, self_lt, self_rb, (255, 255, 255), 1)
    cv2.line(bev, center_loc, front_arrow, (255, 255, 255), 2)
    # draw circle
    for meter in range(0, world_width, 10):
        color = (200, 200, 200)
        bev = cv2.circle(
            bev,
            center_loc,
            int(bev_pixel_meter * meter),
            color,
            1,
        )  # noqa
    # draw stright line
    cv2.line(
        bev,
        (init_bev_size[1] // 2, 0),
        (init_bev_size[1] // 2, init_bev_size[0]),
        color,
        1,
    )
    cv2.line(
        bev, (0, center_loc[1]), (init_bev_size[1], center_loc[1]), color, 1
    )
    return bev


def center_to_corner_box3d(
    centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2
):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim]
    )
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def draw_detection_box_on_image(
    _image,
    calib,
    points,
    color,
    distort_label=False,
    cls_id=None,
):
    image = _image.copy()
    P2, distCoeffs_default = calib["P2"], calib["disCoeffs"]
    Tr_vel2cam = calib["Tr_vel2cam"]
    if not distort_label:
        distCoeffs = 0.0 * np.copy(distCoeffs_default)
    else:
        distCoeffs = np.copy(distCoeffs_default)
    within_fov = []
    for idx, box in enumerate(points):  # noqa
        points_aug = deepcopy(box)
        points_aug = np.concatenate(
            (points_aug, np.ones((points_aug.shape[0], 1))), axis=1
        )
        points_aug[:, 3] = 1
        points_cam = project_velo_to_camera(points_aug, Tr_vel2cam)

        within_fov = points_cam[:, 2] > 0
        points_cam = points_cam[within_fov, :3].astype(np.float32)

        if points_cam.shape[0] == 0:
            continue

        image_pts = camera2image_pinhole(
            points_cam[:, :3], P2, distCoeffs, image.shape[1], image.shape[0]
        )

        if image_pts is not None:
            image_pts = image_pts.astype(np.int32)
            cls_color = (
                color
                if cls_id is None
                else CLS2COLOR_MAP.get(int(cls_id[idx]), color)
            )
            _draw_box_3d(image, image_pts, cls_color, thickness=3)
    return image


def draw_fisheye_detection_box_on_image(
    _image,
    calib,
    points,
    color,
    distort_label=True,
    draw_lidar=False,
    cls_id=None,
):
    """Draw bev3d results on fisheye camera images.

    Args:
        _image (np.ndarray): fisheye image.
        calib (dict): calibration parameters.
        points (np.ndarray): lidar box points.
        distort_label (bool, optional): whether to distort images.
            Defaults to True.
        color (tuple, optional): box colors. Defaults to (0, 255, 0).
        draw_lidar (bool, optional): whether draw lidar box. Defaults to False.

    """
    image = _image.copy()
    image_height, image_width, _ = image.shape
    rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
    tvec = np.zeros(shape=(3, 1), dtype=np.float32)

    P2, distCoeffs_default, Tr_vel2cam = (
        calib["P2"],
        calib["disCoeffs"],
        calib["Tr_vel2cam"],
    )
    if not distort_label:
        distCoeffs = 0.0 * np.copy(distCoeffs_default)
    else:
        distCoeffs = np.copy(distCoeffs_default)

    within_fov = []
    for i, box in enumerate(points):
        points_aug = deepcopy(box)
        points_aug = np.concatenate(
            (points_aug, np.ones((points_aug.shape[0], 1))), axis=1
        )
        points_aug[:, 3] = 1
        points_cam = project_velo_to_camera(points_aug, Tr_vel2cam)

        within_fov = points_cam[:, 2] > 0
        points_cam = points_cam[within_fov, :3].astype(np.float32)

        if points_cam.shape[0] == 0:
            continue
        points_cam = np.expand_dims(points_cam[:, :3], 0)
        fx, fy = P2[0, 0], P2[1, 1]
        u, v = P2[0, 2], P2[1, 2]
        k_ = np.mat([[fx, 0.0, u], [0.0, fy, v], [0.0, 0.0, 1.0]])
        d_ = np.mat(distCoeffs[:4].T)
        image_pts = cv2.fisheye.projectPoints(
            points_cam, np.array(rvec), tvec, k_, d_
        )[0]
        image_pts = np.array(image_pts).squeeze(axis=0)

        if image_pts.shape[0] == 0:
            continue
        within_img0 = np.logical_and(
            image_pts[:, 0] < image_width, image_pts[:, 0] > 1
        )
        within_img1 = np.logical_and(
            image_pts[:, 1] < image_height, image_pts[:, 1] > 1
        )
        old_image_pts = image_pts
        within_img = np.logical_and(within_img1, within_img0)
        within_fov[within_fov] = within_img
        image_pts = image_pts[within_img]
        if image_pts.shape[0] == 0:
            continue
        if draw_lidar:
            cls_color = (
                color
                if cls_id is None
                else CLS2COLOR_MAP.get(int(cls_id[i]), color)
            )
            if image_pts.shape[0] == 8:
                # idx1, idx2 means the bottom and top idx of a
                # 3D box's 8 points, and the (idx1, idx2) can determine
                # a line of boxes, total 12 lines in a box.
                for idx1, idx2 in [
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                    (0, 3),
                    (1, 2),
                    (4, 7),
                    (5, 6),
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                ]:
                    x1 = int(image_pts[idx1, 0])
                    x2 = int(image_pts[idx2, 0])
                    y1 = int(image_pts[idx1, 1])
                    y2 = int(image_pts[idx2, 1])
                    cv2.line(image, (x1, y1), (x2, y2), cls_color, thickness=3)
            else:
                x1 = int(max(min(old_image_pts[:, 0]), 0))
                y1 = int(max(min(old_image_pts[:, 1]), 0))
                x2 = int(min(max(old_image_pts[:, 0]), image_width - 1))
                y2 = int(min(max(old_image_pts[:, 1]), image_height - 1))
                cv2.rectangle(
                    image, (x1, y1), (x2, y2), cls_color, thickness=3
                )
    return image


def _draw_box_3d(image, corners, c=(0, 0, 255), show_arrow=False, thickness=1):
    face_idx = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            try:
                cv2.line(
                    image,
                    (corners[f[j], 0], corners[f[j], 1]),
                    (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]),
                    c,
                    thickness,
                    lineType=cv2.LINE_AA,
                )  # noqa
            except:  # noqa
                continue  # noqa
        if not show_arrow:
            if ind_f == 0:
                try:
                    cv2.line(
                        image,
                        (corners[f[0], 0], corners[f[0], 1]),
                        (corners[f[2], 0], corners[f[2], 1]),
                        c,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )  # noqa
                except:  # noqa
                    continue  # noqa
                try:
                    cv2.line(
                        image,
                        (corners[f[1], 0], corners[f[1], 1]),
                        (corners[f[3], 0], corners[f[3], 1]),
                        c,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )  # noqa
                except:  # noqa
                    continue

        # # show an arrow to indicate 3D orientation of the object
        # if show_arrow:
        #     # 4,5,6,7
        #     p1 = (corners[0, :] + corners[1, :] +
        #           corners[2, :] + corners[3, :]) / 4
        #     p2 = (corners[0, :] + corners[1, :]) / 2
        #     p3 = p2 + (p2 - p1) * 0.5

        #     p1 = p1.astype(np.int32)
        #     p2 = p2.astype(np.int32)
        #     p3 = p3.astype(np.int32)

        #     cv2.line(image, (p1[0], p1[1]), (p3[0], p3[1]),
        #              c, thickness, lineType=cv2.LINE_AA)
    return image


def get_calib(calibs, cam):
    """Get each view's calibration from all calibs].

    Args:
        calibs ([dict]): [calibs contains all cameras]
        cam ([str]): [cam view]

    Returns:
        [dict]: [dict contains cam's calibration]
    """
    res_calib = defaultdict()
    # get intrinics and disCoeffs
    res_calib["P2"] = np.array(calibs[cam]["K"], dtype=np.float32)
    res_calib["disCoeffs"] = np.array(calibs[cam]["d"], dtype=np.float32)
    # get Transformation (lidar->cam)
    Tr_key = "lidar_top_2_" + cam
    res_calib["Tr_vel2cam"] = np.array(calibs[Tr_key], dtype=np.float32)
    res_calib["rotMat"] = res_calib["Tr_vel2cam"][:3, :3]
    res_calib["transMat"] = res_calib["Tr_vel2cam"][:3, 3]
    return res_calib


def get_3dboxcorner_in_vcs_numpy(box3d):
    """Convert from bev3d label to 3dbox corner in vcs coordinate.

    Args:
        box3d (np.ndarray): input boxes, shape=[N,7]
    Returns:
        np.ndarray: the 8 corners on vcs coordinate.
    """

    box3d = box3d[:, [0, 1, 2, 5, 4, 3, 6]]
    # get normalized corners and multiply with dimension
    # (box height, width, length)
    corner = np.array(
        [
            [-1, -1, 1, 1, -1, -1, 1, 1],  # x
            [1, -1, -1, 1, 1, -1, -1, 1],  # y
            [-1, -1, -1, -1, 1, 1, 1, 1],
        ]
    )  # z
    dim = (
        np.expand_dims(box3d[:, 3:6], axis=1) / 2
    )  # modify dim->1/2 dim, shape:[N,1,3]
    corner_nd = (
        dim * corner.transpose()
    )  # generate the corner based on (h,w,l), shape:[N,8,3]
    corner_nd = np.transpose(corner_nd, (0, 2, 1))  # N,3,8

    # Rotation matrix, shape=[N, 3, 3]
    yaw = box3d[:, -1]
    rot_sin = np.sin(yaw)
    rot_cos = np.cos(yaw)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)

    rot_mat = np.stack(
        [
            [rot_cos, -rot_sin, zeros],
            [rot_sin, rot_cos, zeros],
            [zeros, zeros, ones],
        ]
    )  # [3, 3, N]
    rot_mat = np.transpose(rot_mat, (2, 0, 1))  # [N, 3, 3]

    bbox_corner = np.matmul(rot_mat, corner_nd)  # [N,3,8]
    bbox_corner = np.transpose(bbox_corner, (0, 2, 1))  # [N,8,3]

    # NOTE: each box: rot @ corner -> rot corner
    # bbox_corner = np.einsum("aij,jka->aik", corner_nd, rot_mat)

    center = np.stack((box3d[:, 0], box3d[:, 1], box3d[:, 2]), 1)
    center = np.expand_dims(center, 1)

    # shift the normalized box to center
    bbox_corner += center
    return bbox_corner


def draw_color_bbox(
    bboxes,
    bev_image,
    score,
    bev_size,
    bev_range,
    color,
    cls_id=None,
    scor_thr=0.2,
    thickness=1,
):
    """Draw colored image based on vcs corner.

    Args:
        bboxes (np.ndarray): the box info, shape=[N, 8, 3], (8, 3) means:
            each box's 8 corner coordinate (x,y,z) in vcs coord
        bev_image (np.ndarray): image to draw image.
        cls_id (np.ndarray): objs class id.
        score (np.ndarray): objs score.
        bev_size (tuple, optional): bev image map size. Defaults to (512, 512).
        bev_range (tuple, optional): vcs visible range. Defaults to
            (-30.0, -51.2, 72.4, 51.2).
        scor_thr (float, optional): score_threshold. Defaults to 0.2.
        thickness (int, optional): the thickness og line.
        color (tuple, optional): color of box line.

    Returns:
        np.ndarray: drawed image.
    """
    for idx, (bbox, scor) in enumerate(zip(bboxes, score)):
        cls_color = (
            color
            if cls_id is None
            else CLS2COLOR_MAP.get(int(cls_id[idx]), color)
        )
        if scor > scor_thr:
            p1, p2, p3, p4 = bbox[:4, :2]
            (
                p1_index,
                p2_index,
                p3_index,
                p4_index,
            ) = convert_vcs_coord_to_bev_index_numpy(
                np.stack((p1, p2, p3, p4), axis=0),
                bev_size=bev_size,
                bev_range=bev_range,
            )
            cv2.line(
                bev_image,
                tuple(p1_index[::-1]),
                tuple(p4_index[::-1]),
                cls_color,
                thickness,
            )
            cv2.line(
                bev_image,
                tuple(p4_index[::-1]),
                tuple(p3_index[::-1]),
                cls_color,
                thickness,
            )
            # cv2.line(bev_image, tuple(p3_index[
            #     ::-1]), tuple(p4_index[::-1]), cls_color, 1)

            cv2.line(
                bev_image,
                tuple(p3_index[::-1]),
                tuple(p2_index[::-1]),
                cls_color,
                thickness,
            )
    return bev_image


def convert_vcs_coord_to_bev_index_numpy(
    vcs_coord,
    bev_size=(512, 512),  # bev coord y, x
    bev_range=(-30.0, -51.2, 72.4, 51.2),
):
    """Convert vcs coordinate into bev image coordinate.

    Args:
        vcs_coord (np.ndarray, shape=[N,2]): the vcs coordinate, (x,y)
        bev_size (tuple, optional): bev image map size. Defaults to (512, 512).
        bev_range (tuple, optional): vcs visible range. Defaults to
            (-30.0, -51.2, 72.4, 51.2).

    Returns:
        np.ndarray: (N, 2)  (y, x) index of (512, 512) bev image map.
    """

    voxel_size = (
        abs(bev_range[2] - bev_range[0]) / bev_size[0],
        abs(bev_range[3] - bev_range[1]) / bev_size[1],
    )

    bev_index_y = np.floor(
        (bev_range[2] - vcs_coord[:, 0]) / voxel_size[0]
    ).astype(int)
    bev_index_x = np.floor(
        (bev_range[3] - vcs_coord[:, 1]) / voxel_size[1]
    ).astype(int)

    bev_index = np.stack([bev_index_y, bev_index_x], axis=1)
    return bev_index


def cv2_draw_lines(img, lines, colors, thickness, line_type=cv2.LINE_8):
    lines = lines.astype(np.int32)
    for line, color in zip(lines, colors):
        color = list(int(c) for c in color)  # noqa [C400]
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
    return img


def rotation_2d(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def corner_to_standup_nd(boxes_corner):
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.

    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def draw_box_in_bev(
    img, coors_range, boxes, color, thickness=1, labels=None, label_color=None
):
    coors_range = np.array(coors_range)
    bev_corners = center_to_corner_box2d(
        boxes[:, [0, 1]], boxes[:, [3, 4]], boxes[:, 6]
    )
    bev_corners -= coors_range[:2]
    bev_corners *= np.array(img.shape[:2])[::-1] / (
        coors_range[3:5] - coors_range[:2]
    )
    standup = corner_to_standup_nd(bev_corners)
    text_center = standup[:, 2:]
    text_center[:, 1] -= (standup[:, 3] - standup[:, 1]) / 2

    bev_lines = np.concatenate(
        [bev_corners[:, [0, 2, 3]], bev_corners[:, [1, 3, 0]]], axis=2
    )
    bev_lines = bev_lines.reshape(-1, 4)
    colors = np.tile(np.array(color).reshape(1, 3), [bev_lines.shape[0], 1])
    colors = colors.astype(np.int32)
    img = cv2_draw_lines(img, bev_lines, colors, thickness)
    if labels is not None:
        if label_color is None:
            label_color = colors
        else:
            label_color = np.tile(
                np.array(label_color).reshape(1, 3), [bev_lines.shape[0], 1]
            )
            label_color = label_color.astype(np.int32)

        img = cv2_draw_text(  # noqa [F821]
            img, text_center, labels, label_color, thickness * 2
        )
    return img


def draw_kitti_pred_in_bev(
    bbox_pred,
    bev_map,
    point_cloud_range,
    label_pred,
    color=[0, 0, 255],  # noqa [B006]
):
    """Draw kitti predictions in bev.

    The reason why we have draw_kitti_gt_in_bev and
    draw_kitti_pred_in_bev is that we want to add features to
    draw preds and gt for all 3 different categories.

    args
    ----
    bbox_gt: numpy.array of shape (num_objs, 8)
        Ground truth of bbox.
    label_gt: list of labels in natural language
        The labels of the bbox_gt in same order.
    bev_map: numpy.array
        The image which has been drew in previous steps
    point_cloud_range: list
        Show the range of the detection range.

    returns
    -------
    bev_map: numpy.array
    """
    # TODO (runzhou.ge) Use the label_gt to draw different colors for
    # different classes
    if bbox_pred.shape[0] > 0:
        return draw_box_in_bev(bev_map, point_cloud_range, bbox_pred, color, 2)
    else:
        return bev_map


def read_lidar(path, dim=6):
    return (
        np.fromfile(path, dtype=np.double).reshape(-1, dim).astype(np.float32)
    )


# @numba.jit(nopython=True)
def _points_to_bevmap_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    coor_to_voxelidx,
    # coors_2d,
    bev_map,
    height_lowers,
    # density_norm_num=16,
    with_reflectivity=False,
    max_voxels=40000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    height_slice_size = voxel_size[-1]
    coor = np.zeros(shape=(3,), dtype=np.int32)  # DHW
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # coors_2d[voxelidx] = coor[1:]
        bev_map[-1, coor[1], coor[2]] += 1
        height_norm = bev_map[coor[0], coor[1], coor[2]]
        incomimg_height_norm = (
            points[i, 2] - height_lowers[coor[0]]
        ) / height_slice_size
        if incomimg_height_norm > height_norm:
            bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
            if with_reflectivity:
                bev_map[-2, coor[1], coor[2]] = points[i, 3]
    # return voxel_num


def points_to_bev(
    points,
    voxel_size,
    coors_range,
    with_reflectivity=False,
    density_norm_num=16,
    max_voxels=40000,
):
    """Convert kitti points(N, 4) to a bev map. return [C, H, W] map.

    this function based on algorithm in points_to_voxel.
    takes 5ms in a reduced pointcloud with voxel_size=[0.1, 0.1, 0.8]

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3] contain reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        with_reflectivity: bool. if True, will add a intensity map to bev map.
    Returns:
        bev_map: [num_height_maps + 1(2), H, W] float tensor.
            `WARNING`: bev_map[-1] is num_points map, NOT density map,
            because calculate density map need more time in cpu rather
            than gpu. if with_reflectivity is True, bev_map[-2] is
            intensity map.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # DHW format
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    bev_map_shape = list(voxelmap_shape)
    bev_map_shape[0] += 1
    height_lowers = np.linspace(
        coors_range[2], coors_range[5], voxelmap_shape[0], endpoint=False
    )
    if with_reflectivity:
        bev_map_shape[0] += 1
    bev_map = np.zeros(shape=bev_map_shape, dtype=points.dtype)
    _points_to_bevmap_reverse_kernel(
        points,
        voxel_size,
        coors_range,
        coor_to_voxelidx,
        bev_map,
        height_lowers,
        with_reflectivity,
        max_voxels,
    )
    # print(voxel_num)
    return bev_map


def point_to_vis_bev(
    points, voxel_size=None, coors_range=None, max_voxels=80000
):
    if voxel_size is None:
        voxel_size = [0.1, 0.1, 0.1]
    if coors_range is None:
        coors_range = [-50, -50, -3, 50, 50, 1]
    voxel_size[2] = coors_range[5] - coors_range[2]
    bev_map = points_to_bev(
        points, voxel_size, coors_range, max_voxels=max_voxels
    )
    height_map = (bev_map[0] * 255).astype(np.uint8)
    return cv2.cvtColor(height_map, cv2.COLOR_GRAY2RGB)
