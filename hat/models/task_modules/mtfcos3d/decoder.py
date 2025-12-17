# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hat.core.nms.box3d_nms import box2d3d_multiclass_nms
from hat.core.virtual_camera import (
    CameraBase,
    CylindricalCamera,
    FisheyeCamera,
)
from hat.models.base_modules.postprocess import PostProcessorBase
from hat.models.task_modules.fcos.target import distance2bbox, get_points
from hat.models.utils import multi_class_nms
from hat.registry import OBJECT_REGISTRY

__all__ = ["MTFCOS3DDecoder"]


def limit_period(
    val: Tensor, offset: float = 0.5, period: float = np.pi
) -> Tensor:
    """Limit the value into a period for periodic function.

    Args:
        val: The value to be converted.
        offset: Offset to set the value range.
            Defaults is 0.5.
        period: Period of the value. Defaults is np.pi.

    Returns:
        Value in the range of [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


def apply_nms_2d3d(
    pred: Tensor,
    iou_threshold: float,
    replace: bool = False,
    score2d: bool = False,
) -> Tensor:
    score_key = "score2d" if score2d else "score"
    score = pred[score_key][None, ...].float()
    bbox = pred["bbox"][None, ...].float()
    cate_id = pred["category_id"][None, ...].float()
    batch, n = score.shape
    keep = score.new_zeros((batch, n), dtype=torch.int8)
    for bi in range(batch):
        bi_idx = multi_class_nms(
            bbox[bi], score[bi], cate_id[bi], iou_threshold
        )
        keep[bi] = keep[bi].scatter(0, bi_idx, 1)
        if replace:
            score[bi][keep[bi] == 0] = 0
    pred["nms_keep"] = keep[0, :]
    pred[score_key][list(keep[0, :] == 0)] = 0
    return keep


def format_angle(
    angle: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    angle[angle > np.pi] -= 2 * np.pi
    angle[angle < -np.pi] += 2 * np.pi
    return angle


@OBJECT_REGISTRY.register
class MTFCOS3DDecoder(PostProcessorBase):
    """Decoder for Multi-task-fused-FCOS.

    To decode head outputs into 3d and 2d bboxes.

    Args:
        num_classes: Number of categories excluding the background
            category.
        head_channels: A dict to tell which head channel should be outputed.
        strides: A list contains the strides of fcos_head output.
        rescale: Whether to map the prediction result to the orig img.
            Default is True.
        use_direction_classifier: Whether to use direction classifier.
            Default is True.
        dir_offset: A start value for direction regression. Default is pi / 4.
        nms_kwargs: A dict that sets the nms kwargs.
        depth_type: The of representing depth. Choices are "Cartesian" and
            "Cylindrical".
        is_train_3d_branch: Whether the module is training 3d task.
        use_multibin: Whether to use multibin strategy.
        multibin_centers: The centers of bins.
        multibin_margin: The margin of bins.
        output_parser: A parser module to apply tranform to outputs.
        do_bev3d_nms: Whether to do bev3d nms to get kept bbox mask.

    """

    def __init__(
        self,
        num_classes: int,
        head_channels: dict,
        strides: Tuple[int],
        rescale: bool = True,
        use_direction_classifier: bool = True,
        dir_offset: float = 0.7854,  # pi / 4
        nms_kwargs: Optional[dict] = None,
        depth_type: str = "Cartesian",
        is_train_3d_branch: bool = True,
        use_multibin: bool = False,
        multibin_centers: Optional[Tuple[float, ...]] = None,
        multibin_margin: float = 0.0,
        output_parser: Optional[nn.Module] = None,
        do_bev3d_nms: bool = False,
    ):
        super(MTFCOS3DDecoder, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.strides = strides
        self.rescale = rescale
        self.nms_kwargs = nms_kwargs

        self.is_train_3d_branch = is_train_3d_branch
        self.head_channels = head_channels
        self.dir_offset = dir_offset
        # (x, y, z, x_size, y_size, z_size, yaw)
        self.bbox_code_size = 7
        self.use_direction_classifier = use_direction_classifier
        self.group_reg_dims = []
        # get bbox3d channels
        for name, chanels in head_channels.items():
            if "_3d_group" in name:
                self.group_reg_dims.append(chanels[-1])
        assert self.use_direction_classifier, "only suport dir_cls is True"
        self.use_multibin = use_multibin
        self.multibin_centers = multibin_centers
        if self.use_multibin:
            assert (
                self.multibin_centers is not None
            ), "to use multibin, but multibin_centers is None"
            self.multibin_centers = np.array(multibin_centers)
            self.multibin_centers = torch.from_numpy(self.multibin_centers)
        self.multibin_margin = multibin_margin
        self.output_parser = output_parser

        assert depth_type in [
            "Cartesian",
            "Cylindrical",
        ], f"depth encode type must in [Cartesian, Cylindrical], \
                but got {depth_type}"

        self.depth_type = depth_type
        self.do_bev3d_nms = do_bev3d_nms

    @staticmethod
    def decode_alphax_multibin(
        pred_3d_bbox, pred_dir_cls, multibin_centers, with_max=False
    ):

        if with_max:
            _, pred_dir_cls = pred_dir_cls.max(dim=-1)

        angle_centers = multibin_centers[pred_dir_cls.flatten()].view(
            pred_dir_cls.size()
        )
        device = pred_3d_bbox.device
        angle_centers = angle_centers.to(device)
        pred_3d_bbox_angle_offset = pred_3d_bbox[..., 6]
        alphax = pred_3d_bbox_angle_offset + angle_centers
        alphax = limit_period(alphax, 0.5, 2 * np.pi)
        pred_3d_bbox = torch.cat(
            [pred_3d_bbox[..., :6], alphax[..., None]], dim=-1
        )
        return pred_3d_bbox

    @staticmethod
    def decode_alphax_multibin_margin(
        pred_3d_bbox, pred_dir_cls, multibin_centers, with_max=False
    ):
        if with_max:
            _, pred_dir_cls = pred_dir_cls.max(dim=-1)

        angle_centers = multibin_centers[pred_dir_cls.flatten()].view(
            pred_dir_cls.size()
        )
        device = pred_3d_bbox.device
        angle_centers = angle_centers.to(device)
        pred_3d_bbox_info = pred_3d_bbox[..., 6:]
        pred_3d_bbox_info = pred_3d_bbox_info.reshape(
            (-1, len(multibin_centers))
        )
        pred_3d_bbox_angle_offset = pred_3d_bbox_info.gather(
            -1, pred_dir_cls[..., None]
        )
        alphax = pred_3d_bbox_angle_offset + angle_centers[..., None]
        alphax = limit_period(alphax, 0.5, 2 * np.pi)
        pred_3d_bbox = torch.cat([pred_3d_bbox[..., :6], alphax], dim=-1)
        return pred_3d_bbox

    @staticmethod
    def decode_alphax_2bin_sin_differece(
        bbox: Tensor, dir_cls: Tensor, dir_offset: float
    ):
        """Decode 2bin sin differece learned alphax angle.

        Args:
            bbox: Bounding box predictions in shape
                [N, C] with yaws to be decoded.
            dir_cls: Predicted direction classes.
            dir_offset: Direction offset before dividing all the
                directions into several classes.

        Returns:
            Tensor: Bounding boxes with decoded alphax.
        """
        if bbox.shape[0] > 0:
            dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi)
            bbox[..., 6] = (
                dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype)
            )

        return bbox

    @staticmethod
    def decode_roty(loc: Tensor, alphax: Tensor):
        """Decode global yaw angle.

        Args:
            loc: 3D location predictions in shape [N, 3].
            alphax: alphax predictions in shape [N,]

        Returns:
            Tensor: Bounding boxes with decoded yaws.
        """
        theta = torch.atan2(loc[:, 0], loc[:, 2])
        rot_y = theta + alphax
        # limit rot_y in (-pi, pi)
        rot_y = format_angle(rot_y)
        return rot_y

    def forward(
        self, pred: Dict[str, Sequence[Tensor]], label: Dict[str, Any]
    ):
        if not isinstance(pred, dict):
            raise NotImplementedError

        assert len(pred) == len(self.head_channels)

        dir_cls_preds = []
        cls_scores = []
        bbox_preds = []
        bbox_preds_3d = []
        ctrnesses_2d = []
        ctrnesses_3d = []
        for name, value_list in pred.items():
            # dir_cls need read before cls task
            if "dir" in name:
                dir_cls_preds = value_list
            elif "cls" in name:
                cls_scores = value_list
            elif "ctrness_2d" in name:
                ctrnesses_2d = value_list
            elif "ctrness_3d" in name:
                ctrnesses_3d = value_list
            elif "offset_2d_reg" in name:
                bbox_preds = value_list
            else:
                # convert bbox_reg feats together for every levels
                assert "_group_reg" in name, "unkown head name as pre-defined!"
                if bbox_preds_3d == []:
                    bbox_preds_3d = value_list
                else:
                    # x_offset, y_offset, depth, h, w, l, alpha_offset
                    for i, lvl_preds in enumerate(value_list):
                        bbox_preds_3d[i] = torch.cat(
                            [bbox_preds_3d[i], lvl_preds], dim=1
                        )
        assert (
            len(cls_scores)
            == len(bbox_preds)
            == len(bbox_preds_3d)
            == len(dir_cls_preds)
            == len(ctrnesses_2d)
            == len(ctrnesses_3d)
        )
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        result = {}
        det_results = []
        if self.output_parser is not None:
            parsed_result = []
        # loop every image
        for img_id in range(len(cls_scores[0])):
            # pack same pred attibutes on diff lvl feat into list
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_3d_list = [
                bbox_preds_3d[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier:
                # default
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id]
                    .new_full([2, *cls_scores[i][img_id].shape[1:]], 0)
                    .detach()
                    for i in range(num_levels)
                ]
            ctrness_2d_pred_list = [
                ctrnesses_2d[i][img_id].detach() for i in range(num_levels)
            ]
            ctrness_3d_pred_list = [
                ctrnesses_3d[i][img_id].detach() for i in range(num_levels)
            ]

            det_bboxes = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                bbox_pred_3d_list,
                dir_cls_pred_list,
                ctrness_2d_pred_list,
                ctrness_3d_pred_list,
                mlvl_points,
                label,
                img_id,
                self.rescale,
            )
            if self.output_parser is not None:
                parsed_result.append(self.output_parser(det_bboxes))
            if len(result) != 0:
                for k, v in det_bboxes.items():
                    if k == "pred_bboxes":
                        det_results.append(deepcopy(det_bboxes[k]))
                    result[k] = torch.cat([result[k], v[None, ...]], dim=0)
            else:
                for k, v in det_bboxes.items():
                    result[k] = v[None, ...]
                    if k == "pred_bboxes":
                        det_results.append(deepcopy(det_bboxes[k]))
        if not self.is_train_3d_branch:
            result["pred_bboxes"] = det_results
            result["img_name"] = label.get("img_name", "")
            result["img_id"] = label.get("img_id")
        if self.output_parser is not None:
            return parsed_result
        return result

    def _get_bboxes_single(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        bbox_preds_3d: List[Tensor],
        dir_cls_preds: List[Tensor],
        ctrnesses_2d: List[Tensor],
        ctrnesses_3d: List[Tensor],
        mlvl_points: List[Tensor],
        input_meta: dict,
        img_idx: int,
        rescale: bool = False,
    ) -> Dict[str, Tensor]:
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores: Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds_3d: Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W) in
                format [prj_u,prj_v,dep,w,h,l,alpha].
            dir_cls_preds: Box scores for direction class predictions on a
                single scale level with shape (num_points * 2, H, W).
            ctrnesses_2d: 2d Centerness for a single scale level with shape
                (num_points, H, W).
            ctrnesses_3d: 3d Centerness for a single scale level with shape
                (num_points, H, W).
            mlvl_points: Box reference for a single scale level with shape
                (num_total_points, 2).
            input_meta: Metadata of input image.
            img_idx: image index of input image.
            rescale: If True, return boxes in original image space.
                Default is False.

        Returns:
            Predicted 3D boxes, scores, labels.
        """

        try:
            # current camera instance
            virtual_camera: CameraBase = input_meta["virtual_cam"][img_idx]
            # original camera instance
            source_camera: CameraBase = input_meta["source_cam"][img_idx]
            self.is_train_3d_branch = True
        except KeyError:
            self.is_train_3d_branch = False
            # hard code anyway.
            virtual_camera = CylindricalCamera.init_cam_param_by_matrix(
                image_size=[704, 576],  # W * H
                camera_matrix=np.array(
                    [[220, 0, 352], [0, 220, 288], [0, 0, 1]]
                ),
                is_virtual=True,
            )
            source_camera = FisheyeCamera()

        # transform intrinsic to virtual camera
        camera_matrix = torch.tensor(virtual_camera.camera_matrix)

        assert (
            len(cls_scores)
            == len(bbox_preds)
            == len(bbox_preds_3d)
            == len(mlvl_points)
        )
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_bboxes_3d = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_ctrness_2d = []
        mlvl_ctrness_3d = []

        # loop for every level feature
        for (
            cls_score,
            bbox_pred,
            bbox_pred_3d,
            dir_cls_pred,
            ctrness_2d,
            ctrness_3d,
            points,
        ) in zip(
            cls_scores,
            bbox_preds,
            bbox_preds_3d,
            dir_cls_preds,
            ctrnesses_2d,
            ctrnesses_3d,
            mlvl_points,
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = (
                cls_score.permute(1, 2, 0)
                .reshape(-1, self.cls_out_channels)
                .sigmoid()
            )
            # 2d block
            ctrness_2d: Tensor = (
                ctrness_2d.permute(1, 2, 0).reshape(-1).sigmoid()
            )
            bbox_pred: Tensor = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred = bbox_pred.relu()
            # 3d block
            bbox_pred_3d: Tensor = bbox_pred_3d.permute(1, 2, 0).reshape(
                -1, sum(self.group_reg_dims)
            )
            bbox_pred_3d[:, 3:6] = bbox_pred_3d[:, 3:6].relu()
            if self.use_multibin:
                bbox_pred_3d = bbox_pred_3d[..., : sum(self.group_reg_dims)]
            else:
                bbox_pred_3d = bbox_pred_3d[:, : self.bbox_code_size]

            ctrness_3d: Tensor = (
                ctrness_3d.permute(1, 2, 0).reshape(-1).sigmoid()
            )
            if self.use_multibin:
                dir_cls_pred: Tensor = dir_cls_pred.permute(1, 2, 0).reshape(
                    -1, len(self.multibin_centers)
                )
            else:
                dir_cls_pred: Tensor = dir_cls_pred.permute(1, 2, 0).reshape(
                    -1, 2
                )

            # not choice big value for 2bin
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            #############################################
            nms_pre = self.nms_kwargs.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * ctrness_2d[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                scores = scores[topk_inds, :]
                points = points[topk_inds, :]
                # 2d block
                bbox_pred = bbox_pred[topk_inds, :]
                ctrness_2d = ctrness_2d[topk_inds]
                # 3d block
                bbox_pred_3d = bbox_pred_3d[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                ctrness_3d = ctrness_3d[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
            #############################################

            # not limit temp.
            bboxes = distance2bbox(points, bbox_pred, max_shape=None)

            # change the offset to actual center predictions
            bbox_pred_3d[:, :2] = points - bbox_pred_3d[:, :2]

            # depth decode by enc_type
            if self.depth_type == "Cartesian":
                pass
            elif self.depth_type == "Cylindrical":
                fx = camera_matrix[0, 0]
                cu = camera_matrix[0, 2]
                theta = (bbox_pred_3d[:, 0] - cu) / fx
                bbox_pred_3d[:, 2] *= torch.cos(theta)
            else:
                raise NotImplementedError(
                    "depth_type supported [Cartesian, Cylindrical]"
                )
            # 3D loc decode in Cartesian coord.
            bbox_pred_3d[:, :3] = self.pts2Dto3D(
                bbox_pred_3d[:, :3], virtual_camera
            ).to(cls_score.device)

            pred_center2d = virtual_camera.project_cam2pixel(
                bbox_pred_3d[:, :3].clone().cpu().numpy()
            )

            pred_center2d = torch.from_numpy(pred_center2d).to(
                cls_score.device
            )

            mlvl_scores.append(scores)
            mlvl_bboxes.append(bboxes)
            mlvl_ctrness_2d.append(ctrness_2d)
            mlvl_bboxes_3d.append(bbox_pred_3d)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_ctrness_3d.append(ctrness_3d)
            mlvl_centers2d.append(pred_center2d)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_ctrness_2d = torch.cat(mlvl_ctrness_2d)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        mlvl_bboxes_3d = torch.cat(mlvl_bboxes_3d)
        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_ctrness_3d = torch.cat(mlvl_ctrness_3d)

        # decode local yaw to global yaw for 3D nms
        #########################################################
        if self.use_multibin:
            self.multibin_centers = self.multibin_centers.to(
                mlvl_dir_scores.device
            )
            if self.multibin_margin > 0:
                mlvl_bboxes_3d = self.decode_alphax_multibin_margin(
                    mlvl_bboxes_3d, mlvl_dir_scores, self.multibin_centers
                )
            else:
                mlvl_bboxes_3d = self.decode_alphax_multibin(
                    mlvl_bboxes_3d, mlvl_dir_scores, self.multibin_centers
                )
        else:
            mlvl_bboxes_3d = self.decode_alphax_2bin_sin_differece(
                mlvl_bboxes_3d, mlvl_dir_scores, self.dir_offset
            )

        rot_y = self.decode_roty(mlvl_bboxes_3d[:, :3], mlvl_bboxes_3d[:, 6])

        # cat rot_y(global yaw) in 8th ch
        mlvl_bboxes_3d = torch.cat([mlvl_bboxes_3d, rot_y[:, None]], dim=1)

        #########################################################
        # Real3D metric needs
        # loc_y move to bottom center of 3d bbox
        mlvl_bboxes_3d[:, 1] += mlvl_bboxes_3d[:, 3] / 2
        #########################################################
        # mlvl_bboxes camera coord x, y, z, h, w, l, alpha, rot_y
        mlvl_bboxes_3d_for_nms = mlvl_bboxes_3d[:, [0, 2, 4, 5, 7]]

        # no scale_factors in box2d3d_multiclass_nms
        # Then we multiply it from outside
        # accoding to FCOSv2 modifed.
        mlvl_nms_scores2d = torch.sqrt(mlvl_scores * mlvl_ctrness_2d[:, None])
        mlvl_nms_scores3d = torch.sqrt(mlvl_scores * mlvl_ctrness_3d[:, None])
        # TODO(xudong.he) WIP: bev nms
        (
            labels,
            scores2d,
            bboxes2d,
            scores3d,
            bboxes3d,
            center2d_prj,
            _,
            _,
        ) = box2d3d_multiclass_nms(
            mlvl_bboxes_3d,
            mlvl_bboxes_3d_for_nms,
            mlvl_nms_scores3d,
            score_thr=self.nms_kwargs.get("score_thr"),
            nms_thr=self.nms_kwargs.get("nms_thr"),
            mlvl_dir_scores=mlvl_dir_scores,
            mlvl_centers2d=mlvl_centers2d,
            mlvl_scores2d=mlvl_nms_scores2d,
            mlvl_bboxes2d=mlvl_bboxes,
            do_nms_bev=self.do_bev3d_nms,
        )

        output, num_pred_obj = self._get_real3d_style_result(
            source_camera,
            virtual_camera,
            labels,
            scores2d,
            bboxes2d,
            scores3d,
            bboxes3d,
            center2d_prj,
            rescale,
        )

        # 2d bbox nms by 3d bbox projected to image
        if self.nms_kwargs.get("iou_threshold", None) is not None:
            apply_nms_2d3d(
                output,
                self.nms_kwargs.get("iou_threshold"),
                self.nms_kwargs.get("replace"),
                self.nms_kwargs.get("use_score2d"),
            )

        max_per_image = self.nms_kwargs.get("max_per_img")
        if num_pred_obj > max_per_image:
            _, inds = scores2d.sort(descending=True)
            inds = inds[:max_per_image]
            for key, val in output.items():
                output[key] = val[inds]
        else:
            for key, val in output.items():
                rem_size = max_per_image - num_pred_obj
                rem_shape = (
                    (rem_size, val.shape[1])
                    if len(val.shape) == 2
                    else (rem_size,)
                )
                output[key] = torch.cat(
                    [
                        val,
                        val.new_zeros(rem_shape),
                    ],
                )

        if not self.is_train_3d_branch:
            output["pred_bboxes"] = torch.cat(
                [
                    output["bbox"],
                    output["score2d"][:, None],
                    output["category_id"][:, None],
                ],
                dim=1,
            )
        return output

    def _get_real3d_style_result(
        self,
        source_camera: CameraBase,
        virtual_camera: CameraBase,
        labels,
        scores2d,
        bboxes2d,
        scores3d,
        bboxes3d,
        center2d_prj,
        rescale=False,
    ):
        num_pred_obj = bboxes3d.shape[0]
        device = scores2d.device

        real3d_out = OrderedDict()
        real3d_out["location"] = bboxes3d[:, :3]
        real3d_out["dim"] = bboxes3d[:, 3:6]
        real3d_out["alpha"] = bboxes3d[:, 6]
        real3d_out["rotation_y"] = bboxes3d[:, 7]
        real3d_out["dep"] = bboxes3d[:, 2]
        # 3d score limited by eval func
        real3d_out["score"] = scores3d
        real3d_out["center"] = center2d_prj
        real3d_out["category_id"] = labels

        real3d_out["score2d"] = scores2d
        real3d_out["bbox"] = bboxes2d

        if rescale:
            loc = real3d_out["location"].clone().cpu().numpy()
            real3d_out["location"] = torch.from_numpy(
                virtual_camera.project_cam2dstCam(source_camera, loc)
            ).to(device)
            real3d_out["dep"] = real3d_out["location"][:, 2]
            real3d_out["center"] = torch.from_numpy(
                virtual_camera.project_pixel2dstCam(
                    source_camera, real3d_out["center"].clone().cpu().numpy()
                )
            ).to(device)
            real3d_out["bbox"] = torch.from_numpy(
                virtual_camera.project_bbox2d2dstcam(
                    source_camera,
                    real3d_out["bbox"].clone().cpu().numpy(),
                    use_bbox_edge_center=True,
                )
            ).to(device)
        return real3d_out, num_pred_obj

    @staticmethod
    def pts2Dto3D(points: Tensor, virtual_camera=None):
        """Project 2.5D to Camera.

        Args:
            points: points in 2D images, [N, 3], 3 corresponds with x, y
                in the image and depth.
            source_camera: Camera instance.
            virtual_camera: Camera instance.
            scale: whether scale to original image.

        Returns:
            Tensor: points in 3D space. [N, 3], 3 corresponds with x, y, z
                in 3D space.
        """
        assert points.shape[1] == 3

        points2D = points[:, :2].cpu().numpy()
        depths = points[:, 2].view(-1, 1).cpu().numpy()

        # 2.5D loc decode to 3D loc in virtual camera
        points3D = torch.from_numpy(
            virtual_camera.project_pixel2cam(points2D, depth=depths)
        )

        return points3D

    @staticmethod
    def project_corners_to_2d(corners_cam, camera: CameraBase):

        box3d_prj_bev = camera.project_cam2pixel(corners_cam[:4, :])
        # get bev bbox width
        x_bev, _, w_bev, _ = cv2.boundingRect(np.int32(box3d_prj_bev))

        corners_cam_ctr = np.zeros((6, 3))
        # bootom-left
        corners_cam_ctr[0, 0] = corners_cam[[0, 3], 0].mean(axis=0)
        corners_cam_ctr[0, 1] = corners_cam[[0, 3], 1].mean(axis=0)
        corners_cam_ctr[0, 2] = corners_cam[[0, 3], 2].mean(axis=0)
        # bootom-front
        corners_cam_ctr[1, 0] = corners_cam[[0, 1], 0].mean(axis=0)
        corners_cam_ctr[1, 1] = corners_cam[[0, 1], 1].mean(axis=0)
        corners_cam_ctr[1, 2] = corners_cam[[0, 1], 2].mean(axis=0)
        # bootom-right
        corners_cam_ctr[2, 0] = corners_cam[[1, 2], 0].mean(axis=0)
        corners_cam_ctr[2, 1] = corners_cam[[1, 2], 1].mean(axis=0)
        corners_cam_ctr[2, 2] = corners_cam[[1, 2], 2].mean(axis=0)
        # bootom-back
        corners_cam_ctr[3, 0] = corners_cam[[2, 3], 0].mean(axis=0)
        corners_cam_ctr[3, 1] = corners_cam[[2, 3], 1].mean(axis=0)
        corners_cam_ctr[3, 2] = corners_cam[[2, 3], 2].mean(axis=0)
        # top
        corners_cam_ctr[4, 0] = corners_cam[[6, 7, 5, 4], 0].mean(axis=0)
        corners_cam_ctr[4, 1] = corners_cam[[6, 7, 5, 4], 1].mean(axis=0)
        corners_cam_ctr[4, 2] = corners_cam[[6, 7, 5, 4], 2].mean(axis=0)
        # bootom
        corners_cam_ctr[5, 0] = corners_cam[[0, 1, 2, 3], 0].mean(axis=0)
        corners_cam_ctr[5, 1] = corners_cam[[0, 1, 2, 3], 1].mean(axis=0)
        corners_cam_ctr[5, 2] = corners_cam[[0, 1, 2, 3], 2].mean(axis=0)
        box3d_ctr_prj = camera.project_cam2pixel(corners_cam_ctr)

        # get height
        _, y_used, _, h_used = cv2.boundingRect(np.int32(box3d_ctr_prj))
        # x1,y1,x2,y2
        return torch.tensor([x_bev, y_used, x_bev + w_bev, y_used + h_used])
