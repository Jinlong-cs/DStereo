# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm

from hat.core.box3d_utils import compute_2d_box
from hat.metrics.detection3d.basic_struct import ImgObj
from hat.metrics.detection3d.compute_entry import (
    get_tp_fp_nuscense,
    summarize_nuscenes,
)
from hat.metrics.detection3d.generate_results import generate_nuscenes_result
from hat.metrics.detection3d.utils import mioa_ignore
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["HorizonNuscense3DMetric"]


@OBJECT_REGISTRY.register
class HorizonNuscense3DMetric(EvalMetric):
    """Horizon Nuscense3D metric.

    Note: This HorizonNuscense3DMetric is different from the \
        original Nuscense metric in the community. This metric \
        only calculates the value of AP3D.

    Args:
        save_dir: Output dir to save file.
        eval_class: Name of class to evaluate.
        target_recalls: Target recalls to evaluate.
        target_precisions: Target precisions to evaluate.
        eval_cameras: Cameras to evaluate.
        eval_bbox_type: Bbox type, in ["all", "truncated", "non-truncated"].
        image_size: Image size, (width, height) format.
        max_depth: Max depth. Defaults to 300.
        iou_threshold: 3D iou threshold. Defaults to 0.5.
        mioa_thresh: Ignore mask thresh. Defaults to 0.6.
        enable_ignore: Whether enable ignore. Defaults to False.
        horizon_truncate_offset: Truncate offset. Defaults to 0.
        image_tag: images with tag to evaluate under given key-value.
        num_worker: Num workers for calculating metric.
    """

    def __init__(
        self,
        save_dir: str,
        eval_class: str,
        target_recalls: List[float],
        target_precisions: List[float],
        eval_cameras: List[str],
        eval_bbox_type: str,
        image_size: Tuple[int],
        max_depth: float = 300,
        iou_threshold: float = 0.5,
        mioa_thresh: float = 0.6,
        enable_ignore: bool = False,
        horizon_truncate_offset: int = 0,
        image_tag: Optional[Dict[str, Any]] = None,
        num_worker: int = 0,
        **kwargs,
    ):

        assert eval_bbox_type in ["all", "truncated", "non-truncated"]
        self.save_dir = save_dir
        self.eval_class = eval_class
        self.target_recalls = target_recalls
        self.target_precisions = target_precisions
        self.eval_cameras = eval_cameras
        self.image_tag = image_tag if image_tag is not None else {}
        assert isinstance(self.image_tag, dict)
        self.eval_bbox_type = eval_bbox_type
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.max_depth = max_depth
        self.mioa_thresh = mioa_thresh
        self.enable_ignore = enable_ignore
        self.horizon_truncate_offset = horizon_truncate_offset

        self.all_gts = defaultdict(list)
        self.all_dets = defaultdict(list)
        self.gt_count = 0
        self.key_calib = {}
        self.key_distcoeffs = {}
        self.num_worker = num_worker

    def update(self, gts: List[ImgObj], preds: List[ImgObj]):
        return self._parse_gts_and_preds(gts, preds)

    def _parse_gts_and_preds(
        self, gt_anno_list: List[ImgObj], pred_list: List[ImgObj]
    ):
        # filter gt
        all_gts = defaultdict(list)
        all_dets = defaultdict(list)
        gt_count = 0

        key_ignore_mask = {}
        for gt_frame in gt_anno_list:
            image_key = gt_frame.image_key

            # filter by camera name
            if not self._check_match(image_key):
                continue

            # filter by image tag
            gt_image_tag = gt_frame.attrs.get("tags", {})
            if not self._check_image_tag(gt_image_tag):
                continue

            if image_key not in key_ignore_mask.keys():
                key_ignore_mask[image_key] = gt_frame.ignore_mask

            self.key_calib[image_key] = gt_frame.calib
            self.key_distcoeffs[image_key] = gt_frame.dist_coeffs

            for gt_obj in gt_frame.instances:
                if gt_obj.depth >= self.max_depth:
                    gt_obj.ignore = True

                if self.eval_bbox_type in ["truncated", "non-truncated"]:
                    bbox = np.array(gt_obj.bbox)
                    bbox[2:] += bbox[:2]
                    if self.eval_bbox_type == "truncated":
                        if np.all(bbox[2:] < self.image_size) and np.all(
                            bbox[:2] >= [0, 0]
                        ):
                            continue
                    elif self.eval_bbox_type == "non-truncated":
                        new_img_width = (
                            self.image_size[0] - self.horizon_truncate_offset
                        )
                        new_img_height = self.image_size[1]
                        if np.any(
                            bbox[2:] >= [new_img_width, new_img_height]
                        ) or np.any(
                            bbox[:2] < [self.horizon_truncate_offset, 0]
                        ):
                            continue

                if self.enable_ignore:
                    if not gt_obj.ignore:
                        gt_count += 1
                else:
                    gt_count += 1
                all_gts[image_key].append(gt_obj)

        for pred_frame in pred_list:
            image_key = pred_frame.image_key
            # filter by camera name
            if not self._check_match(image_key):
                continue
            for pred_obj in pred_frame.instances:

                if pred_obj.bbox is not None:
                    bbox2d = pred_obj.bbox
                else:
                    bbox2d, _ = compute_2d_box(
                        pred_obj.dimensions,
                        pred_obj.location,
                        pred_obj.rotation_y,
                        self.key_calib[image_key],
                        dist_coeff=self.key_distcoeffs[image_key],
                        fisheye=False,
                        img_wh=self.image_size,
                        z_thresh=1.0,
                    )
                if bbox2d is None:
                    continue
                pred_obj.bbox = bbox2d
                if (
                    np.all(np.array(pred_obj.dimensions) > 0)
                    and pred_obj.depth < self.max_depth
                ):
                    if image_key in key_ignore_mask.keys() and mioa_ignore(
                        pred_obj.bbox,
                        key_ignore_mask[image_key],
                        self.mioa_thresh,
                    ):  # noqa
                        continue
                    all_dets[image_key].append(pred_obj)

        # To save mem usage when num_worker > 1.
        if self.num_worker <= 1:
            self.all_gts.update(all_gts)
            self.all_dets.update(all_dets)
            self.gt_count += gt_count

        return all_gts, all_dets, gt_count

    def single_worker(self, group_imgs_list):
        res_by_img = []
        for v in group_imgs_list:
            res = get_tp_fp_nuscense(
                v["gt"], v["pred"], self.iou_threshold, self.enable_ignore
            )
            res_by_img.append(res)
        return res_by_img

    def get(self):
        logger.info(
            "ALL GT: {} images, {} objects.".format(
                str(len(self.all_gts.keys())),
                str(sum([len(v) for v in self.all_gts.values()])),
            )  # noqa E501
        )
        logger.info(
            "ALL PRED: {} images, {} objects.".format(
                str(len(self.all_dets.keys())),
                str(sum([len(v) for v in self.all_dets.values()])),
            ),  # noqa E501
        )

        group_by_imgs = [
            {"gt": self.all_gts[k], "pred": self.all_dets[k]}
            for k in self.all_gts.keys()
        ]

        res_by_img = []

        if self.num_worker <= 1:
            res_by_img = self.single_worker(group_by_imgs)
        else:
            pool = Pool(self.num_worker)
            ret_list = []
            idx_interval = len(group_by_imgs) // self.num_worker
            for idx_ in range(0, len(group_by_imgs), idx_interval):
                ret = pool.apply_async(
                    self.single_worker,
                    (group_by_imgs[idx_ : idx_ + idx_interval],),
                )
                ret_list.append(ret)
            pool.close()
            for ret in tqdm.tqdm(ret_list, desc="Process: "):
                res_by_img += ret.get()

        res = summarize_nuscenes(
            res_by_img,
            self.gt_count,
            self.target_recalls,
            self.target_precisions,
        )  # noqa E501

        # dump to file
        generate_nuscenes_result(
            results=res,
            output_dir=self.save_dir,
            result_json="result.json",
            result_png="result.png",
            table_json="tables.json",
        )
        ap_res = dict(  # noqa C408
            AP_3D=round(res["AP"], 4),
            AR_3D=round(res["AR"], 4),
        )
        logger.info(
            "Nuscense AP_3D eval done! AP_3D: {}, AR_3D: {}".format(
                ap_res["AP_3D"], ap_res["AR_3D"]
            )
        )
        return ap_res

    def _check_match(self, img_name):
        matched = [int(camera in img_name) for camera in self.eval_cameras]
        return sum(matched) > 0

    def _check_image_tag(self, gt_image_tag):
        keep = True
        for key, value in self.image_tag.items():
            value = value if isinstance(value, list) else [value]
            if gt_image_tag.get(key, None) not in value:
                keep = False
                break
        return keep
