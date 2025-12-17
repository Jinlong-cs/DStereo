import copy
import os
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.core.box_utils import xywh_to_x1y1x2y2
from hat.core.eq_focal_length import cal_equivalent_focal_length_uv_mat
from hat.core.position_embedding_utils import PositionEncoder
from hat.core.undistort_lut import get_undistort_points
from hat.core.utils_3d import (
    Object3d,
    compute_box_3d,
    draw_projected_box3d,
    image_transform,
)
from hat.data.transforms.affine import _pad_array
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import check_packages_available
from .detection import DetInputPadding
from .label_generator import (
    dense3d_pad_after_label_generator,
    label_encoding,
    roi_heatmap_label_encoding,
    roi_heatmap_label_encoding_undistort_uv_depth,
)


@OBJECT_REGISTRY.register
class Image3DTransform(object):
    def __init__(
        self,
        input_wh,
        keep_res,
        shift=None,
        keep_aspect_ratio=False,
        support_wh=(2048, 1280),
        crop_roi=None,
    ):
        self.input_wh = input_wh
        self.keep_res = keep_res
        if shift is None:
            shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self._keep_aspect_ratio = keep_aspect_ratio
        self.support_wh = support_wh
        self.crop_roi = crop_roi

    def __call__(self, data):
        img = data["img"]
        if self.crop_roi is not None:
            x1, y1, x2, y2 = self.crop_roi
            img = img[y1:y2, x1:x2]
        data["ori_img"] = img  # for debug
        orgin_wh = img.shape[:2][::-1]
        if self._keep_aspect_ratio and orgin_wh != self.support_wh:
            resize_wh_ratio = float(self.input_wh[0]) / float(self.input_wh[1])
            orgin_wh_ratio = float(orgin_wh[0]) / float(orgin_wh[1])
            affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            if resize_wh_ratio > orgin_wh_ratio:
                new_wh = (
                    int(orgin_wh[1] * resize_wh_ratio),
                    orgin_wh[1],
                )
                img = cv2.warpAffine(img, affine, new_wh, 0)
            elif resize_wh_ratio < orgin_wh_ratio:
                new_wh = (
                    orgin_wh[0],
                    int(orgin_wh[0] / resize_wh_ratio),
                )
                img = cv2.warpAffine(img, affine, new_wh, 0)
        img, trans_matrix = image_transform(
            img, self.input_wh, self.keep_res, shift=self.shift
        )
        meta = data["anno"]["meta"] if "meta" in data["anno"] else {}
        meta["center"] = trans_matrix["center"]
        meta["size"] = trans_matrix["size"]
        meta["img_wh"] = np.array(img.shape[:2][::-1])
        meta["orgin_wh"] = np.array(orgin_wh)
        meta["trans_matrix"] = trans_matrix["trans_input"]
        data["img"] = img
        data["anno"]["meta"] = meta
        return data


@OBJECT_REGISTRY.register
class ImageTransformWithScale(object):
    def __init__(self, scale_wh, keep_res, shift=None):
        if shift is None:
            self.shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self.scale_wh = scale_wh
        self.keep_res = keep_res

    def __call__(self, data):
        img = data["img"]
        data["ori_img"] = img  # for debug
        orgin_wh = img.shape[:2][::-1]
        input_wh = (
            int(orgin_wh[0] * self.scale_wh[0]),
            int(orgin_wh[1] * self.scale_wh[1]),
        )

        img, trans_matrix = image_transform(
            img, input_wh, self.keep_res, shift=self.shift
        )
        meta = data["meta"] if "meta" in data else {}
        meta["center"] = trans_matrix["center"]
        meta["size"] = trans_matrix["size"]
        meta["img_wh"] = np.array(img.shape[:2][::-1])
        meta["orgin_wh"] = np.array(orgin_wh)
        meta["trans_matrix"] = trans_matrix["trans_input"]
        data["img"] = img
        data["meta"] = meta
        return data


@OBJECT_REGISTRY.register
class MaskImageEdgeTransform(object):
    """
    Perform random mask for image edge.

    Args:
        mask_ranges (Sequence[Tuple[int, int]]):
            image valid pixel area, [left, top, right, bottom].
        seed (int, optional): random seed number. Defaults to 0.
        prob (float, optional):
            the probability of performing random image mask. Defaults to 0.5.
        image_channel_order (str, optional):
            input layout for images. Defaults to hwc.

    Returns:
        data: (Dict): data after image edge mask transform.
    """

    def __init__(
        self,
        mask_ranges: Sequence[Tuple[int, int]],
        seed: int = 0,
        prob: float = 0.5,
        image_channel_order: str = "hwc",
    ) -> None:
        self.mask_ranges = mask_ranges  # left, top, right, bottom
        self.seed = seed
        random.seed(seed)
        self.prob = prob
        self.image_channel_order = image_channel_order

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.prob:
            return data

        if "img" not in data:
            raise KeyError(
                f"Except key `img`, but got {list(data.keys())} in data. "
            )

        valid_area = []  # left, top, right, bottom
        for border in self.mask_ranges:
            if border[0] == border[1]:
                mask_pixel = border[0]
            else:
                mask_pixel = random.randint(*border)
            valid_area.append(mask_pixel)

        image = data["img"]
        image_mask = np.zeros_like(image)
        if self.image_channel_order == "hwc":
            image_mask[
                valid_area[1] : valid_area[3], valid_area[0] : valid_area[2], :
            ] = 1
        elif self.image_channel_order == "chw":
            image_mask[
                :, valid_area[1] : valid_area[3], valid_area[0] : valid_area[2]
            ] = 1
        else:
            raise RuntimeError(f"{self.image_channel_order} not supported.")

        data["img"] = (image * image_mask).astype(np.uint8)

        return data


@OBJECT_REGISTRY.register
class Heatmap3DDetectionLableGenerate(object):
    def __init__(
        self,
        num_classes,
        classid_map,
        normalize_depth,
        focal_length_default,
        alpha_in_degree,
        filtered_name,
        min_box_edge,
        max_depth,
        max_objs,
        down_stride=4,
        use_bbox2d=False,
        enable_ignore_area=False,
        use_project_bbox2d=False,
        shift=None,
        undistort_2dcenter=False,
        undistort_depth_uv=False,
        input_padding=None,
        crop_roi=None,
        vis_label=False,
        pe_config=None,
        keep_meta_keys=None,
        depth_min_option=False,
    ):
        self.num_classes = num_classes
        self.classid_map = classid_map
        self.normalize_depth = normalize_depth
        self.focal_length_default = focal_length_default
        self.alpha_in_degree = alpha_in_degree
        self.filtered_name = filtered_name
        self.down_stride = down_stride
        self.use_bbox2d = use_bbox2d
        self.enable_ignore_area = enable_ignore_area
        if shift is None:
            self.shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self.min_box_edge = min_box_edge
        self.max_depth = max_depth
        self.max_objs = max_objs
        self.use_project_bbox2d = use_project_bbox2d
        self.undistort_2dcenter = undistort_2dcenter
        self.undistort_depth_uv = undistort_depth_uv
        self.input_padding = input_padding
        self.crop_roi = crop_roi
        self.vis_label = vis_label
        self._depth_min_option = depth_min_option

        if keep_meta_keys is None:
            self.keep_meta_keys = ["img_wh"]
        else:
            self.keep_meta_keys = keep_meta_keys

        self.is_with_pe = False
        if pe_config is not None:
            self.keep_meta_keys.append("coordinate_map")
            self.is_with_pe = pe_config["is_with_pe"]
            self.position_encoder = PositionEncoder(
                pe_stride=pe_config["pe_stride"],
                input_hw=pe_config["input_hw"],
                img_resize=pe_config["img_resize"],
                pe_h=pe_config["pe_h"],
                pe_w=pe_config["pe_w"],
                default_intrinsic_mat=pe_config["default_intrinsic_mat"],
                default_distort=pe_config["default_distort"],
                default_pitch=pe_config["default_pitch"],
                default_roll=pe_config["default_roll"],
                default_camera_z=pe_config["default_camera_z"],
                crop_roi=pe_config["crop_roi_3d"],
                verbose=pe_config["verbose"],
            )

    def _show_label(self, data, gt_label, anno, meta):
        if getattr(self, "cc", None) is None:
            self.cc = 0
        img = data["img"]
        img_h, img_w, _ = img.shape
        output_h, output_w = (
            img_h // self.down_stride,
            img_w // self.down_stride,
        )
        calib = copy.deepcopy(meta["calib"])
        if self.crop_roi is not None:
            calib[0, 2] -= self.crop_roi[0]  # center_u
            calib[1, 2] -= self.crop_roi[1]  # center_v
        for obj in anno:
            if "bbox_2d" in obj and obj["bbox_2d"] is not None:
                bbox = obj["bbox_2d"]
                if self.crop_roi is not None:
                    x1 = int(bbox[0] - self.crop_roi[0]) // 2
                    y1 = int(bbox[1] - self.crop_roi[1]) // 2
                else:
                    x1 = int(bbox[0]) // 2
                    y1 = int(bbox[1]) // 2
                x2, y2 = int(x1 + bbox[2] // 2), int(y1 + bbox[3] // 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            obj.update(obj["in_camera"])
            obj_3d = Object3d(obj)
            pts_img, _ = compute_box_3d(obj_3d, calib, meta["distCoeffs"])
            if pts_img is not None:
                pts_img /= 2.0
                draw_projected_box3d(img, pts_img, (0, 255, 0))
        # img = np.pad(img, ((0, 0), (16, 16), (0, 0)), 'constant')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        vis_hms = []
        pad_img = np.zeros((img_h, img_w), dtype=np.uint8)
        for k in [
            "heatmap",
            "box2d_wh",
            "dimensions",
            "location_offset",
            "depth",
            "heatmap_weight",
            "ignore_mask",
        ]:
            if k == "location_offset":
                hm_k = gt_label[k][1, :, :]
            else:
                hm_k = gt_label[k][0, :, :]
            hm_k = cv2.resize(hm_k, (img_w, img_h))
            if k != "ignore_mask":
                hm_k -= np.min(hm_k)
                hm_k /= np.max(hm_k)
            hm_k *= 255
            hm_k = hm_k.astype(np.uint8)
            hm_k = np.stack([hm_k, pad_img, pad_img], axis=-1)
            hm_k = cv2.addWeighted(img, 0.6, hm_k, 0.4, 0.0)
            vis_hms.append(hm_k)

        # draw index used in rotation
        hm_k = np.zeros((output_h, output_w, 3), np.uint8)
        for i in range(len(gt_label["index"])):
            if gt_label["index_mask"][i] > 0:
                cx = gt_label["index"][i] % output_w
                cy = gt_label["index"][i] // output_w
                cv2.circle(hm_k, (cx, cy), 4, (0, 255, 0), -1)
        hm_k = cv2.resize(hm_k, (img_w, img_h))
        hm_k = cv2.addWeighted(img, 0.6, hm_k, 0.4, 0.0)
        vis_hms.append(hm_k)

        vis_hm = np.concatenate(vis_hms, axis=0)
        vis_hm = vis_hm[:, :, ::-1]
        if not os.path.exists("3d_vis_imgs"):
            os.mkdir("3d_vis_imgs")
        cv2.imwrite("3d_vis_imgs/%04d.jpg" % self.cc, vis_hm)
        self.cc += 1

    def __call__(self, data):

        label = data["anno"].pop("objects")
        meta = data["anno"]["meta"]
        meta["calib"] = np.array(meta["calib"])

        gt = label_encoding(
            label,
            meta,
            self.num_classes,
            self.classid_map,
            self.normalize_depth,
            self.focal_length_default,
            self.alpha_in_degree,
            self.down_stride,
            self.use_bbox2d,
            self.enable_ignore_area,
            self._depth_min_option,
            shift=self.shift,
            filtered_name=self.filtered_name,
            min_box_edge=self.min_box_edge,
            max_depth=self.max_depth,
            max_objs=self.max_objs,
            use_project_bbox2d=self.use_project_bbox2d,
            undistort_2dcenter=self.undistort_2dcenter,
            undistort_depth_uv=self.undistort_depth_uv,
            crop_roi=self.crop_roi,
        )
        if self.vis_label:
            self._show_label(data, gt, label, meta)

        # add coordinate 3d map to label
        if self.is_with_pe:
            calib_params = None
            if "calib_all" in meta:
                calib_params = meta["calib_all"]
            elif "extrinsic" in meta:
                calib_params = meta["extrinsic"]
            coordinate3d_map = self.position_encoder(calib_params)
            meta["coordinate_map"] = coordinate3d_map

        for k in list(meta.keys()):
            if k not in self.keep_meta_keys:
                meta.pop(k)
        if self.input_padding is not None:
            gt = dense3d_pad_after_label_generator(
                gt, self.input_padding, self.down_stride
            )
        data = {"img": data["img"], **meta, **gt}
        data["img"] = data["img"].transpose(2, 0, 1)
        data["depth"] = data["depth"].astype(np.float32)
        return data


@OBJECT_REGISTRY.register
class ROIHeatmap3DDetectionLableGenerate:
    def __init__(
        self,
        num_classes,
        classid_map,
        normalize_depth,
        focal_length_default,
        filtered_name,
        min_box_edge,
        max_depth,
        max_gt_boxes_num,
        is_train=True,
        use_bbox2d=False,
        use_project_bbox2d=False,
        undistort_depth_uv=False,
        shift=None,
        crop_roi=None,
        pe_config=None,
        input_padding=None,
        keep_meta_keys=None,
    ):
        self.num_classes = num_classes
        self.classid_map = classid_map
        self.normalize_depth = normalize_depth
        self.focal_length_default = focal_length_default
        self.max_gt_boxes_num = max_gt_boxes_num
        self.use_bbox2d = use_bbox2d
        self.min_box_edge = min_box_edge
        self.use_project_bbox2d = use_project_bbox2d
        self.is_train = is_train
        if shift is None:
            self.shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self.max_depth = max_depth
        self.filtered_name = filtered_name
        self.undistort_depth_uv = undistort_depth_uv
        self.crop_roi = crop_roi

        if keep_meta_keys is None:
            self.keep_meta_keys = ["img_wh", "im_hw"]
        else:
            self.keep_meta_keys = keep_meta_keys

        self.is_with_pe = False
        if pe_config is not None:
            self.keep_meta_keys.append("coordinate_map")
            self.is_with_pe = pe_config["is_with_pe"]
            self.position_encoder = PositionEncoder(
                pe_stride=pe_config["pe_stride"],
                input_hw=pe_config["input_hw"],
                img_resize=pe_config["img_resize"],
                pe_h=pe_config["pe_h"],
                pe_w=pe_config["pe_w"],
                default_intrinsic_mat=pe_config["default_intrinsic_mat"],
                default_distort=pe_config["default_distort"],
                default_pitch=pe_config["default_pitch"],
                default_roll=pe_config["default_roll"],
                default_camera_z=pe_config["default_camera_z"],
                crop_roi=pe_config["crop_roi_3d"],
                verbose=pe_config["verbose"],
            )

        self.input_padding = input_padding

    def __call__(self, data):
        label = data["anno"].pop("objects")
        meta = data["anno"]["meta"]
        meta["calib"] = np.array(meta["calib"])

        if self.undistort_depth_uv:
            self.keep_meta_keys += ["eq_fu", "eq_fv"]
            gt = roi_heatmap_label_encoding_undistort_uv_depth(
                data["img"],
                label=label,
                meta=meta,
                num_classes=self.num_classes,
                classid_map=self.classid_map,
                normalize_depth=self.normalize_depth,
                focal_length_default=self.focal_length_default,
                max_gt_boxes_num=self.max_gt_boxes_num,
                filtered_name=self.filtered_name,
                use_bbox2d=self.use_bbox2d,
                shift=self.shift,
                max_depth=self.max_depth,
                use_project_bbox2d=self.use_project_bbox2d,
                crop_roi=self.crop_roi,
            )
        else:
            gt = roi_heatmap_label_encoding(
                data["img"],
                label=label,
                meta=meta,
                num_classes=self.num_classes,
                classid_map=self.classid_map,
                normalize_depth=self.normalize_depth,
                focal_length_default=self.focal_length_default,
                max_gt_boxes_num=self.max_gt_boxes_num,
                filtered_name=self.filtered_name,
                use_bbox2d=self.use_bbox2d,
                shift=self.shift,
                max_depth=self.max_depth,
                use_project_bbox2d=self.use_project_bbox2d,
                crop_roi=self.crop_roi,
            )

        # add coordinate 3d map to label
        if self.is_with_pe:
            calib_params = None
            if "calib_all" in meta:
                calib_params = meta["calib_all"]
            elif "extrinsic" in meta:
                calib_params = meta["extrinsic"]
            coordinate3d_map = self.position_encoder(calib_params)
            meta["coordinate_map"] = coordinate3d_map

        meta["im_hw"] = meta["img_wh"][::-1].astype(np.float32)
        distCoeffs = meta["distCoeffs"]
        for k in list(meta.keys()):
            if k not in self.keep_meta_keys:
                meta.pop(k)

        if self.is_train:
            data = {
                "img": data["img"],
                **meta,
                **gt,
                "distCoeffs": np.array(distCoeffs),
            }
        else:
            if coco_mask is None:
                check_packages_available("pycocotools")
            ignore_mask = meta["ignore_mask"]
            if not isinstance(ignore_mask, np.ndarray):
                ignore_mask = coco_mask.decode(ignore_mask)
            ignore_mask = ignore_mask.astype(np.uint8)
            trans_matrix = meta.pop("trans_matrix")
            new_wh = (
                int(ignore_mask.shape[1] * trans_matrix[0][0]),
                int(ignore_mask.shape[0] * trans_matrix[1][1]),
            )
            ignore_mask = cv2.warpAffine(ignore_mask, trans_matrix, new_wh, 0)
            meta["ignore_mask"] = coco_mask.encode(
                np.asfortranarray(ignore_mask)
            )
            data = {
                "img": data["img"],
                **meta,
                **gt,
                "ori_img": data["ori_img"],
                "distCoeffs": np.array(distCoeffs),
            }
        data["img"] = data["img"].transpose(2, 0, 1)
        return data


@OBJECT_REGISTRY.register
class RoI3DDetInputPadding(DetInputPadding):
    def __call__(self, data):
        data = super().__call__(data)

        w_scale = 1.0 / data["trans_mat"][0, 0]
        h_scale = 1.0 / data["trans_mat"][1, 1]

        data["calib"][0, 2] += w_scale * self.input_padding[0]
        data["calib"][1, 2] += h_scale * self.input_padding[1]

        return data


@OBJECT_REGISTRY.register
class BBoxGenerate:
    """
    Generate 3d bbox and corresponding info from labeled data.

    Args:
        filtered_name: filtered class name.
        use_bbox2d: use 2d bbox or not.
        undistort_2dcenter: undistort 2d center or not.
        crop_roi: crop the image (x1, y1, x2, y2)

    """

    def __init__(
        self,
        filtered_name: str,
        use_bbox2d: Optional[bool] = False,
        undistort_2dcenter: Optional[bool] = False,
        crop_roi: Optional[list] = None,
    ):
        self.filtered_name = filtered_name
        self.use_bbox2d = use_bbox2d
        self.undistort_2dcenter = undistort_2dcenter
        self.crop_roi = crop_roi

    def __call__(self, data):
        label = data["anno"].pop("objects")
        meta = data["anno"]["meta"]
        meta["calib"] = np.array(meta["calib"])
        meta["orgin_wh"] = np.array(data["img"].shape[:2][::-1])
        use_bbox2d = self.use_bbox2d
        filtered_name = self.filtered_name
        undistort_2dcenter = self.undistort_2dcenter
        crop_roi = self.crop_roi

        calib = copy.deepcopy(meta["calib"])

        if crop_roi is not None:
            calib[0, 2] -= crop_roi[0]  # center_u
            calib[1, 2] -= crop_roi[1]  # center_v

        bboxes = []
        location_offsets = []
        dims = []
        depths = []
        locations = []
        rotation_ys = []
        cls_ids = []
        if "image_key" in meta.keys():
            meta["file_name"] = meta["image_key"]
        if filtered_name not in meta["file_name"]:

            for ann in label:
                in_camera = (
                    ann["in_camera"] if "in_camera" in ann.keys() else ann
                )
                if crop_roi is not None:
                    ann = copy.deepcopy(ann)
                    ann["bbox"][0] -= crop_roi[0]
                    ann["bbox"][1] -= crop_roi[1]
                    if "bbox_2d" in ann and ann["bbox_2d"] is not None:
                        ann["bbox_2d"][0] -= crop_roi[0]
                        ann["bbox_2d"][1] -= crop_roi[1]

                cls_id = int(ann["category_id"])

                # generated by prelabel pipeline
                if (
                    use_bbox2d
                    and "bbox_2d" in ann
                    and ann["bbox_2d"] is not None
                ):
                    # use image 2d bbox
                    bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                    if undistort_2dcenter:
                        a_bbox = ann["bbox_2d"]
                        bbox_cx = a_bbox[0] + a_bbox[2] / 2.0
                        bbox_cy = a_bbox[1] + a_bbox[3] / 2.0
                        proj_p = get_undistort_points(
                            np.array([[bbox_cx, bbox_cy]]),
                            np.array(calib[:3, :3]),
                            np.array(meta["distCoeffs"]),
                            img_wh=meta["orgin_wh"],
                        )
                        proj_p = proj_p.reshape(-1, 2)
                        proj_p = proj_p.astype(np.int64)
                        bbox_cx = proj_p[0, 0]
                        bbox_cy = proj_p[0, 1]
                        cx, cy = calib[0, 2], calib[1, 2]
                        location_offset = np.array(
                            [
                                (bbox_cx - cx)
                                * in_camera["depth"]
                                / calib[0, 0],
                                (bbox_cy - cy)
                                * in_camera["depth"]
                                / calib[1, 1],
                                in_camera["depth"],
                            ]
                        ).astype(np.float64)
                        location_offset[1] += in_camera["dim"][0] / 2.0
                        location_offset = (
                            in_camera["location"] - location_offset
                        )
                else:
                    continue

                bboxes.append(bbox)
                location_offsets.append(location_offset)
                dims.append(in_camera["dim"])
                depths.append(in_camera["depth"])
                locations.append(in_camera["location"])
                rotation_ys.append(in_camera["rotation_y"])
                cls_ids.append(cls_id)

        data["bboxes"] = np.array(bboxes)
        data["location_offsets"] = np.array(location_offsets)
        data["dims"] = np.array(dims)
        data["depths"] = np.array(depths)
        data["locations"] = np.array(locations)
        data["rotation_ys"] = np.array(rotation_ys)
        data["cls_ids"] = np.array(cls_ids)

        data["calib"] = calib
        if "calib_all" in meta.keys():
            data["calib_all"] = meta["calib_all"]
        elif "extrinsic" in meta.keys():
            data["extrinsic"] = meta["extrinsic"]
        if "ignore_mask" in meta.keys():
            data["ignore_mask"] = meta["ignore_mask"]
        data["distCoeffs"] = np.array(data["anno"]["meta"]["distCoeffs"])
        if "uv_map" in data["anno"]["meta"].keys():
            data["uv_map"] = data["anno"]["meta"]["uv_map"]
        data.pop("anno", None)
        data.pop("transform_meta", None)
        return data


@OBJECT_REGISTRY.register
class AffineTransform:
    """
    Affine transform the image and the bbox (resize only).

    Args:
        input_wh: input image size (w, h)
        keep_res: whether to keep the original resolution
        max_objs: max number of objects
        shift: shift the image (w, h)
        keep_aspect_ratio: whether to keep the aspect ratio
        crop_roi: crop the image (x1, y1, x2, y2)
        undistort_depth_uv: whether to undistort the depth
    """

    def __init__(
        self,
        input_wh: tuple,
        keep_res: bool,
        max_objs: int,
        shift: Optional[tuple] = None,
        keep_aspect_ratio: Optional[bool] = False,
        crop_roi: Optional[tuple] = None,
        undistort_depth_uv: Optional[bool] = False,
    ):
        self.input_wh = input_wh
        self.keep_res = keep_res
        self.max_objs = max_objs
        if shift is None:
            shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self._keep_aspect_ratio = keep_aspect_ratio
        self.crop_roi = crop_roi
        self.undistort_depth_uv = undistort_depth_uv

    def __call__(self, data):
        img = data["img"]

        if self.crop_roi is not None:
            x1, y1, x2, y2 = self.crop_roi
            img = img[y1:y2, x1:x2]
        img, trans_matrix = image_transform(
            img, self.input_wh, self.keep_res, shift=self.shift
        )
        img_wh = np.array(img.shape[:2][::-1])
        width, height = img_wh
        if "ignore_mask" not in data:
            ignore_mask = np.zeros((width, height, 1))
        else:
            ignore_mask = data["ignore_mask"]
            if not isinstance(ignore_mask, np.ndarray):
                if coco_mask is None:
                    check_packages_available("pycocotools")
                ignore_mask = coco_mask.decode(ignore_mask)
            ignore_mask = ignore_mask.astype(np.uint8)
            if self.crop_roi is not None:
                x1, y1, x2, y2 = self.crop_roi
                ignore_mask = ignore_mask[y1:y2, x1:x2]
            ignore_mask = cv2.warpAffine(
                ignore_mask,
                trans_matrix["trans_input"],
                (width, height),
                flags=cv2.INTER_NEAREST,
            )

        bboxes = data["bboxes"]
        if len(bboxes) == 0:
            data["bboxes"] = np.zeros((0, 4))
            data["location_offsets"] = np.zeros((0, 3))
            data["dims"] = np.zeros((0, 3))
            data["depths"] = np.zeros((0,))
            data["locations"] = np.zeros((0, 3))
            data["rotation_ys"] = np.zeros((0,))
            data["cls_ids"] = np.zeros((0,), dtype=np.int64)
        else:
            top_left_points = bboxes[:, :2]
            bottom_right_points = bboxes[:, 2:]
            top_left_points = np.insert(top_left_points, 2, 1, axis=1)
            bottom_right_points = np.insert(bottom_right_points, 2, 1, axis=1)
            top_left_points = np.matmul(
                top_left_points, trans_matrix["trans_input"].T
            )
            bottom_right_points = np.matmul(
                bottom_right_points, trans_matrix["trans_input"].T
            )
            new_bboxes = np.concatenate(
                [top_left_points[:, :2], bottom_right_points[:, :2]], axis=1
            )
            data["bboxes"] = np.array(new_bboxes)
        data["img_wh"] = img_wh
        data["im_hw"] = img_wh[::-1].astype(np.float32)
        data["trans_matrix"] = np.array(
            trans_matrix["trans_input"], dtype=np.float64
        )
        data["img"] = img.transpose(2, 0, 1)
        data["ignore_mask"] = ignore_mask

        if self.undistort_depth_uv:
            eq_fu_mat, eq_fv_mat = cal_equivalent_focal_length_uv_mat(
                trans_matrix["size"][0],
                trans_matrix["size"][1],
                data["calib"],
                data["distCoeffs"],
            )
            resized_eq_fu = cv2.warpAffine(
                eq_fu_mat, data["trans_matrix"], (width, height)
            )
            resized_eq_fv = cv2.warpAffine(
                eq_fv_mat, data["trans_matrix"], (width, height)
            )
            if len(bboxes) == 0:
                ctx_eq_fu = np.zeros((0,), dtype=np.float64)
                ctx_eq_fv = np.zeros((0,), dtype=np.float64)
            else:
                origin_ctx = (bboxes[:, 0] + bboxes[:, 2]) / 2
                origin_cty = (bboxes[:, 1] + bboxes[:, 3]) / 2
                ctx_eq_fu = eq_fu_mat[
                    np.floor((origin_cty)).astype(int),
                    np.floor((origin_ctx)).astype(int),
                ]
                ctx_eq_fv = eq_fv_mat[
                    np.floor((origin_cty)).astype(int),
                    np.floor((origin_ctx)).astype(int),
                ]

            data["resized_eq_fu"] = resized_eq_fu
            data["resized_eq_fv"] = resized_eq_fv
            data["ctx_eq_fu"] = ctx_eq_fu
            data["ctx_eq_fv"] = ctx_eq_fv

        data = self.padding(data)
        data.pop("calib_all", None)
        return data

    def padding(self, data):
        data["bboxes"] = self.pad_data(data["bboxes"], "bboxes")
        data["location_offsets"] = self.pad_data(
            data["location_offsets"], "location_offsets"
        )
        data["locations"] = self.pad_data(data["locations"], "locations")
        data["dims"] = self.pad_data(data["dims"], "dims")
        data["rotation_ys"] = self.pad_data(data["rotation_ys"], "rotation_ys")
        data["depths"] = self.pad_data(data["depths"], "depths")
        data["cls_ids"] = self.pad_data(data["cls_ids"], "cls_ids", value=-1)
        if self.undistort_depth_uv:
            data["ctx_eq_fu"] = self.pad_data(data["ctx_eq_fu"], "ctx_eq_fu")
            data["ctx_eq_fv"] = self.pad_data(data["ctx_eq_fv"], "ctx_eq_fv")
        return data

    def pad_data(self, data, name, value=0):
        pad_shape = list(data.shape)
        pad_shape[0] = self.max_objs
        data = _pad_array(data, pad_shape, name, value)
        return data
