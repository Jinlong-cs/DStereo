import copy
import logging
import os
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch

from hat.core.compose_transform import Compose
from hat.data.transforms.affine import _pad_array
from hat.data.transforms.common import Cast
from hat.data.transforms.detection import RandomCrop
from hat.registry import OBJECT_REGISTRY, build_from_registry
from hat.utils.apply_func import _as_list

logger = logging.getLogger(__name__)

__all__ = ["AttrCV2CropInput"]


@OBJECT_REGISTRY.register
class AttrCV2CropInput(object):
    """CV2 crop resize transformer for attribute task.

    Args:
        model_input_hw : tuple of int, The size of model input data.
        detection_task_name : str or list, The name of corresponding detection task name.  # noqa
        is_nchw : bool, Layout of img, `hwc` or `chw`,  default shape HWC.
        norm_ratio : float, The `norm ratio` based on center region.
        norm_method : str, The method of norm, whether to keep shape ratio after cropped.  # noqa
        rsize_shape : int, The cropped shape.
        center_crop_prob : float, center crop ratio.
        transforms : List of transform.
        use_limit_box: bool, whether to use limited box when cropping.

    """

    def __init__(
        self,
        model_input_hw: tuple,
        detection_task_name: Optional[Union[str, Sequence[str]]] = None,
        is_nchw: bool = False,
        norm_ratio: float = 1.25,
        norm_method: str = "longside_square",
        rsize_shape: int = 256,
        center_crop_prob: float = 1,
        transforms: list or dict = None,
        use_limit_box: bool = True,
        add_dummy_img_when_empty: bool = True,
    ):
        super().__init__()
        if detection_task_name is not None:
            self.detection_task_name = _as_list(detection_task_name)

        self._norm_method = norm_method
        self.target_shape = model_input_hw
        self.norm_ratio = norm_ratio
        self.rsize_shape = rsize_shape

        self.is_nchw = is_nchw

        self.use_limit_box = use_limit_box
        self.crop_roi = RandomCrop(
            center_crop_prob=center_crop_prob,
            h_ratio_range=(self.norm_ratio, self.norm_ratio),
            w_ratio_range=(self.norm_ratio, self.norm_ratio),
            size=self.target_shape,
        )

        if transforms is not None:
            transforms = build_from_registry(transforms)
            if isinstance(transforms, (list, tuple)):
                transforms = Compose(transforms)  # noqa
        self.transforms = transforms
        self.add_dummy_img_when_empty = add_dummy_img_when_empty

    def _expand_rects(
        self, bbox, img_shape, expand_ratio=2.0, norm_method=None
    ):
        ori_h = bbox[3] - bbox[1]
        ori_w = bbox[2] - bbox[0]
        assert expand_ratio >= 1
        if norm_method == "longside_ratio":
            exp_h = ori_h * (expand_ratio - 1.0) / 2.0
            exp_w = ori_w * (expand_ratio - 1.0) / 2.0
        elif norm_method == "longside_square":
            length = max(ori_h, ori_w)
            exp_h = (expand_ratio * length - ori_h) / 2.0
            exp_w = (expand_ratio * length - ori_w) / 2.0

        new_bbox = [0, 0, 0, 0]
        expand = [0, 0, 0, 0]

        new_bbox[0] = max(0, bbox[0] - exp_w)
        expand[0] = min(0, bbox[0] - exp_w)
        new_bbox[1] = max(0, bbox[1] - exp_h)
        expand[1] = min(0, bbox[1] - exp_h)
        new_bbox[2] = min(bbox[2] + exp_w, img_shape[1])
        expand[2] = max(0, bbox[2] + exp_w - img_shape[1])
        new_bbox[3] = min(bbox[3] + exp_h, img_shape[0])
        expand[3] = max(0, bbox[3] + exp_h - img_shape[0])

        new_bbox = list(map(int, new_bbox))
        expand = list(map(int, expand))

        return new_bbox, expand

    def crop(self, roi, src_h, src_w, img, rsize_shape=256):
        rects_1, expand = self._expand_rects(
            roi, (src_h, src_w), expand_ratio=2, norm_method=self._norm_method
        )
        # 2. clip extreme rects
        rects_1[0] = max(0, rects_1[0])
        rects_1[1] = max(0, rects_1[1])
        rects_1[2] = src_w - 1 if rects_1[2] >= src_w else rects_1[2]
        rects_1[3] = src_h - 1 if rects_1[3] >= src_h else rects_1[3]

        # 3. crop image with rects and expand it
        img_crop = img[rects_1[1] : rects_1[3], rects_1[0] : rects_1[2], :]
        rects_2 = [rects_1[i] + expand[i] for i in range(len(rects_1))]
        raw_width = rects_2[2] - rects_2[0]
        raw_height = rects_2[3] - rects_2[1]
        if self._norm_method == "longside_ratio":
            raw_shape = (raw_height, raw_width, 3)
        elif self._norm_method == "longside_square":
            length = max(raw_height, raw_width)
            raw_shape = (length, length, 3)
        else:
            raise ValueError("Not supported norm method.")

        img_expand = np.zeros(raw_shape)
        img_expand[
            -expand[1] : raw_height - expand[3],
            -expand[0] : raw_width - expand[2],
            :,
        ] = img_crop

        # 4. resize
        assert img_expand.shape[0] > 0 and img_expand.shape[1] > 0
        img_resize = cv2.resize(img_expand, (rsize_shape, rsize_shape))
        data = {}
        data["img"] = img_resize
        data["img_shape"] = img_resize.shape
        data["layout"] = "hwc"
        if self.use_limit_box:
            limit_box = (
                roi[:4].reshape((-1, 2))
                - np.array(rects_2).reshape((-1, 2))[0]
            ).astype("float64")
            limit_box[:, 0] *= rsize_shape / raw_shape[1]
            limit_box[:, 1] *= rsize_shape / raw_shape[0]
            limit_box = limit_box.flatten()
            data["limit_box"] = limit_box

        data = self.crop_roi(data)
        ts_img = cv2.resize(data["img"], self.target_shape)
        crop_rects = [
            data["crop_bbox"][2],
            data["crop_bbox"][0],
            data["crop_bbox"][3],
            data["crop_bbox"][1],
        ]

        return ts_img, raw_shape, rects_2, crop_rects

    def __call__(self, img_meta):
        bgr_img, org_img_hw = img_meta["img"], img_meta["img_shape"]

        img_anno = img_meta["img_anno"]

        resize_img_hw = [
            int(self.target_shape[0]),
            int(self.target_shape[1]),
        ]

        detection_results = []
        for task_name in self.detection_task_name:
            if task_name not in img_anno:
                logger.warning(f"img_anno has not {self.detection_task_name}.")
            else:
                detection_results.extend(img_anno[task_name])
        all_roi_img = []
        roi_id = []
        for roi_info in detection_results:
            roi_data = np.array(roi_info["data"])
            ts_img, _, _, _ = self.crop(
                roi_data,
                org_img_hw[0],
                org_img_hw[1],
                bgr_img,
                rsize_shape=self.rsize_shape,
            )
            all_roi_img.append(ts_img)
            roi_id.append(roi_info.get("id", "Unknown"))
        if len(all_roi_img) == 0:
            if self.add_dummy_img_when_empty:
                all_roi_img.append(
                    cv2.resize(bgr_img, (resize_img_hw[1], resize_img_hw[0]))
                )
                roi_id.append("Dummy")
            else:
                img_meta.update(
                    {
                        "img": None,
                        "img_height": resize_img_hw[0],
                        "img_width": resize_img_hw[1],
                        "img_anno": None,
                        "obj_id": roi_id,
                    }
                )
                return img_meta

        resize_img_np = np.array(all_roi_img)
        resize_img_tensor = torch.from_numpy(resize_img_np).float()

        if not self.is_nchw:
            resize_img = resize_img_tensor.permute(0, 3, 1, 2)
            img_meta.update(layout="chw")

        img_meta.update(
            {
                "img": resize_img,
                "img_height": resize_img_hw[0],
                "img_width": resize_img_hw[1],
                "obj_id": roi_id,
                "img_anno": None,
            }
        )
        return img_meta


@OBJECT_REGISTRY.register
class RoiTransformCroperAttr(AttrCV2CropInput):
    """roi crop transformer for attribute task.

    Args:
        model_input_hw : tuple of int, The size of model input data.
        detection_task_name : str, The name of corresponding detection task name.  # noqa
        is_nchw : bool, Layout of img, `hwc` or `chw`,  default shape HWC.
        norm_ratio : float, The `norm ratio` based on center region.
        norm_method : str, The method of norm, whether to keep shape ratio after cropped.  # noqa
        rsize_shape : int, The cropped shape.
        center_crop_prob : float, center crop ratio.
        transforms : List of transform.
        use_limit_box: bool, whether to use limited box when cropping.
        mode: str, raining or validation.
        save_crop_image: bool, whether to save copped img during validation.
        save_crop_image_root: str, save root dir.
        save_draw_box: bool, whether to draw box on saved image.

    """

    def __init__(
        self,
        model_input_hw: tuple,
        norm_ratio: float = 1.25,
        norm_method: str = "longside_square",
        rsize_shape: int = 256,
        center_crop_prob: float = 1.0,
        transforms: list or dict = None,
        use_limit_box: bool = True,
        mode: str = "train",
        save_crop_image: bool = False,
        save_crop_image_root: str = None,
        save_draw_box: bool = True,
    ):
        super().__init__(
            model_input_hw=model_input_hw,
            norm_ratio=norm_ratio,
            norm_method=norm_method,
            rsize_shape=rsize_shape,
            center_crop_prob=center_crop_prob,
            transforms=transforms,
            use_limit_box=use_limit_box,
        )
        self.mode = mode
        self.save_crop_image = save_crop_image
        self.save_crop_image_root = save_crop_image_root
        self.save_draw_box = save_draw_box

    def __call__(self, data):
        img = data["img"]
        rec_name = data["data_path"].split("/")[-3]
        task_name = data["data_path"].split("/")[-4].split("_attribute_")[0]
        gt_boxes = data["gt_bboxes"]
        box_attributes = data["attribute_label"]
        img_name = data["img_name"]
        src_h, src_w, _ = img.shape
        rois = gt_boxes.copy()
        assert len(gt_boxes) == len(box_attributes)

        results = []
        for idx, roi in enumerate(rois):
            roi_name = f"{img_name}_{idx}"

            ts_img, raw_shape, expand_box, crop_rects = self.crop(
                roi, src_h, src_w, img, rsize_shape=self.rsize_shape
            )  # noqa

            if len(ts_img.shape) == 3:
                assert ts_img.shape[2] in [
                    1,
                    3,
                ], "img should has the format of HWC"

            pad_shape = list(ts_img.shape)
            pad_shape[0] = self.target_shape[1]
            pad_shape[1] = self.target_shape[0]
            im_hw = np.array(ts_img.shape[:2]).reshape((2,))
            imag = _pad_array(ts_img, pad_shape, "img")

            attribute = np.array(box_attributes[idx])

            cast = Cast(np.float32)

            if self.mode == "train":
                data_info = {
                    "img": imag,
                    "im_hw": cast(im_hw),
                    "attribute": cast(attribute),
                    "labels": cast(attribute),
                    "layout": "hwc",
                    "color_space": "rgb",
                }
            else:
                roi_box = roi[:4]
                dst_box = (
                    roi_box.reshape((-1, 2))
                    - np.array(expand_box).reshape((-1, 2))[0]
                )
                dst_box[:, 0] *= self.rsize_shape / raw_shape[1]
                dst_box[:, 1] *= self.rsize_shape / raw_shape[0]
                dst_box = dst_box - np.array(crop_rects).reshape((-1, 2))[0]
                dst_box[:, 0] *= self.target_shape[1] / (
                    crop_rects[2] - crop_rects[0]
                )
                dst_box[:, 1] *= self.target_shape[0] / (
                    crop_rects[3] - crop_rects[1]
                )
                dst_box = dst_box.flatten()

                if (
                    self.save_crop_image
                    and self.save_crop_image_root is not None
                ):
                    save_image_dir = os.path.join(
                        self.save_crop_image_root, task_name, rec_name
                    )
                    os.makedirs(save_image_dir, exist_ok=True)
                    save_crop_image_path = os.path.join(
                        save_image_dir, roi_name + ".jpg"
                    )
                    save_image = copy.deepcopy(ts_img)
                    if self.save_draw_box:
                        cv2.rectangle(
                            save_image,
                            pt1=(int(dst_box[0]), int(dst_box[1])),
                            pt2=(int(dst_box[2]), int(dst_box[3])),
                            color=(0, 255, 0),
                        )
                    cv2.imwrite(save_crop_image_path, save_image[:, :, ::-1])

                data_info = {
                    "img": imag,
                    "im_hw": cast(im_hw),
                    "attribute": cast(attribute),
                    "labels": cast(attribute),
                    "roi_name": roi_name,
                    "roi_box": roi_box,
                    "dst_box": dst_box,
                    "layout": "hwc",
                    "color_space": "rgb",
                }

            if self.transforms is not None:
                data_info = self.transforms(data_info)
            results.append(data_info)

        return results
