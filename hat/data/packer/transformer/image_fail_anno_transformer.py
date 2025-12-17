import copy
import os
import warnings
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from .anno_ts_utils import (
    _default_check_render_image,
    draw_colorbar,
    draw_mask,
    drawcontour,
    get_map,
    imread,
    imwrite,
)


@OBJECT_REGISTRY.register
class ImageFailGenerateLabelMapAnnoTs(object):
    """
    Generate mask from annotation for image fail parsing.

    Args:
        output_dir:
            Output directory
        src_label:
            Source label ids.
        dst_label:
            Target label ids.
        colors : :py:class:`numpy.ndarray`
            Used in :py:func:`cv2.LUT`
        clsnames:
            Class names.
        anno_to_contours_fn:
            How to convert annotation from polygons to mask.

            This function is called in the following ways:

            .. code-block:: python

                contours, value = anno_to_contours_fn(
                    data, width, height, dst_label)
        reuse_prelabel :
            Whether reuse prelabel, by default True.
        is_merge:
            Whether merge label, by default True.
        merge_config: list, optional
            Config, by default None
    """

    def __init__(
        self,
        output_dir: str,
        src_label: Dict,
        dst_label: Dict,
        colors: np.ndarray,
        clsnames: List[str],
        anno_to_contours_fn: Callable,
        reuse_prelabel: Optional[bool] = False,
        is_merge: Optional[bool] = False,
        merge_config: Optional[list] = None,
    ) -> None:
        self.colors = colors[:, :, ::-1]
        self.anno_map = get_map(src_label, dst_label)

        self.labels = dst_label

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        draw_colorbar(colors, clsnames, os.path.split(self.output_dir)[0])

        self.area_count = copy.deepcopy(self.labels)
        for key in self.area_count.keys():
            self.area_count[key] = 0

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.reuse_prelabel = reuse_prelabel
        self.is_merge = is_merge
        self._anno_to_contours_fn = anno_to_contours_fn
        if is_merge:
            assert (
                merge_config is not None
            ), "merge_config should not be None, when is_merge is True"
            self.merge_config = merge_config

    def _anno_check(
        self,
        imgname: str,
        cts: list,
    ) -> bool:
        """
        Check annotation validity.

        Returns:
            boolean
                true if check is passed.
        """
        result = True
        possible_cts = []
        certain_cts = []
        # split cts by types
        for ct in cts:
            attrs = ct["attrs"]
            if attrs["confidence"] == "possible":
                possible_cts.append(ct)
            elif attrs["confidence"] == "certain":
                certain_cts.append(ct)
            else:
                warnings.warn(
                    f"Invalid anno contour is found in {imgname}: {attrs}!"
                )
        all_cts_num = len(cts)
        # check consistency in number of possible/certain
        if all_cts_num == 0:
            result = False
            warnings.warn(f"Invalid anno: there is no contours in {imgname}!")
        elif len(possible_cts) > len(certain_cts):
            result = False
            warnings.warn(
                f"Invalid anno: the possible/certain attrs are mismatch in {imgname}!"  # noqa
            )
        return result

    def _draw_contours(
        self, mask: Any, cts: list, width: int, height: int, imgname: str
    ) -> Any:
        draw_flag = True  # draw success or not
        # anno check
        if not self._anno_check(imgname, cts):
            draw_flag = False
            return mask, draw_flag
        contours = []
        values = []
        if len(cts) > 0:
            for ct in cts:
                ct_pts, value = self._anno_to_contours_fn(
                    ct, width, height, self.labels
                )
                contours.append(ct_pts)
                values.append(value)
            assert len(contours) == len(values)
            for value, contour in zip(values, contours):
                mask = drawcontour(contour, value, mask)
        return mask, draw_flag

    def _merge_labelmap(
        self,
        label: np.ndarray,
        merge_id_src: int,
        merge_label: np.ndarray,
        merge_id_dst: int,
        merge_iou_thresh: int,
        merge_pixel_thresh: int,
    ) -> np.ndarray:
        merge_label = np.array(merge_label == merge_id_dst) * (label != 255)
        h, w = label.shape
        label_merge = copy.deepcopy(label)
        major = cv2.__version__.split(".")[0]
        if major == "3":
            _, contours, hierarchy = cv2.findContours(
                merge_label.astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        else:
            contours, hierarchy = cv2.findContours(
                merge_label.astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        for contour in contours:
            # M = drawcontour(contour, h, w)
            M = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(M, [contour], -1, (1), -1)
            if merge_pixel_thresh is not None:
                if M.sum() < merge_pixel_thresh:
                    continue
            if merge_iou_thresh is not None:
                intersection = label[M == 1] == merge_id_src
                ratio = intersection.sum() / (M.sum() + 0.0)
                if ratio > merge_iou_thresh:
                    label_merge[M == 1] = merge_id_src
            else:
                label_merge[M == 1] = merge_id_src
        label_merge[label == 255] = 255
        return label_merge

    def __call__(self, item: tuple) -> Any:
        if item is None:
            return None
        image_dir, anno = item
        image_path = os.path.abspath(image_dir)
        image_path = _default_check_render_image(image_path)
        name = os.path.basename(image_path)
        anno["image_key"] = name
        anno["image_url"] = image_path

        imgname = name.replace("." + name.split(".")[-1], "")

        im = imread(image_path)
        if im is None:
            warnings.warn("WARNING! no image: %s, ignore." % image_path)
            return None
        predict_label = image_path.replace(
            "." + image_path.split(".")[-1], "_label.png"
        )
        if self.reuse_prelabel and os.path.isfile(predict_label):
            label = self.anno_map[imread(predict_label)]
        else:
            label = np.zeros(im.shape[0:2], dtype=np.uint8)

        width, height = anno["width"], anno["height"]
        if "parsing" not in anno:
            warnings.warn("WARNING! no paring area: %s." % anno["image_key"])
        else:
            cts = anno["parsing"]
            label, draw_flag = self._draw_contours(
                label, cts, width, height, imgname
            )
        if not draw_flag:
            return None

        if self.is_merge:
            assert "merge_label_url" in anno, "Not exist merge label."
            merge_label = imread(anno["merge_label_url"])
            for merge_info in self.merge_config:
                label = self._merge_labelmap(
                    label,
                    merge_info["merge_id_src"],
                    merge_label,
                    merge_info["merge_id_dst"],
                    merge_info["merge_iou_thresh"],
                    merge_info["merge_pixel_thresh"],
                )
        label_path = self.output_dir + "/" + imgname + "_label.png"
        imwrite(label_path, label)
        label_fusion_path = self.output_dir + "/" + imgname + "_fusion.jpg"
        imwrite(label_fusion_path, draw_mask(im, label, self.colors))

        anno["label_path"] = label_path
        return image_path, anno
