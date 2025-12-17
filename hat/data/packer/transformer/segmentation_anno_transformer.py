import copy
import os
import warnings
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from .anno_ts_utils import (
    _default_check_render_image,
    check_parsing_ignore,
    draw_colorbar,
    draw_mask,
    drawcontour,
    get_image_shape,
    get_map,
    imread,
    imwrite,
    verify_image,
)


class _default_get_label_path_transformer(object):
    """Given the path of an image.

    Return the path of its corresponding label map file.
    `Label_root_path` is specified in this file.

    Args:
        label_root_path:
            the label map root path; if None, use image root path.
    """

    def __init__(self, label_root_path: Optional[Any] = None):
        if label_root_path is not None:
            self.label_root_path = os.path.abspath(label_root_path)
        else:
            self.label_root_path = label_root_path

    def __call__(self, image_path: str) -> str:
        image_path = os.path.expanduser(image_path)
        assert (
            image_path.endswith("jpg")
            or image_path.endswith("jpeg")
            or image_path.endswith("png")
        ), ("invalid ext for image %s, only allow jpg, jpeg, png" % image_path)

        if self.label_root_path is not None:
            image_name = os.path.basename(image_path)
            label_name = os.path.splitext(image_name)[0] + "_label.png"
            label_path = os.path.join(self.label_root_path, label_name)
        else:
            label_path = image_path[: image_path.rindex(".")] + "_label.png"
        return label_path


@OBJECT_REGISTRY.register
class DenseBoxSegAnnoTs(object):
    """Legacy transformer.

    That transform segmentation annotation to densebox detection annotation.

    Args:
        class_ids: list of int
            Class ids for instances in segmentation annotation
        get_label_path_fn:
            The way to get label path. This function is called
            on the following way

            .. code-block:: none

                label_path = get_label_path_fn(image_path)
        verify_image: bool
            Whether to verify image.
    """

    def __init__(
        self,
        class_ids: list,
        get_label_path_fn: Optional[
            Any
        ] = _default_get_label_path_transformer(),  # noqa
        verify_image: Optional[bool] = True,
        verify_label: Optional[bool] = True,
    ):
        assert isinstance(class_ids, list), "expect list, but get %s" % str(
            type(class_ids)
        )
        assert callable(
            get_label_path_fn
        ), "get_label_path_fn should be callable"
        for k, v in enumerate(class_ids):
            assert (
                isinstance(v, int) and v >= 0
            ), "idx should >= 0, but get %s at key %s" % (str(type(v)), k)
        self.class_ids = class_ids
        self.get_label_path_fn = get_label_path_fn
        self.verify_image = verify_image
        self.verify_label = verify_label

    def __call__(self, *item):
        if item[0] is None:
            return None
        if len(item) == 2:
            _, anno = item
        elif len(item) == 1:
            _, anno = item[0]
        else:
            raise RuntimeError

        img_path = anno["image_url"]
        label_path = anno.get("label_path", None)

        img_path = _default_check_render_image(img_path)

        if label_path is None:
            label_path = self.get_label_path_fn(img_path)

        if self.verify_image and not verify_image(img_path):
            return None
        if self.verify_label and not verify_image(label_path):
            return None
        img_h, img_w, img_c = get_image_shape(cv2.imread(img_path))
        l, t, r, b = (0, 0, img_w, img_h)
        cx, cy = (round(0.5 * img_w, 2), round(0.5 * img_h, 2))
        inst = {
            "points_data": [[l, t], [r, t], [r, b], [l, b], [cx, cy]],
            "class_id": self.class_ids,
            "attribute": [],
            "is_hard": [0],
            "is_point_hard": [0] * 5,
        }
        # no ignore region
        anno_dict = {
            "instances": inst,
            "ignore_regions": None,
            "img_url": img_path,
            "parsing_map_urls": label_path,
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "idx": -1,
            "img_attribute": None,
        }
        return anno_dict


@OBJECT_REGISTRY.register
class DefaultGenerateLabelMapAnnoTs(object):
    """Generate mask from annotation.

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
            Whether merge label, by default False.
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
        dst_label_map: Optional[Any] = None,
        reuse_prelabel: Optional[bool] = True,
        is_merge: Optional[bool] = False,
        merge_config: Optional[Any] = None,
        check_parsing_ignore: Optional[bool] = False,
    ) -> None:
        self.colors = colors[:, :, ::-1]
        self.anno_map = get_map(src_label, dst_label)

        self.labels = dst_label
        self.dst_label_map = dst_label_map

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
        self.check_parsing_ignore = check_parsing_ignore
        self._anno_to_contours_fn = anno_to_contours_fn
        if is_merge:
            assert (
                merge_config is not None
            ), "merge_config should not be None, when is_merge is True"
            self.merge_config = merge_config

    def _draw_contours(
        self, mask: Any, cts: list, width: int, height: int
    ) -> Any:
        contours = []
        values = []
        for c in cts:
            if self.dst_label_map is None:
                ct_pts, value = self._anno_to_contours_fn(
                    c, width, height, self.labels
                )
            else:
                ct_pts, value = self._anno_to_contours_fn(
                    c, width, height, self.dst_label_map
                )
            contours.append(ct_pts)
            values.append(value)

        assert len(contours) == len(values)
        for value, contour in zip(values, contours):
            mask = drawcontour(contour, value, mask)

        return mask

    def _merge_labelmap(
        self,
        label: np.ndarray,
        merge_id_src: int,
        merge_label: np.ndarray,
        merge_id_dst: int,
        merge_iou_thresh: int,
        merge_pixel_thresh: int,
    ) -> Any:
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
        if self.reuse_prelabel:
            if "predict_label" in anno:
                predict_label = anno["predict_label"]
            else:
                predict_label = image_path.replace(
                    "." + image_path.split(".")[-1], "_label.png"
                )
            assert os.path.isfile(
                predict_label
            ), f"Not exist predict label path: {predict_label}"
            label = self.anno_map[imread(predict_label)]
        else:
            label = np.zeros(im.shape[0:2], dtype=np.uint8)

        width, height = anno["width"], anno["height"]
        if "parsing" not in anno and "lane" not in anno:
            warnings.warn(
                "WARNING! no paring and lane area: %s." % anno["image_key"]
            )
        else:
            cts = anno["parsing"] if "parsing" in anno else anno["lane"]
            label = self._draw_contours(label, cts, width, height)

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
        if self.check_parsing_ignore and not check_parsing_ignore(label_path):
            if os.path.isfile(label_path):
                os.remove(label_path)
            if os.path.isfile(label_fusion_path):
                os.remove(label_fusion_path)
            return None

        anno["label_path"] = label_path
        return image_path, anno
