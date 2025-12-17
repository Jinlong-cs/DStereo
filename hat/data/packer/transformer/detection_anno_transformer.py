import json
import os
import re
import warnings
from typing import Any, Dict, Optional

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from .anno_ts_utils import (
    _get_box_10_points,
    check_image_completeness,
    check_obj,
)


@OBJECT_REGISTRY.register
class DenseBoxDetAnnoTs(object):
    """Default annotation transformer for object detection.

    Packed in the densebox image record format.

    Args:
        anno_config:
            Configure
        root_dir:
            Image root
    """

    def __init__(
        self,
        anno_config: Dict,
        root_dir: str,
        verbose: Optional[bool] = True,
        skip_invalid: Optional[bool] = True,
    ) -> None:
        self.verbose = verbose
        self.root_dir = root_dir
        self.anno_config = anno_config
        self.skip_invalid = skip_invalid

    def __call__(self, *item: tuple) -> Any:
        if len(item) == 1 and item[0] is not None:
            image_dir, anno = item[0]
        elif len(item) == 2:
            image_dir, anno = item
        else:
            return None

        image_url = os.path.abspath(image_dir)
        if not os.path.exists(image_url):
            if self.skip_invalid:
                if self.verbose:
                    warnings.warn(
                        "WARNING: skip invalid image: %s" % (image_url)
                    )
                return None
            else:
                raise RuntimeError("No such image: %s" % (image_url))

        if not check_image_completeness(image_url):
            if self.verbose:
                warnings.warn(
                    "WARNING: skip premature end image: %s" % (image_url)
                )  # noqa
            return None

        instances = []
        ignore_regions = []
        for obj in anno.get(self.anno_config["base_classname"], []):
            x1, y1, x2, y2 = map(float, obj["data"])
            height = y2 - y1
            width = x2 - x1
            if height <= 0 or width <= 0:
                continue
            points_data = _get_box_10_points(map(float, obj["data"]))
            matched = False
            for class_mapper in self.anno_config["class_mappers"]:
                if check_obj(obj, class_mapper.get("match_condiction")):
                    matched = True
                    class_id = class_mapper["id"]
                    if class_id is None:
                        upsample = class_mapper.get("upsample", 0)
                        if upsample == 0:
                            break
                        assert (
                            upsample < 1
                        ), "ignore supports downsample only! Plese set upsample value < 1"  # noqa
                        if upsample > np.random.rand():
                            continue  # keep upsample of this match_condition and continue mapping  # noqa
                        else:
                            break  # ignore 1-upsample
                    if check_obj(obj, class_mapper.get("ignore_condiction")):
                        ignore_region = {
                            "left_top": points_data[0],
                            "right_bottom": points_data[2],
                            "class_id": [class_id],
                        }
                        ignore_regions.append(ignore_region)
                    else:
                        if check_obj(obj, class_mapper.get("hard_condiction")):
                            is_hard = True
                        else:
                            is_hard = False
                        instance = {
                            "points_data": points_data,
                            "class_id": [class_id],
                            "attribute": [],
                            "is_hard": [int(is_hard)],
                        }
                        upsample = class_mapper.get("upsample", 1)
                        upsample_int = int(upsample)
                        upsample = upsample_int + int(
                            (upsample - upsample_int) > np.random.rand()
                        )
                        instances.extend([instance] * upsample)
                    if not self.anno_config.get("allow_multi_match", False):
                        break
            if not matched and self.verbose:
                warnings.warn("WARNING: not matched obj: %s" % json.dumps(obj))

        if self.anno_config.get("remove_empty_images", False) and not len(
            instances
        ):  # noqa
            if self.verbose:
                warnings.warn("WARNING: skip no data image: %s" % (image_url))
            return None
        np.random.shuffle(instances)

        img = cv2.imread(image_url, cv2.IMREAD_UNCHANGED)

        if img is None:
            if self.verbose:
                warnings.warn("WARNING: skip invalid image: %s" % (image_url))
            return None
        img_url = os.path.relpath(image_url, self.root_dir)
        if self.anno_config.get("remove_zh_image_path", False) and re.findall(
            "[\u4e00-\u9fa5]", img_url
        ):  # noqa
            if self.verbose:
                warnings.warn(
                    "WARNING, deprecated params: skip zh image path: %s"
                    % (image_url)
                )  # noqa
            return None
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_c = img.shape[2] if len(img.shape) == 3 else 1

        if self.anno_config.get("default_ignore_full_image", False):
            class_ids = set(
                map(
                    lambda class_mapper: class_mapper["id"],
                    self.anno_config["class_mappers"],
                )
            )
            for class_id in range(1, self.anno_config["num_classes"] + 1):
                if class_id not in class_ids:
                    ignore_region = {
                        "left_top": (0, 0),
                        "right_bottom": (img_w, img_h),
                        "class_id": [class_id],
                    }
                    ignore_regions.append(ignore_region)

        anno_dict = {
            "img_url": img_url,
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "instances": instances,
            "ignore_regions": ignore_regions,
        }
        return anno_dict
