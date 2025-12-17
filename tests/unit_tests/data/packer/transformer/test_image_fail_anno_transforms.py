import json
import os

import cv2
import numpy as np
import pytest

from hat.data.packer.transformer.anno_ts_utils import (
    image_fail_parsing_anno_to_contours_fn,
)
from hat.data.packer.transformer.image_fail_anno_transformer import (
    ImageFailGenerateLabelMapAnnoTs,
)
from hat.data.packer.utils import get_colors_and_class_names_for_lane_parsing
from hat.utils.config import Config
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


def bucket_exist(path):
    """If user does not have permission of some horzion buckets,
    like HDLTAlgorithm, it will raise PermissionError. This function
    will catch this error.
    """
    try:
        common_exists = os.path.exists(path) and len(os.listdir(path)) > 0
    except PermissionError:
        common_exists = False
    return common_exists


# step 1 prepare data
@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS,
    reason="requiring HAT_BUCKET bucket",
)
def test_image_fail_anno_transformer():
    root = f"{HAT_BUCKET_PATH}/users/zihan.qiu/jenkins_test_transform_data/image_fail_anno_transform"  # noqa
    f = open(os.path.join(root, "data2.json"))
    input = json.load(f)
    input[0] = os.path.join(root, "raw_img.jpg")
    label_map_config = Config.fromfile(
        os.path.join(root, "image_fail_parsing_labelmap_7cls.py")
    )  # noqa
    src_label = label_map_config.src_label
    dst_label = label_map_config.dst_label
    color_map = label_map_config.color_map
    colors, clsnames = get_colors_and_class_names_for_lane_parsing(color_map)
    anno_t = ImageFailGenerateLabelMapAnnoTs(
        output_dir="tmp_out",
        src_label=src_label,
        dst_label=dst_label,
        colors=colors,
        clsnames=clsnames,
        reuse_prelabel=False,
        anno_to_contours_fn=image_fail_parsing_anno_to_contours_fn,
    )

    _output = anno_t(input)
    assert isinstance(_output, tuple)
    label = cv2.imread("tmp_out/raw_img_label.png")
    fusion = cv2.imread("tmp_out/raw_img_fusion.jpg")
    label_gt = cv2.imread(os.path.join(root, "label.png"))
    fusion_gt = cv2.imread(os.path.join(root, "fusion.jpg"))
    print((label - label_gt).max())
    print((label - label_gt).sum())
    assert np.array_equal(label, label_gt)
    assert np.array_equal(fusion, fusion_gt)
