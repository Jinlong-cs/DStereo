from typing import List, Optional

import numpy as np

from hat.data.datasets.legacy_densebox import LegacyDenseBoxImageRecordDataset
from hat.registry import OBJECT_REGISTRY

__all__ = ["TrafficSignDenseBoxImageRecordDataset"]


@OBJECT_REGISTRY.register
class TrafficSignDenseBoxImageRecordDataset(LegacyDenseBoxImageRecordDataset):
    """A dataset that can read densebox image record for traffic sign.

    Args:
        rec_path: Image record path.
        anno_path: Annotation path.
        transforms : List of transform.
        read_only: Whether output raw content, by default False.
        with_img_buf: Whether the raw content with img buf.
        with_seg_label: Whether the raw content with segmentation label.
        to_rgb: Whether output image in rgb color, by default True.
        rec_idx_file_path: Image record index file path.
            if None use:rec_path + '.idx'.
        seg_label_dtype: The output data type of segmentation label.
        task_type: The task type.
        show_image_info: Whether show image info.
    """

    def __init__(
        self,
        rec_path: str,
        anno_path: str,
        transforms: Optional[List] = None,
        read_only: bool = False,
        with_img_buf: bool = False,
        with_seg_label: bool = False,
        to_rgb: bool = True,
        rec_idx_file_path: str = None,
        seg_label_dtype: type = np.uint8,
        task_type="detection",
        show_image_info=False,
    ):
        super().__init__(
            rec_path,
            anno_path,
            read_only,
            with_img_buf,
            with_seg_label,
            to_rgb,
            rec_idx_file_path,
            seg_label_dtype,
        )
        self.transforms = transforms
        self.task_type = task_type
        self.show_image_info = show_image_info
        self.color_space = "bgr" if not to_rgb else "rgb"

    def __getitem__(self, idx):

        res = super().__getitem__(idx)

        if self.with_img_buf:
            img, img_buf, anno = res
        else:
            img, anno = res

        data = {"img": img, "anno": anno, "color_space": self.color_space}

        if self.transforms is not None:
            results = self.transforms(data)

        return results
