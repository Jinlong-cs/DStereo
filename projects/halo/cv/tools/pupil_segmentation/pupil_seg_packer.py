import argparse
import copy
import logging
import os
from typing import List, Optional

import cv2

from hat.utils.logger import init_logger
from projects.halo.cv.tools.landmark.utils import LandmarkVis
from projects.halo.cv.tools.packer import BasePacker, norm_bbox

logger = logging.getLogger(__name__)


class PupilSegPacker(BasePacker):
    """Lmdb packer for pupil segmentation.

    Crop image without resizing.
    Visualize cropped image and annotation in `vis_anno`.

    Args:
        anno_path: annotaion files.
        src_type: "image" or "anno".
        save_dir: lmdb saving path.
        crop_img: crop image according to expanded bbox. Defaults to True.
        expand_ratio: bbox expand ratio. Defaults to 2.0.
        norm_method: bbox expand method. Only `longside ratio` and
            `longsie_square` are supported. Defaults to "longside_square".
        num_workers: Num workers for reading data using multiprocessing.
            Defualts to 8.
        pack_type: The file type for packing. Defaults to 8.
        num_samples: the number of samples you want to pack. Pack all samples
            if num_samples is None. Defaults to None.
    """

    def __init__(
        self,
        anno_path: str,
        src_type: str,
        save_dir: str,
        crop_img: Optional[bool] = True,
        expand_ratio: Optional[float] = 2.0,
        norm_method: Optional[str] = "longside_square",
        num_workers: Optional[int] = 8,
        pack_type: Optional[str] = "lmdb",
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        super(PupilSegPacker, self).__init__(
            anno_path,
            src_type,
            save_dir,
            crop_img,
            expand_ratio,
            norm_method,
            num_workers,
            pack_type,
            num_samples,
            **kwargs,
        )

    def _process_bbox(
        self, gt_bboxes: List[int], rects: List[int]
    ) -> List[int]:
        gt_bboxes[0] -= rects[0]
        gt_bboxes[1] -= rects[1]
        gt_bboxes[2] -= rects[0]
        gt_bboxes[3] -= rects[1]
        return gt_bboxes

    def _pack_anno(self, idx):
        anno = self.annos[idx]
        img_path = anno[0]
        img = cv2.imread(img_path)
        bbox = list(map(float, anno[1:5]))
        pupil_ellipse_param = list(map(float, anno[5:10]))

        rects, expand = norm_bbox(
            copy.deepcopy(bbox), self.norm_method, self.expand_ratio, img.shape
        )
        rects = [rects[i] + expand[i] for i in range(4)]
        gt_bboxes = self._process_bbox(bbox, rects)
        pupil_ellipse_param[0] -= rects[0]
        pupil_ellipse_param[1] -= rects[1]
        pack_data = {
            "gt_bboxes": gt_bboxes,
            "gt_pupil_ellipse_param": pupil_ellipse_param,
        }
        return pack_data

    def vis_anno(self, **kwargs):
        idx = kwargs["idx"]
        img = kwargs["img"]
        anno_data = self._pack_anno(idx)
        gt_bboxes = anno_data["gt_bboxes"]
        pupil_ellipse_param = anno_data["gt_pupil_ellipse_param"]
        img = LandmarkVis.vis_bbox(img, gt_bboxes)
        cv2.ellipse(
            img,
            (
                int(pupil_ellipse_param[0] + 0.5),
                int(pupil_ellipse_param[1] + 0.5),
            ),
            (
                int(pupil_ellipse_param[2] / 2 + 0.5),
                int(pupil_ellipse_param[3] / 2 + 0.5),
            ),
            int(pupil_ellipse_param[-1] + 0.5),
            0,
            360,
            (0, 0, 255),
            -1,
        )
        cv2.imwrite(f"{self.vis_dir}/{idx:0>8d}.jpg", img)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack pupil segmentation dataset."
    )
    parser.add_argument(
        "--anno-path",
        required=True,
        help="The path annotations file.",
    )
    parser.add_argument(
        "--src-type",
        required=True,
        help="src_type must be `image` or `anno`.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="The directory for result of packer.",
    )
    parser.add_argument(
        "--expand-ratio",
        default=2.0,
        help="bbox expand ratio.",
    )
    parser.add_argument(
        "--norm-method",
        default="longside_square",
        help="bbox expand method, `longside ratio` or `longsie_square`",
    )
    parser.add_argument(
        "--pack-type",
        default="lmdb",
        help="The target pack type for result of packer",
    )
    parser.add_argument(
        "--num-workers",
        default=16,
        help="The number of workers to load image.",
    )
    parser.add_argument(
        "--num-samples",
        default=None,
        help="The number of samples.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    init_logger("work_dirs/hat_logss/pupil_segmentation_packer")
    logger.info(args)
    src_type = args.src_type.split(",")
    if not isinstance(src_type, list):
        src_type = [src_type]
    for s_type in src_type:
        assert s_type in ["image", "anno"]
    for s_type in src_type:
        logger.info(f"=================src_type: {s_type}==================")
        pack_path = os.path.join(
            args.save_dir,
            "%s_%s" % (s_type, args.pack_type),
        )
        packer = PupilSegPacker(
            anno_path=args.anno_path,
            src_type=s_type,
            save_dir=pack_path,
            crop_img=True,
            expand_ratio=float(args.expand_ratio),
            norm_method=args.norm_method,
            num_workers=int(args.num_workers),
            pack_type=args.pack_type,
            num_samples=None
            if args.num_samples is None
            else int(args.num_samples),
        )
        packer()
        del packer
