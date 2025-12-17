import argparse
import copy
import logging
import os

import cv2
import numpy as np

from hat.utils.logger import init_logger
from projects.halo.cv.tools.landmark.hand_landmark_packer import (
    HandLdmk3DPacker,
)
from projects.halo.cv.tools.landmark.utils import LandmarkVis
from projects.halo.cv.tools.packer import norm_bbox

logger = logging.getLogger(__name__)


class FaceLdmk2DPacker(HandLdmk3DPacker):
    """Packer for face landmark.

    Inherit from HandLdmk3DPacker, the parameters and functions are the same.
    The only difference is that this class is used to pack 2d face landmark.
    """

    def _pack_anno(self, idx):
        anno = self.annos[idx]
        img_path = anno[0]
        img = cv2.imread(img_path)
        bbox = list(map(float, anno[1:5]))
        ldmk = list(map(float, anno[5 : 5 + self.num_ldmk * 2]))
        rects, expand = norm_bbox(
            copy.deepcopy(bbox), self.norm_method, self.expand_ratio, img.shape
        )
        rects = [rects[i] + expand[i] for i in range(4)]
        gt_ldmk = self._process_ldmk(ldmk, rects)
        gt_bboxes = self._process_bbox(bbox, rects)
        ldmk_attr = list(
            map(float, anno[5 + 2 * self.num_ldmk : 5 + 3 * self.num_ldmk])
        )
        head_pose = anno[-3:]
        pack_data = {
            "gt_ldmk": gt_ldmk,
            "ldmk_attr": ldmk_attr,
            "gt_bboxes": gt_bboxes,
            "head_pose": head_pose,
        }
        return pack_data

    def vis_anno(self, **kwargs):
        idx = kwargs["idx"]
        img = kwargs["img"]
        anno_data = self._pack_anno(idx)
        gt_ldmk = np.array(anno_data["gt_ldmk"]).round().astype(np.int32)
        gt_bboxes = anno_data["gt_bboxes"]
        img = LandmarkVis.vis_landmark(
            img=img, ldmk=gt_ldmk, num_ldmk=self.num_ldmk
        )
        img = LandmarkVis.vis_boundary(
            img=img, ldmk=gt_ldmk, num_ldmk=self.num_ldmk, ldmk_type="face68"
        )
        img = LandmarkVis.vis_bbox(img, gt_bboxes)
        cv2.imwrite(f"{self.vis_dir}/{idx:0>8d}.jpg", img)


def parse_args():
    parser = argparse.ArgumentParser(description="Pack hand ldmk 3d dataset.")
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
        "--num-ldmk",
        default=68,
        help="The number of landmark, default is 68 for face.",
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
    init_logger("work_dirs/hat_logss/landmark_packer")
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
        packer = FaceLdmk2DPacker(
            anno_path=args.anno_path,
            src_type=s_type,
            save_dir=pack_path,
            num_ldmk=int(args.num_ldmk),
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
