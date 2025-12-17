import numpy as np

from hat.core.data_struct.img_structures import (
    BEVObject,
    ImgCls,
    ImgObjDet,
    ImgObjDet3D,
    ImgSemSeg,
)

img = np.random.rand(3, 64, 64)
img_id = 0
layout = "chw"
color_space = "rgb"
img_width = img.shape[2]
img_height = img.shape[1]


def test_ImgCls():
    cls_label = 0
    ImgCls(
        img=img,
        img_id=img_id,
        layout=layout,
        color_space=color_space,
        img_width=img_width,
        img_height=img_height,
        cls_label=cls_label,
    )


gt_bboxes = np.random.rand(6, 5)
ig_regions = np.random.rand(5, 5)
parent_gt_bboxes = np.random.rand(3, 5)
parent_ig_regions = np.random.rand(3, 5)


def test_ImgObjDet():
    ImgObjDet(
        img=img,
        img_id=img_id,
        layout=layout,
        color_space=color_space,
        img_width=img_width,
        img_height=img_height,
        gt_bboxes=gt_bboxes,
        ig_regions=ig_regions,
        parent_gt_bboxes=parent_gt_bboxes,
        parent_ig_regions=parent_ig_regions,
    )


dim = np.random.rand(6, 3)
location = np.random.rand(6, 3)
rotation_y = np.random.rand(6)
distCoeffs = np.random.rand(6, 5)
alpha = np.random.rand(6)
location_offset = np.random.rand(6, 3)


def test_ImgObjDet3D():
    ImgObjDet3D(
        img=img,
        img_id=img_id,
        layout=layout,
        color_space=color_space,
        img_width=img_width,
        img_height=img_height,
        gt_bboxes=gt_bboxes,
        ig_regions=ig_regions,
        dim=dim,
        location=location,
        rotation_y=rotation_y,
        distCoeffs=distCoeffs,
        alpha=alpha,
        location_offset=location_offset,
    )


gt_seg = np.random.rand(64, 64)
gt_seg_stride = 1


def test_ImgSemSeg():
    ImgSemSeg(
        img=img,
        img_id=img_id,
        layout=layout,
        color_space=color_space,
        img_width=img_width,
        img_height=img_height,
        gt_seg=gt_seg,
        gt_seg_stride=gt_seg_stride,
    )


frames = [[]] * 3
homography = np.random.rand(5, 3, 3)
front_mask = np.random.rand(5, 64, 64)
intrinsics = np.random.rand(5, 3, 3)
distortcoef = np.random.rand(5)


def test_BEVObject():
    BEVObject(
        frames=frames,
        homography=homography,
        front_mask=front_mask,
        intrinsics=intrinsics,
        distortcoef=distortcoef,
    )
