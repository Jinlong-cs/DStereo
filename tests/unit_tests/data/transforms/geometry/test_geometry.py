import os

import numpy as np
import pytest

from hat.data.datasets.real3d_dataset import Real3DDataset
from hat.registry import build_from_registry
from tests import (
    AUTO_JENKINS_TEST_BUCKET_EXISTS,
    AUTO_JENKINS_TEST_BUCKET_PATH,
)


@pytest.mark.skipif(
    not AUTO_JENKINS_TEST_BUCKET_EXISTS,
    reason="requiring AUTO_JENKINS_TEST bucket",
)  # noqa: E501
@pytest.mark.parametrize(
    [
        "rec_paths",
    ],
    [
        pytest.param(
            [
                os.path.join(
                    AUTO_JENKINS_TEST_BUCKET_PATH,
                    "test_virtual_camera/fisheye/single_view_real3d_test.rec",  # noqa: E501
                ),
            ]
        ),
    ],
)
def test_real3d_dataset_rec(rec_paths):
    category_id_dict = Real3DDataset.get_category_id_dict(3)
    camera_names = (
        "__fisheye_front__",
        "__fisheye_back__",
        "__fisheye_left__",
        "__fisheye_right__",
    )
    cfg = dict(
        type="Real3DDatasetRec",
        paths=rec_paths,
        num_classes=3,
        transforms=[
            dict(
                type="CreateVirtualCamera",
                task_type="real3d",
                source_cam="Cylindrical",
                virtual_cam="Cylindrical",
                image_size=[704 * 2, 576 * 2],  # W * H
                cameraMatrix=np.array(
                    [[220 * 2, 0, 352 * 2], [0, 220 * 2, 288 * 2], [0, 0, 1]]
                ),
                is_virtual=False,
                calib_json=None,
            ),
            dict(
                type="CameraParamResize",
                scale_x=0.5,  # 1/2 resize
                scale_y=0.5,  # 1/2 resize
                img_scale=[
                    (576, 704),
                ],
            ),
            dict(
                type="CameraPresetParamCrop",
                crop_top=0,
                crop_bottom=0,
                crop_left=0,
                crop_right=0,
            ),
            dict(
                type="CameraParamHorizontalFlip2D",
                flip_ratio=0.5,
            ),
            dict(
                type="FCOS3DCameraPlugin",
                is_train=True,
                num_classes=3,
                category_id_dict=category_id_dict,
                bbox_ct=True,
                rescale=False,
                max_depth=30,
                depth_type="Cylindrical",
                camera_names=camera_names,
                cache_static_map=True,
                is_warp_image=True,  # default True
                keep_org_img=False,
            ),
            dict(type="ImageToTensor", from_numpy=True),
            dict(
                type="ConvertLayout",
                hwc2chw=True,
                keys=["img"],
            ),
        ],
        num_dist=4,
    )

    # test function construction
    dataset = build_from_registry(cfg)

    # test function __len__
    length_dataset = len(dataset)
    assert length_dataset == 6

    # test function __getitem__
    check_keys = [
        "source_cam",
        "virtual_cam",
        "gt_bboxes",
        "gt_bboxes_3d",
        "gt_classes_3d",
        "depths",
        "depth_type",
        "imgs",
        "image_height",
        "image_width",
        "image_name",
    ]
    for key in check_keys:
        assert key in dataset[0]
    assert dataset[0]["image_height"] == 576
    assert dataset[0]["image_width"] == 704
    assert dataset[0]["virtual_cam"].fx == 220
    assert dataset[0]["virtual_cam"].fy == 220


if __name__ == "__main__":
    pytest.main(["-s", __file__])
