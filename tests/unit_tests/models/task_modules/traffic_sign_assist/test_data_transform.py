import os

import pytest

from hat.registry import build_from_registry
from tests import HAT_BUCKET_PATH, HAT_BUCKET_URL_PATH

try:
    import albumentations
except ImportError:
    albumentations = None
try:
    import hatbc
except ImportError:
    hatbc = None


@pytest.mark.skipif(
    not HAT_BUCKET_URL_PATH, reason="HDLTAlgorithm is required"
)  # noqa
@pytest.mark.skipif(
    albumentations is None, reason="albumentations is required"
)
@pytest.mark.skipif(hatbc is None, reason="hatbc is required")
@pytest.mark.parametrize(
    ["anno_file", "rec_file"],
    [
        pytest.param(
            "unit_test_data/mono_traffic_sign_assist_2pe/train.anno.pb_rec",  # noqa
            "unit_test_data/mono_traffic_sign_assist_2pe/train.rec",  # noqa
        )
    ],
)
def test_traffic_sign_2pe_dataset(anno_file, rec_file):
    rec_path = os.path.join(HAT_BUCKET_PATH, rec_file)
    anno_path = os.path.join(HAT_BUCKET_PATH, anno_file)
    input_hw = (128, 128)
    norm_len = 110
    norm_method = "height"
    config = dict(
        type="DenseboxDataset2PETrafficSign",
        data_path=rec_path,
        anno_path=anno_path,
        task_type="detection",
        class_id=[1, 2],
        category={1: 1, 2: 2},
        to_rgb=True,
        ignore_hard=True,
        use_ignore=True,
        rand_sampling_bbox=True,
        roi_lt_id=0,
        roi_rb_id=2,
        gt_lt_id=10,
        gt_rb_id=12,
        image_processing_backend="opencv",
        transforms=[
            dict(
                type="RoiTransformer",
                roi_crop_parm=dict(
                    norm_len=norm_len,
                    norm_method=norm_method,
                    output_wh=input_hw[::-1],
                    input_wh=None,
                    min_crop_scale=0.9,
                    max_crop_scale=1.1,
                    max_coord_jitter_ratio=0.05,
                    img_min_scale=0.1,
                    img_max_scale=10,
                    padd_val=0,
                    random_roi_ratio=0.0,
                    restrict_roi_in_center=False,
                    flip_ratio=0,
                ),
                img_crop_parm=dict(
                    target_wh=input_hw[::-1],
                    inter_method=10,
                    use_pyramid=True,
                    pyramid_min_step=0.7,
                    pyramid_max_step=0.8,
                    pixel_center_aligned=False,
                ),
                bbox_ts_parm=dict(
                    clip=True,
                    min_valid_area=10,
                    min_valid_clip_area_ratio=0.02,
                    min_edge_size=8,
                    label_type="detection",
                ),
            ),
            dict(
                type="TrafficSignDetectionLableTs",
                pad_det_data=True,
                max_gt_boxes_num=100,
                max_ig_regions_num=100,
                regroup_gt_bboxes=True,
                gt_boxes_key="gt_bboxes",
                ig_regions_key="ig_bboxes",
                gt_classes_key="gt_classes",
            ),
            dict(type="ToTensor", to_yuv=False),
            dict(
                type="RenameKeys",
                keys=["gt_bboxes|gt_boxes", "ig_bboxes|ig_regions"],
            ),
        ],
    )
    dataset = build_from_registry(config)
    for data in dataset:
        assert isinstance(data, list)
        assert "img_name" in data[0]
        assert "img_height" in data[0]
        assert "img_width" in data[0]
        assert "img_id" in data[0]
        assert "img" in data[0]
        assert "color_space" in data[0]
        assert "layout" in data[0]
        assert "img_shape" in data[0]
        break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
