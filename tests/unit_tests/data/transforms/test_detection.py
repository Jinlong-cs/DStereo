# Copyright (c) Horizon Robotics. All rights reserved.

from copy import deepcopy

import numpy as np
import pytest
import torch

from hat.data.transforms.detection import (
    AlbuImageOnlyTransform,
    AugmentHSV,
    Batchify,
    ColorJitter,
    DetYOLOv5MixUp,
    DetYOLOXMixUp,
    FixedCrop,
    HueSaturationValue,
    MeanBlur,
    MedianBlur,
    MinIoURandomCrop,
    Mosaic,
    Normalize,
    Pad,
    PadTensorListToBatch,
    PlainCopyPaste,
    RandomBrightnessContrast,
    RandomCrop,
    RandomExpand,
    RandomFlip,
    RandomResizedCrop,
    RandomSizeCrop,
    Resize,
    RGBShift,
    ShiftScaleRotate,
    ToFasterRCNNData,
    ToMultiTaskFasterRCNNData,
    ToTensor,
)
from hat.utils.package_helper import check_packages_available
from tests.utils import (
    check,
    check_range,
    check_shape,
    check_type,
    gen_fake_transforms_data,
)

try:
    import albumentations
except ImportError:
    albumentations = None


def test_resize():
    def common_check(data):
        assert "img" in data and "img_shape" in data and "pad_shape" in data
        assert data["img_shape"] == data["img"].shape
        assert data["pad_shape"] == data["img"].shape

    # single scale, keep_ratio=True
    data = gen_fake_transforms_data(300, 400, layout="hwc")
    img_scale = (200, 300)  # h,w
    resizer = Resize(img_scale, keep_ratio=True)
    resize_data = resizer(data)
    common_check(resize_data)
    # single scale, keep_ratio=False
    data = gen_fake_transforms_data(300, 400, layout="hwc")
    resizer = Resize(img_scale, keep_ratio=False)
    resize_data = resizer(data)
    common_check(resize_data)
    assert resize_data["img"].shape[:2] == img_scale
    # multiscale mode 1
    data = gen_fake_transforms_data(300, 400, layout="hwc")
    resizer = Resize(
        img_scale, multiscale_mode="range", ratio_range=(0.5, 2.0)
    )
    resize_data = resizer(data)
    common_check(resize_data)
    # multiscale mode 2
    data = gen_fake_transforms_data(300, 400, layout="hwc")
    img_scale = ((100, 200), (400, 500))
    resizer = Resize(img_scale, multiscale_mode="range", keep_ratio=False)
    resize_data = resizer(data)
    common_check(resize_data)
    assert min(resize_data["img_shape"][:2]) >= 100
    assert max(resize_data["img_shape"][:2]) <= 500
    # multiscale mode 3
    data = gen_fake_transforms_data(300, 400, layout="hwc")
    img_scale = ((100, 200), (300, 400), (400, 500))
    resizer = Resize(img_scale, multiscale_mode="value", keep_ratio=False)
    resize_data = resizer(data)
    common_check(resize_data)
    assert resize_data["img_shape"][:2] in img_scale

    data = gen_fake_transforms_data(300, 400, layout="hwc")
    img_scale = (200, 301)  # h,w
    data["gt_bboxes"] = np.array([[-10, -10, 20, 20], [20, -20, 40, 40]])
    data["gt_classes"] = np.array([1, 0])
    resizer = Resize(
        img_scale, keep_ratio=False, divisor=2, rm_neg_coords=False
    )
    resize_data = resizer(data)
    common_check(resize_data)
    assert resize_data["img_shape"][:2] == (200, 300)
    assert resize_data["gt_bboxes"].min() < 0


@pytest.mark.parametrize(
    ["px", "py", "c"],
    [
        pytest.param(0, 1, 3),
        pytest.param(1, 0, 3),
        pytest.param(0, 1, 6),
        pytest.param(1, 0, 6),
    ],
)
def test_random_flip(px, py, c):
    data = gen_fake_transforms_data(300, 400, layout="hwc", c=c)
    src_data = data.copy()
    fliper = RandomFlip(px, py)
    flip_data = fliper(data).copy()
    flip_flip_data = fliper(flip_data).copy()
    assert (flip_flip_data["img"] == src_data["img"]).all()
    assert (flip_flip_data["gt_bboxes"] == src_data["gt_bboxes"]).all()
    assert (flip_flip_data["gt_seg"] == src_data["gt_seg"]).all()
    assert (flip_flip_data["gt_flow"] == src_data["gt_flow"]).all()
    assert (flip_flip_data["gt_mask"] == src_data["gt_mask"]).all()
    assert (flip_flip_data["gt_ldmk"] == src_data["gt_ldmk"]).all()
    assert (flip_flip_data["gt_img"] == src_data["gt_img"]).all()
    assert flip_flip_data["cur_pattern"] == src_data["raw_pattern"]
    assert (
        flip_flip_data["gt_eye_cls_labels"] == src_data["gt_eye_cls_labels"]
    ).all()
    assert (flip_flip_data["gt_ldmk_attr"] == src_data["gt_ldmk_attr"]).all()
    assert (flip_flip_data["eye_status"] == src_data["eye_status"]).all()
    assert (
        flip_flip_data["eye_vis_labels"] == src_data["eye_vis_labels"]
    ).all()
    for flip_flip_gt_line, gt_line in zip(
        flip_flip_data["gt_lines"], src_data["gt_lines"]
    ):
        assert (flip_flip_gt_line == gt_line).all()
    for flip_flip_gt_polygon, gt_polygon in zip(
        flip_flip_data["gt_polygons"], src_data["gt_polygons"]
    ):
        assert (flip_flip_gt_polygon == gt_polygon).all()
    assert (
        flip_flip_data["gt_pupil_ellipse_param"]
        == src_data["gt_pupil_ellipse_param"]
    ).all()


@pytest.mark.parametrize(
    ["size", "divisor", "pad_val", "seg_pad_val", "target_h", "target_w"],
    [
        pytest.param((500, 600), 1, 100, 100, 500, 600),
        pytest.param((500, 600), 64, 100, 100, 512, 640),
    ],
)
def test_pad(size, divisor, pad_val, seg_pad_val, target_h, target_w):
    data = gen_fake_transforms_data(300, 400, layout="hwc", fill_value=100)
    target_data = gen_fake_transforms_data(
        target_w, target_h, layout="hwc", fill_value=100
    )
    padder = Pad(size, divisor, pad_val, seg_pad_val)
    padded_data = padder(data).copy()
    assert (padded_data["img"] == target_data["img"]).all()
    assert (padded_data["gt_seg"] == target_data["gt_seg"]).all()


@pytest.mark.parametrize(
    ["mean", "std", "return_tensor", "raw_norm"],
    [
        pytest.param(128, 128, True, False),
        pytest.param([128, 128, 128], [128, 128, 128], True, False),
        pytest.param([128, 128, 128], 128, False, False),
        pytest.param(128, 128, True, True),
    ],
)
def test_normalize(mean, std, return_tensor, raw_norm):
    data = gen_fake_transforms_data(
        300, 400, layout="hwc", return_tensor=return_tensor
    )
    if raw_norm:
        data["bit_nums_lower"] = [0]
    normalizer = Normalize(mean, std, raw_norm)
    normalized_data = normalizer(data).copy()
    if return_tensor:
        assert isinstance(normalized_data["img"], torch.Tensor)
    else:
        assert isinstance(normalized_data["img"], np.ndarray)


@pytest.mark.parametrize(
    [
        "size",
        "c",
        "layout",
        "crop_around_gt",
        "repeat_times",
        "inclusion_rate",
    ],
    [
        pytest.param((500, 600), 3, "hwc", False, 1, 0.0),
        pytest.param((200, 300), 3, "hwc", False, 1, 0.0),
        pytest.param((500, 600), 6, "hwc", False, 1, 0.0),
        pytest.param((200, 300), 6, "hwc", False, 1, 0.0),
        pytest.param((500, 600), 3, "chw", False, 1, 0.0),
        pytest.param((200, 300), 3, "chw", False, 1, 0.0),
        pytest.param((500, 600), 6, "chw", False, 1, 0.0),
        pytest.param((200, 300), 6, "chw", False, 1, 0.0),
        pytest.param((500, 600), 3, "hwc", True, 1, 0.0),
        pytest.param((200, 300), 3, "hwc", True, 10, 0.0),
        pytest.param((500, 600), 6, "hwc", True, 20, 0.0),
        pytest.param((200, 300), 6, "hwc", True, 30, 0.0),
        pytest.param((500, 600), 3, "chw", True, 1, 0.1),
        pytest.param((200, 300), 3, "chw", True, 10, 0.2),
        pytest.param((500, 600), 6, "chw", True, 20, 0.3),
        pytest.param((200, 300), 6, "chw", True, 30, 1.0),
    ],
)
def test_random_crop(
    size, c, layout, crop_around_gt, repeat_times, inclusion_rate
):
    data = gen_fake_transforms_data(300, 400, layout=layout, c=c)
    random_croper = RandomCrop(
        size,
        crop_around_gt=crop_around_gt,
        repeat_times=repeat_times,
        inclusion_rate=inclusion_rate,
    )
    croped_data = random_croper(data).copy()
    if layout == "hwc":
        assert croped_data["img"].shape[0] <= 400
        assert croped_data["img"].shape[1] <= 300
        assert croped_data["gt_flow"].shape[0] <= 400
        assert croped_data["gt_flow"].shape[1] <= 300
        assert croped_data["gt_disp"].shape[0] <= 400
        assert croped_data["gt_disp"].shape[1] <= 300
    else:
        assert croped_data["img"].shape[1] <= 400
        assert croped_data["img"].shape[2] <= 300
        assert croped_data["gt_flow"].shape[1] <= 400
        assert croped_data["gt_flow"].shape[2] <= 300
        assert croped_data["gt_disp"].shape[0] <= 400
        assert croped_data["gt_disp"].shape[1] <= 300


@pytest.mark.parametrize(
    [
        "origin_size",
        "size",
        "scale_factor",
        "center_shake",
        "layout",
        "repeat_times",
    ],
    [
        pytest.param(
            (3840, 2160), (704, 1472), 0.65, (10, 100, 20, 30), "hwc", 1
        ),
        pytest.param(
            (1280, 720), (704, 1472), 0.5, (10, 100, 20, 30), "hwc", 10
        ),
        pytest.param(
            (1920, 1080), (704, 1472), 0.5, (10, 100, 20, 30), "hwc", 1
        ),
        pytest.param(
            (3824, 2048), (704, 1472), 0.5, (10, 100, 20, 30), "hwc", 1
        ),
        pytest.param(
            (3840, 2160), (704, 1472), 0.65, (10, 100, 20, 30), "chw", 10
        ),
        pytest.param(
            (1280, 720), (704, 1472), 0.5, (10, 100, 20, 30), "chw", 1
        ),
        pytest.param(
            (1920, 1080), (704, 1472), 0.5, (10, 100, 20, 30), "chw", 1
        ),
        pytest.param(
            (3824, 2048), (704, 1472), 0.5, (10, 100, 20, 30), "chw", 10
        ),
    ],
)
def test_random_center_crop(
    origin_size, size, scale_factor, center_shake, layout, repeat_times
):
    data = gen_fake_transforms_data(*origin_size, layout=layout)
    data["scale_factor"] = [scale_factor] * 4
    random_croper = RandomCrop(
        size,
        min_area=-1,
        min_iou=-1,
        center_crop_prob=0.5,
        center_shake=center_shake,
        repeat_times=repeat_times,
    )
    croped_data = random_croper(data).copy()
    img_hw = (
        croped_data["img_shape"][:2]
        if layout == "hwc"
        else croped_data["img_shape"][1:]
    )
    assert img_hw[0] <= size[0] and img_hw[1] <= size[1]


@pytest.mark.parametrize(
    [
        # resize
        "origin_size",
        "max_scale",
        "keep_ratio",
        "multiscale_mode",
    ],
    [
        pytest.param(
            (3840, 2160),
            (1080, 2280),
            True,
            "max_size",
        ),
        pytest.param(
            (1280, 720),
            (1080, 2280),
            True,
            "max_size",
        ),
        pytest.param(
            (1920, 1080),
            (1080, 2280),
            True,
            "max_size",
        ),
        pytest.param(
            (3824, 2048),
            (1080, 2280),
            True,
            "max_size",
        ),
    ],
)
def test_resize_max_size(origin_size, max_scale, keep_ratio, multiscale_mode):
    # check the target resize
    data = gen_fake_transforms_data(*origin_size, layout="hwc")
    resize = Resize(
        max_scale=max_scale,
        keep_ratio=keep_ratio,
        multiscale_mode=multiscale_mode,
    )
    resized_data = resize(data).copy()
    assert resized_data["img"].shape[0] <= max_scale[0]
    assert resized_data["img"].shape[1] <= max_scale[1]


@pytest.mark.parametrize(
    [
        "w",
        "h",
        "c",
        "layout",
        "return_tensor",
        "to_yuv",
        "use_yuv_v2",
        "target_w",
        "target_h",
    ],
    [
        pytest.param(500, 600, 3, "hwc", True, False, False, 500, 600),
        pytest.param(400, 555, 3, "chw", False, True, False, 400, 554),
        pytest.param(400, 555, 3, "chw", False, True, True, 400, 554),
        pytest.param(500, 600, 6, "hwc", True, False, False, 500, 600),
        pytest.param(400, 555, 6, "chw", False, True, False, 400, 554),
        pytest.param(400, 555, 6, "chw", False, True, True, 400, 554),
    ],
)
def test_to_tensor(
    w, h, c, layout, return_tensor, to_yuv, use_yuv_v2, target_w, target_h
):
    data = gen_fake_transforms_data(
        w, h, layout, return_tensor=return_tensor, c=c
    )
    to_tensor = ToTensor(to_yuv)
    tensor_data = to_tensor(data).copy()
    assert tensor_data["layout"] == "chw"
    if to_yuv:
        assert tensor_data["color_space"] == "yuv"
    assert isinstance(tensor_data["img"], torch.Tensor)
    assert isinstance(tensor_data["gt_bboxes"], torch.Tensor)
    assert isinstance(tensor_data["gt_classes"], torch.Tensor)
    assert isinstance(tensor_data["gt_seg"], torch.Tensor)
    assert isinstance(tensor_data["gt_flow"], torch.Tensor)
    assert tensor_data["img"].shape[1:] == (target_h, target_w)
    assert tensor_data["img"].shape[1:] == tensor_data["gt_seg"].shape
    assert tensor_data["img"].shape[1:] == tensor_data["gt_flow"].shape[1:]
    assert (
        tensor_data["img_shape"][1:] == np.array([target_h, target_w])
    ).all()


@pytest.mark.parametrize(
    [
        "w",
        "h",
        "layout",
        "return_tensor",
        "size",
        "divisor",
        "repeat",
        "target_w",
        "target_h",
    ],
    [
        pytest.param(450, 556, "hwc", True, (576, 480), 1, 1, 480, 576),
        pytest.param(520, 440, "chw", False, (575, 478), 32, 2, 576, 480),
    ],
)
def test_batchify(
    w, h, layout, return_tensor, size, divisor, repeat, target_w, target_h
):
    data = gen_fake_transforms_data(w, h, layout, return_tensor=return_tensor)
    batchifyer = Batchify(size, divisor, repeat=repeat)
    collated_data = batchifyer(data).copy()
    # check keys
    assert "img" not in collated_data
    assert "imgs" in collated_data and len(collated_data["imgs"]) == repeat
    # check layout
    assert collated_data["layout"] == layout
    # check shape
    assert (collated_data["pad_shape"] == collated_data["imgs"][0].shape).all()
    h_index, w_index = layout.index("h"), layout.index("w")
    assert collated_data["pad_shape"][w_index] == target_w
    assert collated_data["pad_shape"][h_index] == target_h
    if "gt_seg" in collated_data:
        assert (
            collated_data["pad_shape"][h_index]
            == collated_data["gt_seg"].shape[0]
        )  # noqa
        assert (
            collated_data["pad_shape"][w_index]
            == collated_data["gt_seg"].shape[1]
        )  # noqa


def test_resize_and_crop():
    # front
    data = gen_fake_transforms_data(3840, 2160, layout="hwc")
    img_scale = (540, 960)  # h, w
    resized_data = Resize(img_scale, keep_ratio=True)(data)
    assert resized_data["img"].shape == (540, 960, 3)
    cropped_data = FixedCrop([0, 0, 960, 512])(resized_data)
    assert cropped_data["img"].shape == (512, 960, 3)
    assert cropped_data["before_crop_shape"] == (540, 960, 3)
    assert cropped_data["crop_offset"] == [0, 0, 0, 0]

    # side
    data = gen_fake_transforms_data(2048, 1280, layout="hwc")
    img_scale = (600, 960)  # h, w
    resized_data = Resize(img_scale, keep_ratio=True)(data)
    assert resized_data["img"].shape == (600, 960, 3)
    cropped_data = FixedCrop([0, 88, 960, 512])(resized_data)
    assert cropped_data["img"].shape == (512, 960, 3)
    assert cropped_data["before_crop_shape"] == (600, 960, 3)
    assert cropped_data["crop_offset"] == [0, 88, 0, 88]


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_color_aug():
    data = gen_fake_transforms_data(512, 512, layout="chw")
    data["img"] = torch.from_numpy(data["img"])
    data = ColorJitter()(data)
    assert "img" in data


def test_random_expand():
    data = gen_fake_transforms_data(512, 512, layout="hwc")
    data = RandomExpand()(data)
    assert "img" in data
    assert "gt_bboxes" in data
    assert data["gt_bboxes"].shape[0] == 2


def test_miniou_randomcrop():
    data = gen_fake_transforms_data(512, 512, layout="hwc")
    data = MinIoURandomCrop()(data)
    assert "img" in data
    assert "gt_bboxes" in data
    assert data["gt_bboxes"].shape[1] == 4


def test_hsv_aug():
    data = gen_fake_transforms_data(512, 512, layout="hwc")
    data = AugmentHSV()(data)
    assert "img" in data
    data = gen_fake_transforms_data(512, 512, layout="chw")
    data = AugmentHSV()(data)
    assert "img" in data


def test_inverse_transform():
    data = gen_fake_transforms_data(128, 64, layout="hwc")
    fix_croper = FixedCrop((8, 4, 120, 60))  # (120, 60, 3)
    random_croper = RandomCrop((60, 120), center_crop_prob=0.5)
    resizer = Resize((128, 64), ratio_range=(0.5, 1.5), keep_ratio=True)
    for _ in range(100):
        for op in [fix_croper, random_croper, resizer]:
            gt = op(deepcopy(data))
            gt_bboxes = op.inverse_transform(
                torch.tensor(gt["gt_bboxes"]), "detection", gt
            )
            assert (gt_bboxes.long().numpy() == data["gt_bboxes"]).all()


@pytest.mark.parametrize(
    [
        "max_gt_boxes_num",
        "max_ig_regions_num",
    ],
    [
        pytest.param(3, 3),
        pytest.param(1000, 1000),
    ],
)
def test_to_faster_rcnn_data(max_gt_boxes_num, max_ig_regions_num):
    data = gen_fake_transforms_data(640, 352, layout="hwc")
    data["gt_classes"] += 1
    _gt_bboxes = data["gt_bboxes"]
    _gt_classes = data["gt_classes"]
    data["gt_bboxes"] = np.vstack([_gt_bboxes, _gt_bboxes + 100])
    data["gt_classes"] = np.hstack([_gt_classes, _gt_classes])
    num_gt = (data["gt_classes"] > 0).sum()
    cvter = ToFasterRCNNData(
        max_gt_boxes_num=max_gt_boxes_num,
        max_ig_regions_num=max_ig_regions_num,
    )
    faster_rcnn_data = cvter(data)
    assert len(faster_rcnn_data["gt_boxes"]) == max_gt_boxes_num
    assert len(faster_rcnn_data["ig_regions"]) == max_ig_regions_num
    if num_gt > max_gt_boxes_num:
        assert 0 == faster_rcnn_data["gt_boxes"][0][-1]
        assert 0 == faster_rcnn_data["gt_boxes_num"][0]
    else:
        assert num_gt == faster_rcnn_data["gt_boxes_num"][0]


@pytest.mark.parametrize(
    [
        "max_gt_boxes_num",
        "max_ig_regions_num",
    ],
    [
        pytest.param(3, 3),
        pytest.param(1000, 1000),
    ],
)
def test_to_multi_task_faster_rcnn_data(max_gt_boxes_num, max_ig_regions_num):
    data = gen_fake_transforms_data(640, 352, layout="hwc")
    # set all cls to -1
    data["gt_classes"][...] = -1
    # set det1 task data: cls 1
    data["gt_classes"][0] = 1
    # set det2 task data: cls 2
    data["gt_classes"][1] = 2
    # set ldmk task data: cls 1
    data["gt_ldmk"] = np.random.randint(low=0, high=3, size=(1, 15, 3))

    task_clsidx_map = {"det1": 1, "det2": 2, "ldmk": 1}
    det1_num_gt = (data["gt_classes"] == 1).sum()
    det2_num_gt = (data["gt_classes"] == 2).sum()
    ldmk_num_gt = (data["gt_classes"] == 1).sum()
    cvter = ToMultiTaskFasterRCNNData(
        taskname_clsidx_map=task_clsidx_map,
        max_gt_boxes_num=max_gt_boxes_num,
        max_ig_regions_num=max_ig_regions_num,
    )
    mult_task_data = cvter(data)
    assert (
        ("det1" in mult_task_data.keys())
        and ("det2" in mult_task_data.keys())
        and ("ldmk" in mult_task_data.keys())
    )
    assert det1_num_gt == mult_task_data["det1"]["gt_boxes_num"][0]
    assert det2_num_gt == mult_task_data["det2"]["gt_boxes_num"][0]
    assert ldmk_num_gt == mult_task_data["ldmk"]["gt_boxes_num"][0]


def test_pad_tensor_list_to_batch():
    img1 = torch.rand((3, 100, 200))
    img2 = torch.rand((3, 150, 90))
    img_tensor_list = [img1, img2]
    batch_img = PadTensorListToBatch()({"img": img_tensor_list})
    num_img = len(img_tensor_list)
    max_h = max([img.shape[1] for img in img_tensor_list])
    max_w = max([img.shape[2] for img in img_tensor_list])
    assert batch_img["img"].shape == (num_img, 3, max_h, max_w)


@pytest.mark.parametrize(
    [
        "min_ins_num",
        "cp_prob",
    ],
    [pytest.param(3, 1.0), pytest.param(4, 1.0)],
)
def test_plain_copy_paste(min_ins_num, cp_prob):
    data = gen_fake_transforms_data(704, 576, layout="hwc")
    data["gt_bboxes"][1, :2] += 10
    data2 = data.copy()
    data2["gt_bboxes"] = data["gt_bboxes"] + 10
    plainCP = PlainCopyPaste(min_ins_num, cp_prob)
    data_ = plainCP(data)
    assert (data_["gt_bboxes"] == data["gt_bboxes"]).all()
    assert (data_["gt_classes"] == data["gt_classes"]).all()
    data2_ = plainCP(data2)
    assert len(data2_["gt_bboxes"]) == min_ins_num
    assert len(data2_["gt_classes"]) == min_ins_num


def test_huesaturationvalue():
    data = gen_fake_transforms_data(512, 512, layout="hwc")
    data = HueSaturationValue()(data)
    assert "img" in data


@pytest.mark.parametrize(
    ["p", "ksize"],
    [
        pytest.param(1, 3),
        pytest.param(1, 5),
    ],
)
def test_mean_blur(p, ksize):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    mean_blur = MeanBlur(
        p=p,
        ksize=ksize,
    )
    aug_data = mean_blur(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["p", "ksize"],
    [
        pytest.param(1, 3),
        pytest.param(1, 5),
    ],
)
def test_median_blur(p, ksize):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    median_blur = MedianBlur(
        p=p,
        ksize=ksize,
    )
    aug_data = median_blur(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["r_shift_limit", "g_shift_limit", "b_shift_limit", "p"],
    [
        pytest.param((-20, 20), (-20, 20), (-20, 20), 0.5),
        pytest.param((-10, 20), (-40, 50), (-30, 20), 1.0),
        pytest.param((-20, 30), (-20, 10), (-10, 30), 0.0),
    ],
)
def test_rgb_shift(r_shift_limit, g_shift_limit, b_shift_limit, p):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    rgb_shift = RGBShift(
        r_shift_limit=r_shift_limit,
        g_shift_limit=g_shift_limit,
        b_shift_limit=b_shift_limit,
        p=p,
    )
    aug_data = rgb_shift(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["brightness_limit", "contrast_limit", "brightness_by_max", "p"],
    [
        pytest.param((-0.2, 0.2), (-0.1, 0.1), True, 1.0),
        pytest.param((-0.2, 0.2), (-0.1, 0.1), False, 1.0),
        pytest.param((-0.1, 0.1), (-0.2, 0.2), True, 0.0),
        pytest.param((-0.1, 0.1), (-0.2, 0.2), False, 0.0),
    ],
)
def test_random_brightness_contrast(
    brightness_limit, contrast_limit, brightness_by_max, p
):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    random_brightness_contrast = RandomBrightnessContrast(
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        brightness_by_max=brightness_by_max,
        p=p,
    )
    aug_data = random_brightness_contrast(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["shift_limit", "scale_limit", "rotate_limit", "p"],
    [
        pytest.param((-0.2, 0.2), (-0.1, 0.1), (-40, 40), 1.0),
        pytest.param((-0.2, 0.2), (-0.1, 0.1), (-40, 40), 1.0),
        pytest.param((-0.1, 0.1), (-0.2, 0.2), (-40, 40), 0.0),
        pytest.param((-0.1, 0.1), (-0.2, 0.2), (-40, 40), 0.0),
    ],
)
def test_shift_scale_rotate(shift_limit, scale_limit, rotate_limit, p):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    shift_scale_rotate = ShiftScaleRotate(
        shift_limit=shift_limit,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        p=p,
    )
    aug_data = shift_scale_rotate(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["height", "width", "scale", "ratio", "p"],
    [
        pytest.param(100, 200, (0.8, 1.2), (1.7, 2.7), 1.0),
        pytest.param(400, 300, (0.8, 1.2), (1.7, 2.7), 1.0),
    ],
)
def test_random_resized_crop(height, width, scale, ratio, p):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    random_resized_crop = RandomResizedCrop(
        height=height, width=width, scale=scale, ratio=ratio, p=p
    )
    aug_data = random_resized_crop(data).copy()
    check(aug_data["img"], check_shape, shape=(height, width, 3))
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.skipif(
    albumentations is None, reason="albumentations is required"
)
@pytest.mark.parametrize(
    [
        "albu_params",
    ],
    [
        pytest.param(
            [
                dict(
                    name="RandomBrightnessContrast",
                    p=0.3,
                ),
                dict(
                    name="ToGray",
                    p=0.2,
                ),
            ]
        ),
    ],
)
def test_albu_image_only_transform(albu_params):
    data = gen_fake_transforms_data(640, 352, layout="hwc")
    gt_bboxes_ori = deepcopy(data["gt_bboxes"])
    cvter = AlbuImageOnlyTransform(albu_params)
    data = cvter(data)
    gt_bboxes_after = data["gt_bboxes"]
    assert "img" in data
    assert np.allclose(gt_bboxes_ori, gt_bboxes_after)


@pytest.mark.parametrize(
    ["w", "h", "c", "layout", "min_size", "max_size"],
    [
        pytest.param(500, 600, 3, "hwc", 300, 600),
        pytest.param(300, 400, 3, "chw", 100, 500),
    ],
)
def test_random_size_crop(w, h, c, layout, min_size, max_size):
    data = gen_fake_transforms_data(w, h, layout=layout, c=c)
    random_croper = RandomSizeCrop(min_size, max_size)
    croped_data = random_croper(data).copy()
    if layout == "hwc":
        assert croped_data["img"].shape[0] <= max_size
        assert croped_data["img"].shape[1] <= max_size
        assert croped_data["img"].shape[0] >= min_size
        assert croped_data["img"].shape[1] >= min_size
    else:
        assert croped_data["img"].shape[1] <= max_size
        assert croped_data["img"].shape[2] <= max_size
        assert croped_data["img"].shape[1] >= min_size
        assert croped_data["img"].shape[2] >= min_size


@pytest.mark.parametrize(
    ["img_size_cache", "img_size"],
    [
        pytest.param((500, 900), (1400, 700)),
        pytest.param((704, 576), (512, 800)),
    ],
)
def test_yolov5_mixup(img_size_cache, img_size):
    data_cache = gen_fake_transforms_data(*img_size_cache, layout="hwc")
    num_gt_cache = len(data_cache["gt_bboxes"])
    data = gen_fake_transforms_data(*img_size, layout="hwc")
    num_gt = len(data["gt_bboxes"])
    MixUp = DetYOLOv5MixUp(p=1)
    # run mixup only when len of cached data > 4
    for _ in range(4):
        data_aug = MixUp(deepcopy(data_cache))
        assert (data_aug["gt_bboxes"] == data_cache["gt_bboxes"]).all()
        assert (data_aug["gt_classes"] == data_cache["gt_classes"]).all()
    data_aug = MixUp(deepcopy(data))
    assert len(data_aug["gt_bboxes"]) == num_gt_cache + num_gt
    assert data_aug["img"].shape == data["img"].shape


@pytest.mark.parametrize(
    ["img_size_cache", "img_size"],
    [
        pytest.param((500, 900), (1400, 700)),
        pytest.param((704, 576), (512, 800)),
    ],
)
def test_yolox_mixup(img_size_cache, img_size):
    data_cache = gen_fake_transforms_data(*img_size_cache, layout="hwc")
    num_gt_cache = len(data_cache["gt_bboxes"])
    data = gen_fake_transforms_data(*img_size, layout="hwc")
    num_gt = len(data["gt_bboxes"])
    MixUp = DetYOLOXMixUp(p=1)
    # run mixup only when len of cached data > 4
    for _ in range(4):
        data_aug = MixUp(deepcopy(data_cache))
        assert (data_aug["gt_bboxes"] == data_cache["gt_bboxes"]).all()
        assert (data_aug["gt_classes"] == data_cache["gt_classes"]).all()
    data_aug = MixUp(deepcopy(data))
    assert len(data_aug["gt_bboxes"]) >= num_gt
    assert len(data_aug["gt_bboxes"]) <= num_gt + num_gt_cache
    assert data_aug["img"].shape == data["img"].shape


@pytest.mark.parametrize(
    ["image_size", "degrees", "translate", "scale", "shear"],
    [
        pytest.param(100, 10, 0.1, 0.1, 10),
        pytest.param(200, 20, 0.2, 0.2, 5),
    ],
)
def test_mosaic(image_size, degrees, translate, scale, shear):
    data = [gen_fake_transforms_data(640, 420, layout="hwc") for i in range(5)]
    mosaic = Mosaic(image_size, degrees, translate, scale, shear)
    data_aug = mosaic(data)
    assert "img" in data_aug
    assert data_aug["img"].shape == (image_size, image_size, 3)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
