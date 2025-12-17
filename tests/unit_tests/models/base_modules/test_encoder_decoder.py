import pytest
import torch

from hat.models.base_modules.bbox_decoder import XYWHBBoxDecoder
from hat.models.base_modules.label_encoder import (
    MatchLabelFlankEncoder,
    MatchLabelGroundLineEncoder,
    OneHotClassEncoder,
    RCNN3DLabelFromMatch,
    RCNNBinDetLabelFromMatch,
    RCNNKPSLabelFromMatch,
    RCNNMultiBinDetLabelFromMatch,
    XYWHBBoxEncoder,
)
from hat.models.task_modules.roi_modules.flank_point_decoder import (
    FlankPointDecoder,
)


@pytest.mark.parametrize(
    ["class_agnostic_neg", "exclude_background"],
    [
        pytest.param(False, False),
        pytest.param(False, True),
        pytest.param(True, False),
        pytest.param(True, True),
    ],
)
def test_one_hot_class_encoder(class_agnostic_neg, exclude_background):

    batch_size = 100
    num_classes = 5

    encoder = OneHotClassEncoder(
        num_classes=num_classes,
        class_agnostic_neg=class_agnostic_neg,
        exclude_background=exclude_background,
    )

    # generate fake data
    label = torch.randint(-num_classes + 1, num_classes, (batch_size,))
    encoded = encoder(label)

    # 1. check shape
    target_dim = num_classes - exclude_background
    assert encoded.shape[0] == batch_size and encoded.shape[1] == target_dim

    # 2. check content
    # positive part
    pos_mask = label > 0
    pos_encoded = encoded[pos_mask]
    pos_label = label[pos_mask]
    rr = torch.arange(pos_mask.sum())
    assert torch.all(pos_encoded[rr, pos_label - int(exclude_background)] == 1)

    # zero part
    if not exclude_background:
        z_encoded = encoded[label == 0]
        assert torch.all(z_encoded[:, 0] == 1)

    # negative part
    neg_mask = label < 0
    neg_encoded = encoded[neg_mask]
    neg_label = -label[neg_mask]
    if class_agnostic_neg:
        assert torch.all(neg_encoded == -1)
    else:
        rr = torch.arange(neg_mask.sum())
        assert torch.all(
            neg_encoded[rr, neg_label - int(exclude_background)] == -1
        )


@pytest.mark.parametrize(
    ["box_shape", "legacy_bbox"],
    [
        pytest.param((4, 4), False),
        pytest.param((2, 2, 4), True),
    ],
)
def test_xywh_encoder_decoder(box_shape, legacy_bbox):

    # generate fake box data
    box1 = torch.rand(*box_shape) * 100
    box1[..., 2:] = box1[..., :2] + torch.randint(30, 60, (2,))

    box2 = box1 + torch.randn(*box_shape)

    encoder = XYWHBBoxEncoder(
        legacy_bbox=legacy_bbox, reg_std=(0.1, 0.1, 0.2, 0.2)
    )
    decoder = XYWHBBoxDecoder(
        legacy_bbox=legacy_bbox, reg_std=(0.1, 0.1, 0.2, 0.2)
    )

    # make sure the data remain identical after encoding and decoding
    box_delta = encoder(box1, box2)
    pred_box = decoder(box1, box_delta)

    assert torch.max(torch.abs(pred_box - box2)) < 0.001


def test_match_label_encoder():
    # TODO(tian.li): test case for match label encoder
    pass


@pytest.mark.parametrize(
    ["feat_h", "feat_w", "kps_num"],
    [pytest.param(8, 8, 2), pytest.param(16, 16, 4)],
)
def test_rcnn_kps_label_from_match(feat_h, feat_w, kps_num):
    batch_size = 10
    num_boxes = 256
    padded_gt_num = 200
    ignore_labels = (0, 3)
    roi_expand_param = 1.2
    gauss_threshold = 0.6

    label_encoder = RCNNKPSLabelFromMatch(
        feat_h=feat_h,
        feat_w=feat_w,
        kps_num=kps_num,
        ignore_labels=ignore_labels,
        roi_expand_param=roi_expand_param,
        gauss_threshold=gauss_threshold,
    )

    # generate fake data
    boxes = torch.rand(batch_size, num_boxes, 4)
    gt_boxes = torch.rand(batch_size, padded_gt_num, 4 + kps_num * 3)
    match_pos_flag = torch.randint(-1, 2, (batch_size, num_boxes))
    match_gt_id = torch.randint(0, padded_gt_num, (batch_size, num_boxes))

    result = label_encoder(
        boxes=boxes,
        gt_boxes=gt_boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
    )

    # check shape
    assert result["kps_cls_label"].shape == torch.Size(
        [batch_size, num_boxes, kps_num, feat_h, feat_w]
    )
    assert result["kps_cls_label_weight"].shape == torch.Size(
        [batch_size, num_boxes, kps_num, feat_h, feat_w]
    )
    assert result["kps_reg_label"].shape == torch.Size(
        [batch_size, num_boxes, kps_num * 2, feat_h, feat_w]
    )
    assert result["kps_reg_label_weight"].shape == torch.Size(
        [batch_size, num_boxes, kps_num * 2, feat_h, feat_w]
    )


@pytest.mark.parametrize(
    ["feature_h", "feature_w", "num_classes", "allow_low_quality_heatmap"],
    [pytest.param(8, 8, 2, False), pytest.param(16, 16, 4, True)],
)
def test_rcnn_bin_det_label_from_match(
    feature_h, feature_w, num_classes, allow_low_quality_heatmap
):
    batch_size = 10
    num_boxes = 256
    padded_gt_num = 200

    roi_h_zoom_scale = 1.0
    roi_w_zoom_scale = 1.0
    cls_on_hard = False

    # generate fake data
    boxes = torch.rand(batch_size, num_boxes, 4)
    gt_boxes = torch.rand(batch_size, padded_gt_num, 4 + num_classes)
    match_pos_flag = torch.randint(-1, 2, (batch_size, num_boxes))
    match_gt_id = torch.randint(0, padded_gt_num, (batch_size, num_boxes))

    label_encoder = RCNNBinDetLabelFromMatch(
        roi_h_zoom_scale=roi_h_zoom_scale,
        roi_w_zoom_scale=roi_w_zoom_scale,
        feature_h=feature_h,
        feature_w=feature_w,
        num_classes=num_classes,
        cls_on_hard=cls_on_hard,
        allow_low_quality_heatmap=allow_low_quality_heatmap,
    )

    result = label_encoder(
        boxes=boxes,
        gt_boxes=gt_boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
    )

    # check shape
    assert result["label_map"].shape == torch.Size(
        [batch_size * num_boxes, num_classes, feature_h, feature_w]
    )
    assert result["offset"].shape == torch.Size(
        [batch_size * num_boxes, 4, feature_h, feature_w]
    )
    assert result["mask"].shape == torch.Size(
        [batch_size * num_boxes, num_classes]
    )


@pytest.mark.parametrize(
    [
        "feature_h",
        "feature_w",
        "num_classes",
        "max_subbox_num",
        "roi_h_zoom_scale",
        "roi_w_zoom_scale",
    ],
    [pytest.param(8, 8, 2, 100, 1, 1), pytest.param(16, 16, 4, 50, 2, 2)],
)
def test_rcnn_multi_bin_det_label_from_match(
    feature_h,
    feature_w,
    num_classes,
    max_subbox_num,
    roi_h_zoom_scale,
    roi_w_zoom_scale,
):
    batch_size = 10
    num_boxes = 256
    padded_gt_num = 200

    # generate fake data
    boxes = torch.rand(batch_size, num_boxes, 4)
    gt_boxes = torch.rand(
        batch_size, padded_gt_num, max_subbox_num, 4 + num_classes
    )
    match_pos_flag = torch.randint(-1, 2, (batch_size, num_boxes))
    match_gt_id = torch.randint(0, padded_gt_num, (batch_size, num_boxes))

    label_encoder = RCNNMultiBinDetLabelFromMatch(
        feature_h=feature_h,
        feature_w=feature_w,
        num_classes=num_classes,
        max_subbox_num=max_subbox_num,
        cls_on_hard=True,
        reg_on_hard=True,
        use_ig_region=True,
        roi_h_zoom_scale=roi_h_zoom_scale,
        roi_w_zoom_scale=roi_w_zoom_scale,
    )

    result = label_encoder(
        boxes=boxes,
        gt_boxes=gt_boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
        ig_regions=None,
        ig_regions_num=None,
    )

    # check shape
    assert result["label_map"].shape == torch.Size(
        [batch_size * num_boxes, num_classes, feature_h, feature_w]
    )
    assert result["offset"].shape == torch.Size(
        [batch_size * num_boxes, 4, feature_h, feature_w]
    )
    assert result["label_mask"].shape == torch.Size(
        [batch_size * num_boxes, num_classes, feature_h, feature_w]
    )
    assert result["offset_mask"].shape == torch.Size(
        [batch_size * num_boxes, 4, feature_h, feature_w]
    )


def test_match_label_ground_line_encoder():

    limit_reg_length = False
    cls_use_pos_only = True
    cls_on_hard = False
    reg_on_hard = False

    batch_size = 10
    num_boxes = 128
    padded_gt_num = 200

    # generate fake data
    boxes = torch.rand(batch_size, num_boxes, 4)
    gt_boxes = torch.rand(batch_size, padded_gt_num, 5)
    gt_flanks = torch.rand(batch_size, padded_gt_num, 9)
    match_pos_flag = torch.randint(-1, 2, (batch_size, num_boxes))
    match_gt_id = torch.randint(0, padded_gt_num, (batch_size, num_boxes))

    label_encoder = MatchLabelGroundLineEncoder(
        limit_reg_length=limit_reg_length,
        cls_use_pos_only=cls_use_pos_only,
        cls_on_hard=cls_on_hard,
        reg_on_hard=reg_on_hard,
    )

    result = label_encoder(
        boxes=boxes,
        gt_boxes=gt_boxes,
        gt_flanks=gt_flanks,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
    )

    # check shape
    assert result["reg_label"].shape == torch.Size([batch_size, num_boxes, 2])
    assert result["reg_label_mask"].shape == torch.Size(
        [batch_size, num_boxes, 2]
    )


@pytest.mark.parametrize(
    ["feat_h", "feat_w", "undistort_depth_uv"],
    [pytest.param(8, 8, False), pytest.param(16, 16, True)],
)
def test_rcnn_3d_label_from_match(feat_h, feat_w, undistort_depth_uv):
    batch_size = 10
    num_boxes = 128
    padded_gt_num = 100
    kps_num = 1

    label_encoder = RCNN3DLabelFromMatch(
        feat_h=feat_h,
        feat_w=feat_w,
        kps_num=kps_num,
        gauss_threshold=0.6,
        gauss_3d_threshold=0.6,
        gauss_depth_threshold=0.6,
        roi_expand_param=1.2,
        undistort_depth_uv=undistort_depth_uv,
    )

    # generate fake data
    boxes = torch.rand(batch_size, num_boxes, 4)
    gt_boxes = (
        torch.rand(batch_size, padded_gt_num, 15)
        if undistort_depth_uv
        else torch.rand(batch_size, padded_gt_num, 14)
    )
    match_pos_flag = torch.randint(-1, 2, (batch_size, num_boxes))
    match_gt_id = torch.randint(0, padded_gt_num, (batch_size, num_boxes))

    result = label_encoder(
        boxes=boxes,
        gt_boxes=gt_boxes,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
    )

    # check shape
    assert result["kps_cls_label"].shape == torch.Size(
        [batch_size, num_boxes, kps_num, feat_h, feat_w]
    )
    assert result["kps_cls_label_weight"].shape == torch.Size(
        [batch_size, num_boxes, kps_num, feat_h, feat_w]
    )
    assert result["kps_2d_offset"].shape == torch.Size(
        [batch_size, num_boxes, kps_num * 2, feat_h, feat_w]
    )
    assert result["kps_2d_offset_weight"].shape == torch.Size(
        [batch_size, num_boxes, kps_num * 2, feat_h, feat_w]
    )
    assert result["kps_3d_offset"].shape == torch.Size(
        [batch_size, num_boxes, kps_num * 2, feat_h, feat_w]
    )
    assert result["kps_3d_offset_weight"].shape == torch.Size(
        [batch_size, num_boxes, kps_num * 2, feat_h, feat_w]
    )
    if undistort_depth_uv:
        assert result["kps_depth_u"].shape == torch.Size(
            [batch_size, num_boxes, kps_num, feat_h, feat_w]
        )
        assert result["kps_depth_v"].shape == torch.Size(
            [batch_size, num_boxes, kps_num, feat_h, feat_w]
        )
    else:
        assert result["kps_depth"].shape == torch.Size(
            [batch_size, num_boxes, kps_num, feat_h, feat_w]
        )
    assert result["kps_depth_weight"].shape == torch.Size(
        [batch_size, num_boxes, kps_num, feat_h, feat_w]
    )
    assert result["dim_loc_r_y"].shape == torch.Size(
        [batch_size, num_boxes, 7]
    )
    assert result["pos_match"].shape == torch.Size(
        [batch_size, num_boxes * kps_num]
    )
    assert result["rois"].shape == torch.Size([batch_size, num_boxes, 4])


def test_match_label_flank_encoder():

    cls_on_hard = False
    use_cls = True
    reg_on_hard = False
    reg_on_inside_only = False

    batch_size = 10
    num_boxes = 128
    padded_gt_num = 200
    num_points = 2

    # generate fake data
    boxes = torch.rand(batch_size, num_boxes, 4)
    gt_boxes = torch.rand(batch_size, padded_gt_num, 5)
    gt_flanks = torch.rand(batch_size, padded_gt_num, num_points, 3)
    match_pos_flag = torch.randint(-1, 2, (batch_size, num_boxes))
    match_gt_id = torch.randint(0, padded_gt_num, (batch_size, num_boxes))

    label_encoder = MatchLabelFlankEncoder(
        cls_on_hard=cls_on_hard,
        use_cls=use_cls,
        reg_on_hard=reg_on_hard,
        reg_on_inside_only=reg_on_inside_only,
    )

    result = label_encoder(
        boxes=boxes,
        gt_boxes=gt_boxes,
        gt_flanks=gt_flanks,
        match_pos_flag=match_pos_flag,
        match_gt_id=match_gt_id,
    )

    # check shape
    assert result["reg_label"].shape == torch.Size(
        [batch_size, num_boxes, num_points, 2]
    )
    assert result["reg_label_mask"].shape == torch.Size(
        [batch_size, num_boxes, num_points, 2]
    )

    decoder = FlankPointDecoder(roi_expand_param=1.0)
    decode_result = decoder(
        batch_rois=boxes,
        head_out=dict(
            rcnn_cls_pred=result["cls_label"].unsqueeze(-1).unsqueeze(-1),
            rcnn_reg_pred=result["reg_label"],
        ),
    )
    assert "pred_flank" in decode_result
