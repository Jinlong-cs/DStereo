import pytest
import torch

from hat.registry import build_from_registry
from projects.isp.app.det.model_select_j5 import (
    get_mixvargenet_config,
    get_unet_config,
    large_mixvargenet_stride2channels,
    mixvargenet_stride64_example,
)
from tests.utils import gen_fake_det_label_pred_data


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_auto_assign_fcos(mode):
    config = dict(
        type="AutoAssignFCOS",
        backbone=get_mixvargenet_config(
            net_config=mixvargenet_stride64_example,
            output_list=[0, 1, 2, 3, 4, 5],
            input_channels=3,
        ),
        neck=get_unet_config(large_mixvargenet_stride2channels),
        head=dict(
            type="AutoAssignHead",
            num_classes=2,
            in_strides=[4, 8, 16, 32, 64],
            out_strides=[4, 8, 16, 32, 64],
            stride2channels={
                4: 32,
                8: 64,
                16: 96,
                32: 160,
                64: 320,
            },
            feat_channels=32,
            stacked_convs=2,
            use_sigmoid=True,
            share_bn=False,
            share_conv=False,
            upscale_bbox_pred=True,
        ),
        targets=dict(
            type="AutoAssignTarget",
            strides=[4, 8, 16, 32, 64],
            cls_out_channels=2,
            center_prior=dict(
                type="CenterPrior",
                force_topk=False,
                num_classes=2,
                strides=[4, 8, 16, 32, 64],
            ),
            norm_on_bbox=True,
        )
        if mode == "train"
        else None,
        loss=dict(
            type="AutoAssignLoss",
            loss_name=["loss_pos", "loss_neg", "loss_center"],
            pos_loss=dict(
                type="PosLoss",
                reg_loss=dict(
                    type="CIoULoss",
                    loss_name="loss_bbox",
                    loss_weight=5.0,
                    reduction="none",
                ),
                loss_weight=0.25,
            ),
            neg_loss=dict(
                type="NegLoss",
                loss_weight=0.75,
            ),
            center_loss=dict(
                type="CenterLoss",
                loss_weight=0.75,
            ),
        )
        if mode == "train"
        else None,
        post_process=dict(
            type="FCOSDecoder",
            num_classes=2,
            strides=[4, 8, 16, 32, 64],
            nms_use_centerness=True,
            nms_sqrt=True,
            transforms=[dict(type="Resize")],
            inverse_transform_key=["scale_factor"],
            test_cfg=dict(
                nms_pre=1000,
                nms=dict(name="nms", iou_threshold=0.6, max_per_img=100),
                score_thr=0.3,
                min_bbox_size=0,
            ),
            filter_score_mul_centerness=True,
        )
        if mode != "train"
        else None,
    )

    detector = build_from_registry(config)

    h = w = 1024
    strides = [4, 8, 16, 32, 64]
    num_classes = 2

    label, _ = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    label["img"] = torch.rand(2, 3, h, w)

    if mode == "train":
        loss = detector(label)
        assert "loss_pos" in loss
        assert "loss_neg" in loss
        assert "loss_center" in loss

    if mode == "val" or mode == "test":
        results = detector(label)
        assert "pred_bboxes" in results
        assert results["pred_bboxes"][0].shape == (0, 6)


@pytest.mark.parametrize(
    ["me"],
    [
        pytest.param(
            dict(
                type="NISP",
                in_channels=3,
                out_channels=3,
                block_num=2,
            ),
        ),
    ],
)
def test_isp_me_detector(me):
    config = dict(
        type="IspMeDetector",
        me=me,
        detector=dict(
            type="AutoAssignFCOS",
            backbone=get_mixvargenet_config(
                net_config=mixvargenet_stride64_example,
                output_list=[0, 1, 2, 3, 4, 5],
                input_channels=3,
            ),
            neck=get_unet_config(large_mixvargenet_stride2channels),
            head=dict(
                type="AutoAssignHead",
                num_classes=2,
                in_strides=[4, 8, 16, 32, 64],
                out_strides=[4, 8, 16, 32, 64],
                stride2channels={
                    4: 32,
                    8: 64,
                    16: 96,
                    32: 160,
                    64: 320,
                },
                feat_channels=32,
                stacked_convs=2,
                use_sigmoid=True,
                share_bn=False,
                share_conv=False,
                upscale_bbox_pred=True,
            ),
            targets=dict(
                type="AutoAssignTarget",
                strides=[4, 8, 16, 32, 64],
                cls_out_channels=2,
                center_prior=dict(
                    type="CenterPrior",
                    force_topk=False,
                    num_classes=2,
                    strides=[4, 8, 16, 32, 64],
                ),
                norm_on_bbox=True,
            ),
            loss=dict(
                type="AutoAssignLoss",
                loss_name=["loss_pos", "loss_neg", "loss_center"],
                pos_loss=dict(
                    type="PosLoss",
                    reg_loss=dict(
                        type="CIoULoss",
                        loss_name="loss_bbox",
                        loss_weight=5.0,
                        reduction="none",
                    ),
                    loss_weight=0.25,
                ),
                neg_loss=dict(
                    type="NegLoss",
                    loss_weight=0.75,
                ),
                center_loss=dict(
                    type="CenterLoss",
                    loss_weight=0.75,
                ),
            ),
            post_process=dict(
                type="FCOSDecoder",
                num_classes=2,
                strides=[4, 8, 16, 32, 64],
                nms_use_centerness=True,
                nms_sqrt=True,
                transforms=[dict(type="Resize")],
                inverse_transform_key=["scale_factor"],
                test_cfg=dict(
                    nms_pre=1000,
                    nms=dict(name="nms", iou_threshold=0.6, max_per_img=100),
                    score_thr=0.3,
                    min_bbox_size=0,
                ),
                filter_score_mul_centerness=True,
            ),
        ),
    )
    me_detector = build_from_registry(config)
    h = w = 1024
    strides = [4, 8, 16, 32, 64]
    num_classes = 2

    label, _ = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    label["img"] = torch.rand(2, 3, h, w)

    loss = me_detector(label)
    assert "loss_pos" in loss
    assert "loss_neg" in loss
    assert "loss_center" in loss


@pytest.mark.parametrize(
    ["lutnet", "me", "deploy"],
    [
        pytest.param(
            dict(
                type="LutNet",
                in_channels=3,
                down_size=(256, 256),
                pregamma=1 / 3.0,
                track_coarse_steps=1000,
                norm_symmetric=True,
            ),
            dict(
                type="NISP",
                in_channels=3,
                out_channels=3,
                block_num=2,
            ),
            False,
        ),
        pytest.param(
            None,
            dict(
                type="NISP",
                in_channels=3,
                out_channels=3,
                block_num=2,
            ),
            True,
        ),
        pytest.param(
            dict(
                type="LutNet",
                in_channels=3,
                down_size=(256, 256),
                pregamma=1 / 3.0,
                track_coarse_steps=1000,
                norm_symmetric=True,
            ),
            None,
            False,
        ),
        pytest.param(
            dict(
                type="LutNet",
                in_channels=3,
                down_size=(256, 256),
                pregamma=1 / 3.0,
                track_coarse_steps=1000,
                norm_symmetric=True,
            ),
            None,
            True,
        ),
    ],
)
def test_lut_me_detector(lutnet, me, deploy):
    config = dict(
        type="LutMeDetector",
        lutnet=lutnet,
        me=me,
        detector=dict(
            type="AutoAssignFCOS",
            backbone=get_mixvargenet_config(
                net_config=mixvargenet_stride64_example,
                output_list=[0, 1, 2, 3, 4, 5],
                input_channels=3,
            ),
            neck=get_unet_config(large_mixvargenet_stride2channels),
            head=dict(
                type="AutoAssignHead",
                num_classes=2,
                in_strides=[4, 8, 16, 32, 64],
                out_strides=[4, 8, 16, 32, 64],
                stride2channels={
                    4: 32,
                    8: 64,
                    16: 96,
                    32: 160,
                    64: 320,
                },
                feat_channels=32,
                stacked_convs=2,
                use_sigmoid=True,
                share_bn=False,
                share_conv=False,
                upscale_bbox_pred=True,
            ),
            targets=dict(
                type="AutoAssignTarget",
                strides=[4, 8, 16, 32, 64],
                cls_out_channels=2,
                center_prior=dict(
                    type="CenterPrior",
                    force_topk=False,
                    num_classes=2,
                    strides=[4, 8, 16, 32, 64],
                ),
                norm_on_bbox=True,
            ),
            loss=dict(
                type="AutoAssignLoss",
                loss_name=["loss_pos", "loss_neg", "loss_center"],
                pos_loss=dict(
                    type="PosLoss",
                    reg_loss=dict(
                        type="CIoULoss",
                        loss_name="loss_bbox",
                        loss_weight=5.0,
                        reduction="none",
                    ),
                    loss_weight=0.25,
                ),
                neg_loss=dict(
                    type="NegLoss",
                    loss_weight=0.75,
                ),
                center_loss=dict(
                    type="CenterLoss",
                    loss_weight=0.75,
                ),
            ),
            post_process=dict(
                type="FCOSDecoder",
                num_classes=2,
                strides=[4, 8, 16, 32, 64],
                nms_use_centerness=True,
                nms_sqrt=True,
                transforms=[dict(type="Resize")],
                inverse_transform_key=["scale_factor"],
                test_cfg=dict(
                    nms_pre=1000,
                    nms=dict(name="nms", iou_threshold=0.6, max_per_img=100),
                    score_thr=0.3,
                    min_bbox_size=0,
                ),
                filter_score_mul_centerness=True,
            ),
        ),
        deploy=deploy,
    )
    lut_me_detector = build_from_registry(config)
    h = w = 1024
    strides = [4, 8, 16, 32, 64]
    num_classes = 2

    label, _ = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    label["img"] = torch.rand(2, 3, h, w)

    loss = lut_me_detector(label)
    assert "loss_pos" in loss
    assert "loss_neg" in loss
    assert "loss_center" in loss
