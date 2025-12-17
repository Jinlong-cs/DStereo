import pytest
import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.registry import build_from_registry
from tests.utils import gen_fake_det_label_pred_data


@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_single_stage(mode):

    range_multiplier = 0.5
    default_regress_ranges = (
        (-1, 64),
        (64, 128),
        (128, 256),
        (256, 1e8),
    )
    regress_ranges = tuple(
        (
            (x * range_multiplier, y * range_multiplier)
            for x, y in (default_regress_ranges)
        )
    )
    num_classes = 1
    config = dict(
        type="SingleStageDetector",
        backbone=dict(
            type="VargNetV2",
            input_channels=3,
            input_sequence_length=1,
            num_classes=1000,
            factor=2,
            alpha=0.5,
            bias=True,
            bn_kwargs={"eps": 1e-5, "momentum": 0.1},
            group_base=8,
            include_top=False,
            head_factor=2,
        ),
        neck=dict(
            type="BiFPN",
            fpn_name="bifpn_sum",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[4, 8, 16, 32, 64],
            stride2channels=get_vargnetv2_stride2channels(0.5),
            out_channels=32,
            stack=3,
            start_level=1,
            end_level=-1,
            num_outs=5,
        ),
        box_module=dict(
            type="OutputModule",
            head=dict(
                type="FCOSHead",
                num_classes=1,
                in_strides=[4, 8, 16, 32, 64],
                out_strides=[8, 16, 32, 64],
                stride2channels={4: 32, 8: 32, 16: 32, 32: 32, 64: 32},
                feat_channels=32,
                stacked_convs=2,
                use_sigmoid=True,
                share_bn=False,
                upscale_bbox_pred=False,
            ),
            target=dict(
                type="FCOSTarget",
                strides=[8, 16, 32, 64],
                regress_ranges=regress_ranges,
                cls_out_channels=num_classes,
                background_label=num_classes,
                use_iou_replace_ctrness=True,
                norm_on_bbox=True,
            )
            if mode == "train"
            else None,
            loss=dict(
                type="FCOSLoss",
                cls_loss=dict(
                    type="FocalLoss",
                    loss_name="loss_cls",
                    num_classes=num_classes + 1,
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=1,
                ),
                reg_loss=dict(
                    type="GIoULoss",
                    loss_name="loss_bbox",
                    loss_weight=1,
                ),
                centerness_loss=dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=True,
                    loss_name="loss_iou",
                    loss_weight=1,
                ),
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="FCOSDecoder",
                num_classes=num_classes,
                strides=[8, 16, 32, 64],
                test_cfg=dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.0,
                    nms=dict(name="nms", iou_threshold=0.7),
                    max_per_img=30,
                ),
                nms_sqrt=False,
                transforms=None,
                inverse_transform_key=[
                    "scale_factor",
                    "crop_offset",
                    "before_crop_shape",
                ],
                filter_score_mul_centerness=False,
            )
            if mode != "train"
            else None,
        ),
    )

    single_stage_detector = build_from_registry(config)

    h, w = 576, 704
    strides = [8, 16, 32, 64]
    num_classes = 1

    label, _ = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    label["img"] = torch.rand(2, 3, h, w)

    if mode == "train":
        loss = single_stage_detector(label)
        assert "OutputModule__loss_cls" in loss.keys()
        assert "OutputModule__loss_bbox" in loss.keys()
        assert "OutputModule__loss_iou" in loss.keys()

    if mode == "val" or mode == "test":
        results = single_stage_detector(label)
        assert isinstance(results["OutputModule_predict_pred_bboxes"], list)


if "__main__" == __name__:
    test_single_stage("val")
