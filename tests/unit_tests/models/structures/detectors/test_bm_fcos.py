import pytest
import torch

from hat.core.data_struct.app_struct import DetObjects
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test
from tests.utils import gen_fake_det_label_pred_data

try:
    import hatbc
except ImportError:
    hatbc = None


@pytest.mark.skipif(hatbc is None, reason="need hatbc")
@pytest.mark.parametrize(
    ["mode"],
    [
        pytest.param("train"),
        pytest.param("val"),
    ],
)
def test_bmfcos(mode):
    task_name = "person_detection"
    mean, std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
    decoder_transforms = [
        dict(type="Resize", img_scale=(1536, 2816)),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="Pad", size=(1536, 2816), pad_val=0),
    ]

    config = dict(
        type="BMFCOS",
        backbone=dict(
            type="efficientnet",
            model_type="b0",
            num_classes=1000,
            activation="swish",
            use_se_block=True,
            include_top=False,
            flat_output=False,
            bn_kwargs={},
        ),
        neck=dict(
            type="BiFPN",
            fpn_name="bifpn_sum",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[8, 16, 32, 64, 128],
            stride2channels={2: 16, 4: 24, 8: 40, 16: 112, 32: 320},
            out_channels={8: 256, 16: 256, 32: 256, 64: 256, 128: 256},
            start_level=2,
            end_level=-1,
            num_outs=5,
            stack=1,
        ),
        head=dict(
            type="FCOSHead",
            num_classes=1,
            in_strides=[8, 16, 32, 64, 128],
            out_strides=[8, 16, 32, 64, 128],
            stride2channels={
                4: 256,
                8: 256,
                16: 256,
                32: 256,
                64: 256,
                128: 256,
            },
            upscale_bbox_pred=True,
            feat_channels=256,
            stacked_convs=4,
            share_conv=True,
            use_sigmoid=True,
            share_bn=True,
            dequant_output=True,
            int8_output=False,
            use_plain_conv=True,
            use_gn=True,
            use_scale=True,
        ),
        target=dict(
            type="DynamicFcosTarget",
            center_sampling=True,
            center_sampling_radius=2.5,
            strides=[8, 16, 32, 64, 128],
            cls_out_channels=1,
            background_label=1,
            topK=10,
            loss_cls=dict(
                type="FocalLoss",
                loss_name="cls",
                num_classes=1 + 1,
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                reduction="none",
            ),
            loss_reg=dict(
                type="GIoULoss",
                loss_name="reg",
                loss_weight=3.0,
                reduction="none",
            ),
        )
        if mode == "train"
        else None,
        loss=dict(
            type="FCOSLoss",
            cls_loss=dict(
                type="FocalLoss",
                loss_name="loss_cls",
                num_classes=1 + 1,
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
            ),
            reg_loss=dict(
                type="GIoULoss",
                loss_name="loss_bbox",
                loss_weight=2.0,
            ),
            centerness_loss=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_name="loss_iou",
                loss_weight=1.0,
            ),
        )
        if mode == "train"
        else None,
        postprocess=dict(
            type="MultiInputSequential",
            modules=[
                dict(
                    type="FCOSDecoder",
                    num_classes=1,
                    strides=[8, 16, 32, 64, 128],
                    nms_use_centerness=True,
                    nms_sqrt=False,
                    filter_score_mul_centerness=True,
                    transforms=decoder_transforms,
                    inverse_transform_key=["scale_factor"],
                    test_cfg=dict(
                        score_thr=0.05,
                        nms_pre=1000,
                        nms=dict(
                            name="nms", iou_threshold=0.65, max_per_img=100
                        ),
                    ),
                ),
                dict(
                    type="FCOSConverter",
                    task_name=task_name,
                    cls_name_mapping={0: "_".join(task_name.split("_")[:-1])},
                    node_name=f"{task_name}_converter",
                ),
            ],
        )
        if mode != "train"
        else None,
    )

    bmfcos_detector = build_from_registry(config)

    h = w = 1024
    strides = [8, 16, 32, 64, 128]
    num_classes = 1

    label, _ = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    label["img"] = torch.rand(2, 3, h, w)

    if mode == "train":
        loss = bmfcos_detector(label)
        assert "loss_cls" in loss
        assert "loss_bbox" in loss
        assert "loss_iou" in loss

        # see test_fcos_head.py
        with pytest.raises(NotImplementedError):
            qat_test(bmfcos_detector, label, with_quantized=False)

    if mode == "val" or mode == "test":
        results = bmfcos_detector(label)
        assert isinstance(results[0], DetObjects)
