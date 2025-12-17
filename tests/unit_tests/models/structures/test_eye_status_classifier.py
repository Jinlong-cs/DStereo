import copy

import pytest
import torch

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.registry import build_from_registry


@pytest.mark.parametrize(
    "eye_structure",
    [
        "EyeStatusClassifier",
        "EyeStatusSingleHeadClassifier",
        "EyeStatusMultiTaskClassifier",
    ],
)
def test_eye_structure(eye_structure):
    bn_kwargs = dict(eps=1e-3, momentum=0.01)
    in_channels = [32, 32, 32, 64, 128]
    out_channels = [32, 32, 64, 128, 256]
    alpha = 0.5
    num_ldmk = 16
    in_channels = [int(x * alpha) for x in in_channels]
    out_channels = [int(x * alpha) for x in out_channels]
    ldmk_in_channels = out_channels[1:]

    data_cfg = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="EyeStatusDataset",
            rec_paths=[
                "./tmp_orig_data/face/eye_status/train_02.rec",
                "./tmp_orig_data/face/eye_status/train_03.rec",
            ],
            transforms=[
                dict(
                    type="RandomFlip",
                    px=0.5,
                ),
                dict(
                    type="RandomShiftRotateScale",
                    rotate_prob=0.4,
                    max_rotate_angle=10.0,
                    shift_prob=0.4,
                    max_shift_range=(0.05, 0.05),
                ),
                dict(
                    type="CropRecROI",
                    crop_type="scale",
                    target_shape=(96, 160, 3),
                    base_roi=[112, 186, 208, 346],
                    crop_jitter_range=0.1,
                    center_shift_range=0,
                ),
                dict(
                    type="GenerateGaussianHeatmap",
                    num_ldmk=16,
                    feat_stride=4,
                    heatmap_shape=[24, 40],
                ),
                dict(
                    type="GenerateGaussianVector",
                    num_ldmk=16,
                    feat_stride=4,
                    vector_size=[40, 24],
                ),
                dict(
                    type="ToTensor",
                ),
                dict(
                    type="TorchVisionAdapter",
                    interface="ColorJitter",
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                ),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(
                    type="TorchVisionAdapter",
                    interface="Normalize",
                    mean=128.0,
                    std=128.0,
                ),
            ],
        ),
        # sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=4,
        num_workers=0,
        pin_memory=True,
    )

    net_config = [
        [
            MixVarGENetConfig(
                in_channels=in_channels[0],
                out_channels=out_channels[0],
                head_op="mixvarge_f2",
                stack_ops=[],
                stack_factor=1,
                stride=1,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 2
        [
            MixVarGENetConfig(
                in_channels=in_channels[1],
                out_channels=out_channels[1],
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 4
        [
            MixVarGENetConfig(
                in_channels=in_channels[2],
                out_channels=out_channels[2],
                head_op="mixvarge_f4",
                stack_ops=["mixvarge_f4", "mixvarge_f4"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 8
        [
            MixVarGENetConfig(
                in_channels=in_channels[3],
                out_channels=out_channels[3],
                head_op="mixvarge_f2_gb16",
                stack_ops=[
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                ],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 16
        [
            MixVarGENetConfig(
                in_channels=in_channels[4],
                out_channels=out_channels[4],
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                stack_factor=1,
                stride=2,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],  # stride 32
    ]

    cls_net_config = [
        [
            MixVarGENetConfig(
                in_channels=out_channels[-1],
                out_channels=int(320 * alpha),
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                stack_factor=1,
                stride=1,
                fusion_strides=[],
                extra_downsample_num=0,
            ),  # noqa
        ],
    ]

    mixvargenet_cls_head = dict(
        type="EyeMixVarGEClsHead",
        net_config=cls_net_config,
        num_classes=5,
        in_channels=int(320 * alpha),
        out_channels=1280,
        bn_kwargs=bn_kwargs,
        pool_size=(3, 5),
        include_top=True,
        flat_output=False,
    )

    cls_head = dict(
        type="EyeMultiBranchHead",
        l_cls_head=copy.deepcopy(mixvargenet_cls_head),
        r_cls_head=copy.deepcopy(mixvargenet_cls_head),
    )

    single_net_config = [
        [
            MixVarGENetConfig(
                in_channels=128,
                out_channels=160,
                head_op="mixvarge_f2_gb16",
                stack_ops=["mixvarge_f2_gb16"],
                stack_factor=1,
                stride=1,
                fusion_strides=[],
                extra_downsample_num=0,
            ),
        ],
    ]

    left_single_head = dict(
        type="EyeMixVarGEClsHead",
        net_config=single_net_config,
        num_classes=5,
        bn_kwargs={},
        in_channels=160,
        out_channels=160,
        pool_size=(3, 5),
        bias=False,
        dropout_rate=0.2,
        include_top=False,
    )

    right_single_head = dict(
        type="EyeMixVarGEClsHead",
        net_config=single_net_config,
        num_classes=5,
        bn_kwargs={},
        in_channels=160,
        out_channels=160,
        pool_size=(3, 5),
        bias=False,
        dropout_rate=0.2,
        include_top=False,
        flat_output=True,
    )

    single_cls_head = dict(
        type="EyeMultiBranchSingleHead",
        l_cls_head=left_single_head,
        r_cls_head=right_single_head,
        num_classes=5,
        bn_kwargs={},
        in_channels=160,
        out_channels=160,
        pool_size=(3, 5),
    )

    if (
        eye_structure == "EyeStatusClassifier"
        or eye_structure == "EyeStatusSingleHeadClassifier"
    ):
        config = dict(
            type=eye_structure,
            backbone=dict(
                type="MixVarGENet",
                net_config=net_config,
                num_classes=1000,
                bn_kwargs=bn_kwargs,
                include_top=False,
                bias=True,
            ),
            ldmk_neck=dict(
                type="FPEM_FFM",
                bn_kwargs=bn_kwargs,
                in_channels_list=[16, 32, 64, 128],
                out_channels=128,
                fpem_repeat=1,
                act_type="relu",
            ),
            cls_head=cls_head
            if eye_structure == "EyeStatusClassifier"
            else single_cls_head,
            ldmk_head=dict(
                type="EyeLdmkHead",
                heatmap_head=dict(
                    type="EyeLdmkHeatmapHead",
                    in_channels=16,
                    num_ldmk=num_ldmk,
                ),
                vector_head=dict(
                    type="EyeLdmkVectorHead",
                    num_ldmk=num_ldmk,
                    in_channels=16,
                ),
            ),
            left_cls_losses=dict(
                type="SoftTargetCrossEntropy",
            ),
            right_cls_losses=dict(
                type="SoftTargetCrossEntropy",
            ),
            ldmk_losses=dict(
                type="SmoothL1Loss",
                beta=1.0,
            ),
            heatmap_losses=dict(
                type="SmoothL1Loss",
                beta=1.0,
            ),
            num_classes=5,
        )

        data_loader = build_from_registry(data_cfg)
        model = build_from_registry(config)
        for batch in data_loader:
            outputs, losses = model(batch)
            if eye_structure == "EyeStatusClassifier":
                assert outputs["l_feat"].shape == (4, 5, 1, 1)
                assert outputs["r_feat"].shape == (4, 5, 1, 1)
            else:
                assert outputs["l_feat"].shape == (4, 5)
                assert outputs["r_feat"].shape == (4, 5)
            assert outputs["x_dist"].shape == (4, 16, 1, 40)
            assert outputs["y_dist"].shape == (4, 16, 24, 1)
            assert outputs["heatmap"].shape == (4, 16, 24, 40)
            assert "left_cls_loss" in losses.keys()
            assert "right_cls_loss" in losses.keys()
            assert "heatmap_loss" in losses.keys()
            assert "ldmk_loss" in losses.keys()
            break
    else:
        bin_cls_head = dict(
            type="EyeMixVarGEBinClsHead",
            bn_kwargs=bn_kwargs,
            pool_size=(3, 5),
            in_channels=int(256 * alpha),
            out_channels=320,
            bias=False,
            flat_output=True,
            dropout_rate=0.2,
        )

        multi_bin_head = dict(
            type="EyeMultiBinBranchHead",
            l_cls_head=copy.deepcopy(bin_cls_head),
            r_cls_head=copy.deepcopy(bin_cls_head),
        )

        config = dict(
            type="EyeStatusMultiTaskClassifier",
            backbone=dict(
                type="MixVarGENet",
                net_config=net_config,
                num_classes=1000,
                bn_kwargs=bn_kwargs,
                include_top=False,
                bias=True,
            ),
            ldmk_neck=dict(
                type="FPEM_FFM",
                bn_kwargs=bn_kwargs,
                in_channels_list=ldmk_in_channels,
                out_channels=128,
                # in_channels_list=[24, 48, 96, 192],
                # out_channels=192,
                fpem_repeat=1,
                act_type="relu",
            ),
            cls_head=cls_head,
            bin_cls_head=multi_bin_head,
            ldmk_head=dict(
                type="EyeLdmkHead",
                heatmap_head=dict(
                    type="EyeLdmkHeatmapHead",
                    in_channels=16,
                    num_ldmk=num_ldmk,
                ),
                vector_head=dict(
                    type="EyeLdmkVectorHead",
                    num_ldmk=num_ldmk,
                    in_channels=16,
                ),
            ),
            left_cls_losses=dict(
                type="SoftTargetCrossEntropy",
            ),
            right_cls_losses=dict(
                type="SoftTargetCrossEntropy",
            ),
            left_bin_cls_losses=dict(
                type=torch.nn.BCEWithLogitsLoss,
                reduction="mean",
            ),
            right_bin_cls_losses=dict(
                type=torch.nn.BCEWithLogitsLoss,
                reduction="mean",
            ),
            ldmk_losses=dict(
                type="SmoothL1Loss",
                beta=1.0,
            ),
            heatmap_losses=dict(
                type="SmoothL1Loss",
                beta=1.0,
            ),
            num_classes=5,
        )

        data_loader = build_from_registry(data_cfg)
        model = build_from_registry(config)
        for batch in data_loader:
            outputs, losses = model(batch)
            assert outputs["l_feat"].shape == (4, 5, 1, 1)
            assert outputs["r_feat"].shape == (4, 5, 1, 1)
            assert outputs["x_dist"].shape == (4, 16, 1, 40)
            assert outputs["y_dist"].shape == (4, 16, 24, 1)
            assert outputs["heatmap"].shape == (4, 16, 24, 40)
            assert outputs["l_bin_feat"].shape == (4, 1)
            assert outputs["r_bin_feat"].shape == (4, 1)
            assert "left_cls_loss" in losses.keys()
            assert "right_cls_loss" in losses.keys()
            assert "heatmap_loss" in losses.keys()
            assert "ldmk_loss" in losses.keys()
            assert "left_bin_loss" in losses.keys()
            assert "right_bin_loss" in losses.keys()
            break
