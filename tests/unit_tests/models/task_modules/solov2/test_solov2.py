import numpy as np
import pytest
import torch

from hat.core.data_struct.app_struct import DetObjects
from hat.core.data_struct.base_struct import ClsLabels, Mask, Masks
from hat.core.mask2polygon import Mask2Polygon
from hat.models.losses.cross_entropy_loss import CrossEntropyLoss
from hat.models.losses.focal_loss import FocalLossV2
from hat.models.task_modules.solov2 import (
    InstanceSegToMsg,
    InstanceSegToParsing,
    SOLOV2Decoder,
    SOLOV2Head,
    SOLOV2Loss,
    SOLOV2Target,
)
from hat.models.task_modules.solov2.loss import MaskLoss
from hat.utils.package_helper import check_packages_available
from tests.utils import gen_fake_feats

hatbc_available = check_packages_available("hatbc", raise_exception=False)

if hatbc_available:
    from hatbc.message import Instance


@pytest.mark.skipif(
    not hatbc_available,
    reason="hatbc is not available, skip test",
)
@pytest.mark.parametrize(
    ["dict_to_hat_struct", "with_polygon"],
    [
        pytest.param(False, True),
        pytest.param(False, False),
        pytest.param(True, False),
    ],
)
@pytest.mark.parametrize(
    ["attr_name2num", "upsample_mask_logit"],
    [
        pytest.param({"occlusion": 2}, True),
        pytest.param({}, False),
    ],
)
def test_solov2(
    attr_name2num, upsample_mask_logit, dict_to_hat_struct, with_polygon
):
    cls_num = 3
    batch_size = 2
    in_channels = 32
    kernel_out_channels = 32
    strides = [4, 8]
    num_grids = [16, 8]
    obj_name = "obj"

    attr_names = ["type"] + list(attr_name2num.keys())
    cls_name_mapping = {"type": {i: f"type_{i}" for i in range(cls_num)}}
    cls_name_mapping.update(
        {
            k: {i: f"{k}_{i}" for i in range(v)}
            for k, v in attr_name2num.items()
        }
    )

    ori_shape = [512, 640]
    resized_shape = [3, 256, 256]

    solov2_head = SOLOV2Head(
        num_classes=cls_num,
        feat_channels=32,
        in_channels=in_channels,
        feat_channels_attr=32,
        num_grids=num_grids,
        attr_name2num=attr_name2num,
        kernel_out_channels=kernel_out_channels,
        mask_feature_head_updater=dict(
            in_channels=in_channels,
            feat_channels=32,
            out_channels=kernel_out_channels,
            stacked_convs=1,
            start_level=0,
            end_level=1,
        ),
        upsample_mask_logit=upsample_mask_logit,
    )

    solov2_decoder = SOLOV2Decoder(
        num_classes=cls_num,
        strides=strides,
        object_name=obj_name,
        num_grids=num_grids,
        combine=True,
        attr_names=attr_names,
        cls_name_mapping=cls_name_mapping,
        kernel_out_channels=kernel_out_channels,
        upsample_mask_logit=upsample_mask_logit,
        dict_to_hat_struct=dict_to_hat_struct,
        mask2polygon_fn=Mask2Polygon() if with_polygon else None,
        test_cfg_updater=dict(
            nms_pre=10,
            score_thr=0.0,
            mask_thr=0.01,
            filter_thr=0.0,
            kernel="gaussian",
            sigma=2.0,
            max_per_img=10,
        ),
    )
    # train
    attr_loss_names = [f"loss_ce_{k}" for k in attr_name2num.keys()]
    solov2_loss = SOLOV2Loss(
        cls_loss=FocalLossV2(),
        mask_loss=MaskLoss(),
        attr_loss=[CrossEntropyLoss(ignore_index=255)] * len(attr_name2num),
        mask_loss_name="loss_mask",
        cls_loss_name="loss_cls",
        attr_loss_names=attr_loss_names,
    )

    solov2_target = SOLOV2Target(
        num_classes=cls_num,
        num_grids=num_grids,
        upsample_mask_logit=upsample_mask_logit,
        attr_name2num=attr_name2num,
        ignore_index=255,
    )

    fake_neck_feat, _, _ = gen_fake_feats(
        resized_shape[2], resized_shape[1], strides, batch_size, in_channels
    )

    fake_attrs = [
        torch.tensor([1, 1] + [1] * len(attr_name2num)),
        torch.tensor([2, 2] + [1] * len(attr_name2num)),
    ]
    fake_seg = torch.tensor(
        np.random.randint(0, 3, [resized_shape[1], resized_shape[2]])
    )
    fake_label = dict(
        img_shape=[np.array(resized_shape)] * batch_size,
        img_height=[ori_shape[0]] * batch_size,
        img_width=[ori_shape[1]] * batch_size,
        gt_labels=[fake_attrs] * batch_size,
        gt_seg=[fake_seg] * batch_size,
    )

    pred = solov2_head(fake_neck_feat)
    results = solov2_decoder(pred, fake_label)
    assert len(results) == batch_size

    if dict_to_hat_struct:
        for result in results:
            assert isinstance(result, DetObjects)
            pred_masks = getattr(result, f"{obj_name}_instanceseg")
            assert isinstance(pred_masks, Masks)
            ins_num = pred_masks.masks.shape[0]
            assert pred_masks.masks.shape[1] == ori_shape[0]
            assert pred_masks.masks.shape[2] == ori_shape[1]
            for attr_name in attr_names:
                pred_attr = getattr(result, f"{obj_name}_{attr_name}")
                assert isinstance(pred_attr, ClsLabels)
                assert len(pred_attr) == ins_num

        solov2_to_parsing = InstanceSegToParsing(task_name="obj_instanceseg")
        parsing_res = solov2_to_parsing(results)
        assert len(parsing_res) == batch_size
        for res in parsing_res:
            assert isinstance(res, Mask)
    else:
        if with_polygon:
            for result in results:
                assert "polygons" in result

        solov2_to_msg = InstanceSegToMsg(task_name="obj_instanceseg")
        for result in results:
            assert isinstance(result, dict)
        msg_res = solov2_to_msg(results)
        assert len(msg_res) == batch_size
        for res in msg_res:
            for res_i in res:
                assert isinstance(res_i, Instance)

    target = solov2_target(fake_label, pred)
    loss = solov2_loss(pred, target)
    assert len(target) == 3
    assert "loss_cls" in loss
    assert "loss_mask" in loss
    for attr_loss_name in attr_loss_names:
        assert attr_loss_name in loss
