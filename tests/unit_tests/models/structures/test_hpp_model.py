import copy
import json

import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.timeout(600)
def test_hpp_model_task():
    alpha = 0.5
    bn_kwargs = dict(eps=2e-5, momentum=0.1)
    channel_list_ = [32, 32, 64, 128, 256]
    out_channels_ = int(channel_list_[2] * alpha)
    config = dict(
        type="HppModel",
        out_indices=2,
        backbone=dict(
            type="VargNetV2",
            input_channels=3,
            input_sequence_length=1,
            num_classes=1000,
            factor=2,
            alpha=alpha,
            bias=True,
            bn_kwargs=bn_kwargs,
            group_base=8,
            include_top=False,
            head_factor=2,
        ),
        decode_head=dict(
            type="HPPDecodeHead",
            block_num=4,
            in_channels=out_channels_,
            out_channels=out_channels_,
        ),
        losses=dict(
            type="HppLoss",
            ins_embedding_channel=4,
            weight_offset=0.2,
            weight_exist=1.0,
            weight_nonexist=1.0,
            weight_attention=0.1,
            weight_sisc=0.5,
        ),
    )
    hpp_model_train = build_from_registry(config)
    hpp_model_train.train()
    feat_stride = 8
    img_h, img_w = (256, 512)
    grid_h, grid_w = img_h // feat_stride, img_w // feat_stride
    x = {
        "img": torch.rand(2, 3, img_h, img_w),
        "labels": (
            torch.rand(2, 3, grid_h, grid_w),
            torch.randint(0, 3, (2, 1, grid_h * grid_w, grid_h * grid_w)),
        ),
    }
    result_train = hpp_model_train(x)
    assert "hpp_exist_loss" in result_train
    assert "hpp_nonexist_loss" in result_train
    assert "hpp_offset_loss" in result_train
    assert "hpp_sisc_loss" in result_train
    assert "hpp_attention_loss" in result_train

    qat_test(hpp_model_train, x, with_quantized=False)

    # deploy model
    config_deploy_model = copy.deepcopy(config)
    config_deploy_model["losses"] = None
    params_desc = dict(
        image_h=img_h,
        image_w=img_w,
        ego_near_length=10,
        ego_far_length=70,
        ego_left=8,
        ego_right=8,
        up_pixel=10,
        ipm_roi=[-8, 70, 16, 60],
    )
    add_desc_pp = dict(
        type="AddDesc",
        per_tensor_desc=[
            # num of desc == num of pred output tensors (one tensor per out stride)  # noqa
            json.dumps(
                dict(
                    task="hpp_traj_pred_prob",
                    score_threshold=0.6,
                    **params_desc,
                )
            ),
            json.dumps(
                dict(
                    task="hpp_traj_pred_offset",
                    **params_desc,
                )
            ),
        ],
    )
    config_deploy_model["post_process"] = add_desc_pp
    hpp_model_deploy = build_from_registry(config_deploy_model)
    hpp_model_deploy.eval()
    results_deploy = hpp_model_deploy(x)
    assert results_deploy[0].shape[-2:] == results_deploy[1].shape[-2:]


if __name__ == "__main__":
    pytest.main(["-s", __file__])
