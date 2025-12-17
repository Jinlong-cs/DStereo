# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict

import torch

from hat.models.task_modules.traj_pred.backbones.vectornet_backbone import (
    VectorNetBackbone,
)
from hat.models.task_modules.traj_pred.heads.basic_heads import BasicMlpDecoder
from hat.models.task_modules.traj_pred.structures.vectornet import (
    BasicVectorNet,
    VectorNetV2,
)


def gen_example_vectornet_model(version: int = 1):
    # TODO(shengzhe.dai): delete this when removing BasicVectorNet from HAT.
    if version == 1:
        structure_cls = BasicVectorNet
    elif version == 2:
        structure_cls = VectorNetV2
    else:
        raise ValueError(f"Undefined structure version {version}")

    pkl_fps = 2
    context_frames = 4
    gt_traj_len = 12
    target_freq = 10
    traj_len = int(gt_traj_len * target_freq / pkl_fps)
    max_obs_num = 32
    polyline_seg_len = 10
    max_num_ele_seg = 256
    traj_sample_ratio = int(target_freq / pkl_fps)
    traj_polyline_len = traj_sample_ratio * (context_frames - 1)
    # road feats: start_x, start_y, end_x, end_y, turn_dir, pre_pre_x,
    #             pre_pre_y
    # traj feats: start_x, start_y, end_x, end_y, time_stamp, pid
    road_feat_dim = 7
    traj_feat_dim = 6

    # -- Parameters about models
    subgraph_hidden_size = 128
    globalgraph_hidden_size = 128
    use_out_fc = True
    num_attn_head = 4
    backbone_fc_out_channels = 128
    mlp_predictor_hidden_size = 128
    mlp_input_channels = backbone_fc_out_channels

    enable_scale_trils = True
    model_heads = OrderedDict(
        normal_head=BasicMlpDecoder(
            in_channels=mlp_input_channels,
            traj_len=traj_len,
            hidden_size=mlp_predictor_hidden_size,
            num_hidden_layers=3,
            enable_scale_trils=enable_scale_trils,
            log_std_clamp_min=-2,
            log_std_clamp_max=2,
            is_int_infer_model=True,
        )
    )
    enable_itp_gt = True

    model = structure_cls(
        backbone=VectorNetBackbone(
            road_feat_channels=road_feat_dim,
            road_polyline_len=polyline_seg_len - 1,
            road_polyline_num=max_num_ele_seg,
            traj_feat_channels=traj_feat_dim,
            traj_polyline_len=traj_polyline_len,
            traj_polyline_num=max_obs_num,
            num_hidden_units=subgraph_hidden_size,
            num_attn_hidden_units=globalgraph_hidden_size,
            num_sub_graph_layers=3,
            num_attention_heads=num_attn_head,
            use_out_fc=use_out_fc,
            fc_out_channels=mlp_input_channels,
        ),
        heads=model_heads,
        post_process=None,
        losses=None,
        itp_ratio=traj_sample_ratio,
        enable_itp_gt=enable_itp_gt,
        is_int_infer_model=True,
    )
    return model


def test_vectornet_structure(version: int = 1):
    pkl_fps = 2
    context_frames = 4
    target_freq = 10
    max_obs_num = 32
    polyline_seg_len = 10
    max_num_ele_seg = 256
    traj_sample_ratio = int(target_freq / pkl_fps)
    traj_polyline_len = traj_sample_ratio * (context_frames - 1)
    hdm_max_num_obs = 32
    road_feat_dim = 7
    traj_feat_dim = 6
    num_all_poly = max_num_ele_seg + max_obs_num
    traj_len = 60

    # We just test the infer mode here.
    test_inputs = dict(
        struct_road_feats=torch.randn(
            (
                hdm_max_num_obs,
                road_feat_dim,
                max_num_ele_seg,
                polyline_seg_len - 1,
            )
        ),
        struct_traj_feats=torch.randn(
            (hdm_max_num_obs, traj_feat_dim, max_obs_num, traj_polyline_len)
        ),
        attention_mask=torch.ones(
            (hdm_max_num_obs, 1, num_all_poly, num_all_poly)
        ),
    )

    model = gen_example_vectornet_model(version=version)
    model.fuse_model()
    model.set_qconfig()
    output = model(test_inputs)
    output = output._asdict()
    assert "normal_head_probabilities" in output
    assert "normal_head_mean_var" in output
    assert output["normal_head_mean_var"].shape == (
        max_obs_num,
        1,
        traj_len,
        2,
    )
    assert output["normal_head_probabilities"].shape == (
        max_obs_num,
        1,
        traj_len,
        2,
        2,
    )


def test_vectornet_v2_structure():
    test_vectornet_structure(version=2)
