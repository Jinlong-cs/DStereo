from collections import OrderedDict

import pytest
import torch

from hat.models.task_modules.e2e_dynamic.deformable_transformer_plus_cropped import (  # noqa
    CroppedDeformableTransformer,
)
from hat.models.task_modules.e2e_dynamic.e2e_dynamic import E2EDynamicModule
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import MLP, Linear
from hat.models.task_modules.e2e_dynamic.memory_bank import MemoryBankModule
from hat.models.task_modules.e2e_dynamic.postprocess import (
    TrackerManager,
    TrackerPostProcess,
)
from hat.models.task_modules.e2e_dynamic.qim import (
    QuerySpatialInteractionModule,
)
from hat.models.task_modules.e2e_dynamic.set_criterion import (
    E2EClipMatcher,
    E2EHungarianMatcher,
)


def get_input_data():
    inputs = {}
    inputs["motr_targets"] = [
        {
            "bev_tracking": [
                {
                    "obj_idxes": torch.ones(100, dtype=torch.int64) * -99,
                    "labels": torch.zeros(100, dtype=torch.int64),
                    "boxes": torch.rand((100, 4), dtype=torch.float32),
                    "yaws": torch.rand((100, 2), dtype=torch.float32),
                    "bev_loc_z": torch.rand(100, dtype=torch.float32),
                    "heights": torch.rand(100, dtype=torch.float32),
                    "scores": torch.rand(100, dtype=torch.float32),
                }
            ]
        }
    ]
    inputs["bev_stage2_feats"] = [
        [
            torch.rand((1, 48, 256, 256)),
            torch.rand((1, 48, 128, 128)),
            torch.rand((1, 96, 64, 64)),
            torch.rand((1, 192, 32, 32)),
            torch.rand((1, 192, 16, 16)),
        ]
    ]
    inputs["bev_stage2_3d_vehicle_head_predict_bev3d_ct"] = torch.randn(
        4, 300, 2
    )
    inputs["bev_stage2_3d_vrumerge_head_predict_bev3d_ct"] = torch.randn(
        4, 300, 2
    )
    inputs["bev_stage2_3d_vehicle_head_predict_bev3d_score"] = torch.randn(
        4, 300
    )
    inputs["bev_stage2_3d_vrumerge_head_predict_bev3d_score"] = torch.randn(
        4, 300
    )
    inputs["odo_info"] = [torch.rand(12, 3)]
    inputs["timestamp"] = [torch.zeros(1, 1)]
    return inputs


@pytest.mark.parametrize(
    [
        "e2e_out_velocity",
        "e2e_out_trajectory",
        "e2e_use_qim",
        "e2e_use_mb",
        "init_query_feat",
    ],
    [
        pytest.param(False, False, False, False, False, id="detection"),
        pytest.param(False, False, False, False, True, id="detection"),
        pytest.param(False, False, True, False, False, id="tracking"),
        pytest.param(False, False, True, False, True, id="tracking"),
        pytest.param(True, False, True, True, False, id="velocity"),
        pytest.param(True, False, True, True, True, id="velocity"),
        pytest.param(
            True, True, True, True, False, id="velocity and trajectory"
        ),
        pytest.param(
            True, True, True, True, True, id="velocity and trajectory"
        ),
    ],
)
def test_e2e_dynamic_module(
    e2e_out_velocity,
    e2e_out_trajectory,
    e2e_use_qim,
    e2e_use_mb,
    init_query_feat,
):
    clip_datas = get_input_data()

    transformer = CroppedDeformableTransformer(
        feat_shape=((128, 128), (64, 64), (32, 32), (16, 16)),
        embedding_dim=256,
        num_head=8,
        num_queries=360 if e2e_use_qim else 300,
        num_det_queries=300,
        num_decoder_layers=6,
        feedforward_dim=512,
        dropout_ratio=0.1,
        return_intermediate_dec=True,
    )
    subtask_head = OrderedDict(
        bbox_embed_xy=MLP(
            input_dim=256,
            hidden_dim=256,
            output_dim=2,
            num_layers=3,
            is_output=False,
        ),
        bbox_embed_wh=MLP(
            input_dim=256,
            hidden_dim=256,
            output_dim=2,
            num_layers=3,
            is_output=False,
        ),
        class_embed=Linear(in_channels=256, out_channels=3),
        yaw_embed=Linear(in_channels=256, out_channels=2),
        zheight_embed=MLP(
            input_dim=256,
            hidden_dim=256,
            output_dim=2,
            num_layers=3,
            is_output=True,
        ),
    )
    extra_subtask_head = {}
    if e2e_out_velocity:
        extra_subtask_head.update(
            velocity_embed=MLP(
                input_dim=288,
                hidden_dim=256,
                output_dim=3,
                num_layers=3,
                is_output=True,
            ),
            K_embed=MLP(
                input_dim=288,
                hidden_dim=256,
                output_dim=48,
                num_layers=3,
                is_output=False,
            ),
        )
    if e2e_out_trajectory:
        extra_subtask_head.update(
            trajpred_embed=MLP(
                input_dim=288,
                hidden_dim=256,
                output_dim=120,
                num_layers=2,
                is_output=True,
            ),
            trajpred_logits=MLP(
                input_dim=288,
                hidden_dim=256,
                output_dim=1,
                num_layers=2,
                is_output=True,
            ),
        )
    e2e_query_kwargs = {
        "num_veh_queries": 200,
        "num_vru_queries": 100,
        "veh_filter_scores": 0.2,
        "vru_filter_scores": 0.2,
        "num_track_queries": 60,
        "query_drop_ratio": 0.2,
        "query_fp_ratio": 0.1,
        "num_max_queries": 1000,
        "fp_rand_ref_pts_value": 0.02,
        "fp_ref_pts_bias": 0.03,
    }
    matcher = E2EHungarianMatcher(
        iou_loss_type="ciou",
    )
    criterion = E2EClipMatcher(
        matcher=matcher,
        match_indices_each_layer=False,
    )

    query_interaction_module = QuerySpatialInteractionModule(
        update_query_pos=True,
        dropout_ratio=0.1,
        dim_in=256,
        hidden_dim=512,
        dim_out=512,
        replace_identity_with_tgt=True,
    )
    memory_bank_module = MemoryBankModule(
        memory_bank_len=10,
        memory_bank_with_temp_attn=True,
        memory_bank_with_self_attn=True,
        num_heads=8,
        dim_in=256,
        hidden_dim=512,
        dim_out=512,
    )

    trajectory_kwargs = {
        "num_traj_modal": 5,
        "num_frames_for_pred": 60,
    }

    e2e_nms_setting = {
        "letnms": [
            {
                "p_t": 0.35,
                "min_t": 4.0,
                "max_t": 12.0,
                "radius": 3.0,
                "e_loc_threshold": 0.4,
                "angle_threshold": 0.3,
                "area_threshold": 0.4,
                "ct_nms_param": {
                    "scale_l": 0.85,
                    "scale_w": 0.6,
                    "use_yaw_filter": True,
                    "yaw_threshold": 0.1,
                    "use_mutual_ctnms": True,
                },
            },
            {
                "p_t": 0.35,
                "min_t": 4.0,
                "max_t": 12.0,
                "radius": 3.0,
                "e_loc_threshold": 0.4,
                "angle_threshold": 0.3,
                "area_threshold": 0.4,
                "ct_nms_param": {
                    "scale_l": 0.85,
                    "scale_w": 0.6,
                    "use_yaw_filter": True,
                    "yaw_threshold": 0.1,
                    "use_mutual_ctnms": True,
                },
            },
            {
                "p_t": 0.35,
                "min_t": 4.0,
                "max_t": 8.0,
                "radius": 1.6,
                "e_loc_threshold": 0.3,
                "angle_threshold": 0.3,
                "area_threshold": 0.4,
                "ct_nms_param": {
                    "scale_l": 0.85,
                    "scale_w": 0.6,
                    "use_yaw_filter": True,
                    "yaw_threshold": 0.1,
                    "use_mutual_ctnms": True,
                },
            },
        ],
        "nms": {"nms_thresh": 0.3},
    }
    e2e_track_manager_kwargs = {
        # 0: vehicle, 1: cyclist 2:pedestrian
        "cls_score_thr": {
            0: dict(score_threshold=0.3, filter_score_thresh=0.25),
            1: dict(score_threshold=0.3, filter_score_thresh=0.25),
            2: dict(
                score_threshold=0.3,
                filter_score_thresh=0.25,
            ),
        },
        "miss_tolerance": 5,
        "nms_setting": {
            "agnostic": True,
            "let_nms": e2e_nms_setting["letnms"],
        },
        "vcs_range": [-51.2, -51.2, 51.2, 51.2],
        "valid_range": [-51.2, -51.2, 51.2, 51.2],
        "traj_thickness": 1.0,
    }
    decode_rot_setting = {
        "psc_rot": {
            "N_steps_PSC_rot": 3,
        }
    }

    track_manager = TrackerManager(
        decode_rot_setting=decode_rot_setting, **e2e_track_manager_kwargs
    )
    post_process = TrackerPostProcess(
        vcs_range=(-30.0, -51.2, 72.4, 51.2, -3, 5),
        decode_rot_setting=None,
    )
    e2e_dynamic = E2EDynamicModule(
        transformer=transformer,
        subtask_head=subtask_head,
        extra_subtask_head=extra_subtask_head,
        criterion=criterion,
        in_strides=[2, 4, 8, 16, 32],  # noqa
        out_strides=[2, 4, 8, 16, 32],  # noqa
        decoder_strides=[4, 8, 16, 32],
        num_channels=[48, 48, 96, 192, 192],
        wlh_anchor_size={
            "car": [0.0178, 0.0468, 0.02],
            "cyclist": [0.0178, 0.0468, 0.02],
            "pedestrian": [0.0178, 0.0468, 0.02],
        },
        odometry_channels=[2, 32, 32],
        bev_size=[256, 256],
        default_time_delta=0.1,
        vcs_range=(-30.0, -51.2, 72.4, 51.2),
        use_auxiliary_loss=True,
        use_box_refine=False,
        e2e_query_kwargs=e2e_query_kwargs,
        query_interaction_module=query_interaction_module
        if e2e_use_qim
        else None,
        memory_bank_module=memory_bank_module if e2e_use_mb else None,
        e2e_out_velocity=e2e_out_velocity,
        e2e_out_trajectory=e2e_out_trajectory,
        trajectory_kwargs=trajectory_kwargs,
        init_query_feat=init_query_feat,
    )
    out = e2e_dynamic(clip_datas)
    assert "output_for_losses" in out[0]
    assert "clip_num_samples" in out[0]

    val_e2e_dynamic = E2EDynamicModule(
        transformer=transformer,
        subtask_head=subtask_head,
        extra_subtask_head=extra_subtask_head,
        in_strides=[2, 4, 8, 16, 32],  # noqa
        out_strides=[2, 4, 8, 16, 32],  # noqa
        decoder_strides=[4, 8, 16, 32],
        num_channels=[48, 48, 96, 192, 192],
        wlh_anchor_size={
            "car": [0.0178, 0.0468, 0.02],
            "cyclist": [0.0178, 0.0468, 0.02],
            "pedestrian": [0.0178, 0.0468, 0.02],
        },
        odometry_channels=[2, 32, 32],
        bev_size=[256, 256],
        default_time_delta=0.1,
        vcs_range=(-30.0, -51.2, 72.4, 51.2),
        use_auxiliary_loss=False,
        use_box_refine=False,
        e2e_query_kwargs=e2e_query_kwargs,
        track_manager=track_manager,
        post_process=post_process,
        query_interaction_module=query_interaction_module
        if e2e_use_qim
        else None,
        memory_bank_module=memory_bank_module if e2e_use_mb else None,
        e2e_out_velocity=e2e_out_velocity,
        e2e_out_trajectory=e2e_out_trajectory,
        trajectory_kwargs=trajectory_kwargs,
        init_query_feat=init_query_feat,
    )
    val_e2e_dynamic.eval()
    with torch.no_grad():
        val_out = val_e2e_dynamic(clip_datas)
    assert "output_for_losses" not in val_out[0]
    assert "clip_num_samples" not in val_out[0]
    assert "obj_idxes" in val_out[0]
