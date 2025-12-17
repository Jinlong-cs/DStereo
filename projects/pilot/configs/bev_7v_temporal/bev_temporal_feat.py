"""NOTE: This file JUST output bev common temporal feature for compile."""
import os

from projects.pilot.configs.bev_7v_temporal.common import (
    backbone,
    bev_fusion,
    camera_view_names,
    deploy_head,
    deploy_mode,
    deploy_narrow_head,
    deploy_side_head,
    deploy_stage2_inputs_key,
    front_camera_view_names,
    head,
    identity_head,
    multi_view_collect,
    narrow_backbone,
    narrow_camera_view_names,
    narrow_head,
    narrow_pafpn_neck,
    pafpn_neck,
    side_backbone,
    side_camera_view_names,
    side_head,
    side_pafpn_neck,
    temporal_fusion,
)

cfg_dir = os.path.dirname(__file__)

# -------------------------- TASK ---------------------------
if not deploy_mode:
    raise NotImplementedError(f'"{__file__}": just for deploy or pack_infer.')

task_name = "bev_temporal_feat"

data_loader = None
metric_updater = None


# -------------------------- MODEL --------------------------
inputs = dict(
    train={},
    val={},
    deploy={},
)


def get_model(mode):
    multi_view_module = dict(
        img=dict(
            type="BEVStageOneModule",
            backbone=backbone,
            neck=pafpn_neck,
            head=head,
        ),
        side_img=dict(
            type="BEVStageOneModule",
            backbone=side_backbone,
            neck=side_pafpn_neck,
            head=side_head,
        ),
        narrow_img=dict(
            type="BEVStageOneModule",
            backbone=narrow_backbone,
            neck=narrow_pafpn_neck,
            head=narrow_head,
        ),
    )
    bevfusion_pick_keys = None
    if mode == "deploy":
        multi_view_module = dict()
        for key in camera_view_names:
            if key in front_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=backbone,
                    neck=pafpn_neck,
                    head=deploy_head,
                )
            elif key in side_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=side_backbone,
                    neck=side_pafpn_neck,
                    head=deploy_side_head,
                )
            elif key in narrow_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=narrow_backbone,
                    neck=narrow_pafpn_neck,
                    head=deploy_narrow_head,
                )
            else:
                raise TypeError
        bevfusion_pick_keys = deploy_stage2_inputs_key

    model = dict(
        type="MultiViewTwoStageBEVModule",
        # view_img_key to module
        multi_view_module=multi_view_module,
        multi_view_collect=multi_view_collect,
        bevfusion_pick_keys=bevfusion_pick_keys,
        bev_fusion_module=dict(
            type="BEVStageTwoModule",
            bevfusion=bev_fusion[mode],
            temporal_fusion=temporal_fusion[mode],
            temporal_output=identity_head,
        ),
    )

    return model
