import os

from projects.pilot.configs.bev_5v.common import (
    backbone,
    camera_view_names,
    deploy_head,
    deploy_mode,
    deploy_round_head,
    fisheye_camera_view_names,
    front_camera_view_names,
    head,
    multi_view_collect,
    pafpn_neck,
    round_backbone,
    round_head,
    round_pafpn_neck,
)

cfg_dir = os.path.dirname(__file__)

# -------------------------- TASK ---------------------------
if not deploy_mode:
    raise NotImplementedError(f'"{__file__}": just for deploy or pack_infer.')

task_name = "bev_each_view_feat"

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
        round_img=dict(
            type="BEVStageOneModule",
            backbone=round_backbone,
            neck=round_pafpn_neck,
            head=round_head,
        ),
    )

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
            elif key in fisheye_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=round_backbone,
                    neck=round_pafpn_neck,
                    head=deploy_round_head,
                )
            else:
                raise TypeError

    model = dict(
        type="MultiViewTwoStageBEVModule",
        # view_img_key to module
        multi_view_module=multi_view_module,
        multi_view_collect=multi_view_collect,
        bev_fusion_module=None,
    )

    return model
