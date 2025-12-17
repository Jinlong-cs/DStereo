import os

from hdflow.plugins.mono.data_management_and_deploy.base.structure import (  # noqa
    CLASS_NAMES_DETECTION,
    DataDeployTasksCfgs,
)

__all__ = [
    "TASK_NAME_TO_TS_CFG",
]

root_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)))


def get_kps_ann_ts_config(**kwargs):
    args = kwargs.get("args", None)
    skip_invalid = args.workflow_pass_invalid if args is not None else False
    ann_ts_kps = dict(
        type="Compose",
        transformer=[
            dict(
                type="CMKpsAnnoJsonTs",
                max_num_kps=10,
                conf_ignore_list=[],
                class_name="traffic_sign",
                kps_prefix="traffic_sign_keypoint_",
            ),
            dict(
                type="CMKeyPointAnnoTs",
                max_num_kps=10,
                kps_corner_status_id_dict={
                    "very_indistinct": 0,
                    "indistinct": 1,
                    "occluded": 2,
                    "full_visible": 3,
                    "clearly": 4,
                    "ignore": 5,
                },  # noqa
                min_roi_width=10,
                min_roi_height=10,
                ignore_cls_cn_medium_type=None,
                ignore_cls_cn_sub_type=None,
                ignore_occ_type=["invisible"],
                ignore_kps_corner_status=None,
                skip_invalid=skip_invalid,
            ),
        ],
    )
    return ann_ts_kps


def get_kps_packing_configs(**kwargs):
    args = kwargs.get("args", None)
    skip_invalid = args.workflow_pass_invalid if args is not None else False
    num_workers = kwargs["num_workers"]
    pack_configs = dict(
        # pack image data
        image_packer=dict(
            type="HorizonImageDataPacker",
            uri=kwargs["output_rec_path"],
            num_worker=num_workers,
            skip_invalid=skip_invalid,
        ),
        # dataset for read packed image data
        image_dataset=dict(
            type="HorizonImageDataDataset",
            rec_path=kwargs["output_rec_path"],
            rec_idx_file=kwargs["output_rec_path"] + ".idx",
            to_rgb=True,
            decode_img=False,
        ),
        # packer for pack annotation data
        anno_packer=dict(
            type="KPSAnnoPacker",
            uri=kwargs["output_anno_path"],
            num_worker=num_workers,
        ),
    )
    return pack_configs


def get_kps_viz_dataset_config(default):
    return dict(
        type="KPSDataset",
        img_rec_path=default["rec_path"],
        anno_rec_path=default["anno_path"],
        to_rgb=True,
    )


def get_kps_viz_visualizer_config(default):
    from hdflow.badcase.basic.visualizer.class_metas.base import _COLORS

    default["task_type"] = "keypoint"

    point_ids = [3, 4, 5, 6, 7]
    class_name = [f"traffic_sign_{i}" for i in point_ids]
    keypoint_task_to_names = dict(
        traffic_sign_kps_detection=[f"traffic_sign_kps_{i}" for i in point_ids]
    )
    keypoint_names = dict()
    parent_name2kpsname = dict()
    for i in point_ids:
        color_i = dict(
            name=[f"p{j}" for j in range(i)],
            color=[_COLORS[j] for j in range(i)],
        )
        keypoint_names.update({f"traffic_sign_kps_{i}": color_i})
        parent_name2kpsname.update(
            {f"traffic_sign_{i}": f"traffic_sign_kps_{i}"}
        )

    default["viz_class_id"] = point_ids
    default["class_name"] = class_name

    default["kps_color_maps"] = dict(
        keypoint_task_to_names=keypoint_task_to_names,
        keypoint_names=keypoint_names,
        keypoint_connection_rules=dict(),
        parent_name2kpsname=parent_name2kpsname,
    )
    return default


TASK_NAME_TO_TS_CFG = {
    # ----------------------------------------
    # cyclist
    # ----------------------------------------
    "cloud_model_traffic_sign_kps": DataDeployTasksCfgs(
        # no yaml type
        task_ts_cfg=None,
        packing_task_mode="default",
        task_owners_email=["hao.chen@horizon.ai"],
        data_pack_update_fn=lambda default, **kwargs: get_kps_packing_configs(
            **kwargs
        ),
        anno_transformer_update_fn=lambda default, **kwargs: get_kps_ann_ts_config(  # noqa
            **kwargs
        ),
        viz_dataset_config_update_fn=lambda default, **kwargs: get_kps_viz_dataset_config(  # noqa
            default
        ),
        viz_visualizer_config_update_fn=lambda default, **kwargs: get_kps_viz_visualizer_config(  # noqa
            default
        ),
        class_name=[],
        eval_setting_config=None,
        evalset_config=None,
    ),
}
