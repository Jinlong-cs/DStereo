import argparse
import os
import pprint
from functools import partial

import horizon_plugin_pytorch as horizon
import torch
from hatbc.utils import _as_list
from hpflow.callbacks import PickleDump, PickleUpload
from hpflow.modules.desc import AnchorBoxDetTaskDesc, KeyPoint2DTaskDesc
from hpflow.op import get_expand_roi_bbox
from hpflow.scope import get_current_config_scope_attribute
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import (
    TakeByKey,
    filter_bbox_by_attributes,
    get_bbox2d_visualizer,
    get_visualizer,
)
from hpflow.utils.symbol_desc_parser import parse_hat_cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="argument for workflow builder"
    )
    parser.add_argument("--cfg_path", default=None)
    parser.add_argument("--model_ckpt", default=None)
    parser.add_argument("--project_id", default="PDT20220004")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--qat_mode", default=True)
    parser.add_argument(
        "--march", default=horizon.quantization.March.BERNOULLI2
    )  # noqa
    args, _ = parser.parse_known_args()
    return args


args = get_parser()


current_path = os.path.dirname(os.path.abspath(__file__))
cfg_path = f"{current_path}/../multitask.py"

global_config = get_current_config_scope_attribute()
eval_task = global_config.get("task", PTypeEnum.kAll)

cfg_path = args.cfg_path
model_ckpt = args.model_ckpt
model_name = args.model_name or model_ckpt.split("/")[-2]
qat_mode = bool(args.qat_mode)
march = args.march

global_config = get_current_config_scope_attribute()
task = global_config.get("task", PTypeEnum.kAll)
skip_img_model = global_config.get("skip_img_model", False)
dataset = global_config.get("dataset")
savedir = global_config.get("savedir")
overwrite = global_config.get("overwrite", False)

project_sensor = "mono3.0_x8b"

ctx = global_config.get("context", torch.device("cpu"))
roi_desc_parsed = parse_hat_cfg(cfg_path, is_roi_model=True)

roi_model_depends_on_ptype = PTypeEnum.kTrafficSignBBox2D

for task_desc_i in roi_desc_parsed.task_descs:
    if isinstance(task_desc_i, AnchorBoxDetTaskDesc):
        task_desc_i.set_aux_kwargs_for_symbol_desc(
            vis_score_threshold=0.3,
            score_threshold_per_class=[
                0.1,
            ]
            * len(task_desc_i.ptypes),
            feature_layout="reg_cls",
            clip_bbox=True,
        )
    elif isinstance(task_desc_i, KeyPoint2DTaskDesc):
        task_desc_i.set_aux_kwargs_for_symbol_desc(
            feature_stride=8,
            kps_pos_distance_xy=(12.5, 12.5),
            render_conf=False,
        )

# model meta
roi_task_descs = roi_desc_parsed.task_descs
roi_desc_parsed.padding = [0.0, 0.0, 0.0, 0.0]

pprint.pprint(roi_desc_parsed.__dict__)

# roi filter
valid_biaozhu_medium_types = [
    "W_Warning",
    "Prohibit_I",
    "P_Other1",
    "P_Unknown1",
    "P_ElectricalSpeedLimit",
    "P_Other_SpeedLimele",
    "P_Unknow_SpeedLimele",
    "P_SpeedLimit",
    "PSL_Other",
    "PSL_Unknown",
    "ProhibitRemove",
    "PR_Other",
    "PR_Unknown",
    "Min_SpeedLim",
    "MSL_Other",
    "MSL_Unknown",
    "I_IndicationCircle",
    "I_Other_circle",
    "I_Unknown_circle",
    "I_IndicationRectangle",
    "I_Other_rectangle",
    "I_Unknown_rectangle",
]

valid_biaozhu_sub_types = [
    "W_Working",
    "I_MinSpeedLim100",
    "I_MinSpeedLim110",
    "I_MinSpeedLim60",
    "I_MinSpeedLim70",
    "I_MinSpeedLim80",
    "I_MinSpeedLim90",
    "P_NoPassingRev",
    "Other_Indications",
    "P_NoParking",
    "Prohibit_I",
    "W_Warning",
    "P_SpeedLim10",
    "P_SpeedLim100",
    "P_SpeedLim100ele",
    "P_SpeedLim110",
    "P_SpeedLim120",
    "P_SpeedLim120ele",
    "P_SpeedLim15",
    "P_SpeedLim20",
    "P_SpeedLim25",
    "P_SpeedLim30",
    "P_SpeedLim35",
    "P_SpeedLim40",
    "P_SpeedLim40ele",
    "P_SpeedLim5",
    "P_SpeedLim50",
    "P_SpeedLim50ele",
    "P_SpeedLim60",
    "P_SpeedLim60ele",
    "P_SpeedLim65",
    "P_SpeedLim70",
    "P_SpeedLim80",
    "P_SpeedLim80ele",
    "P_SpeedLim90",
    "P_SpeedLim90ele",
    "P_SpeedLimRev",
    "I_MinSpeedLim50",
    "P_SpeedLim5ele",
]

# 需要与2pe cls 模型的类别描述一致，mono2.x 和mono3.x 里面是不一样的
valid_2pe_cls_types_mono2_x = [
    "MinSpeedLim100",
    "MinSpeedLim110",
    "MinSpeedLim50",
    "MinSpeedLim60",
    "MinSpeedLim70",
    "MinSpeedLim80",
    "MinSpeedLim90",
    "NoPassingRev",
    "Other_IndicationsCircle",
    "Other_NoParking",
    "Other_Prohibit",
    "Other_Warning",
    "SpeedLim10",
    "SpeedLim100",
    "SpeedLim100_electric",
    "SpeedLim110",
    "SpeedLim110_electric",
    "SpeedLim120",
    "SpeedLim120_electric",
    "SpeedLim15",
    "SpeedLim20",
    "SpeedLim25",
    "SpeedLim30",
    "SpeedLim30_electric",
    "SpeedLim35",
    "SpeedLim40",
    "SpeedLim40_electric",
    "SpeedLim5",
    "SpeedLim50",
    "SpeedLim50_electric",
    "SpeedLim55",
    "SpeedLim5_electric",
    "SpeedLim60",
    "SpeedLim60_electric",
    "SpeedLim65",
    "SpeedLim70",
    "SpeedLim70_electric",
    "SpeedLim80",
    "SpeedLim80_electric",
    "SpeedLim90",
    "SpeedLim90_electric",
    "SpeedLimRev",
    "Other_IndicationsRectangle",
    "P_NoEntry",
    "P_Noway",
    "P_SlowFor",
    "P_StopFor",
    "P_WeightLim",
    "P_WeightLimWheel",
    "W_MergeLeft",
    "W_MergeRight",
    "W_SplitLeft",
    "W_SplitRight",
    "W_SlowDown",
]
valid_2pe_cls_types_mono3_x = [
    "P_NoParking",
    "I_MinSpeedLim40",
    "I_MinSpeedLim50",
    "I_MinSpeedLim60",
    "I_MinSpeedLim70",
    "I_MinSpeedLim80",
    "I_MinSpeedLim90",
    "I_MinSpeedLim100",
    "I_MinSpeedLim110",
    "P_SpeedLim5",
    "P_SpeedLim10",
    "P_SpeedLim15",
    "P_SpeedLim20",
    "P_SpeedLim25",
    "P_SpeedLim30",
    "P_SpeedLim35",
    "P_SpeedLim40",
    "P_SpeedLim45",
    "P_SpeedLim50",
    "P_SpeedLim55",
    "P_SpeedLim60",
    "P_SpeedLim65",
    "P_SpeedLim70",
    "P_SpeedLim75",
    "P_SpeedLim80",
    "P_SpeedLim85",
    "P_SpeedLim90",
    "P_SpeedLim95",
    "P_SpeedLim100",
    "P_SpeedLim105",
    "P_SpeedLim110",
    "P_SpeedLim115",
    "P_SpeedLim120",
    "P_SpeedLimRev5",
    "P_SpeedLimRev10",
    "P_SpeedLimRev15",
    "P_SpeedLimRev20",
    "P_SpeedLimRev25",
    "P_SpeedLimRev30",
    "P_SpeedLimRev35",
    "P_SpeedLimRev40",
    "P_SpeedLimRev45",
    "P_SpeedLimRev50",
    "P_SpeedLimRev55",
    "P_SpeedLimRev60",
    "P_SpeedLimRev65",
    "P_SpeedLimRev70",
    "P_SpeedLimRev75",
    "P_SpeedLimRev80",
    "P_SpeedLimRev85",
    "P_SpeedLimRev90",
    "P_SpeedLimRev95",
    "P_SpeedLimRev100",
    "P_SpeedLimRev105",
    "P_SpeedLimRev110",
    "P_SpeedLimRev115",
    "P_SpeedLimRev120",
    "P_SpeedLim40ele",
    "P_SpeedLim50ele",
    "P_SpeedLim60ele",
    "P_SpeedLim70ele",
    "P_SpeedLim80ele",
    "P_SpeedLim90ele",
    "P_SpeedLim100ele",
    "P_SpeedLim110ele",
    "P_SpeedLim120ele",
    "P_Other_SpeedLimele",
]

valid_2pe_cls_types = valid_2pe_cls_types_mono2_x + valid_2pe_cls_types_mono3_x

if not skip_img_model:
    #  for 4pe infer + 2pe cls  ->  2pe assist
    attribute_key = PTypeEnum.kTrafficSignCategory
    valid_attributes = valid_2pe_cls_types
else:
    # for biaozhu -> 2pe assist
    attribute_key = "cn_medium_type"
    valid_attributes = valid_biaozhu_medium_types

roi_filter_func = partial(
    filter_bbox_by_attributes,
    attribute_key=attribute_key,
    valid_attributes=valid_attributes,
)

roi_input_func = TakeByKey(
    roi_model_depends_on_ptype,
    as_list=False,
    filter_func=roi_filter_func,
)

# ---- seperate line ----

# transformer: decode image bytes
transformer = dict(
    type="DefaultInputTransform",
    depends_on_ptype=roi_model_depends_on_ptype,
    add_annotated_attrs=True,
)


# workflow
workflow = dict(
    type="RoIModel",
    roi_transformer=dict(
        type="RoINormalizer",
        dst_wh=roi_desc_parsed.image_size[::-1],
        norm_method=roi_desc_parsed.norm_method,
        norm_length=64
        if roi_desc_parsed.norm_length is None
        else roi_desc_parsed.norm_length,  # noqa
        score_thresh=None,
        padding=roi_desc_parsed.padding,
        roi_expand_func=get_expand_roi_bbox,
        roi_expand_kwargs={
            "start_x": "roi_center_x",
            "start_y": "roi_y1",
            "offset_x_ref_edge": "roi_width",
            "offset_x1_coeff": -1,
            "offset_x2_coeff": 1,
            "offset_y_ref_edge": "roi_height",
            "offset_y1_coeff": 0,
            "offset_y2_coeff": 3,
        },
    ),
    model=dict(
        type="ImgModel",
        model=dict(
            type="TorchModel",
            cfg=cfg_path,
            model_ckpt=model_ckpt,
            device=ctx,
            qat=qat_mode,
            get_input_func=TakeByKey(["img"]),
            is_roi_model=True,
            march=march,
            allow_miss=True,
        ),
        img_transformer=dict(
            type="DefaultImgFormatter",
            tensor_type="torch",
            ctx=ctx,
            layout="NCHW",
            mean=(128,) * 3,
            std=(128,) * 3,
        ),
        decoders=[
            task_desc_i.get_decoder_block()
            for task_desc_i in _as_list(roi_task_descs)
        ],
    ),
    fusion_block=dict(
        type="FusionBBox2DsAndRecongnitions",
        ptype2fusion_blocks={
            PTypeEnum.kTrafficSignAssist_IR_Ramp: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kTrafficSignAssist_IR_Ramp,
            ),
            PTypeEnum.kTrafficSignAssist_Other_GuideSign_WhiteBlack: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kTrafficSignAssist_Other_GuideSign_WhiteBlack,  # noqa
            ),
        },
        ptype=PTypeEnum.kTrafficSignAssist,
    ),
    roi_input_func=roi_input_func,
    keep_original_rois=True,
)


# visualizer
attr_ptype2vis = dict()
for task_desc_i in roi_task_descs:
    attr_ptype2vis.update(task_desc_i.get_ptype2visualizer())

attr_visualizer = get_visualizer(attr_ptype2vis, allow_missing=True)
ptype2vis = dict()
ptype2vis[roi_model_depends_on_ptype] = get_bbox2d_visualizer(
    roi_model_depends_on_ptype
)  # noqa
ptype2vis[roi_model_depends_on_ptype].update(
    dict(attr_visualizer=attr_visualizer)
)  # noqa
ptype2vis[PTypeEnum.kTrafficSignAssist] = get_bbox2d_visualizer(
    roi_model_depends_on_ptype
)  # noqa
ptype2vis[PTypeEnum.kTrafficSignAssist].update(
    dict(attr_visualizer=attr_visualizer)
)  # noqa
visualizer = get_visualizer(ptype2vis, allow_missing=True)

name = f"{model_name}"
tag = ["traffic_sign_assist_det", "merge"]

callbacks = [
    PickleDump(
        savedir=os.path.join(savedir, name, dataset.savedir),
        filename="results.pkl",
        overwrite=overwrite,
    ),
    PickleUpload(
        savedir=os.path.join(savedir, name, dataset.savedir),
        filename="results.pkl",
        eval_ptypes=dataset.eval_ptypes,
        dataset_id=dataset.dataset_id,
        dataset_tag=dataset.dataset_tag,
        adapt_num_workers=4,
        writter=dataset.eval_platform_result_writter,
        adaptor=None,
        project_id="PDT20220004",
        predname=name,
        enable_upload=True,
        overwrite=overwrite,
        tag=tag,
    ),
]

env = dict(
    savedir=f"{name}/",  # noqa
    eval_platform=dict(
        tag=["2pe_traffic_sign_assist_det"],
        predname=f"{project_sensor}_{name}",  # noqa
        reportname=f"{project_sensor}_{name}",  # noqa
    ),
    backend="torch",
)
