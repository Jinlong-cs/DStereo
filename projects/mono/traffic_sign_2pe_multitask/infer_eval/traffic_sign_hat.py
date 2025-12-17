import argparse
import logging
import os

import horizon_plugin_pytorch as horizon
import torch
from hatbc.utils import _as_list
from hpflow.callbacks import PickleDump, PickleUpload
from hpflow.modules.desc import AnchorBoxDetTaskDesc, KeyPoint2DTaskDesc
from hpflow.scope import get_current_config_scope_attribute
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import TakeByKey, get_bbox2d_visualizer, get_visualizer
from hpflow.utils.symbol_desc_parser import parse_hat_cfg

logger = logging.getLogger(__name__)


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
dataset = global_config.get("dataset")
savedir = global_config.get("savedir")
overwrite = global_config.get("overwrite", False)

ctx = global_config.get("context", torch.device("cpu"))
roi_desc_parsed = parse_hat_cfg(cfg_path, is_roi_model=True)

roi_model_depends_on_ptype = PTypeEnum.kTrafficSignBBox2D

for task_desc_i in roi_desc_parsed.task_descs:
    if isinstance(task_desc_i, AnchorBoxDetTaskDesc):
        task_desc_i.set_aux_kwargs_for_symbol_desc(
            vis_score_threshold=0.1,
            score_threshold_per_class=[
                0.0,
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
padding = roi_desc_parsed.padding or [0, 0, 0, 0]

# transformer: decode image bytes
transformer = dict(
    type="DefaultInputTransform",
    depends_on_ptype=roi_model_depends_on_ptype,
)

# workflow

workflow = dict(
    type="RoIModel",
    roi_transformer=dict(
        type="RoINormalizer",
        dst_wh=roi_desc_parsed.image_size[::-1],
        norm_method=roi_desc_parsed.norm_method,  # noqa
        norm_length=64
        if roi_desc_parsed.norm_length is None
        else roi_desc_parsed.norm_length,  # noqa
        score_thresh=None,
        padding=padding,
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
            PTypeEnum.kTrafficSignCategory: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kTrafficSignCategory,
            ),
            PTypeEnum.kTrafficSignOcclusion: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kTrafficSignOcclusion,
            ),
        },
        ptype=roi_model_depends_on_ptype,
    ),
)

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
visualizer = get_visualizer(ptype2vis, allow_missing=True)

name = model_name
tag = ["traffic_sign", "merge"]
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
        adapt_num_workers=8,
        writter=dataset.eval_platform_result_writter,
        adaptor=None,
        project_id=args.project_id,
        predname=name,
        enable_upload=True,
        overwrite=overwrite,
        tag=tag,
    ),
]

env = dict(
    savedir=name,
    symbolic_mode=True,
    eval_platform=dict(
        tag=tag,
        predname=name,
        reportname=name,
        diff_report_id=None,
        enable_report=True,
        enable_upload=False,
    ),
    backend="torch",
)
