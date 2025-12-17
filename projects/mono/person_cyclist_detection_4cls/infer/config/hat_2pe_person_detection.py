import copy
import logging
import os

import horizon_plugin_pytorch as horizon
import mxnet as mx
import torch
from hatbc.utils import _as_list

# from person_adapters import PersonAdapter, TakePredictionByKey
from hpflow.callbacks import PickleDump, PickleUpload
from hpflow.modules.desc import AnchorBoxDetTaskDesc, KeyPoint2DTaskDesc
from hpflow.scope import get_current_config_scope_attribute
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import (
    TakeByKey,
    get_bbox2d_ptype2vis,
    get_ptype2vis,
    get_visualizer,
)
from hpflow.utils.symbol_desc_parser import parse_hat_cfg

logger = logging.getLogger(__name__)

# add desc infos

global_config = get_current_config_scope_attribute()
task = global_config.get("task", PTypeEnum.kAll)
dataset = global_config.get("dataset")
savedir = global_config.get("savedir")
overwrite = global_config.get("overwrite", False)

cfg_path = "projects/mono/person_cyclist_detection_4cls/multitask.py"  # noqa
os.environ[
    "PYTHONPATH"
] = f"{os.path.dirname(cfg_path)}:{os.environ['PYTHONPATH']}"

model_ckpt = "qat-checkpoint-last-fbe45f75.pth.tar"  # noqa
name = "hat-mono-person-detection"

ctx = copy.deepcopy(global_config.get("context", torch.device("cpu")))
if isinstance(ctx, type(mx.gpu(0))):
    ctx = torch.device(ctx.device_id)

roi_desc_parsed = parse_hat_cfg(cfg_path, is_roi_model=True)

roi_model_depends_on_ptype = PTypeEnum.kPedBBox2D
model_ckpt = model_ckpt.format(MODEL_NAME=model_ckpt)

for task_desc_i in roi_desc_parsed.task_descs:
    if isinstance(task_desc_i, AnchorBoxDetTaskDesc):
        task_desc_i.set_aux_kwargs_for_symbol_desc(
            vis_score_threshold=0.45,
            score_threshold_per_class=[
                0.5,
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
        dst_wh=[128, 128],
        # dst_wh=roi_desc_parsed.image_size[::-1],
        norm_method=roi_desc_parsed.norm_method,  # noqa
        norm_length=roi_desc_parsed.norm_length,  # noqa
        score_thresh=None,
    ),
    model=dict(
        type="ImgModel",
        model=dict(
            type="TorchModel",
            cfg=cfg_path,
            model_ckpt=model_ckpt,
            device=ctx,
            qat=True if "qat" in model_ckpt else False,
            get_input_func=TakeByKey(["img"]),
            is_roi_model=True,
            march=horizon.quantization.March.BERNOULLI2,
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
            PTypeEnum.kPed2PE: dict(
                type="AssignAttributesToBBox2Ds", attr_ptype=PTypeEnum.kPed2PE
            ),
            PTypeEnum.kCyc2PE: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kCyc2PE,
            ),
            PTypeEnum.kMotorCyc2PE: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kMotorCyc2PE,
            ),
            PTypeEnum.kTriCyc2PE: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kTriCyc2PE,
            ),
        },
        ptype=roi_model_depends_on_ptype,
    ),
    post_block=dict(
        type="BBox2DNMS",
        iou_thresh=0.5,
        multiple_cls=True,
        class_names=[
            PTypeEnum.kPed2PE,
            PTypeEnum.kCyc2PE,
            PTypeEnum.kMotorCyc2PE,
            PTypeEnum.kTriCyc2PE,
        ],
    ),
)

# visualizer

attr_ptype2vis = get_ptype2vis(
    task, roi_desc_parsed.task_descs, return_4pe_tasks=False
)  # noqa
attr_visualizer = get_visualizer(attr_ptype2vis, allow_missing=True)
ptype2vis = get_bbox2d_ptype2vis(
    roi_model_depends_on_ptype, attr_visualizer=attr_visualizer
)  # noqa
visualizer = get_visualizer(ptype2vis, allow_missing=True)
# env config
tag = ["mono_kj_2pe_detection_person_cyclist_multitask", "merge"]
enable_upload = True
enable_report = True

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
        adapt_num_workers=1,
        writter=dataset.eval_platform_result_writter,
        adaptor=None,
        project_id="PDT2020005",
        predname=name,
        enable_upload=enable_upload,
        overwrite=overwrite,
        tag=tag,
    ),
]

env = dict(
    savedir=name,
    eval_platform=dict(
        tag=tag,
        predname=name,
        reportname=name,
        diff_report_id=None,
        report=enable_report,
    ),
    backend="torch",
)
