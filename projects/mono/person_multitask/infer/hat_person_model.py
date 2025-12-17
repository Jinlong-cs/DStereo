import argparse
import logging
import os

# import horizon_plugin_pytorch as horizon
import torch
from hpflow.modules.desc import AnchorBoxDetTaskDesc, KeyPoint2DTaskDesc
from hpflow.scope import get_current_config_scope_attribute
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import TakeByKey, get_visualizer
from hpflow.utils.symbol_desc_parser import parse_hat_cfg
from person_adapters import PersonAdapter, TakePredictionByKey

from hat.utils.apply_func import _as_list

logger = logging.getLogger(__name__)

# add desc infos
global_config = get_current_config_scope_attribute()
eval_task = global_config.get("task", PTypeEnum.kAll)

current_path = os.path.dirname(os.path.abspath(__file__))
cfg_path = f"{current_path}/../entry.py"


def get_parser():
    parser = argparse.ArgumentParser(
        description="argument for workflow builder"
    )
    parser.add_argument("--eval-cfg-model-name", default="")
    parser.add_argument("--eval-qat", default="qat")
    parser.add_argument("--eval-joint-4pe-2pe", action="store_true")
    parser.add_argument("--eval-joint-4pe-score", type=float, default=0.35)
    parser.add_argument("--eval-dataset-task", default=None)
    args, argv = parser.parse_known_args()
    if argv:
        logger.warning(
            f"unrecognized arguments in person"
            f" workflow parser : {' '.join(argv)}"
        )

    return args


args = get_parser()


model_name = args.eval_cfg_model_name
eval_qat = args.eval_qat
model_step = 0
model_ckpt = f"http://fm-shu-zhang.train.hogpu.cc/plat_gpu/{model_name}/output/models/mono_multitask_2pe_person/qat-checkpoint-step-4999.pth.tar"  # noqa
if eval_qat == "qat":
    model_step = 9999
    model_ckpt = f"http://fm-shu-zhang.train.hogpu.cc/plat_gpu/{model_name}/output/models/person_multitask/qat-checkpoint-step-{model_step}.pth.tar"  # noqa
elif eval_qat == "float":
    model_step = 24999
    model_ckpt = f"http://fm-shu-zhang.train.hogpu.cc/plat_gpu/{model_name}/output/models/person_multitask/float-checkpoint-step-{model_step}.pth.tar"  # noqa

model_name = f"{eval_qat}_{model_step}_{model_name}"
save_dir = f'{global_config.get("savedir")}/person_multitask_2pe/{model_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

extra_leaderboard_flags = os.environ.get(
    "HPFLOW_UPLOAD_FLAGS", model_name
).split(
    ","
)  # noqa
ctx = global_config.get("context", torch.device("cpu"))
roi_desc_parsed = parse_hat_cfg(cfg_path, is_roi_model=True)
roi_model_depends_on_ptype = PTypeEnum.kPedBBox2D
# model_ckpt = model_ckpt.format(MODEL_NAME=args.eval_cfg_model_name)

for task_desc_i in roi_desc_parsed.task_descs:
    if isinstance(task_desc_i, AnchorBoxDetTaskDesc):
        task_desc_i.set_aux_kwargs_for_symbol_desc(
            vis_score_threshold=0.1,
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
            qat=True if eval_qat != "float" else False,
            get_input_func=TakeByKey(["img"]),
            is_roi_model=True,
            march="bernoulli2",
            allow_miss=True,
            convert=False,
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
            PTypeEnum.kPedHeadBBox2D: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kPedHeadBBox2D,
            ),
            PTypeEnum.kPedAge: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kPedAge,
            ),
            PTypeEnum.kPedPose: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kPedPose,
            ),
            PTypeEnum.kPedOrientation: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kPedOrientation,
            ),
            PTypeEnum.kPedOcclusion: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kPedOcclusion,
            ),
            PTypeEnum.kPedPosNeg: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kPedPosNeg,
            ),
            PTypeEnum.kPedPosNegOcclusion: dict(
                type="AssignAttributesToBBox2Ds",
                attr_ptype=PTypeEnum.kPedPosNegOcclusion,
            ),
            PTypeEnum.kPed2PE: dict(
                type="AssignAttributesToBBox2Ds", attr_ptype=PTypeEnum.kPed2PE
            ),
            PTypeEnum.kCyc2PE: dict(
                type="AssignAttributesToBBox2Ds", attr_ptype=PTypeEnum.kCyc2PE
            ),
        },
        ptype=roi_model_depends_on_ptype,
    ),
)

# visualizer
ptype2vis = dict()
visualizer = get_visualizer(ptype2vis, allow_missing=True)
eval_task = args.eval_dataset_task
# env config
tags = ["2pe_person_multitask", "merge"] + extra_leaderboard_flags
env = dict(
    savedir=save_dir,
    symbolic_mode=True,
    eval_platform=dict(
        tag=tags,
        predname=model_name,
        reportname=model_name,
        adaptor=PersonAdapter(
            key=eval_task,
            take_input_func=TakePredictionByKey(eval_task),
            joint_4pe_2pe=args.eval_joint_4pe_2pe,
            score_thresh_4pe=args.eval_joint_4pe_score,
        ),
    ),
    backend="torch",
)
