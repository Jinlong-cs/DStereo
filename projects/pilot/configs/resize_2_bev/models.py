from collections import OrderedDict
from copy import deepcopy
from importlib import import_module

import torch
from bev_common import (
    bev_fusion_input_name,
    bev_fusion_input_stride,
    warp_sizes,
)
from common import (
    front_input_hw,
    input_hw,
    split_mode,
    tasks,
    val_decoders,
    view_num,
    with_cam_standiardization,
)

march = "bayes"

task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

inputs = dict(img=torch.zeros((1, 3, *input_hw)))


opt_inputs = dict(
    train=dict(),
    val=dict(),
    test=dict(),
)

if view_num == 1:
    traced_inputs = dict(
        img=torch.zeros((view_num, 3, *input_hw)),
    )
    if with_cam_standiardization:
        traced_inputs["uv_map"] = torch.zeros((view_num, *input_hw, 2))
else:
    traced_inputs = dict()
    if view_num in [4, 5]:
        traced_inputs.update(
            {
                "img_%d" % i: torch.zeros((1, 3, *input_hw))
                for i in range(view_num)
            }
        )
    elif view_num == 6:
        traced_inputs.update({"img_0": torch.zeros((1, 3, *front_input_hw))})
        traced_inputs.update(
            {
                "img_%d" % i: torch.zeros((1, 3, *input_hw))
                for i in range(1, view_num)
            }
        )
    traced_inputs.update(
        {"img_%d" % i: torch.zeros((1, 3, *input_hw)) for i in range(view_num)}
    )
    if with_cam_standiardization:
        traced_inputs.update(
            {
                "uv_map_%d" % i: torch.zeros((1, *input_hw, 2))
                for i in range(view_num)
            }
        )
if any(["bev" in t for t in task_names]):
    traced_inputs.update(
        {
            "homo_offset_%d" % i: torch.randn((1, s[0], s[1], 2))
            for i, s in enumerate(warp_sizes)
        }
    )

traced_inputs_split = [{}, {}]

traced_inputs_split[0].update({"img": torch.zeros((1, 3, *input_hw))})
if with_cam_standiardization:
    traced_inputs_split[0].update({"uv_map": torch.zeros((1, *input_hw, 2))})

traced_inputs_split[1].update(
    {
        "%s_%d"
        % (bev_fusion_input_name, i): torch.zeros(
            (
                1,
                16,
                int(input_hw[0] / bev_fusion_input_stride),
                int(input_hw[1] / bev_fusion_input_stride),
            )
        )
        for i in range(view_num)
    }
)
traced_inputs_split[1].update(
    {
        "homo_offset_%d" % i: torch.randn((1, s[0], s[1], 2))
        for i, s in enumerate(warp_sizes)
    }
)


def get_model(mode):
    converted_decoders = OrderedDict(
        {
            (tuple(task_names), group): decoder
            for group, (task_names, decoder) in val_decoders.items()
        }
    )

    if view_num == 1 or mode != "test":
        if view_num == 5:
            inputs = dict(img=torch.zeros((view_num, 3, *input_hw)))
        elif view_num == 6:
            inputs = dict(
                img=torch.zeros((1, 3, *front_input_hw)),
                side_img=torch.zeros((5, 3, *input_hw)),
            )
        inputs = dict(img=torch.zeros((view_num, 3, *input_hw)))
        if with_cam_standiardization:
            opt_inputs[mode]["uv_map"] = None
    else:
        inputs = dict()
        if view_num in [4, 5]:
            inputs.update(
                {
                    "img_%d" % i: torch.zeros((1, 3, *input_hw))
                    for i in range(view_num)
                }
            )
            if with_cam_standiardization:
                opt_inputs[mode].update(
                    {
                        "uv_map_%d" % i: torch.zeros((1, *input_hw, 2))
                        for i in range(view_num)
                    }
                )
        elif view_num == 6:
            inputs.update({"img_0": torch.zeros((1, 3, *front_input_hw))})
            inputs.update(
                {
                    "img_%d" % i: torch.zeros((1, 3, *input_hw))
                    for i in range(1, view_num)
                }
            )
            if with_cam_standiardization:
                opt_inputs[mode].update(
                    {"uv_map_0": torch.zeros((1, *front_input_hw, 2))}
                )
                opt_inputs[mode].update(
                    {
                        "uv_map_%d" % i: torch.zeros((1, *input_hw, 2))
                        for i in range(1, view_num)
                    }
                )

    # inputs = dict(img=torch.zeros((view_num, 3, *input_hw)))
    if any(["bev" in t for t in task_names]):
        if mode in ["train", "val"]:
            opt_inputs[mode]["homo_offset"] = torch.randn(
                (view_num, 256, 256, 2)
            )
        elif mode == "test":
            opt_inputs[mode].update(
                {
                    "homo_offset_%d" % i: torch.randn((1, s[0], s[1], 2))
                    for i, s in enumerate(warp_sizes)
                }
            )
    return dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.inputs[mode] for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode) for T in TASK_CONFIGS},
        funnel_modules=converted_decoders if "val" in mode else None,
        flatten_outputs="val" not in mode,
        lazy_forward=False,
        force_cpu_init=True,  # currently init on gpu for multitask will cause cuda oom
        # __build_recursive=False,
    )


def get_model_split(mode):
    assert mode == "test", "split model only for test mode"
    inputs_stage1 = dict(
        img=torch.zeros((1, 3, *input_hw)),
    )
    if with_cam_standiardization:
        inputs_stage1["uv_map"] = torch.zeros((1, *input_hw, 2))
    inputs_stage2 = dict()
    inputs_stage2.update(
        {
            "%s_%d"
            % (bev_fusion_input_name, i): torch.randn(
                (
                    1,
                    16,
                    int(input_hw[0] / bev_fusion_input_stride),
                    int(input_hw[1] / bev_fusion_input_stride),
                )
            )
            for i in range(view_num)
        }
    )
    inputs_stage2.update(
        {
            "homo_offset_%d" % i: torch.randn((1, s[0], s[1], 2))
            for i, s in enumerate(warp_sizes)
        }
    )

    def rebuild_stage1(model):
        if model["type"] == "ListInputModelWraper":
            model = model["model"]
        if model["type"] == "TwoStageBEVModule":
            stage1_model = model["stage1_module"]
        else:
            stage1_model = model
        return stage1_model

    def rebuild_stage2(model):
        if model["type"] == "ListInputModelWraper":
            model = model["model"]
        if model["type"] == "TwoStageBEVModule":
            stage2_model = model["stage2_module"]
        else:
            raise NotImplementedError(
                "BEV stage2 should exist in split mode, but%s" % model["type"]
            )
        return stage2_model

    stage1_tasks = [T for T in TASK_CONFIGS if "bev" not in T.task_name]
    for T in TASK_CONFIGS:
        if "bev" in T.task_name:
            stage1_tasks.append(T)
            break
    stage1_model = dict(
        type="MultitaskGraphModel",
        inputs=inputs_stage1,
        task_inputs={T.task_name: T.inputs[mode] for T in stage1_tasks},
        task_modules={
            T.task_name: rebuild_stage1(T.get_model(mode))
            for T in TASK_CONFIGS
        },
        funnel_modules=None,
        flatten_outputs=True,
        lazy_forward=False,
        # __build_recursive=False,
        force_cpu_init=True,  # currently init on gpu for multitask will cause cuda oom
    )

    stage2_tasks = [T for T in TASK_CONFIGS if "bev" in T.task_name]
    stage2_model = dict(
        type="MultitaskGraphModel",
        inputs=inputs_stage2,
        task_inputs={T.task_name: T.inputs[mode] for T in stage2_tasks},
        task_modules={
            T.task_name: rebuild_stage2(T.get_model(mode))
            for T in stage2_tasks
        },
        funnel_modules=None,
        flatten_outputs=True,
        lazy_forward=False,
        # __build_recursive=False,
        force_cpu_init=True,  # currently init on gpu for multitask will cause cuda oom
    )
    return dict(
        type="BEVSplitModuleWrapper",
        mid_feature_name=bev_fusion_input_name,
        stage1_module=stage1_model,
        stage2_module=stage2_model,
        view_num=view_num,
        remap_prefix=(("input_cat.quant", "backbone.quant"),),
    )


model = deepcopy(get_model("train"))
val_model = deepcopy(get_model("val"))
test_model = deepcopy(
    get_model_split("test") if split_mode else get_model("test")
)
