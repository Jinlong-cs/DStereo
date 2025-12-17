# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from hatbc.utils import _as_list
    from hatbc.workflow import make_traceable
except ImportError:
    _as_list = None
    make_traceable = property
from horizon_plugin_pytorch.quantization import QuantStub

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import flatten

__all__ = [
    "TwoStageBEVModule",
    "ListInputModelWraper",
    "ListInputPreprocess",
    "BEVSplitModuleWrapper",
    "MultiViewTwoStageBEVModule",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BEVStageOneModule(nn.Module):
    """
    The basic structure of stage1 in TwoStageBEVModule.

    We commonly extract features in image view in stage1.

    Args:
        backbone: Stage1 backbone module.
        neck: Stage1 neck module.
        head: Stage1 head module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
    ):
        super(BEVStageOneModule, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, uv_map=None):
        feat = self.backbone(x, uv_map=uv_map)
        feat = self.neck(feat)
        return self.head([feat])

    def fuse_model(self):
        for module in self.children():
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    @property
    def input_warping(self):
        return (
            hasattr(self.backbone, "warping_module")
            and self.backbone.warping_module is not None
        )

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class BEVStageTwoModule(nn.Module):
    """
    The basic structure of stage2 in BEVStageTwoModule.

    We usually apply transform on image view features
    and obtain BEV output in stage2.

    Args:
        bevfusion: BEV fusion module.
        backbone: Stage2 backbone module.
        neck: Stage2 neck module.
        head: Stage2 head module.
        temporal_fusion: Temporal fusion module.
        bev_fusion_upsample: Upsample module.
    """

    def __init__(
        self,
        bevfusion: nn.Module,
        backbone: nn.Module = None,
        neck: nn.Module = None,
        head: Optional[nn.Module] = None,
        roi_resizes: Optional[Union[nn.Module, List[nn.Module]]] = None,
        bev_fusion_upsample: Optional[nn.Module] = None,
        temporal_fusion: Optional[nn.Module] = None,
        temporal_output: Optional[nn.Module] = None,
    ):
        super(BEVStageTwoModule, self).__init__()

        self.bevfusion = bevfusion
        self.temporal_fusion = temporal_fusion
        self.bev_fusion_upsample = bev_fusion_upsample

        self.backbone = backbone
        self.neck = neck
        self.roi_resizes = _as_list(roi_resizes) if roi_resizes else None
        self.head = head
        self.temporal_output = temporal_output

    def forward(self, feats, inputs=None):
        bev_rot_mat, bev_fusion_out = self.bevfusion(feats, meta=inputs)

        if self.temporal_fusion is not None:
            temporal_fusion_out, hidden_feats = self.temporal_fusion(
                bev_fusion_out,
                meta=inputs,
            )
            bev_fusion_out = temporal_fusion_out
            if self.temporal_output:
                hidden_feats = self.temporal_output(hidden_feats)
                return hidden_feats

        if self.bev_fusion_upsample is not None:
            bev_fusion_out = self.bev_fusion_upsample(bev_fusion_out)

        bev_stage2_feats = self.backbone(bev_fusion_out)
        bev_stage2_feats = self.neck(bev_stage2_feats)

        bev_feats = [bev_stage2_feats]
        if self.roi_resizes:
            resize_feats = []
            for feat in bev_feats:
                feats = [roi_resize(feat) for roi_resize in self.roi_resizes]
                resize_feats.append(feats)
            bev_feats = resize_feats

        bev_pred = self.head(
            bev_feats,
            inputs,  # target
        )

        return bev_pred

    def fuse_model(self):
        for module in self.children():
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class TwoStageBEVModule(nn.Module):
    """
    The basic structure of TwoStageBEVModule.

    Args:
        stage1_module: stage1 module contains feature extraction on image.
        stage2_module: Stage2 module contains feature fusion and
                       subtasks in BEV space.
        stage1_input_keys: stage1 input keys.
        bev_fusion_input_name: Name for fusion input.
        view_num: Input sequence length.
    """

    def __init__(
        self,
        stage1_module: nn.Module,
        stage2_module: nn.Module,
        stage1_input_keys: Optional[Union[str, Sequence[str]]] = "img",
        bev_fusion_input_name: Optional[str] = "",
        view_num: Optional[int] = 1,
    ):
        super(TwoStageBEVModule, self).__init__()

        self.stage1 = stage1_module
        self.stage2 = stage2_module

        self.bev_fusion_input_name = bev_fusion_input_name
        self.view_num = view_num
        if isinstance(stage1_input_keys, str):
            self.stage1_input_keys = [stage1_input_keys]
        else:
            self.stage1_input_keys = stage1_input_keys

    def forward(self, data: Dict[str, Any]):
        feat_list = []
        for key in self.stage1_input_keys:
            stage1_input = {"img": data[key]}
            feat = self.stage1(stage1_input)
            feat_list += feat[self.bev_fusion_input_name]
        if "homo_offset" not in data.keys():
            data["homo_offset"] = [
                data["homo_offset_%d" % i] for i in range(self.view_num)
            ]
        stage2_input = OrderedDict(
            {
                self.bev_fusion_input_name: feat_list,
                "homo_offset": data["homo_offset"],
            }
        )
        stage2_label = (
            {"gt_bev_3d": data["gt_bev_3d"]} if "gt_bev_3d" in data else None
        )
        return self.stage2(stage2_input, stage2_label)

    def fuse_model(self):
        for module in [self.stage1, self.stage2]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.stage1, self.stage2]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class BEVSplitModuleWrapper(TwoStageBEVModule):
    """
    Wrapper for TwoStageBEVModule during separate trace or sequence infer.

    Args:
        mid_feature_name: Name for middle feature betwwen stage1 and stage2.
        stage1_module: stage1 module contains feature extraction on image.
        stage2_module: Stage2 module contains feature fusion and
                       subtasks in BEV space.
        view_num: Input sequence length.
        split_index: whether to infer stage1 model or stage2 model only.
        remap_prefix: paramter key tuples,
    """

    def __init__(
        self,
        mid_feature_name: str = "",
        stage1_module: nn.Module = None,
        stage2_module: nn.Module = None,
        view_num: int = 1,
        split_index: int = None,
        remap_prefix: Tuple[Tuple[str]] = (),
    ):
        super(BEVSplitModuleWrapper, self).__init__(
            stage1_module,
            stage2_module,
            mid_feature_name,
            view_num=view_num,
        )
        self.mid_feature_name = mid_feature_name
        assert split_index is None or split_index in [0, 1]
        self.split_index = split_index
        for r in remap_prefix:
            assert len(r) == 2
        self.remap_prefix = remap_prefix

        def hook(state_dict, prefix, *args):
            # prefix remapping
            if any(self.remap_prefix):
                for name, param in state_dict.items():
                    matched_prefix = [
                        r for r in self.remap_prefix if name.startswith(r[0])
                    ]
                    if any(matched_prefix):
                        key = name.replace(
                            matched_prefix[0][0], matched_prefix[0][1]
                        )
                        state_dict[key] = param

            # add extra prefix
            for name, _ in self.state_dict().items():
                if (
                    name.startswith("stage1.") or name.startswith("stage2.")
                ) and name not in state_dict:
                    key = name[7:]
                    if key in state_dict:
                        state_dict[name] = state_dict[key]
                        del state_dict[key]

        self._register_load_state_dict_pre_hook(hook)

    def forward(
        self,
        inputs: Dict[str, Any],
        out_names: Optional[Union[str, Sequence[str]]] = None,
    ) -> namedtuple:  # noqa: D205,D400

        # partially infer with split model for model trace
        if self.split_index is not None:
            return (self.stage1, self.stage2)[self.split_index](
                inputs, out_names
            )

        # sequence infer
        regroup_outputs = []
        for i in range(self.view_num):
            inputs["img"] = inputs["img_%d" % i]
            outputs = self.stage1(inputs, out_names)
            mid_feature_stored = False
            for output in outputs:
                if any(
                    [k for k in output._fields if self.mid_feature_name in k]
                ):
                    if not mid_feature_stored:
                        assert (
                            len(output) == 1
                        ), "only outputs single mid feature map"
                        inputs["%s_%d" % (self.mid_feature_name, i)] = output[
                            0
                        ][0]
                        mid_feature_stored = True
                else:
                    regroup_outputs.append(output)
            assert mid_feature_stored
        regroup_outputs.extend(self.stage2(inputs, out_names))
        return tuple(regroup_outputs)


@OBJECT_REGISTRY.register
class ListInputModelWraper(nn.Module):
    """
    Wraper to support different type of inputs.

    Currently if we train model in concat input and compile
    in list input, a cat op is required in complile which is not
    contained in training. In this module, we make it consistent
    between compile and training.
    input images should be prepared in "img_%d" item in input dict.

    Args:
        model: Wrapped module.
        seq_len: Input sequence length.
        preprocess: Preprocess module.
    """

    def __init__(
        self,
        model: nn.Module,
        seq_len: int,
        preprocess: nn.Module = None,
    ):
        super(ListInputModelWraper, self).__init__()
        self.seq_len = seq_len
        self.model = model
        self.preprocess = preprocess

    def forward(self, data: Dict[str, Any]):

        if self.seq_len == 1:
            return self.model(data)
        else:
            if "img" in data or "side_img" in data:
                img_proc = self.preprocess(data["img"])
                data["img"] = img_proc
                if "side_img" in data:
                    cat_side_img = self.preprocess(data["side_img"])
                    data["side_img"] = cat_side_img
                # _input = torch.split(data["img"], 1, dim=0)
            else:
                _input = [data["img_%d" % i] for i in range(self.seq_len)]
                img_proc = self.preprocess(_input)
                data["img"] = img_proc
            return self.model(data)

    def fuse_model(self):
        for module in self.children():
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class ListInputPreprocess(nn.Module):
    """
    Preprocess module for list input wrapper.

    Basically we quantize input and cat img list.
    When input is a batch tensor, we split it first.

    Args:
        need_quant: Whether to quant input.
        need_cat: Whether to cat input.
    """

    def __init__(
        self,
        need_quant: bool = True,
        need_cat: bool = True,
    ):
        super(ListInputPreprocess, self).__init__()
        self.need_quant = need_quant
        self.need_cat = need_cat
        if need_quant:
            self.quant = QuantStub(1.0 / 128)
        else:
            self.quant = None

        self.cat = nn.quantized.FloatFunctional()

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = torch.split(inputs, 1, dim=0)
        input_list = []
        for i in range(len(inputs)):
            input_list.append(
                self.quant(inputs[i]) if self.need_quant else inputs[i]
            )

        if self.need_cat:
            return self.cat.cat(input_list, 0)
        else:
            return input_list

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class FlattenCollect(nn.Module):
    """Preprocess multi-view input for bev struct module."""

    def __init__(self):
        super(FlattenCollect, self).__init__()
        self.flatten = lambda x: flatten(x)[0]

    def forward(self, inputs):
        out = self.flatten(inputs)
        return out


@OBJECT_REGISTRY.register
class MultiViewTwoStageBEVModule(nn.Module):
    """
    Preprocess multi-view input for bev struct module.

    Args:
        multi_view_module: Bev stage one module.
        bev_fusion_module: Bev stage two module.
        multi_view_collect: Collector for multi views.
        bevfusion_pick_keys: Keys of homo offset for deploy.
    """

    def __init__(
        self,
        multi_view_module: Dict[str, nn.Module],
        bev_fusion_module: nn.Module = None,
        multi_view_collect: nn.Module = None,
        bevfusion_pick_keys: Optional[List[str]] = None,
    ):
        super(MultiViewTwoStageBEVModule, self).__init__()
        self.multi_view_module = multi_view_module
        self.bev_fusion_module = bev_fusion_module
        if multi_view_collect is None:
            multi_view_collect = lambda x: flatten(x)[0]
            multi_view_collect = make_traceable(multi_view_collect)
        self.multi_view_collect = multi_view_collect
        self.bevfusion_pick_keys = None
        if bevfusion_pick_keys is not None:
            assert isinstance(bevfusion_pick_keys, list)
            self.bevfusion_pick_keys = bevfusion_pick_keys

    def forward(self, inputs):
        # forward multi view to multi modules.
        multi_view_feats = []
        for view_img_key, module in self.multi_view_module.items():
            if len(inputs[view_img_key][0]) == 0 and "img" in inputs:
                inputs[view_img_key] = inputs["img"]
            assert view_img_key in inputs

            if hasattr(module, "input_warping") and module.input_warping:
                uv_map_name = f"{view_img_key}_uv_map"
                if uv_map_name not in inputs:
                    inputs[uv_map_name] = inputs["uv_map"]
                assert uv_map_name in inputs
                _view_out = module(inputs[view_img_key], inputs[uv_map_name])
            else:
                _view_out = module(inputs[view_img_key])
            multi_view_feats.append(_view_out)

        bev_fusion_input = self.multi_view_collect(multi_view_feats)
        if self.bevfusion_pick_keys is not None:
            inputs = {k: inputs[k] for k in self.bevfusion_pick_keys}
        if self.bev_fusion_module is not None:
            bev_fusion_out = self.bev_fusion_module(bev_fusion_input, inputs)
            return bev_fusion_out
        else:
            return bev_fusion_input
