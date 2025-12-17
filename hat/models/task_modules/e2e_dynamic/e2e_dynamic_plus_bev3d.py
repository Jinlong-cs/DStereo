from typing import List, Optional, Union

from hatbc.utils import _as_list
from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.trace import combine_dict, get_part_dict


@OBJECT_REGISTRY.register
class E2EPlusBEV3DHead(nn.Module):
    """Head contains bev3d and e2e tasks.

    Args:
        veh_head: module of vehicle task output head.
        vru_head: module of vrumerge task output head.
        e2e_dynamic_head: module of e2e task output head.
        veh_roi_resizes: roi resizers in vehicle task.
        vru_roi_resizes: roi resizers in vrumerge task.
        compile_mode: whether is compile mode.
    """

    def __init__(
        self,
        veh_head: nn.Module,
        vru_head: nn.Module,
        e2e_dynamic_head: nn.Module,
        veh_roi_resizes: Optional[Union[nn.Module, List[nn.Module]]] = None,
        vru_roi_resizes: Optional[Union[nn.Module, List[nn.Module]]] = None,
        compile_mode: bool = False,
    ):
        super().__init__()
        self.veh_head = veh_head
        self.vru_head = vru_head
        self.e2e_dynamic_head = e2e_dynamic_head
        self.veh_roi_resizes = (
            _as_list(veh_roi_resizes) if veh_roi_resizes else None
        )
        self.vru_roi_resizes = (
            _as_list(vru_roi_resizes) if vru_roi_resizes else None
        )
        self.compile_mode = compile_mode

    def forward(self, feats, inputs=None):

        if self.compile_mode:
            tracker_feats = self.e2e_dynamic_head(feats[0])
            return tracker_feats

        if self.veh_roi_resizes:
            veh_feats = []
            for feat in feats:
                feats = [
                    roi_resize(feat) for roi_resize in self.veh_roi_resizes
                ]
                veh_feats.append(feats)
        else:
            veh_feats = feats

        if self.vru_roi_resizes:
            vru_feats = []
            for feat in feats:
                feats = [
                    roi_resize(feat) for roi_resize in self.vru_roi_resizes
                ]
                vru_feats.append(feats)
        else:
            vru_feats = feats

        if "veh_gt" in inputs:
            veh_pred = self.veh_head(veh_feats, inputs.pop("veh_gt"))
        else:
            veh_pred = self.veh_head(veh_feats, inputs)
        if "vru_gt" in inputs:
            vru_pred = self.vru_head(vru_feats, inputs.pop("vru_gt"))
        else:
            vru_pred = self.vru_head(vru_feats, inputs)

        inputs.update({"veh_feats": veh_feats, "vru_feats": vru_feats})
        inputs = combine_dict(inputs, veh_pred, vru_pred)

        e2e_dynamic_pred = self.e2e_dynamic_head(inputs)

        veh_head_out_for_return = get_part_dict(
            veh_pred,
            [
                "bev_stage2_3d_vehicle_head__bev3d_hm_loss",
                "bev_stage2_3d_vehicle_head__bev3d_ct_offset_loss",
            ],
        )
        vru_head_out_for_return = get_part_dict(
            vru_pred,
            [
                "bev_stage2_3d_vrumerge_head__bev3d_hm_loss",
                "bev_stage2_3d_vrumerge_head__bev3d_ct_offset_loss",
            ],
        )

        res = combine_dict(
            e2e_dynamic_pred, veh_head_out_for_return, vru_head_out_for_return
        )

        return res

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
