from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.qtensor import QTensor
from torch.quantization import DeQuantStub

from hat.core.box_utils import box_corner_to_center
from hat.models.task_modules.dino.cdn import dn_post_process, prepare_for_cdn
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["DINO"]


@OBJECT_REGISTRY.register
class DINO(nn.Module):
    """Object Detection method DINO.

    Args:
        backbone: backbone module
        position_embedding: position embedding module
        neck: neck module to handle the intermediate outputs features
        transformer: transformer module
        embed_dim: dimension of embedding
        num_classes: Number of total categories.
        num_queries: Number of proposal dynamic anchor boxes in Transformer
        post_process: Post process module for inference.
        criterion: Criterion for calculating the total losses.
        aux_loss: Whether to calculate auxiliary loss in criterion.
        dn_number: denoise instance number.
        label_noise_ratio: label noise ratio for cdn loss.
        box_noise_scale: box noise scale for cdn loss.
        as_two_stage: Whether using two stage, refering to deform detr.
        with_box_refine: Whether with box refine, refering to deform detr.
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        post_process: Optional[nn.Module] = None,
        criterion: Optional[nn.Module] = None,
        aux_loss: bool = True,
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        as_two_stage: bool = True,
        with_box_refine: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.position_embedding = position_embedding

        self.neck = neck

        self.set_int16_qconfig = True

        self.num_queries = num_queries
        self.embed_dim = embed_dim

        self.transformer = transformer

        self.num_classes = num_classes

        self.aux_loss = aux_loss
        self.criterion = criterion

        # denoising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self.as_two_stage = as_two_stage
        self.with_box_refine = with_box_refine

        self.post_process = post_process

        self.dequant = DeQuantStub()

        self.coord_add = FloatFunctional()

        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

    @fx_wrap()
    def gen_img_masks(self, batched_inputs: dict):
        images = batched_inputs["img"]
        if self.training:
            batch_size, _, H, W = images.shape
            img_masks = images.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                if "before_pad_shape" in batched_inputs:
                    img_h, img_w = batched_inputs["before_pad_shape"][img_id]
                elif "img_shape" in batched_inputs:
                    _, img_h, img_w = batched_inputs["img_shape"][img_id]
                else:
                    img_h, img_w = self.input_shape
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.shape
            img_masks = torch.zeros(batch_size, H, W).to(images.device)
        return img_masks

    @fx_wrap()
    def compute_pos_emdding(
        self, multi_level_feats: List, batched_inputs: Dict
    ):
        img_masks = self.gen_img_masks(batched_inputs)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:])
                .to(torch.bool)
                .squeeze(0)
            )
            multi_level_position_embeddings.append(
                self.position_embedding(multi_level_masks[-1])
            )
        return multi_level_masks, multi_level_position_embeddings

    @fx_wrap()
    def prepare_for_cdn_wrap(self, batched_inputs: dict):
        if self.training:
            targets = self.prepare_targets(batched_inputs)
            (
                input_query_label,
                input_query_bbox,
                attn_mask,
                dn_meta,
            ) = prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )
            targets = None
        return input_query_label, input_query_bbox, attn_mask, dn_meta, targets

    @fx_wrap()
    def for_training_do_nothing(self, inter_states: List[torch.Tensor]):
        if self.training and not isinstance(inter_states[0], QTensor):
            # hack implementation for distributed training, but not for qat
            inter_states[0] = (
                self.label_enc.weight[0, 0] * 0.0 + inter_states[0].clone()
            )
        return inter_states

    @fx_wrap()
    def tmp_addref2(self, lvl, tmp, reference):
        tmp[..., :2] = self.coord_add[lvl].add(tmp[..., :2], reference)

    def forward(self, batched_inputs: dict):
        images = batched_inputs["img"]

        # original features
        features = self.backbone(images)  # output feature dict

        multi_level_feats = self.neck(features)

        multi_level_masks, multi_level_pos_embeds = self.compute_pos_emdding(
            multi_level_feats, batched_inputs
        )

        # denoising preprocessing
        # prepare label query embedding
        (
            input_query_label,
            input_query_bbox,
            attn_mask,
            dn_meta,
            targets,
        ) = self.prepare_for_cdn_wrap(batched_inputs)
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            inter_bboxes,
            init_reference_unact,
            inter_references_unact,
            enc_state,
            enc_reference_unact,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embeds,
            attn_masks=[attn_mask, None],
        )
        length = self.transformer.decoder.num_layers

        inter_states = self.for_training_do_nothing(inter_states)

        output_classes = self.transformer.decoder.class_embed[length - 1](
            inter_states[length - 1]
        )
        output_coords = self.coord_add.add(
            inter_bboxes[length - 1], inter_references_unact[length - 2]
        )

        # # Calculate output coordinates and classes.
        inter_state_list = []
        reference_list = []
        inter_boxes_list = []
        for lvl in range(self.transformer.decoder.num_layers - 1):
            if lvl == 0:
                reference = init_reference_unact
            else:
                reference = inter_references_unact[lvl - 1]
            reference_list.append(self.dequant(reference))
            inter_boxes_list.append(self.dequant(inter_bboxes[lvl]))
            inter_state_list.append(self.dequant(inter_states[lvl]))

        final_classes = self.dequant(output_classes)
        final_coords = self.dequant(output_coords)

        # denoising postprocessing
        outputs = self._post_process(
            batched_inputs,
            final_classes,
            final_coords,
            inter_state_list,
            inter_boxes_list,
            reference_list,
            enc_state,
            enc_reference_unact,
            dn_meta,
            targets,
        )
        return outputs

    @fx_wrap()
    def _post_process(
        self,
        batched_inputs: dict,
        final_classes: torch.Tensor,
        final_coords: torch.Tensor,
        inter_state_list: List[torch.Tensor],
        inter_boxes_list: List[torch.Tensor],
        reference_list: List[torch.Tensor],
        enc_states: torch.Tensor,
        enc_reference_unact: torch.Tensor,
        dn_meta: Optional[dict] = None,
        targets: Optional[dict] = None,
    ):
        if self.training:
            outputs_classes = []
            outputs_coords = []
            for lvl in range(self.transformer.decoder.num_layers - 1):
                pred_class = self.transformer.decoder.class_embed[lvl](
                    inter_state_list[lvl]
                )
                outputs_classes.append(pred_class)

                coords = inter_boxes_list[lvl] + reference_list[lvl]
                outputs_coords.append(coords)
            outputs_coords.append(final_coords)
            outputs_classes.append(final_classes)
            outputs_class = torch.stack(outputs_classes)

            outputs_coord = torch.stack(
                [coord.sigmoid() for coord in outputs_coords]
            )

            if dn_meta is not None:
                outputs_class, outputs_coord = dn_post_process(
                    outputs_class, outputs_coord, dn_meta, self.aux_loss
                )

            # prepare for loss computation
            output = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
            if self.aux_loss:
                output["aux_outputs"] = self._set_aux_loss(
                    outputs_class, outputs_coord
                )

            # prepare two stage output
            if self.as_two_stage:
                interm_class = self.dequant(
                    self.transformer.decoder.class_embed[-1](enc_states)
                )
                enc_reference_unact = self.dequant(enc_reference_unact)
                interm_coord = enc_reference_unact.sigmoid()
                output["enc_outputs"] = {
                    "pred_logits": interm_class,
                    "pred_boxes": interm_coord,
                }
            loss_dict = self.criterion(output, targets, dn_meta)
            return loss_dict
        else:
            box_cls = final_classes
            box_pred = final_coords
            if not self.post_process:
                return box_cls, box_pred

            results = self.post_process(
                batched_inputs,
                box_cls,
                box_pred,
            )
            return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_targets(self, targets):
        bs = len(targets["gt_classes"])
        if "before_pad_shape" in targets:
            shapes = targets["before_pad_shape"]
        else:
            shapes = targets["img_shape"]
        for i in range(bs):
            bbox = targets["gt_bboxes"][i].float()
            h, w = shapes[i][-2:]
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=bbox.device
            )
            bbox = bbox / image_size_xyxy
            bbox = box_corner_to_center(bbox)
            targets["gt_classes"] = [c.long() for c in targets["gt_classes"]]
            targets["gt_bboxes"][i] = bbox
        return targets

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if self.set_int16_qconfig:
            self.transformer.decoder.bbox_embed[-1].layers[
                -1
            ].qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )  # encoder bbox head

            self.coord_add.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
        else:
            from hat.utils import qconfig_manager

        for i in range(self.transformer.decoder.num_layers):
            self.transformer.decoder.class_embed[i].qconfig = None
        self.transformer.decoder.class_embed[
            self.transformer.decoder.num_layers - 1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        self.transformer.decoder.bbox_embed[
            self.transformer.decoder.num_layers - 1
        ].layers[-1].qconfig = qconfig_manager.get_default_qat_out_qconfig()

        for module in [self.backbone, self.neck, self.transformer]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
