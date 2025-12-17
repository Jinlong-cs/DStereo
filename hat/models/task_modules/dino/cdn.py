import torch
import torch.nn as nn

from .layers import inverse_sigmoid


def prepare_for_cdn(
    targets: dict,
    dn_number: int,
    label_noise_ratio: float,
    box_noise_scale: float,
    num_queries: int,
    num_classes: int,
    hidden_dim: int,
    label_enc: nn.Module,
):
    """Prepare data for Contrastive Denoising Training (CDN).

    Args:
        targets: A dictionary containing ground truth classes
            and bounding boxes.
        dn_number: Number of denoising boxes number.
        label_noise_ratio: Ratio of label noise to be applied.
        box_noise_scale: Scale of box noise to be applied.
        num_queries: Number of queries.
        num_classes: Number of classes in the dataset.
        hidden_dim: Dimension of the hidden layers.
        label_enc: Encoder for label information.

    Returns:
        input_query_label (torch.Tensor): Input tensor for label queries.
        input_query_bbox (torch.Tensor): Input tensor for bounding box queries.
        attn_mask (torch.Tensor): Attention mask.
        dn_meta (dict): Metadata for CDN training.
    """
    if dn_number <= 0:
        return None, None, None, None
        # positive and negative dn queries
    dn_number = dn_number * 2
    known = [(torch.ones_like(t)).cuda() for t in targets["gt_classes"]]
    batch_size = len(known)
    known_num = [sum(k) for k in known]
    if int(max(known_num)) == 0:
        return None, None, None, None

    dn_number = dn_number // (int(max(known_num) * 2))

    if dn_number == 0:
        dn_number = 1
    unmask_bbox = unmask_label = torch.cat(known)
    labels = torch.cat([t for t in targets["gt_classes"]])  # noqa [C416]
    boxes = torch.cat([t for t in targets["gt_bboxes"]])  # noqa [C416]
    batch_idx = torch.cat(
        [
            torch.full_like(t.long(), i)
            for i, t in enumerate(targets["gt_classes"])
        ]
    )

    known_indice = torch.nonzero(unmask_label + unmask_bbox)
    known_indice = known_indice.view(-1)

    known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
    known_labels = labels.repeat(2 * dn_number, 1).view(-1)
    known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
    known_bboxs = boxes.repeat(2 * dn_number, 1)
    known_labels_expaned = known_labels.clone()
    known_bbox_expand = known_bboxs.clone()

    if label_noise_ratio > 0:
        p = torch.rand_like(known_labels_expaned.float())
        chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
            -1
        )  # half of bbox prob
        new_label = torch.randint_like(
            chosen_indice, 0, num_classes
        )  # randomly put a new one here
        known_labels_expaned.scatter_(0, chosen_indice, new_label)
    single_padding = int(max(known_num))

    pad_size = int(single_padding * 2 * dn_number)
    positive_idx = (
        torch.tensor(range(len(boxes)))
        .long()
        .cuda()
        .unsqueeze(0)
        .repeat(dn_number, 1)
    )
    positive_idx += (
        (torch.tensor(range(dn_number)) * len(boxes) * 2)
        .long()
        .cuda()
        .unsqueeze(1)
    )

    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + len(boxes)

    if box_noise_scale > 0:
        known_bbox_ = torch.zeros_like(known_bboxs)
        known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
        known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

        diff = torch.zeros_like(known_bboxs)
        diff[:, :2] = known_bboxs[:, 2:] / 2
        diff[:, 2:] = known_bboxs[:, 2:] / 2

        rand_sign = (
            torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32)
            * 2.0
            - 1.0
        )
        rand_part = torch.rand_like(known_bboxs)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign
        known_bbox_ = (
            known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
        )
        known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
        known_bbox_expand[:, :2] = (
            known_bbox_[:, :2] + known_bbox_[:, 2:]
        ) / 2
        known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

    m = known_labels_expaned.long().to("cuda")
    input_label_embed = label_enc(m)
    input_bbox_embed = inverse_sigmoid(known_bbox_expand)

    padding_label = torch.zeros(pad_size, hidden_dim).cuda()
    padding_bbox = torch.zeros(pad_size, 4).cuda()

    input_query_label = padding_label.repeat(batch_size, 1, 1)
    input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

    map_known_indice = torch.tensor([]).to("cuda")
    if len(known_num):
        map_known_indice = torch.cat(
            [torch.tensor(range(num)) for num in known_num]
        )  # [0,1,0,1,2]
        map_known_indice = torch.cat(
            [
                map_known_indice + single_padding * i
                for i in range(2 * dn_number)
            ]
        ).long()
    # print("dtype", input_query_bbox.dtype, input_bbox_embed.dtype)
    if len(known_bid):
        input_query_label[
            (known_bid.long(), map_known_indice)
        ] = input_label_embed
        input_query_bbox[
            (known_bid.long(), map_known_indice)
        ] = input_bbox_embed

    tgt_size = pad_size + num_queries
    attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
    # match query cannot see the reconstruct
    attn_mask[pad_size:, :pad_size] = True
    # reconstruct cannot see each other
    for i in range(dn_number):
        if i == 0:
            attn_mask[
                single_padding * 2 * i : single_padding * 2 * (i + 1),
                single_padding * 2 * (i + 1) : pad_size,
            ] = True
        if i == dn_number - 1:
            attn_mask[
                single_padding * 2 * i : single_padding * 2 * (i + 1),
                : single_padding * i * 2,
            ] = True
        else:
            attn_mask[
                single_padding * 2 * i : single_padding * 2 * (i + 1),
                single_padding * 2 * (i + 1) : pad_size,
            ] = True
            attn_mask[
                single_padding * 2 * i : single_padding * 2 * (i + 1),
                : single_padding * 2 * i,
            ] = True

    dn_meta = {
        "single_padding": single_padding * 2,
        "dn_num": dn_number,
    }

    return input_query_label, input_query_bbox, attn_mask, dn_meta


@torch.jit.unused
def _set_aux_loss(outputs_class, outputs_coord):
    # this is a workaround to make torchscript happy, as torchscript
    # doesn't support dictionary with non-homogeneous values, such
    # as a dict having both a Tensor and a list.
    return [
        {"pred_logits": a, "pred_boxes": b}
        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
    ]


def dn_post_process(
    outputs_class: torch.Tensor,
    outputs_coord: torch.Tensor,
    dn_metas: dict,
    aux_loss: bool,
):
    """Post-processes outputs from Contrastive Denoising Training (CDN).

    Args:
        outputs_class: Tensor containing class predictions.
        outputs_coord: Tensor containing bounding box predictions.
        dn_metas: Metadata dictionary from CDN training.
        aux_loss: Boolean indicating whether to calculate auxiliary losses.

    Returns:
        outputs_class: Processed class prediction tensor.
        outputs_coord: Processed bounding box prediction tensor.
    """
    if dn_metas and dn_metas["single_padding"] > 0:
        padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
        output_known_class = outputs_class[:, :, :padding_size, :]
        output_known_coord = outputs_coord[:, :, :padding_size, :]
        outputs_class = outputs_class[:, :, padding_size:, :]
        outputs_coord = outputs_coord[:, :, padding_size:, :]

        out = {
            "pred_logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(
                output_known_class, output_known_coord
            )
        dn_metas["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord
