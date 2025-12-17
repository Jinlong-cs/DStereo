import torch

from hat.utils.apply_func import _as_list

__all__ = [
    "reorder_first_dim",
    "get_vectors_from_vcs_box",
    "ct_dist_nms",
    "do_ct_dist_nms",
]


def reorder_first_dim(input_data, a):
    """Reorder data from [a*b,...] to [b*a,...]."""
    if a == 1:
        return input_data
    a_b = input_data.shape[0]
    assert a_b % a == 0
    b = int(a_b / a)

    _input_reshape = input_data.reshape(a, b, -1)
    _input_transpose = _input_reshape.transpose(0, 1)
    output_data = _input_transpose.reshape(input_data.shape)
    return output_data


def get_vectors_from_vcs_box(
    wh: torch.tensor, rot: torch.tensor
) -> torch.tensor:
    """Compute the unit vectors along the wh directions.

    Args:
        wh: the length and with of predicted box.
            (order (length, width))
        rot: the rotation angle of predicted box.

    Returns:
        unit vector of length and witdth
    """
    w, h = wh
    v_w = torch.stack([w * torch.cos(rot), -w * torch.sin(rot)])
    v_h = torch.stack([h * torch.sin(rot), h * torch.cos(rot)])
    v_w_norm = torch.norm(v_w)
    v_h_norm = torch.norm(v_h)
    u_w = v_w / v_w_norm
    u_h = v_h / v_h_norm

    return u_w, u_h


def ct_dist_nms(
    pred_scores: torch.tensor,
    pred_cts: torch.tensor,
    pred_whs: torch.tensor,
    pred_rots: torch.tensor,
    scale: float = 1.0,
    rot_match_threshold: float = -1,
    kernel_from_singlebox: bool = True,
):
    """Center distance nms.

    The scheme and results refer to
    https://horizonrobotics.feishu.cn/wiki/wikcnQfB8GaOiYrxd9t5me7KbJg.

    pred_scores: Scores of predicted boxes.
    pred_cts: Centers of predicted boxes.
    pred_whs: Width and height of predicted boxes.
    pred_rots: Rotataion angels of predicted boxes.
    scale: Center distance scale.
    rot_match_threshold: Rotataion angel match threshold,
        for group predicted boxes.
    kernel_from_singlebox: Whether to use multiple pred boxes
        to get kernel size.

    Returns:
        Tensor: Keeped indices of predicted boxes.
    """
    scores = pred_scores.clone()
    cts = pred_cts.clone()
    whs = pred_whs.clone()
    rots = pred_rots.clone()
    keep = torch.ones(rots.shape[0], dtype=torch.bool).to(rots.device)
    if rot_match_threshold > 0:
        # use rot difference to group preds
        rot_diff = torch.abs(rots.unsqueeze(dim=1) - rots.unsqueeze(dim=0))
        num_pred = rots.shape[0]
        # get grouped preds, shape is numpred x numpred
        valid_rot_match = torch.triu(
            rot_diff <= rot_match_threshold, diagonal=0
        )
    else:
        valid_rot_match = torch.ones_like(keep).unsqueeze(dim=0)
        num_pred = 1
    for i in range(num_pred):
        if not keep[i]:
            continue
        match_ind = valid_rot_match[i]
        # if no other pred with lower score matches this pred, continue
        if match_ind.sum() <= 1:
            continue
        # do ctnms in same matched group
        match_whs = whs[match_ind]
        match_rots = rots[match_ind]
        match_cts = cts[match_ind]
        match_socres = scores[match_ind]
        match_ind_idx = torch.where(match_ind > 0)[0]
        orders = match_socres.clone()
        while orders.numel() > 0:
            u_w, u_h = get_vectors_from_vcs_box(
                match_whs[0, :2], match_rots[0]
            )
            if kernel_from_singlebox:
                # get kernel size from highest score pred
                kernel_m = match_whs[:1, :2] * scale
            else:
                # get kernel size from highest score pred and its matched
                kernel_m = match_whs[:1, :2] * scale + match_whs[1:, :2] * (
                    1 - scale
                )

            ct_dist = match_cts[0, :] - match_cts[1:, :]
            ct_dist_w = (ct_dist * u_w[None]).sum(dim=-1, keepdim=True)
            ct_dist_h = (ct_dist * u_h[None]).sum(dim=-1, keepdim=True)
            ct_dist_orien = torch.cat([ct_dist_w, ct_dist_h], dim=-1).abs()
            remain_idxs = (ct_dist_orien > kernel_m).any(dim=1)
            orders = orders[1:][remain_idxs]
            match_cts = match_cts[1:][remain_idxs]
            match_whs = match_whs[1:][remain_idxs]
            match_rots = match_rots[1:][remain_idxs]
            keep[match_ind_idx[1:][~remain_idxs]] = False
            match_ind_idx = match_ind_idx[1:][remain_idxs]

    return keep


def do_ct_dist_nms(
    pred: dict,
    ct_dist_scale: float = 1.0,
    class_ids: list = -1,
    rot_match_threshold: float = -1.0,
    kernel_from_singlebox: bool = True,
) -> dict:
    """Do ct_dist_nms.

    Args:
        pred: the data of predicted box.
        ct_dist_scale: scale to compute kernel
            size, default 1.0.
        class_ids: class to do ctnms
            deafault -1.
        rot_match_threshold: rot diff threshold to group preds,
            only do ctnms in same group, default -1.
        kernel_from_singlebox: whether to use multiple pred boxes
            to get kernel size in ctnms.

    Returns:
        the nms result of pred.
    """
    batch_size = len(pred["pred_bev_discobj_score"])
    for bs_idx in range(batch_size):
        pred_scores = pred["pred_bev_discobj_score"][bs_idx].clone()
        pred_cts = pred["pred_bev_discobj_ct"][bs_idx].clone()
        pred_whs = pred["pred_bev_discobj_wh"][bs_idx].clone()
        pred_rots = pred["pred_bev_discobj_rot"][bs_idx].clone()
        if class_ids == -1:
            # do ctnms between all classes
            keep = ct_dist_nms(
                pred_scores,
                pred_cts,
                pred_whs,
                pred_rots,
                scale=ct_dist_scale,
                rot_match_threshold=rot_match_threshold,
                kernel_from_singlebox=kernel_from_singlebox,
            )
        else:
            # do ctnms in same class
            class_ids = _as_list(class_ids)
            keep = torch.ones(pred_rots.shape[0], dtype=torch.bool).to(
                pred_rots.device
            )
            pred_class_ids = pred["pred_bev_discobj_cls_id"][bs_idx].clone()
            for cls_id in class_ids:
                ind = pred_class_ids == cls_id
                if ind.sum() == 0:
                    continue
                ind_cls_idx = torch.where(ind > 0)[0]
                pred_cls_rots = pred_rots[ind]
                pred_cls_whs = pred_whs[ind]
                pred_cls_cts = pred_cts[ind]
                pred_cls_socres = pred_scores[ind]
                keep_cls = ct_dist_nms(
                    pred_cls_socres,
                    pred_cls_cts,
                    pred_cls_whs,
                    pred_cls_rots,
                    scale=ct_dist_scale,
                    rot_match_threshold=rot_match_threshold,
                    kernel_from_singlebox=kernel_from_singlebox,
                )
                keep[ind_cls_idx[~keep_cls]] = False
        # update pred
        for key in pred.keys():
            if key == "pred_bev_discobj_cls_id":
                tmp_data = torch.ones_like(pred[key][bs_idx]) * -1
            else:
                tmp_data = torch.zeros_like(pred[key][bs_idx])
            tmp_data[keep] = pred[key][bs_idx][keep]
            pred[key][bs_idx] = tmp_data

    return pred
